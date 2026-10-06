import os

import torch
import pandas as pd
import torchaudio
import torchaudio.transforms as T
import webdataset as wds
from torch import nn
from torch.utils.data import DataLoader
from torchcodec.decoders import AudioDecoder, VideoDecoder
from torchvision.transforms import v2

SAMPLE_RATE = 16000
HOP_LENGTH = 160
N_FFT = 400
N_MELS = 128
FPS = 25
CLIP_SECONDS = 8


RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]


VGGSOUND_SPEC_MEAN = -20.437003
VGGSOUND_SPEC_STD = 24.496246


class BlockTubeMasking(nn.Module):

    def __init__(self, patch_size=(2, 16, 16), mask_ratio=0.85, mask_val=0.0):
        super().__init__()
        self.patch_t, self.patch_h, self.patch_w = patch_size
        self.mask_ratio = mask_ratio
        self.mask_val = mask_val

    def forward(self, x):
        if self.mask_ratio <= 0:
            return x

        single = x.dim() == 4
        if single:
            x = x.unsqueeze(0)

        B, C, T_, H, W = x.shape
        grid_t = T_ // self.patch_t
        grid_h = H // self.patch_h
        grid_w = W // self.patch_w
        target_area = self.mask_ratio * grid_h * grid_w

        spatial_mask = torch.zeros(B, grid_h, grid_w, device=x.device)
        for b in range(B):
            log_r = torch.empty(1).uniform_(-0.5, 0.5).item()
            r = torch.tensor(log_r).exp().item()
            bh = max(1, min(grid_h, round((target_area * r) ** 0.5)))
            bw = max(1, min(grid_w, round((target_area / r) ** 0.5)))
            h0 = torch.randint(0, max(1, grid_h - bh + 1), (1,)).item()
            w0 = torch.randint(0, max(1, grid_w - bw + 1), (1,)).item()
            spatial_mask[b, h0 : h0 + bh, w0 : w0 + bw] = 1.0

        mask = spatial_mask.unsqueeze(1).expand(B, grid_t, grid_h, grid_w).contiguous()
        mask = (
            mask.repeat_interleave(self.patch_t, dim=1)
            .repeat_interleave(self.patch_h, dim=2)
            .repeat_interleave(self.patch_w, dim=3)
        )
        if mask.shape[1:] != (T_, H, W):
            mask = torch.nn.functional.pad(
                mask,
                (0, W - mask.shape[3], 0, H - mask.shape[2], 0, T_ - mask.shape[1]),
                value=0,
            )

        mask = mask.unsqueeze(1).expand_as(x)
        result = x * (1 - mask) + (mask * self.mask_val)
        return result.squeeze(0) if single else result


class VideoAudioPipeline:
    def __init__(
        self,
        label_csv_path,
        is_train=True,
        debug=False,
        frame_size=(224, 224),
        num_frames=16,
        num_global_views=2,
        num_local_views=2,
        num_eval_clips=4,
        classes=None,
        video_mask_ratio=0.80,
        freq_mask_param=64,
        time_mask_param=256,
        modality_drop_prob=0.5,
        spec_aug_global=False,
        global_rrc_min_scale=0.0,
        csv_format="vggsound",
        spec_mean=VGGSOUND_SPEC_MEAN,
        spec_std=VGGSOUND_SPEC_STD,
        audio_only=False,
        return_keys=False,
        cross_modal=False,
        clean_survivor=False,
    ):
        self.is_train = is_train
        self.debug = debug
        self.return_keys = return_keys
        self.audio_only = audio_only
        self.sample_rate = SAMPLE_RATE
        self.fps = FPS
        self.frame_size = frame_size
        self.num_frames = num_frames
        self.num_global_views = num_global_views
        self.num_local_views = num_local_views
        self.num_eval_clips = num_eval_clips
        self.spec_mean = spec_mean
        self.spec_std = spec_std
        self.csv_format = csv_format
        self.audio_len = self.sample_rate * CLIP_SECONDS

        self.classes = []
        if csv_format == "audioset":
            self.labels_map = self._load_audioset_labels_map(label_csv_path, classes=classes)
        else:
            self.labels_map = self._load_labels_map(label_csv_path, classes=classes)

        self.spectrogram_transform = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_mels=N_MELS,
                n_fft=N_FFT,
                win_length=N_FFT,
                hop_length=HOP_LENGTH,
                window_fn=torch.hamming_window,
            ),
            torchaudio.transforms.AmplitudeToDB(),
        )

        if global_rrc_min_scale > 0:
            global_aug = [
                v2.RandomResizedCrop(frame_size, scale=(global_rrc_min_scale, 1.0), antialias=True),
                v2.RandomHorizontalFlip(),
            ]
        else:
            global_aug = [
                v2.Resize(256, antialias=True),
                v2.CenterCrop(frame_size),
                v2.RandomHorizontalFlip(),
            ]
        to_tensor = [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=RGB_MEAN, std=RGB_STD)]
        self.global_video_transform = v2.Compose(global_aug + to_tensor)
        self.local_video_transform = v2.Compose(
            [
                v2.RandomResizedCrop(frame_size, scale=(0.4, 1.0), antialias=True),
                v2.RandomHorizontalFlip(),
            ]
            + to_tensor
        )
        self.val_video_transform = v2.Compose(
            [v2.Resize(frame_size, antialias=True), v2.CenterCrop(frame_size)] + to_tensor
        )

        self.audio_aug = torch.nn.Sequential(
            T.FrequencyMasking(freq_mask_param=freq_mask_param),
            T.TimeMasking(time_mask_param=time_mask_param),
        )
        self.video_mask = BlockTubeMasking(patch_size=(2, 16, 16), mask_ratio=video_mask_ratio)
        self.modality_drop_prob = modality_drop_prob
        self.spec_aug_global = spec_aug_global
        self.cross_modal = cross_modal
        self.clean_survivor = clean_survivor
        self._resamplers = {}

    def _load_labels_map(self, label_csv_path, classes=None):
        df = pd.read_csv(label_csv_path, header=None, names=["filename", "label"])
        self.classes = classes if classes is not None else sorted(df["label"].unique())
        label_to_idx = {label: i for i, label in enumerate(self.classes)}
        keys = df["filename"].astype(str).apply(lambda x: os.path.splitext(x)[0])
        labels = df["label"].map(label_to_idx)
        return dict(zip(keys, labels))

    def _load_audioset_labels_map(self, csv_path, classes=None):
        rows = []
        with open(csv_path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split(", ", 3)
                start_ms = round(float(parts[1]) * 1000)
                end_ms = round(float(parts[2]) * 1000)
                rows.append((f"{parts[0]}_{start_ms}_{end_ms}", parts[3].strip('"').split(",")))

        if classes is not None:
            self.classes = classes
        else:
            self.classes = sorted({lid for _, label_ids in rows for lid in label_ids})
        label_to_idx = {label: i for i, label in enumerate(self.classes)}

        labels_map = {}
        for key, label_ids in rows:
            multi_hot = torch.zeros(len(self.classes), dtype=torch.float32)
            for lid in label_ids:
                if lid in label_to_idx:
                    multi_hot[label_to_idx[lid]] = 1.0
            labels_map[key] = multi_hot
        return labels_map

    def _get_normalized_key(self, sample):
        return os.path.basename(sample["__key__"])

    def has_label(self, sample):
        return self._get_normalized_key(sample) in self.labels_map

    def _decode_clips_batched(self, video_decoder, total_frames, stride, start_indices):
        all_indices, clip_lengths = [], []
        for si in start_indices:
            ideal = si + torch.arange(self.num_frames) * stride
            valid = ideal[ideal < total_frames]
            all_indices.append(valid)
            clip_lengths.append(len(valid))

        all_frames = video_decoder.get_frames_at(indices=torch.cat(all_indices)).data

        clips, offset = [], 0
        for length in clip_lengths:
            clip = all_frames[offset : offset + length]
            frames_needed = self.num_frames - clip.shape[0]
            if frames_needed > 0:
                padding = torch.zeros(frames_needed, *clip.shape[1:], dtype=clip.dtype)
                clip = torch.cat([clip, padding], dim=0)
            clips.append(clip)
            offset += length
        return clips

    def _resample_waveform(self, waveform, original_sample_rate):
        if original_sample_rate != self.sample_rate:
            if original_sample_rate not in self._resamplers:
                self._resamplers[original_sample_rate] = torchaudio.transforms.Resample(
                    original_sample_rate, self.sample_rate
                )
            waveform = self._resamplers[original_sample_rate](waveform)
        return waveform

    def _slice_audio(self, waveform, start_idx):
        start_sample = int((start_idx / self.fps) * self.sample_rate)
        end_sample = start_sample + self.audio_len
        if waveform.shape[1] >= end_sample:
            return waveform[:, start_sample:end_sample]
        clip = waveform[:, start_sample:]
        if clip.shape[1] < self.audio_len:
            clip = torch.nn.functional.pad(clip, (0, self.audio_len - clip.shape[1]))
        return clip[:, : self.audio_len]

    def _make_spectrogram(self, audio):
        current_len = audio.shape[1]
        if current_len > self.audio_len:
            audio = audio[:, : self.audio_len]
        elif current_len < self.audio_len:
            audio = torch.nn.functional.pad(audio, (0, self.audio_len - current_len))
        spec = self.spectrogram_transform(audio)
        spec = (spec - self.spec_mean) / self.spec_std
        return audio, spec

    @torch.no_grad()
    def process(self, sample):
        key = self._get_normalized_key(sample)
        if key not in self.labels_map:
            return None
        video_bytes = sample.get("mp4")
        if video_bytes is None:
            return None

        try:
            video_decoder = VideoDecoder(video_bytes, device="cpu")
            total_frames = len(video_decoder)
            stride = int((CLIP_SECONDS * self.fps) / self.num_frames)
            max_start_video = max(0, total_frames - (stride * self.num_frames))

            audio_samples = AudioDecoder(video_bytes).get_all_samples()
            waveform = audio_samples.data
            original_sample_rate = audio_samples.sample_rate
            del audio_samples
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            waveform = self._resample_waveform(waveform, original_sample_rate)

            max_start_audio = max(
                0, int((waveform.shape[1] - self.audio_len) / self.sample_rate * self.fps)
            )
            max_start = min(max_start_video, max_start_audio)
            label = self.labels_map[key]
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label)

            if self.is_train:
                result = self._process_train(
                    video_decoder, total_frames, stride, max_start, waveform, label
                )
            else:
                result = self._process_val(
                    video_decoder, total_frames, stride, max_start, waveform, label
                )
                if self.return_keys:
                    result["key"] = key
            del video_decoder
            return result
        except Exception as e:
            print(f"Skipping corrupt sample {sample.get('__key__', '?')}: {e}")
            return None

    def _process_train(self, video_decoder, total_frames, stride, max_start, waveform, label):
        n_total = self.num_global_views + self.num_local_views
        if max_start > 0:
            all_start_indices = torch.randint(0, max_start + 1, (n_total,)).tolist()
        else:
            all_start_indices = [0] * n_total
        global_starts = all_start_indices[: self.num_global_views]
        local_starts = all_start_indices[self.num_global_views :]

        if self.audio_only:
            return self._process_train_audio_only(waveform, global_starts, local_starts, label)

        decoded_clips = self._decode_clips_batched(
            video_decoder, total_frames, stride, all_start_indices
        )
        global_clips = decoded_clips[: self.num_global_views]
        local_clips = decoded_clips[self.num_global_views :]

        global_videos, global_specs = [], []
        for raw_video, si in zip(global_clips, global_starts):
            global_videos.append(self.global_video_transform(raw_video).permute(1, 0, 2, 3))
            _, spec = self._make_spectrogram(self._slice_audio(waveform, si))
            if self.spec_aug_global:
                spec = self.audio_aug(spec)
            global_specs.append(spec)
        global_videos = torch.stack(global_videos)
        global_specs = torch.stack(global_specs)

        if self.num_local_views == 0:

            return {
                "global_video": global_videos,
                "global_spectrogram": global_specs,
                "local_video": global_videos.new_zeros((0, *global_videos.shape[1:])),
                "local_spectrogram": global_specs.new_zeros((0, *global_specs.shape[1:])),
                "label": label,
            }

        if self.cross_modal:

            local_video_list, local_spec_list = [], []
            for idx, (raw_video, si) in enumerate(zip(local_clips, local_starts)):
                lv = self.local_video_transform(raw_video).permute(1, 0, 2, 3)
                _, spec = self._make_spectrogram(self._slice_audio(waveform, si))
                if idx % 2 == 0:
                    local_video_list.append(torch.zeros_like(lv))
                    local_spec_list.append(spec)
                else:
                    local_video_list.append(lv)
                    local_spec_list.append(torch.zeros_like(spec))
            local_videos = torch.stack(local_video_list)
            local_specs = torch.stack(local_spec_list)
        else:

            local_views, local_spec_list = [], []
            for raw_video, si in zip(local_clips, local_starts):
                local_views.append(self.local_video_transform(raw_video).permute(1, 0, 2, 3))
                _, spec = self._make_spectrogram(self._slice_audio(waveform, si))
                local_spec_list.append(self.audio_aug(spec))
            lv = torch.stack(local_views)
            local_specs_raw = torch.stack(local_spec_list)

            if self.modality_drop_prob > 0:

                K = lv.shape[0]
                rolls = torch.rand(K)
                half_p = self.modality_drop_prob / 2
                drop_video = rolls < half_p
                drop_audio = (rolls >= half_p) & (rolls < self.modality_drop_prob)

                if self.clean_survivor and drop_video.any():
                    local_specs_raw[drop_video] = torch.stack(
                        [
                            self._make_spectrogram(self._slice_audio(waveform, local_starts[k]))[1]
                            for k in range(K)
                            if drop_video[k]
                        ]
                    )
                local_videos = self.video_mask(lv)
                local_specs = local_specs_raw.clone()
                if self.clean_survivor and drop_audio.any():
                    local_videos[drop_audio] = lv[drop_audio]
                for k in range(K):
                    if drop_video[k]:
                        local_videos[k] = 0.0
                    elif drop_audio[k]:
                        local_specs[k] = 0.0
            else:
                local_videos = self.video_mask(lv)
                local_specs = local_specs_raw

        return {
            "global_video": global_videos,
            "global_spectrogram": global_specs,
            "local_video": local_videos,
            "local_spectrogram": local_specs,
            "label": label,
        }

    def _process_train_audio_only(self, waveform, global_starts, local_starts, label):
        zero_video = torch.zeros(3, self.num_frames, *self.frame_size)
        global_specs = []
        for si in global_starts:
            _, spec = self._make_spectrogram(self._slice_audio(waveform, si))
            if self.spec_aug_global:
                spec = self.audio_aug(spec)
            global_specs.append(spec)
        local_specs = []
        for si in local_starts:
            _, spec = self._make_spectrogram(self._slice_audio(waveform, si))
            local_specs.append(self.audio_aug(spec))
        return {
            "global_video": zero_video.expand(self.num_global_views, -1, -1, -1, -1).clone(),
            "global_spectrogram": torch.stack(global_specs),
            "local_video": zero_video.expand(self.num_local_views, -1, -1, -1, -1).clone(),
            "local_spectrogram": torch.stack(local_specs),
            "label": label,
        }

    def _process_val(self, video_decoder, total_frames, stride, max_start, waveform, label):
        if max_start > 0:
            start_indices = torch.randint(0, max_start + 1, (self.num_eval_clips,)).tolist()
        else:
            start_indices = [0] * self.num_eval_clips

        if self.audio_only:
            decoded_clips = [None] * len(start_indices)
        else:
            decoded_clips = self._decode_clips_batched(
                video_decoder, total_frames, stride, start_indices
            )

        videos, specs, waveforms = [], [], []
        for raw_v, si in zip(decoded_clips, start_indices):
            if raw_v is None:
                videos.append(torch.zeros(3, self.num_frames, *self.frame_size))
            else:
                videos.append(self.val_video_transform(raw_v).permute(1, 0, 2, 3))
            a, s = self._make_spectrogram(self._slice_audio(waveform, si))
            specs.append(s)
            waveforms.append(a)

        result = {
            "video": torch.stack(videos),
            "spectrogram": torch.stack(specs),
            "label": label,
        }
        if not self.debug:
            result["waveform"] = torch.stack(waveforms)
        return result


def compute_audioset_pos_weight(csv_path, classes):
    label_to_idx = {label: i for i, label in enumerate(classes)}
    pos_counts = torch.zeros(len(classes), dtype=torch.float32)
    n_samples = 0
    with open(csv_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            for lid in line.strip().split(", ", 3)[3].strip('"').split(","):
                if lid in label_to_idx:
                    pos_counts[label_to_idx[lid]] += 1.0
            n_samples += 1
    return (n_samples - pos_counts) / pos_counts.clamp(min=1.0)


def get_dataloader(
    tar_path,
    csv_path,
    test_tar_path=None,
    test_csv_path=None,
    debug=False,
    batch_size=64,
    num_workers=2,
    num_workers_test=2,
    frame_size=(224, 224),
    num_frames=16,
    num_global_views=2,
    num_local_views=2,
    num_eval_clips=4,
    train_size=None,
    test_size=None,
    video_mask_ratio=0.80,
    freq_mask_param=64,
    time_mask_param=256,
    modality_drop_prob=0.5,
    clean_survivor=False,
    cross_modal=False,
    world_size=1,
    spec_aug_global=False,
    global_rrc_min_scale=0.0,
    csv_format="vggsound",
    spec_mean=VGGSOUND_SPEC_MEAN,
    spec_std=VGGSOUND_SPEC_STD,
    audio_only=False,
    return_test_keys=False,
    prefetch_factor=3,
):
    if prefetch_factor < 1:
        raise ValueError("prefetch_factor must be >= 1")

    def create_loader(tars, pipeline, is_train, num_samples):
        dataset = wds.WebDataset(
            tars,
            shardshuffle=100 if is_train else False,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
            empty_check=is_train,
        )
        if is_train:
            dataset = dataset.shuffle(400)
        dataset = (
            dataset.select(pipeline.has_label)
            .map(pipeline.process)
            .select(lambda x: x is not None)
            .batched(batch_size)
        )
        if num_samples is not None:
            num_batches = int(num_samples // batch_size // world_size)
            dataset = dataset.with_epoch(num_batches).with_length(num_batches)

        n_workers = num_workers if is_train else num_workers_test
        loader_kwargs = {
            "batch_size": None,
            "num_workers": n_workers,
            "persistent_workers": False,
            "pin_memory": True,
        }
        if n_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor
        return DataLoader(dataset, **loader_kwargs)

    print(
        f"Data: batch={batch_size} workers={num_workers}/{num_workers_test} "
        f"frames={num_frames} G={num_global_views} K={num_local_views} "
        f"cross_modal={cross_modal} clean_survivor={clean_survivor} "
        f"modality_drop={modality_drop_prob} video_mask={video_mask_ratio} "
        f"freq/time_mask={freq_mask_param}/{time_mask_param} audio_only={audio_only}"
    )

    train_pipeline = VideoAudioPipeline(
        csv_path,
        is_train=True,
        debug=debug,
        frame_size=frame_size,
        num_frames=num_frames,
        num_global_views=num_global_views,
        num_local_views=num_local_views,
        video_mask_ratio=video_mask_ratio,
        freq_mask_param=freq_mask_param,
        time_mask_param=time_mask_param,
        modality_drop_prob=modality_drop_prob,
        spec_aug_global=spec_aug_global,
        global_rrc_min_scale=global_rrc_min_scale,
        csv_format=csv_format,
        spec_mean=spec_mean,
        spec_std=spec_std,
        audio_only=audio_only,
        cross_modal=cross_modal,
        clean_survivor=clean_survivor,
    )
    train_loader = create_loader(tar_path, train_pipeline, True, train_size)

    if test_tar_path and test_csv_path:
        test_pipeline = VideoAudioPipeline(
            test_csv_path,
            is_train=False,
            debug=debug,
            frame_size=frame_size,
            num_frames=num_frames,
            num_eval_clips=num_eval_clips,
            classes=train_pipeline.classes,
            csv_format=csv_format,
            spec_mean=spec_mean,
            spec_std=spec_std,
            audio_only=audio_only,
            return_keys=return_test_keys,
        )
        test_loader = create_loader(test_tar_path, test_pipeline, False, test_size)
        return train_loader, test_loader, train_pipeline.classes

    return train_loader, train_pipeline.classes
