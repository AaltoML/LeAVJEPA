import argparse
import csv
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

from configs import audio_config, transformer_config, video_config
from data import CLIP_SECONDS, HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE
from models import Echo
from probe_heads import VJEPAProbe

AUDIO_LEN = SAMPLE_RATE * CLIP_SECONDS
NUM_CLASSES = 50

_spec_transform = torch.nn.Sequential(
    torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT,
        win_length=N_FFT, hop_length=HOP_LENGTH, window_fn=torch.hamming_window,
    ),
    torchaudio.transforms.AmplitudeToDB(),
)


def load_encoder(checkpoint_path, vit_size, gradient_checkpointing=False):
    encoder = Echo(audio_config(), video_config(), transformer_config(vit_size),
                   gradient_checkpointing=gradient_checkpointing)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    enc_state = {}
    for k, v in state.items():
        if k.startswith("encoder."):
            k = k[len("encoder."):]
            enc_state[k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k] = v
    encoder.load_state_dict(enc_state, strict=True)
    return encoder


class ESC50Raw(Dataset):

    def __init__(self, root):
        self.root = root
        with open(os.path.join(root, "meta", "esc50.csv")) as f:
            self.items = [(row["filename"], int(row["fold"]), int(row["target"]))
                          for row in csv.DictReader(f)]
        assert len(self.items) == 2000, f"expected 2000 clips, got {len(self.items)}"

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        fname, fold, target = self.items[i]
        wav, sr = torchaudio.load(os.path.join(self.root, "audio", fname))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        if wav.shape[1] < AUDIO_LEN:
            wav = wav.repeat(1, AUDIO_LEN // wav.shape[1] + 1)
        wav = wav[:, :AUDIO_LEN]
        return _spec_transform(wav), target, fold


def cache_mels(root, num_workers):
    loader = DataLoader(ESC50Raw(root), batch_size=32, num_workers=num_workers)
    mels, labels, folds = [], [], []
    for spec, target, fold in loader:
        mels.append(spec)
        labels.append(target)
        folds.append(fold)
    return torch.cat(mels), torch.cat(labels), torch.cat(folds)


@torch.no_grad()
def extract_tokens(encoder, mels, mean, std, device, batch_size):
    feats = []
    for i in range(0, mels.shape[0], batch_size):
        spec = ((mels[i:i + batch_size] - mean) / std).to(device)
        _, patches = encoder(None, spec, return_patches=True)
        feats.append(patches.half().cpu())
    return torch.cat(feats)


def fit_attentive_probe(train_x, train_y, test_x, test_y, seed, epochs, lr, wd,
                        depth, batch_size, device):
    torch.manual_seed(seed)
    dim = train_x.shape[-1]
    num_heads = {768: 12, 1024: 16}[dim]
    head = VJEPAProbe(dim, num_heads, NUM_CLASSES, depth=depth).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    n = train_x.shape[0]
    total_steps = max(1, math.ceil(n / batch_size) * epochs)
    warmup = max(1, int(total_steps * 0.05))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (s + 1) / warmup if s < warmup
        else 0.5 * (1 + np.cos(np.pi * min(1.0, (s - warmup) / max(1, total_steps - warmup)))),
    )
    train_x, train_y = train_x.to(device), train_y.to(device)
    g = torch.Generator().manual_seed(seed + 1)
    head.train()
    for _ in range(epochs):
        order = torch.randperm(n, generator=g)
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = head(train_x[idx])
                loss = F.cross_entropy(logits.float(), train_y[idx], label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    head.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, test_x.shape[0], batch_size):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = head(test_x[i:i + batch_size].to(device)).float()
            correct += (logits.argmax(dim=1).cpu() == test_y[i:i + batch_size]).sum().item()
    return correct / test_x.shape[0]


def main():
    p = argparse.ArgumentParser(description="ESC-50 frozen attentive probe")
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--vit_size", type=str, default="base", choices=["base", "large"])
    p.add_argument("--esc50_root", type=str, default=os.environ.get("ESC50_ROOT", "data/ESC-50"))
    p.add_argument("--batch_size", type=int, default=32, help="feature-extraction batch")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--num_seeds", type=int, default=3)
    p.add_argument("--probe_epochs", type=int, default=100)
    p.add_argument("--probe_lr", type=float, default=1e-3)
    p.add_argument("--probe_wd", type=float, default=1e-4)
    p.add_argument("--probe_depth", type=int, default=4)
    p.add_argument("--probe_batch", type=int, default=128)
    p.add_argument("--folds", type=str, default="1,2,3,4,5")
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    device = torch.device("cuda")
    encoder = load_encoder(args.checkpoint_path, args.vit_size).eval().to(device)
    mels, labels, folds = cache_mels(args.esc50_root, args.num_workers)

    fold_ids = [int(f) for f in args.folds.split(",")]
    per_fold = {}
    for fold in fold_ids:
        tr, te = folds != fold, folds == fold
        mean, std = mels[tr].mean().item(), mels[tr].std().item()
        feats = extract_tokens(encoder, mels, mean, std, device, args.batch_size)
        accs = [fit_attentive_probe(feats[tr], labels[tr], feats[te], labels[te], seed,
                                    args.probe_epochs, args.probe_lr, args.probe_wd,
                                    args.probe_depth, args.probe_batch, device)
                for seed in range(args.num_seeds)]
        per_fold[fold] = dict(accs=accs, spec_mean=round(mean, 4), spec_std=round(std, 4))
        print(f"fold {fold}: acc={np.mean(accs):.4f} "
              f"(seeds: {', '.join(f'{a:.4f}' for a in accs)})", flush=True)

    seed_means = [np.mean([per_fold[f]["accs"][s] for f in fold_ids])
                  for s in range(args.num_seeds)]
    result = dict(
        mean_acc=round(float(np.mean(seed_means)), 4),
        std_across_seeds=round(float(np.std(seed_means)), 4),
        per_fold=per_fold,
        args=vars(args),
    )
    print(json.dumps(result, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
