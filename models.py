import lightning as L
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchmetrics

from fusion import AudioEmbeddings, EarlyFusionEmbeddings, VideoEmbeddings
from transformer import Encoder


def _init_vit_weights(module, std):
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d, nn.Embedding)):
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def _trunc_normal_(params, std):
    for param in params:
        param.data = nn.init.trunc_normal_(
            param.data.to(torch.float32), mean=0.0, std=std
        ).to(param.dtype)


class Echo(nn.Module):

    def __init__(self, a_config, v_config, t_config, gradient_checkpointing=False):
        super().__init__()
        self.t_config = t_config
        self.embedding = EarlyFusionEmbeddings(a_config, v_config, t_config)
        self.encoder = Encoder(t_config, gradient_checkpointing=gradient_checkpointing)
        self.apply(self._init_weights)

    def forward(self, video_x, audio_x, return_patches=False):
        x = self.encoder(self.embedding(video_x, audio_x))
        if return_patches:
            return x[:, 0], x[:, 1:]
        return x[:, 0]

    def _init_weights(self, module):
        std = self.t_config["initializer_range"]
        _init_vit_weights(module, std)
        if isinstance(module, EarlyFusionEmbeddings):
            _trunc_normal_(
                [
                    module.video_time_embed,
                    module.video_spatial_embed,
                    module.audio_freq_pos_embed,
                    module.audio_time_pos_embed,
                    module.cls_token,
                ],
                std,
            )


class DualEcho(nn.Module):

    def __init__(self, a_config, v_config, t_config, gradient_checkpointing=False):
        super().__init__()
        self.t_config = t_config
        self.audio_embedding = AudioEmbeddings(a_config, t_config)
        self.audio_encoder = torch.compile(
            Encoder(t_config, gradient_checkpointing=gradient_checkpointing)
        )
        self.video_embedding = VideoEmbeddings(v_config, t_config)
        self.video_encoder = torch.compile(
            Encoder(t_config, gradient_checkpointing=gradient_checkpointing)
        )
        self.apply(self._init_weights)

    def forward_audio(self, audio_x, return_patches=False):
        x = self.audio_encoder(self.audio_embedding(audio_x))
        return (x[:, 0], x[:, 1:]) if return_patches else x[:, 0]

    def forward_video(self, video_x, return_patches=False):
        x = self.video_encoder(self.video_embedding(video_x))
        return (x[:, 0], x[:, 1:]) if return_patches else x[:, 0]

    def forward(self, video_x, audio_x, return_patches=False):
        if return_patches:
            audio_cls, audio_patches = self.forward_audio(audio_x, return_patches=True)
            video_cls, video_patches = self.forward_video(video_x, return_patches=True)
            return (audio_cls + video_cls) / 2, torch.cat([video_patches, audio_patches], dim=1)
        return (self.forward_audio(audio_x) + self.forward_video(video_x)) / 2

    def _init_weights(self, module):
        std = self.t_config["initializer_range"]
        _init_vit_weights(module, std)
        if isinstance(module, (AudioEmbeddings, VideoEmbeddings)):
            _trunc_normal_(
                [
                    p
                    for name, p in module.named_parameters()
                    if "pos_embed" in name
                    or "time_embed" in name
                    or "spatial_embed" in name
                    or "freq" in name
                    or "cls_token" in name
                ],
                std,
            )


class SIGReg(torch.nn.Module):

    def __init__(self, knots=17, num_slices=2048):
        super().__init__()
        self.num_slices = num_slices
        t = torch.linspace(-5, 5, knots, dtype=torch.float32)
        window = torch.exp(-0.5 * t.square())
        self.register_buffer("t", t)
        self.register_buffer("phi", window)

    def forward(self, proj, global_step):
        dev = proj.device
        g = torch.Generator(device=dev)
        g.manual_seed(global_step)

        A = torch.randn(proj.size(-1), self.num_slices, generator=g, device=dev)
        A = A / A.norm(p=2, dim=0)
        x_t = (proj.float() @ A).unsqueeze(-1) * self.t
        cos_mean = x_t.cos().mean(dim=0)
        sin_mean = x_t.sin().mean(dim=0)

        if dist.is_initialized():
            dist.all_reduce(cos_mean, op=dist.ReduceOp.AVG)
            dist.all_reduce(sin_mean, op=dist.ReduceOp.AVG)

        err = (cos_mean - self.phi).square() + sin_mean.square()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        n_global = proj.size(0) * world_size

        statistic = torch.trapz(err * self.phi, self.t, dim=-1) * n_global
        return statistic.mean()


class Projector(nn.Module):

    def __init__(self, in_dim=768, hidden_dim=2048, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class AttentiveProbe(nn.Module):

    def __init__(self, hidden_size, num_heads, num_classes):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, patches):
        query = self.query.expand(patches.size(0), -1, -1)
        out, _ = self.cross_attn(query, patches, patches)
        return self.head(self.norm(out.squeeze(1)))


def modality_tokens(patches, num_video_tokens, modality):
    if num_video_tokens is None:
        return patches
    if modality == "video":
        return patches[:, :num_video_tokens]
    return patches[:, num_video_tokens:]


def _classification_metrics(multi_label, num_classes, names):
    metrics = {}
    for name in names:
        if multi_label:
            metrics[f"{name}_map"] = torchmetrics.AveragePrecision(
                task="multilabel", num_labels=num_classes
            )
        else:
            metrics[f"{name}_acc"] = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes
            )
            metrics[f"{name}_acc5"] = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes, top_k=5
            )
    return nn.ModuleDict(metrics)


class EchoTrainer(L.LightningModule):

    def __init__(
        self,
        a_config,
        v_config,
        t_config,
        lr=5e-4,
        weight_decay=5e-2,
        lambd=0.05,
        num_views=2,
        batch_size=32,
        epochs=50,
        proj_dim=128,
        num_classes: int = 309,
        probe_lr: float = 1e-3,
        probe_weight_decay: float = 0.0,
        wd_exclude_norm_embed: bool = False,
        cross_modal: bool = False,
        unimodal_token_drop: bool = False,
        gradient_checkpointing: bool = False,
        attentive_probe: bool = False,
        dual_encoder: bool = False,
        multi_label: bool = False,
        total_samples: int = 183_730,
        pos_weight: torch.Tensor = None,
        num_global_views: int = 2,
        no_probe: bool = False,
        **unused_hparams,
    ):

        super().__init__()
        self.save_hyperparameters(ignore=["pos_weight"])
        if unimodal_token_drop and (not cross_modal or dual_encoder):
            raise ValueError("unimodal_token_drop needs the shared-encoder cross_modal path")
        self.cross_modal = cross_modal
        self.unimodal_token_drop = unimodal_token_drop
        self.attentive_probe = attentive_probe
        self.dual_encoder = dual_encoder
        self.multi_label = multi_label
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

        if dual_encoder:
            self.encoder = DualEcho(
                a_config, v_config, t_config, gradient_checkpointing=gradient_checkpointing
            )
            self.num_video_tokens = None
        else:
            self.encoder = torch.compile(
                Echo(a_config, v_config, t_config, gradient_checkpointing=gradient_checkpointing),
                dynamic=False if unimodal_token_drop else None,
            )
            self.num_video_tokens = (v_config["num_frames"] // v_config["tubelet_size"]) * (
                v_config["image_size"] // v_config["patch_size"]
            ) ** 2
        self.projector = torch.compile(
            Projector(in_dim=t_config["hidden_size"], hidden_dim=2048, out_dim=proj_dim)
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.wd_exclude_norm_embed = wd_exclude_norm_embed
        self.lambd = lambd
        self.num_views = num_views
        self.total_samples = total_samples
        self.epochs = epochs
        self.sigreg = SIGReg()

        hidden_size = t_config["hidden_size"]
        self.probe_norm = torch.compile(nn.LayerNorm(hidden_size))
        self.probe_head = torch.compile(nn.Linear(hidden_size, num_classes))
        self.probe_lr = probe_lr
        self.probe_weight_decay = probe_weight_decay

        self.no_probe = bool(no_probe)
        if self.no_probe:
            self.probe_norm.requires_grad_(False)
            self.probe_head.requires_grad_(False)

        if multi_label:
            self.probe_criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        else:
            self.probe_criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
        self.probe_metrics = _classification_metrics(
            multi_label, num_classes, ["train", "test", "test_audio", "test_video"]
        )

        if attentive_probe:
            self.att_probe = torch.compile(
                AttentiveProbe(hidden_size, t_config["num_attention_heads"], num_classes)
            )
            if multi_label:
                self.att_probe_criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            else:
                self.att_probe_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            self.att_probe_metrics = _classification_metrics(
                multi_label, num_classes, ["train", "test", "test_audio", "test_video"]
            )

        if cross_modal:

            self.unimodal_probe_metrics = _classification_metrics(
                multi_label, num_classes, ["audio", "video"]
            )

    def forward(self, video, audio):
        return self.encoder(video, audio)

    def _update_and_log(self, metrics, prefix, logits, targets, key_prefix, **log_kw):
        if self.multi_label:
            m = metrics[f"{prefix}_map"]
            m(logits, targets.long())
            self.log(f"{key_prefix}map", m, sync_dist=True, **log_kw)
        else:
            m1, m5 = metrics[f"{prefix}_acc"], metrics[f"{prefix}_acc5"]
            m1(logits.argmax(dim=1), targets)
            m5(logits, targets)
            self.log(f"{key_prefix}acc", m1, sync_dist=True, **log_kw)
            self.log(f"{key_prefix}acc_top5", m5, sync_dist=True, **log_kw)

    def _linear_probe(self, features):
        return self.probe_head(self.probe_norm(features))

    def _unimodal_probe_loss(self, audio_cls, video_cls, labels):
        audio_logits = self._linear_probe(audio_cls)
        video_logits = self._linear_probe(video_cls)
        audio_loss = self.probe_criterion(audio_logits, labels)
        video_loss = self.probe_criterion(video_logits, labels)
        self.log("train/probe_audio_loss", audio_loss, sync_dist=True)
        self.log("train/probe_video_loss", video_loss, sync_dist=True)
        self._update_and_log(
            self.unimodal_probe_metrics, "audio", audio_logits, labels, "train/probe_audio_"
        )
        self._update_and_log(
            self.unimodal_probe_metrics, "video", video_logits, labels, "train/probe_video_"
        )
        return 0.5 * (audio_loss + video_loss)

    def _sigreg_mean(self, views):
        return torch.stack([self.sigreg(z, self.global_step) for z in views]).mean()

    def training_step(self, batch, batch_idx):
        global_video = batch["global_video"]
        global_spectrogram = batch["global_spectrogram"]
        local_video = batch["local_video"]
        local_spectrogram = batch["local_spectrogram"]
        labels = batch["label"]

        B, G = global_video.shape[:2]
        K = local_video.shape[1]
        global_video_flat = global_video.view(B * G, *global_video.shape[2:])
        global_spectrogram_flat = global_spectrogram.view(B * G, *global_spectrogram.shape[2:])

        if self.dual_encoder:
            if self.attentive_probe:
                audio_cls_g, audio_patches_g = self.encoder.forward_audio(
                    global_spectrogram_flat, return_patches=True
                )
                video_cls_g, video_patches_g = self.encoder.forward_video(
                    global_video_flat, return_patches=True
                )
                patches_global = torch.cat([video_patches_g, audio_patches_g], dim=1)
            else:
                audio_cls_g = self.encoder.forward_audio(global_spectrogram_flat)
                video_cls_g = self.encoder.forward_video(global_video_flat)
            cls_global = (audio_cls_g + video_cls_g) / 2
        elif self.attentive_probe:
            cls_global, patches_global = self.encoder(
                global_video_flat, global_spectrogram_flat, return_patches=True
            )
        else:
            cls_global = self.encoder(global_video_flat, global_spectrogram_flat)

        probe_loss = None
        if not self.no_probe:
            probe_features = cls_global.view(B, G, -1).mean(dim=1).detach()
            probe_logits = self._linear_probe(probe_features)
            probe_loss = self.probe_criterion(probe_logits, labels)
            self._update_and_log(self.probe_metrics, "train", probe_logits, labels, "train/probe_")
            self.log("train/probe_loss", probe_loss, sync_dist=True)
        att_probe_loss = None
        if self.attentive_probe:
            patches_for_att = (
                patches_global.view(B, G, *patches_global.shape[1:]).mean(dim=1).detach()
            )
            att_logits = self.att_probe(patches_for_att)
            att_probe_loss = self.att_probe_criterion(att_logits, labels)
            self._update_and_log(
                self.att_probe_metrics, "train", att_logits, labels, "train/att_probe_"
            )
            self.log("train/att_probe_loss", att_probe_loss, sync_dist=True)

        if self.dual_encoder:
            inv_loss, sigreg_loss, embed_std, unimodal_probe_loss = self._dual_encoder_loss(
                audio_cls_g, video_cls_g, local_video, local_spectrogram, labels, B, G, K
            )
        else:
            inv_loss, sigreg_loss, embed_std, unimodal_probe_loss = self._shared_encoder_loss(
                cls_global, local_video, local_spectrogram, labels, B, G, K
            )

        loss = (1 - self.lambd) * inv_loss + self.lambd * sigreg_loss

        self.log("train/embed_std", embed_std, sync_dist=True)
        self.log("train/lejepa_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/inv_loss", inv_loss, sync_dist=True)
        self.log("train/sigreg_loss", sigreg_loss, sync_dist=True)
        self.log("train/inv_loss_weighted", (1 - self.lambd) * inv_loss, sync_dist=True)
        self.log("train/sigreg_loss_weighted", self.lambd * sigreg_loss, sync_dist=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)

        total_loss = loss
        for extra in (probe_loss, unimodal_probe_loss, att_probe_loss):
            if extra is not None:
                total_loss = total_loss + extra
        self.log("train/joint_loss", total_loss, sync_dist=True)
        return total_loss

    def _shared_encoder_loss(self, cls_global, local_video, local_spectrogram, labels, B, G, K):
        z_global = self.projector(cls_global).view(B, G, -1).permute(1, 0, 2)

        if K == 0:

            z_local_cls = cls_global.new_zeros((0, cls_global.shape[-1]))
        elif self.unimodal_token_drop:

            spec_a = local_spectrogram[:, 0::2]
            vid_v = local_video[:, 1::2]
            Ka, Kv = spec_a.shape[1], vid_v.shape[1]
            if Ka > 0:
                z_audio = self.encoder(None, spec_a.reshape(B * Ka, *spec_a.shape[2:]))
            if Kv > 0:
                z_video = self.encoder(vid_v.reshape(B * Kv, *vid_v.shape[2:]), None)
            ref = z_audio if Ka > 0 else z_video
            z_local_cls = ref.new_empty(B, K, ref.shape[-1])
            if Ka > 0:
                z_local_cls[:, 0::2] = z_audio.view(B, Ka, -1)
            if Kv > 0:
                z_local_cls[:, 1::2] = z_video.view(B, Kv, -1)
            z_local_cls = z_local_cls.reshape(B * K, -1)
        else:

            z_local_cls = self.encoder(
                local_video.view(B * K, *local_video.shape[2:]),
                local_spectrogram.view(B * K, *local_spectrogram.shape[2:]),
            )

        unimodal_probe_loss = None
        if self.cross_modal and not self.no_probe:
            local_cls = z_local_cls.view(B, K, -1).detach()
            unimodal_probe_loss = self._unimodal_probe_loss(
                local_cls[:, 0::2].mean(dim=1), local_cls[:, 1::2].mean(dim=1), labels
            )
            self.log("train/probe_unimodal_loss", unimodal_probe_loss, sync_dist=True)

        if K > 0:
            z_local = self.projector(z_local_cls).view(B, K, -1).permute(1, 0, 2)
        else:
            z_local = z_global.new_zeros((0,) + z_global.shape[1:])

        center = z_global.mean(dim=0)
        all_views = torch.cat([z_global, z_local], dim=0)
        inv_loss = (center - all_views).square().mean()
        sigreg_loss = self._sigreg_mean(all_views)
        embed_std = all_views.std(dim=1).mean()
        return inv_loss, sigreg_loss, embed_std, unimodal_probe_loss

    def _dual_encoder_loss(
        self, audio_cls_g, video_cls_g, local_video, local_spectrogram, labels, B, G, K
    ):
        Ka = K // 2
        Kv = K - Ka
        audio_spec = local_spectrogram[:, 0::2].reshape(B * Ka, *local_spectrogram.shape[2:])
        video_vid = local_video[:, 1::2].reshape(B * Kv, *local_video.shape[2:])
        z_audio_local = self.encoder.forward_audio(audio_spec)
        z_video_local = self.encoder.forward_video(video_vid)

        unimodal_probe_loss = self._unimodal_probe_loss(
            z_audio_local.view(B, Ka, -1).mean(dim=1).detach(),
            z_video_local.view(B, Kv, -1).mean(dim=1).detach(),
            labels,
        )
        self.log("train/probe_unimodal_loss", unimodal_probe_loss, sync_dist=True)

        z_audio_global = self.projector(audio_cls_g).view(B, G, -1).permute(1, 0, 2)
        z_video_global = self.projector(video_cls_g).view(B, G, -1).permute(1, 0, 2)
        z_audio_local = self.projector(z_audio_local).view(B, Ka, -1).permute(1, 0, 2)
        z_video_local = self.projector(z_video_local).view(B, Kv, -1).permute(1, 0, 2)

        audio_embs = torch.cat([z_audio_global, z_audio_local], dim=0)
        video_embs = torch.cat([z_video_global, z_video_local], dim=0)
        audio_center = z_audio_global.mean(dim=0)
        video_center = z_video_global.mean(dim=0)
        inv_loss = (
            (video_center - audio_embs).square().mean()
            + (audio_center - video_embs).square().mean()
        ) / 2
        sigreg_loss = (self._sigreg_mean(audio_embs) + self._sigreg_mean(video_embs)) / 2
        embed_std = (audio_embs.std(dim=1).mean() + video_embs.std(dim=1).mean()) / 2
        return inv_loss, sigreg_loss, embed_std, unimodal_probe_loss

    def configure_optimizers(self):
        linear_probe_params = list(self.probe_head.parameters()) + list(
            self.probe_norm.parameters()
        )
        att_probe_params = list(self.att_probe.parameters()) if self.attentive_probe else []
        probe_ids = {id(p) for p in linear_probe_params + att_probe_params}
        backbone_named = [(n, p) for n, p in self.named_parameters() if id(p) not in probe_ids]

        if self.wd_exclude_norm_embed:

            no_decay_names = (
                "pos_embed",
                "video_time_embed",
                "video_spatial_embed",
                "cls_token",
                "modality_type_embeddings",
            )
            decay, no_decay = [], []
            for name, p in backbone_named:
                if p.ndim <= 1 or any(k in name for k in no_decay_names):
                    no_decay.append(p)
                else:
                    decay.append(p)
            param_groups = [
                {"params": decay, "lr": self.lr, "weight_decay": self.weight_decay},
                {"params": no_decay, "lr": self.lr, "weight_decay": 0.0},
            ]
        else:
            param_groups = [
                {
                    "params": [p for _, p in backbone_named],
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                }
            ]
        if not self.no_probe:
            param_groups.append(
                {
                    "params": linear_probe_params,
                    "lr": self.probe_lr,
                    "weight_decay": self.probe_weight_decay,
                }
            )
        if self.attentive_probe:
            param_groups.append(
                {
                    "params": att_probe_params,
                    "lr": self.probe_lr,
                    "weight_decay": self.probe_weight_decay,
                }
            )
        optimizer = optim.AdamW(param_groups)

        accumulation_steps = self.trainer.accumulate_grad_batches or 1
        effective_batch_size = self.hparams.batch_size * self.trainer.world_size * accumulation_steps
        steps_per_epoch = self.total_samples // effective_batch_size
        total_steps = steps_per_epoch * self.trainer.max_epochs
        warmup_steps = int(total_steps * 0.15)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, total_iters=warmup_steps
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-6
                ),
            ],
            milestones=[warmup_steps],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def validation_step(self, batch, batch_idx):

        return self.test_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        metrics = self.trainer.callback_metrics
        reported = [
            f"{key}={float(metrics[key]):.4f}"
            for key in (
                "probe/test/acc",
                "probe/test/acc_top5",
                "probe/test/map",
                "att_probe/test/acc",
                "att_probe/test/acc_top5",
                "att_probe/test/map",
            )
            if key in metrics
        ]
        if reported:
            print(f"[eval] epoch {self.current_epoch}: " + "  ".join(reported), flush=True)

    def test_step(self, batch, batch_idx):
        video = batch["video"]
        audio = batch["spectrogram"]
        targets = batch["label"]
        B, N = video.shape[:2]
        video = video.view(B * N, *video.shape[2:])
        audio = audio.view(B * N, *audio.shape[2:])
        log_kw = dict(on_step=False, on_epoch=True)

        def clip_mean(logits):
            return logits.view(B, N, -1).mean(dim=1)

        def encode(v, a):
            if self.attentive_probe:
                return self.encoder(v, a, return_patches=True)
            return self.encoder(v, a), None

        with torch.no_grad():
            cls_tokens, patch_tokens = encode(video, audio)
            logits = clip_mean(self._linear_probe(cls_tokens))
            audio_cls, audio_patches = encode(torch.zeros_like(video), audio)
            video_cls, video_patches = encode(video, torch.zeros_like(audio))
            audio_logits = clip_mean(self._linear_probe(audio_cls))
            video_logits = clip_mean(self._linear_probe(video_cls))

        loss = self.probe_criterion(logits, targets)
        self.log("probe/test/loss", loss, sync_dist=True, **log_kw)
        self._update_and_log(self.probe_metrics, "test", logits, targets, "probe/test/", **log_kw)
        self._update_and_log(
            self.probe_metrics, "test_audio", audio_logits, targets, "probe/test/audio_", **log_kw
        )
        self._update_and_log(
            self.probe_metrics, "test_video", video_logits, targets, "probe/test/video_", **log_kw
        )

        if self.attentive_probe:
            with torch.no_grad():
                att_logits = clip_mean(self.att_probe(patch_tokens))
                att_audio_logits = clip_mean(
                    self.att_probe(modality_tokens(audio_patches, self.num_video_tokens, "audio"))
                )
                att_video_logits = clip_mean(
                    self.att_probe(modality_tokens(video_patches, self.num_video_tokens, "video"))
                )
            att_loss = self.att_probe_criterion(att_logits, targets)
            self.log("att_probe/test/loss", att_loss, sync_dist=True, **log_kw)
            self._update_and_log(
                self.att_probe_metrics, "test", att_logits, targets, "att_probe/test/", **log_kw
            )
            self._update_and_log(
                self.att_probe_metrics,
                "test_audio",
                att_audio_logits,
                targets,
                "att_probe/test/audio_",
                **log_kw,
            )
            self._update_and_log(
                self.att_probe_metrics,
                "test_video",
                att_video_logits,
                targets,
                "att_probe/test/video_",
                **log_kw,
            )
        return loss


class EchoFineTuner(L.LightningModule):

    def __init__(
        self,
        encoder,
        hidden_size,
        num_classes=309,
        lr=2e-4,
        backbone_lr_scale=0.1,
        weight_decay=5e-2,
        warmup_fraction=0.05,
        label_smoothing=0.1,
        batch_size=32,
        total_samples=183_730,
        epochs=20,
        attentive_probe=False,
        num_attention_heads=12,
        multi_label=False,
        pos_weight=None,
        mixup_alpha=0.0,
        modality_drop_prob=0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "pos_weight"])
        self.encoder = encoder

        emb = getattr(encoder, "embedding", None)
        if emb is not None and hasattr(emb, "num_temporal_patches"):
            self.num_video_tokens = emb.num_temporal_patches * emb.num_spatial_patches
        else:
            self.num_video_tokens = None
        self.attentive_probe = attentive_probe
        self.multi_label = multi_label
        self.mixup_alpha = mixup_alpha
        self.modality_drop_prob = modality_drop_prob

        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, num_classes)

        if multi_label:
            if pos_weight is not None:
                self.register_buffer("pos_weight", pos_weight.clone().detach())
            else:
                self.pos_weight = None
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.lr = lr
        self.backbone_lr_scale = backbone_lr_scale
        self.weight_decay = weight_decay
        self.warmup_fraction = warmup_fraction
        self.total_samples = total_samples
        self.epochs = epochs

        names = ["train", "test", "test_audio", "test_video"]
        self.metrics = _classification_metrics(multi_label, num_classes, names)
        if attentive_probe:
            self.att_probe = AttentiveProbe(hidden_size, num_attention_heads, num_classes)
            if multi_label:
                self.att_probe_criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            else:
                self.att_probe_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            self.att_metrics = _classification_metrics(multi_label, num_classes, names)

    def _update_and_log(self, metrics, prefix, logits, targets, key_prefix, **log_kw):
        if self.multi_label:
            m = metrics[f"{prefix}_map"]
            m(logits, targets.long())
            self.log(f"{key_prefix}map", m, sync_dist=True, **log_kw)
        else:
            m1, m5 = metrics[f"{prefix}_acc"], metrics[f"{prefix}_acc5"]
            m1(logits.argmax(dim=1), targets)
            m5(logits, targets)
            self.log(f"{key_prefix}acc", m1, sync_dist=True, **log_kw)
            self.log(f"{key_prefix}acc_top5", m5, sync_dist=True, **log_kw)

    def training_step(self, batch, batch_idx):
        video = batch["global_video"][:, 0]
        audio = batch["global_spectrogram"][:, 0]
        targets = batch["label"]

        if self.modality_drop_prob > 0.0:
            rolls = torch.rand(video.size(0), device=video.device)
            half_p = self.modality_drop_prob / 2
            drop_video = rolls < half_p
            drop_audio = (rolls >= half_p) & (rolls < self.modality_drop_prob)
            if drop_video.any():
                video = video.clone()
                video[drop_video] = 0.0
            if drop_audio.any():
                audio = audio.clone()
                audio[drop_audio] = 0.0

        mixed = False
        if self.mixup_alpha > 0.0:
            mix_lam = float(
                torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().item()
            )
            mix_perm = torch.randperm(video.size(0), device=video.device)
            video = mix_lam * video + (1.0 - mix_lam) * video[mix_perm]
            audio = mix_lam * audio + (1.0 - mix_lam) * audio[mix_perm]
            if self.multi_label:
                targets = mix_lam * targets + (1.0 - mix_lam) * targets[mix_perm]
            mixed = True

        def criterion(fn, logits):
            if mixed and not self.multi_label:
                return mix_lam * fn(logits, targets) + (1.0 - mix_lam) * fn(
                    logits, targets[mix_perm]
                )
            return fn(logits, targets)

        if self.attentive_probe:
            cls_tokens, patch_tokens = self.encoder(video, audio, return_patches=True)
        else:
            cls_tokens = self.encoder(video, audio)
        logits = self.head(self.norm(cls_tokens))
        loss = criterion(self.criterion, logits)
        if not mixed:
            self._update_and_log(self.metrics, "train", logits, targets, "train/")

        if self.attentive_probe:
            att_logits = self.att_probe(patch_tokens)
            att_loss = criterion(self.att_probe_criterion, att_logits)
            loss = loss + att_loss
            self.log("train/att_probe_loss", att_loss, sync_dist=True)
            if not mixed:
                self._update_and_log(self.att_metrics, "train", att_logits, targets, "train/att_")

        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        video = batch["video"]
        audio = batch["spectrogram"]
        targets = batch["label"]
        B, N = video.shape[:2]
        video = video.view(B * N, *video.shape[2:])
        audio = audio.view(B * N, *audio.shape[2:])
        log_kw = dict(on_step=False, on_epoch=True)

        def clip_mean(logits):
            return logits.view(B, N, -1).mean(dim=1)

        def encode(v, a):
            if self.attentive_probe:
                return self.encoder(v, a, return_patches=True)
            return self.encoder(v, a), None

        with torch.no_grad():
            cls_tokens, patch_tokens = encode(video, audio)
            audio_cls, audio_patches = encode(video.new_zeros(video.shape), audio)
            video_cls, video_patches = encode(video, audio.new_zeros(audio.shape))
            logits = clip_mean(self.head(self.norm(cls_tokens)))
            audio_logits = clip_mean(self.head(self.norm(audio_cls)))
            video_logits = clip_mean(self.head(self.norm(video_cls)))

        self.log("test/loss", self.criterion(logits, targets), sync_dist=True, **log_kw)
        self._update_and_log(self.metrics, "test", logits, targets, "test/", **log_kw)
        self._update_and_log(self.metrics, "test_audio", audio_logits, targets, "test/audio_", **log_kw)
        self._update_and_log(self.metrics, "test_video", video_logits, targets, "test/video_", **log_kw)

        if self.attentive_probe:
            with torch.no_grad():
                att_logits = clip_mean(self.att_probe(patch_tokens))
                att_audio_logits = clip_mean(
                    self.att_probe(modality_tokens(audio_patches, self.num_video_tokens, "audio"))
                )
                att_video_logits = clip_mean(
                    self.att_probe(modality_tokens(video_patches, self.num_video_tokens, "video"))
                )
            self._update_and_log(self.att_metrics, "test", att_logits, targets, "test/att_", **log_kw)
            self._update_and_log(
                self.att_metrics, "test_audio", att_audio_logits, targets, "test/att_audio_", **log_kw
            )
            self._update_and_log(
                self.att_metrics, "test_video", att_video_logits, targets, "test/att_video_", **log_kw
            )

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)

    def configure_optimizers(self):
        head_params = list(self.norm.parameters()) + list(self.head.parameters())
        if self.attentive_probe:
            head_params += list(self.att_probe.parameters())
        head_ids = {id(p) for p in head_params}
        backbone_params = [p for p in self.encoder.parameters() if id(p) not in head_ids]
        optimizer = optim.AdamW(
            [
                {
                    "params": backbone_params,
                    "lr": self.lr * self.backbone_lr_scale,
                    "weight_decay": self.weight_decay,
                },
                {"params": head_params, "lr": self.lr, "weight_decay": 0.0},
            ]
        )

        accumulation_steps = self.trainer.accumulate_grad_batches or 1
        effective_batch_size = self.hparams.batch_size * self.trainer.world_size * accumulation_steps
        steps_per_epoch = self.total_samples // effective_batch_size
        total_steps = steps_per_epoch * self.epochs
        warmup_steps = int(total_steps * self.warmup_fraction)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, total_iters=warmup_steps
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-7
                ),
            ],
            milestones=[warmup_steps],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


def load_pretrained(checkpoint_path, **overrides):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = dict(ckpt["hyper_parameters"])
    hparams.update(overrides)
    model = EchoTrainer(**hparams)
    state = ckpt["state_dict"]
    missing, _ = model.load_state_dict(state, strict=False)
    core_missing = [k for k in missing if k.startswith(("encoder.", "projector."))]
    if core_missing:
        raise RuntimeError(
            f"checkpoint is missing {len(core_missing)} encoder/projector tensors, "
            f"e.g. {core_missing[:3]}"
        )
    return model
