import argparse
import json
import os

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchmetrics
from torch.utils.data import DataLoader, Dataset

from esc50_probe import NUM_CLASSES, cache_mels, load_encoder


class ESC50Split(Dataset):

    def __init__(self, mels, labels, mean, std, train, freq_mask=0, time_mask=0,
                 random_shift=False):
        self.mels, self.labels = mels, labels
        self.mean, self.std = mean, std
        self.random_shift = train and random_shift
        aug = []
        if train and freq_mask > 0:
            aug.append(torchaudio.transforms.FrequencyMasking(freq_mask))
        if train and time_mask > 0:
            aug.append(torchaudio.transforms.TimeMasking(time_mask))
        self.augment = torch.nn.Sequential(*aug) if aug else None

    def __len__(self):
        return self.mels.shape[0]

    def __getitem__(self, i):
        spec = (self.mels[i] - self.mean) / self.std
        if self.random_shift:
            spec = torch.roll(spec, int(torch.randint(spec.shape[-1], (1,))), dims=-1)
        if self.augment is not None:
            spec = self.augment(spec)
        return spec, self.labels[i]


class ESC50FineTuner(L.LightningModule):
    def __init__(self, encoder, hidden_size, lr, backbone_lr_scale, weight_decay,
                 warmup_fraction, label_smoothing, batch_size, total_samples,
                 epochs, mixup_alpha, zero_video=False):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])
        self.encoder = encoder
        self.zero_video = zero_video
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, NUM_CLASSES)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.mixup_alpha = mixup_alpha
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.test_acc5 = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES,
                                               top_k=5)

    def _logits(self, audio):
        video = None
        if self.zero_video:
            emb = self.encoder.embedding
            video = torch.zeros(audio.shape[0], 3, emb.video_patch_embed.num_frames,
                                emb.video_patch_embed.image_size,
                                emb.video_patch_embed.image_size,
                                device=audio.device, dtype=audio.dtype)
        return self.head(self.norm(self.encoder(video, audio)))

    def training_step(self, batch, batch_idx):
        audio, targets = batch
        mixed = False
        if self.mixup_alpha > 0.0:
            lam = float(torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha)
                        .sample().item())
            perm = torch.randperm(audio.size(0), device=audio.device)
            audio = lam * audio + (1.0 - lam) * audio[perm]
            mixed = True
        logits = self._logits(audio)
        if mixed:
            loss = lam * self.criterion(logits, targets) + \
                (1.0 - lam) * self.criterion(logits, targets[perm])
        else:
            loss = self.criterion(logits, targets)
            self.train_acc(logits.argmax(dim=1), targets)
            self.log("train/acc", self.train_acc)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        audio, targets = batch
        logits = self._logits(audio)
        self.test_acc(logits.argmax(dim=1), targets)
        self.test_acc5(logits, targets)
        self.log("test/acc", self.test_acc)
        self.log("test/acc_top5", self.test_acc5)

    def configure_optimizers(self):
        hp = self.hparams
        optimizer = optim.AdamW([
            {"params": list(self.encoder.parameters()),
             "lr": hp.lr * hp.backbone_lr_scale, "weight_decay": hp.weight_decay},
            {"params": list(self.norm.parameters()) + list(self.head.parameters()),
             "lr": hp.lr, "weight_decay": 0.0},
        ])
        total_steps = max(1, hp.total_samples // hp.batch_size) * hp.epochs
        warmup_steps = int(total_steps * hp.warmup_fraction)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1,
                                                  total_iters=warmup_steps),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-7),
            ],
            milestones=[warmup_steps],
        )
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}


def main():
    p = argparse.ArgumentParser(description="ESC-50 fine-tuning of a pretrained encoder")
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--vit_size", type=str, default="base", choices=["base", "large"])
    p.add_argument("--esc50_root", type=str, default=os.environ.get("ESC50_ROOT", "data/ESC-50"))
    p.add_argument("--folds", type=str, default="1,2,3,4,5")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--lr", type=float, default=1e-3, help="head learning rate")
    p.add_argument("--backbone_lr_scale", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=5e-2)
    p.add_argument("--warmup_fraction", type=float, default=0.05)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--mixup_alpha", type=float, default=0.5)
    p.add_argument("--freq_mask_param", type=int, default=48)
    p.add_argument("--time_mask_param", type=int, default=150)
    p.add_argument("--no_random_shift", action="store_true",
                   help="disable the random circular time shift")
    p.add_argument("--zero_video", action="store_true",
                   help="feed zeroed video frames instead of omitting the video tokens")
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="activation checkpointing in the encoder (ViT-L memory)")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    mels, labels, folds = cache_mels(args.esc50_root, num_workers=8)

    fold_accs, fold_accs5 = {}, {}
    for fold in [int(f) for f in args.folds.split(",")]:
        L.seed_everything(args.seed + fold)
        tr, te = folds != fold, folds == fold
        mean, std = mels[tr].mean().item(), mels[tr].std().item()
        train_ds = ESC50Split(mels[tr], labels[tr], mean, std, train=True,
                              freq_mask=args.freq_mask_param,
                              time_mask=args.time_mask_param,
                              random_shift=not args.no_random_shift)
        test_ds = ESC50Split(mels[te], labels[te], mean, std, train=False)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 num_workers=args.num_workers)

        encoder = load_encoder(args.checkpoint_path, args.vit_size,
                               gradient_checkpointing=args.gradient_checkpointing)
        model = ESC50FineTuner(
            encoder, hidden_size=encoder.t_config["hidden_size"], lr=args.lr,
            backbone_lr_scale=args.backbone_lr_scale, weight_decay=args.weight_decay,
            warmup_fraction=args.warmup_fraction, label_smoothing=args.label_smoothing,
            batch_size=args.batch_size, total_samples=int(tr.sum()),
            epochs=args.epochs, mixup_alpha=args.mixup_alpha, zero_video=args.zero_video)
        trainer = L.Trainer(devices=1, max_epochs=args.epochs, precision="bf16-mixed",
                            logger=False, enable_checkpointing=False,
                            enable_model_summary=False, num_sanity_val_steps=0)
        trainer.fit(model, train_loader)
        metrics = trainer.test(model, test_loader)[0]
        fold_accs[fold] = metrics["test/acc"]
        fold_accs5[fold] = metrics["test/acc_top5"]
        print(f"fold {fold}: acc={fold_accs[fold]:.4f} top5={fold_accs5[fold]:.4f}", flush=True)
        del model, encoder

    accs = list(fold_accs.values())
    result = dict(
        mean_acc=round(float(np.mean(accs)), 4),
        std_across_folds=round(float(np.std(accs)), 4),
        mean_acc_top5=round(float(np.mean(list(fold_accs5.values()))), 4),
        per_fold_acc={f: round(a, 4) for f, a in fold_accs.items()},
        args=vars(args),
    )
    print(json.dumps(result, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
