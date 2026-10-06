import argparse
import glob
import json
import math
import os
import time

import torch
import torch.nn as nn

from dataset_config import DATASETS
from probe_heads import VJEPAProbe


def parse_args():
    p = argparse.ArgumentParser(description="Frozen attentive probe on cached tokens")
    p.add_argument("--mode", type=str, required=True, choices=["extract", "fit", "test"])
    p.add_argument("--cache_dir", type=str, required=True,
                   help="Holds meta.json, the train_v*_p*.pt token parts and results.")
    p.add_argument("--heads_dir", type=str, default=None, help="Default: <cache_dir>/heads")
    p.add_argument("--results_json", type=str, default=None)
    p.add_argument("--dataset", type=str, default="audioset_20k", choices=list(DATASETS.keys()))
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--pretrained_checkpoint", type=str, default=None)
    p.add_argument("--vit_size", type=str, default="base", choices=["base", "large"])
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--frame_size", type=int, default=224)
    p.add_argument("--modality", type=str, default="av", choices=["av", "audio", "video"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--num_workers_test", type=int, default=8)
    p.add_argument("--train_views", type=int, default=2)
    p.add_argument("--part_size", type=int, default=1024, help="Clips per cache part file")
    p.add_argument("--num_eval_clips", type=int, default=4)
    p.add_argument("--max_batches", type=int, default=0, help="Stop each pass early (smoke)")
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--fit_lr", type=float, default=3e-4)
    p.add_argument("--fit_wd", type=float, default=1e-4)
    p.add_argument("--fit_epochs", type=int, default=40)
    p.add_argument("--fit_batch", type=int, default=128)
    p.add_argument("--warmup_fraction", type=float, default=0.05)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--fit_log_every", type=int, default=5)
    args = p.parse_args()
    if args.mode in ("extract", "test") and args.pretrained_checkpoint is None:
        p.error(f"--mode {args.mode} requires --pretrained_checkpoint")
    return args


args = parse_args()
torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

ds = DATASETS[args.dataset]
META_PATH = os.path.join(args.cache_dir, "meta.json")
HEADS_DIR = args.heads_dir or os.path.join(args.cache_dir, "heads")
HEAD_NAME = f"vjepa_d{args.depth}_lr{args.fit_lr:g}_wd{args.fit_wd:g}"


def count_shards(pattern: str) -> int:
    from braceexpand import braceexpand
    return len(list(braceexpand(pattern)))


def build_encoder():
    from configs import audio_config, transformer_config, video_config
    from models import load_pretrained

    print(f"Loading pretrained checkpoint: {args.pretrained_checkpoint}", flush=True)
    encoder = load_pretrained(
        args.pretrained_checkpoint,
        a_config=audio_config(),
        v_config=video_config(args.num_frames, args.frame_size),
        t_config=transformer_config(args.vit_size),
    ).encoder
    encoder = encoder.to(device).eval()
    encoder.requires_grad_(False)
    torch.cuda.empty_cache()
    return encoder


def select_modality(video, audio):
    if args.modality == "audio":
        return None, audio
    if args.modality == "video":
        return video, None
    return video, audio


def build_loaders():
    from data import get_dataloader

    num_workers = min(args.num_workers, count_shards(ds.train_tar))
    num_workers_test = min(args.num_workers_test, count_shards(ds.test_tar))
    return get_dataloader(
        tar_path=ds.train_tar,
        csv_path=ds.train_csv,
        test_tar_path=ds.test_tar,
        test_csv_path=ds.test_csv,
        batch_size=args.batch_size,
        num_workers=num_workers,
        num_workers_test=num_workers_test,
        frame_size=(args.frame_size, args.frame_size),
        num_frames=args.num_frames,
        num_global_views=1,
        num_local_views=1,
        num_eval_clips=args.num_eval_clips,
        train_size=None,
        test_size=None,
        world_size=1,
        video_mask_ratio=0.0,
        freq_mask_param=0,
        time_mask_param=0,
        spec_aug_global=False,
        global_rrc_min_scale=0.0,
        modality_drop_prob=0.0,
        clean_survivor=False,
        cross_modal=False,
        csv_format=ds.csv_format,
        spec_mean=ds.spec_mean,
        spec_std=ds.spec_std,
    )


def extract():
    os.makedirs(args.cache_dir, exist_ok=True)
    existing = glob.glob(os.path.join(args.cache_dir, "train_v*_p*.pt"))
    if existing and not args.overwrite:
        raise SystemExit(f"{len(existing)} cache parts already in {args.cache_dir} (use --overwrite)")
    for f in existing:
        os.remove(f)

    encoder = build_encoder()
    train_loader, _, classes = build_loaders()

    def flush_part(view, part_idx, toks, clss, ys):
        path = os.path.join(args.cache_dir, f"train_v{view}_p{part_idx:03d}.pt")
        torch.save({"tokens": torch.cat(toks), "cls": torch.cat(clss), "y": torch.cat(ys)}, path)
        print(f"    wrote {path}", flush=True)

    meta = None
    for v in range(args.train_views):
        print(f"Extracting train tokens, view {v + 1}/{args.train_views}", flush=True)
        toks, clss, ys, n_part, part_idx, n_seen = [], [], [], 0, 0, 0
        t0 = time.time()
        for bi, batch in enumerate(train_loader):
            video, audio = select_modality(
                batch["global_video"][:, 0].to(device, non_blocking=True),
                batch["global_spectrogram"][:, 0].to(device, non_blocking=True),
            )
            with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
                cls, patches = encoder(video, audio, return_patches=True)
            toks.append(patches.half().cpu())
            clss.append(cls.half().cpu())
            y = batch["label"]
            ys.append(y.to(torch.uint8) if ds.multi_label else y.long())
            n_part += cls.shape[0]
            n_seen += cls.shape[0]
            if meta is None:
                meta = {
                    "checkpoint": args.pretrained_checkpoint,
                    "dataset": ds.name,
                    "multi_label": ds.multi_label,
                    "classes": list(classes),
                    "hidden_size": int(patches.shape[-1]),
                    "num_tokens": int(patches.shape[1]),
                    "num_heads": encoder.t_config["num_attention_heads"],
                    "vit_size": args.vit_size,
                    "train_views": args.train_views,
                    "modality": args.modality,
                }
            if n_part >= args.part_size:
                flush_part(v, part_idx, toks, clss, ys)
                toks, clss, ys, n_part, part_idx = [], [], [], 0, part_idx + 1
            if args.max_batches and bi + 1 >= args.max_batches:
                break
        if n_part:
            flush_part(v, part_idx, toks, clss, ys)
        print(f"  [view {v}] {n_seen} clips in {(time.time() - t0) / 60:.1f} min", flush=True)

    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def load_meta():
    with open(META_PATH) as f:
        return json.load(f)


def average_precision(logits, y):
    import torchmetrics

    metric = torchmetrics.AveragePrecision(task="multilabel", num_labels=y.shape[1])
    return metric(logits.float(), y.long()).item()


def fit():
    meta = load_meta()
    if not meta["multi_label"]:
        raise SystemExit("token_probe.py implements the multi-label (AudioSet) protocol")
    d, num_heads = meta["hidden_size"], meta["num_heads"]
    num_classes = len(meta["classes"])
    os.makedirs(HEADS_DIR, exist_ok=True)

    part_files = sorted(glob.glob(os.path.join(args.cache_dir, "train_v*_p*.pt")))
    if not part_files:
        raise SystemExit(f"No cache parts in {args.cache_dir}")
    parts = []
    for path in part_files:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        view = int(os.path.basename(path).split("_")[1][1:])
        parts.append({"view": view, "tokens": blob["tokens"], "y": blob["y"]})

    g = torch.Generator().manual_seed(args.seed)
    v0_rows = [(i, r) for i, part in enumerate(parts) if part["view"] == 0
               for r in range(part["tokens"].shape[0])]
    perm = torch.randperm(len(v0_rows), generator=g)
    n_val = max(1, int(len(v0_rows) * args.val_frac))
    val_by_part = {}
    for pi, r in (v0_rows[i] for i in perm[:n_val].tolist()):
        val_by_part.setdefault(pi, []).append(r)
    val_tok = torch.cat([parts[pi]["tokens"][torch.tensor(rs)] for pi, rs in val_by_part.items()])
    val_y = torch.cat([parts[pi]["y"][torch.tensor(rs)] for pi, rs in val_by_part.items()])
    for pi, part in enumerate(parts):
        keep = torch.ones(part["tokens"].shape[0], dtype=torch.bool)
        if pi in val_by_part:
            keep[torch.tensor(val_by_part[pi])] = False
        part["train_idx"] = keep.nonzero(as_tuple=True)[0]
    n_fit = sum(part["train_idx"].numel() for part in parts)
    print(f"fit={n_fit} rows / val={val_tok.shape[0]} rows", flush=True)

    crit = nn.BCEWithLogitsLoss()

    @torch.no_grad()
    def eval_head(head):
        head.eval()
        outs = []
        eval_batch = min(args.fit_batch, 128)
        for i in range(0, val_tok.shape[0], eval_batch):
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                outs.append(head(val_tok[i:i + eval_batch].to(device)).float().cpu())
        head.train()
        return average_precision(torch.cat(outs), val_y)

    torch.manual_seed(args.seed)
    head = VJEPAProbe(d, num_heads, num_classes, depth=args.depth).to(device)

    steps_per_epoch = sum(math.ceil(part["train_idx"].numel() / args.fit_batch) for part in parts)
    total_steps = max(1, steps_per_epoch * args.fit_epochs)
    warmup = max(1, int(total_steps * args.warmup_fraction))
    opt = torch.optim.AdamW(head.parameters(), lr=args.fit_lr, weight_decay=args.fit_wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (s + 1) / warmup if s < warmup
        else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warmup) / max(1, total_steps - warmup)))),
    )
    g_ep = torch.Generator().manual_seed(args.seed + 1)
    curve = []
    t0 = time.time()
    for ep in range(args.fit_epochs):

        for pi in torch.randperm(len(parts), generator=g_ep).tolist():
            x_gpu = parts[pi]["tokens"].to(device, non_blocking=True)
            y_gpu = parts[pi]["y"].to(device, non_blocking=True)
            idx = parts[pi]["train_idx"]
            sub = idx[torch.randperm(idx.numel(), generator=g_ep)]
            for i in range(0, sub.numel(), args.fit_batch):
                b = sub[i:i + args.fit_batch].to(device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits = head(x_gpu[b])
                loss = crit(logits.float(), y_gpu[b].float())
                loss.backward()
                opt.step()
                sched.step()
            del x_gpu, y_gpu
        if args.fit_log_every and ((ep + 1) % args.fit_log_every == 0 or ep + 1 == args.fit_epochs):
            v = eval_head(head)
            curve.append({"epoch": ep + 1, "val": v})
            print(f"  ep{ep + 1:<3d} val mAP={v:.4f}", flush=True)
    val = eval_head(head)
    print(f"{HEAD_NAME}: val mAP={val:.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)

    torch.save({"depth": args.depth, "lr": args.fit_lr, "wd": args.fit_wd, "name": HEAD_NAME,
                "val": val, "state_dict": {k: t.cpu() for k, t in head.state_dict().items()}},
               os.path.join(HEADS_DIR, f"{HEAD_NAME}.pt"))
    out_path = args.results_json or os.path.join(args.cache_dir, "fit_results.json")
    with open(out_path, "w") as f:
        json.dump({"meta": {k: v for k, v in meta.items() if k != "classes"},
                   "n_fit": n_fit, "n_val": int(val_tok.shape[0]), "head": HEAD_NAME,
                   "fit_epochs": args.fit_epochs, "fit_batch": args.fit_batch,
                   "val": val, "curve": curve}, f, indent=2)


def test():
    meta = load_meta()
    if meta["modality"] != args.modality:
        raise SystemExit(f"cache was extracted with --modality {meta['modality']}, "
                         f"test launched with --modality {args.modality}")
    d, num_heads = meta["hidden_size"], meta["num_heads"]
    num_classes = len(meta["classes"])
    blob = torch.load(os.path.join(HEADS_DIR, f"{HEAD_NAME}.pt"), map_location="cpu",
                      weights_only=True)
    head = VJEPAProbe(d, num_heads, num_classes, depth=blob["depth"])
    head.load_state_dict(blob["state_dict"])
    head = head.to(device).eval()

    encoder = build_encoder()
    _, test_loader, _ = build_loaders()

    logits_acc, labels_acc, n, t0 = [], [], 0, time.time()
    with torch.no_grad():
        for bi, batch in enumerate(test_loader):
            video, audio = batch["video"], batch["spectrogram"]
            b, nc = video.shape[:2]
            v, a = select_modality(
                video.view(b * nc, *video.shape[2:]).to(device, non_blocking=True),
                audio.view(b * nc, *audio.shape[2:]).to(device, non_blocking=True),
            )
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                _, patches = encoder(v, a, return_patches=True)
                logits = head(patches).float()
            logits_acc.append(logits.view(b, nc, num_classes).mean(dim=1).cpu())
            labels_acc.append(batch["label"].cpu())
            n += b
            if args.max_batches and bi + 1 >= args.max_batches:
                break

    test_map = average_precision(torch.cat(logits_acc), torch.cat(labels_acc))
    print(f"{HEAD_NAME} [{args.modality}] test mAP = {test_map * 100:.2f} "
          f"({n} videos x {args.num_eval_clips} clips, {(time.time() - t0) / 60:.1f} min)",
          flush=True)
    out_path = args.results_json or os.path.join(args.cache_dir, "test_results.json")
    with open(out_path, "w") as f:
        json.dump({"head": HEAD_NAME, "modality": args.modality, "val": blob["val"],
                   "test_map": test_map, "n_videos": n, "num_eval_clips": args.num_eval_clips,
                   "checkpoint": args.pretrained_checkpoint}, f, indent=2)


if __name__ == "__main__":
    {"extract": extract, "fit": fit, "test": test}[args.mode]()
