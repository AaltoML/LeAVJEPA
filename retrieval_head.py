import argparse
import json
import math
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset_config import DATASETS
from retrieval import (
    DATASET,
    build_loader,
    build_subset_keys,
    compute_retrieval_metrics,
    extract_embeddings,
    format_metrics,
    load_model,
    load_subset_file,
)


def parse_args():
    p = argparse.ArgumentParser(description="InfoNCE alignment head on a frozen encoder")
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--subset_file", type=str,
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "datafiles",
                                        "audioset_eval_5_per_class_for_retrieval_cleaned.json"))
    p.add_argument("--batch_size", type=int, default=32, help="Feature-extraction batch")
    p.add_argument("--num_workers", type=int, default=8)

    p.add_argument("--train_per_class", type=int, default=300)
    p.add_argument("--train_clips", type=int, default=2)
    p.add_argument("--train_seed", type=int, default=1)
    p.add_argument("--eval_clips", type=int, default=4)

    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--temp_init", type=float, default=0.07)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--head_batch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="./outputs/retrieval_head")
    p.add_argument("--cache_dir", type=str, default=None,
                   help="Feature cache directory (default: <output_dir>/cache)")
    return p.parse_args()


class ProjHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class AlignmentHead(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, temp_init=0.07, dropout=0.0):
        super().__init__()
        self.audio_head = ProjHead(in_dim, hidden_dim, out_dim, dropout=dropout)
        self.video_head = ProjHead(in_dim, hidden_dim, out_dim, dropout=dropout)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temp_init)))

    def encode(self, A, V):
        return F.normalize(self.audio_head(A), dim=-1), F.normalize(self.video_head(V), dim=-1)

    def forward(self, A, V):
        za, zv = self.encode(A, V)
        return self.logit_scale.clamp(max=4.6052).exp() * za @ zv.t()


def clip_loss(logits):
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


@torch.no_grad()
def head_metrics(head, A, V, device):
    head.eval()
    za, zv = head.encode(A.to(device), V.to(device))
    return compute_retrieval_metrics(za.cpu(), zv.cpu())


def mean_r1(m):
    return 0.5 * (m["a2v_R@1"] + m["v2a_R@1"])


SWEEP_GRID = [
    ("baseline", dict()),
    ("drop0.3", dict(dropout=0.3)),
    ("drop0.5", dict(dropout=0.5)),
    ("wd1e-2", dict(weight_decay=1e-2)),
    ("wd1e-1", dict(weight_decay=1e-1)),
    ("drop0.3+wd1e-2", dict(dropout=0.3, weight_decay=1e-2)),
    ("proj128", dict(proj_dim=128)),
    ("batch4096", dict(head_batch=4096)),
    ("drop0.3+proj128+wd1e-2+ep80", dict(dropout=0.3, proj_dim=128, weight_decay=1e-2, epochs=80)),
    ("drop0.5+wd1e-2+batch4096", dict(dropout=0.5, weight_decay=1e-2, head_batch=4096)),
]


def make_cfg(args, **overrides):
    cfg = dict(
        proj_dim=args.proj_dim, hidden_dim=args.hidden_dim, dropout=args.dropout,
        temp_init=args.temp_init, epochs=args.epochs, head_batch=args.head_batch,
        lr=args.lr, weight_decay=args.weight_decay, seed=args.seed,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def train_head(A_tr, V_tr, A_val, V_val, cfg, device):
    head = AlignmentHead(A_tr.shape[1], cfg.hidden_dim, cfg.proj_dim,
                         temp_init=cfg.temp_init, dropout=cfg.dropout).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    A_tr, V_tr = A_tr.to(device), V_tr.to(device)
    n = A_tr.shape[0]
    g = torch.Generator(device=device).manual_seed(cfg.seed)

    best_state, best_score, best_metrics, best_epoch = None, -1.0, None, -1
    for epoch in range(cfg.epochs):
        head.train()
        perm = torch.randperm(n, generator=g, device=device)
        for i in range(0, n, cfg.head_batch):
            idx = perm[i : i + cfg.head_batch]
            if idx.numel() < 2:
                continue
            loss = clip_loss(head(A_tr[idx], V_tr[idx]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()

        m = head_metrics(head, A_val, V_val, device)
        if mean_r1(m) > best_score:
            best_score, best_metrics, best_epoch = mean_r1(m), m, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    head.load_state_dict(best_state)
    return head, best_metrics, best_epoch


def get_features(model, device, args, ds, split, cache_path):
    if os.path.exists(cache_path):
        print(f"Loaded cached {split} features <- {cache_path}")
        return torch.load(cache_path)
    if split == "train":
        keys = build_subset_keys(ds.train_csv, args.train_per_class, args.train_seed)
        loader = build_loader(ds, keys, ds.train_csv, ds.train_tar, args.train_clips,
                              args.batch_size, args.num_workers)
    else:
        keys = load_subset_file(args.subset_file, ds)
        loader = build_loader(ds, keys, ds.test_csv, ds.test_tar, args.eval_clips,
                              args.batch_size, args.num_workers)
    print(f"Extracting frozen {split} features ({len(keys)} keys)...")
    A, V = extract_embeddings(model, loader, device, use_projector=False)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save({"A": A, "V": V}, cache_path)
    return {"A": A, "V": V}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    ds = DATASETS[DATASET]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = args.cache_dir or os.path.join(args.output_dir, "cache")
    ckpt_id = os.path.splitext(os.path.basename(args.checkpoint_path))[0].replace("=", "")
    tr_cache = os.path.join(cache_dir, f"{ckpt_id}_train_{args.train_per_class}pc_"
                                       f"seed{args.train_seed}_clips{args.train_clips}.pt")
    ev_cache = os.path.join(cache_dir, f"{ckpt_id}_eval_clips{args.eval_clips}.pt")

    model = None
    if not (os.path.exists(tr_cache) and os.path.exists(ev_cache)):
        model = load_model(args.checkpoint_path, device)
    feats_ev = get_features(model, device, args, ds, "eval", ev_cache)
    feats_tr = get_features(model, device, args, ds, "train", tr_cache)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    A_ev, V_ev = feats_ev["A"], feats_ev["V"]
    print("\n=== Raw [CLS] features, no head ===")
    print(format_metrics(compute_retrieval_metrics(A_ev, V_ev)))

    A_all, V_all = feats_tr["A"], feats_tr["V"]
    n = A_all.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    n_val = max(1, int(round(args.val_frac * n)))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    A_tr, V_tr, A_val, V_val = A_all[tr_idx], V_all[tr_idx], A_all[val_idx], V_all[val_idx]
    print(f"\nHead training on {len(tr_idx)} pairs, selection on {len(val_idx)} held-out pairs")

    runs, selected = [], None
    for name, overrides in SWEEP_GRID:
        cfg = make_cfg(args, **overrides)
        head, val_m, epoch = train_head(A_tr, V_tr, A_val, V_val, cfg, device)
        ev = head_metrics(head, A_ev, V_ev, device)
        runs.append({"name": name, "config": vars(cfg), "best_epoch": epoch,
                     "val_metrics": val_m, "eval_metrics": ev})
        print(f"  {name:30s} held-out R@1={mean_r1(val_m):5.2f} (ep {epoch})")
        if selected is None or mean_r1(val_m) > mean_r1(selected["val_metrics"]):
            selected = runs[-1]
            selected_head = head

    print(f"\n=== Selected head [{selected['name']}] on the retrieval pool ===")
    print(format_metrics(selected["eval_metrics"]))

    out = os.path.join(args.output_dir, "retrieval_infonce_head.json")
    with open(out, "w") as f:
        json.dump({"config": vars(args), "selected": selected, "runs": runs}, f, indent=2)
    torch.save(selected_head.state_dict(), os.path.join(args.output_dir, "retrieval_infonce_head.pt"))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
