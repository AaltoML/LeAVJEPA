import argparse
import json
import os
from collections import defaultdict
import random

import torch
import torch.nn.functional as F
import webdataset as wds
from torch.utils.data import DataLoader

from data import VideoAudioPipeline
from dataset_config import DATASETS
from models import load_pretrained

DATASET = "audioset"


def parse_args():
    p = argparse.ArgumentParser(description="Zero-shot audio<->video retrieval")
    p.add_argument("--checkpoint_path", type=str, required=True,
                   help="Pretraining (EchoTrainer) checkpoint")
    p.add_argument("--subset_file", type=str,
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "datafiles",
                                        "audioset_eval_5_per_class_for_retrieval_cleaned.json"),
                   help="Retrieval pool: CAV-MAE-style json (data[].video_id)")
    p.add_argument("--num_eval_clips", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--output_dir", type=str, default="./outputs/retrieval")
    return p.parse_args()


def _read_audioset_csv(csv_path):
    with open(csv_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split(", ", 3)
            if len(parts) < 4:
                continue
            start_ms = round(float(parts[1]) * 1000)
            end_ms = round(float(parts[2]) * 1000)
            yield f"{parts[0]}_{start_ms}_{end_ms}", parts[0], parts[3].strip('"').split(",")


def load_subset_file(path, ds):
    with open(path) as f:
        ids = {entry["video_id"] for entry in json.load(f)["data"]}
    keys, matched = set(), set()
    for key, ytid, _ in _read_audioset_csv(ds.test_csv):
        if ytid in ids:
            keys.add(key)
            matched.add(ytid)
    if ids - matched:
        print(f"  {len(ids - matched)} of {len(ids)} subset ids not in the eval CSV")
    return keys


def build_subset_keys(csv_path, samples_per_class, seed):
    rng = random.Random(seed)
    rows = list(_read_audioset_csv(csv_path))
    class_counts = defaultdict(int)
    for _, _, label_ids in rows:
        for lid in label_ids:
            class_counts[lid] += 1
    by_class = defaultdict(list)
    for key, _, label_ids in rows:
        if label_ids:
            by_class[min(label_ids, key=lambda l: class_counts[l])].append(key)
    chosen = set()
    for cls in sorted(by_class.keys()):
        keys = by_class[cls]
        rng.shuffle(keys)
        chosen.update(keys[:samples_per_class])
    return chosen


def build_loader(ds, keys, csv_path, tar_path, num_eval_clips, batch_size, num_workers):
    pipeline = VideoAudioPipeline(
        csv_path,
        is_train=False,
        debug=True,
        num_eval_clips=num_eval_clips,
        csv_format=ds.csv_format,
        spec_mean=ds.spec_mean,
        spec_std=ds.spec_std,
    )
    pipeline.labels_map = {k: v for k, v in pipeline.labels_map.items() if k in keys}
    print(f"  pool: {len(pipeline.labels_map)} clips with labels ({len(keys)} requested)")
    dataset = (
        wds.WebDataset(
            tar_path,
            shardshuffle=False,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
            empty_check=False,
        )
        .select(pipeline.has_label)
        .map(pipeline.process)
        .select(lambda x: x is not None)
        .batched(batch_size)
    )
    return DataLoader(dataset, batch_size=None, num_workers=num_workers,
                      persistent_workers=False, pin_memory=True)


def load_model(checkpoint_path, device):
    print(f"Loading checkpoint: {checkpoint_path}")
    return load_pretrained(checkpoint_path).to(device).eval()


@torch.no_grad()
def extract_embeddings(model, loader, device, use_projector=True):
    audio_embs, video_embs = [], []
    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        audio = batch["spectrogram"].to(device, non_blocking=True)
        B, N = video.shape[:2]
        a = model.encoder(None, audio.view(B * N, *audio.shape[2:]))
        v = model.encoder(video.view(B * N, *video.shape[2:]), None)
        if use_projector:
            a, v = model.projector(a), model.projector(v)
        audio_embs.append(a.view(B, N, -1).mean(dim=1).float().cpu())
        video_embs.append(v.view(B, N, -1).mean(dim=1).float().cpu())
    return torch.cat(audio_embs), torch.cat(video_embs)


def compute_retrieval_metrics(A, V, ks=(1, 5, 10)):
    S = F.normalize(A, dim=-1) @ F.normalize(V, dim=-1).t()

    def ranks(sim):
        return ((sim > sim.diag().unsqueeze(1)).sum(dim=1) + 1).float()

    out = {"N": S.shape[0]}
    for direction, r in (("a2v", ranks(S)), ("v2a", ranks(S.t()))):
        for k in ks:
            out[f"{direction}_R@{k}"] = (r <= k).float().mean().item() * 100
        out[f"{direction}_mean_rank"] = r.mean().item()
        out[f"{direction}_median_rank"] = float(r.median().item())
    return out


def format_metrics(m):
    lines = [f"N = {m['N']}"]
    for d in ("a2v", "v2a"):
        lines.append(
            f"  {d.upper()}: " + "  ".join(f"R@{k}={m[f'{d}_R@{k}']:5.2f}" for k in (1, 5, 10))
            + f"   mean={m[f'{d}_mean_rank']:6.1f}   median={m[f'{d}_median_rank']:5.0f}"
        )
    return "\n".join(lines)


def main():
    args = parse_args()
    ds = DATASETS[DATASET]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint_path, device)

    keys = load_subset_file(args.subset_file, ds)
    loader = build_loader(ds, keys, ds.test_csv, ds.test_tar, args.num_eval_clips,
                          args.batch_size, args.num_workers)
    A, V = extract_embeddings(model, loader, device)
    metrics = compute_retrieval_metrics(A, V)
    print("\n=== Zero-shot retrieval (projected [CLS], cosine) ===")
    print(format_metrics(metrics))

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "retrieval_zeroshot.json")
    with open(out, "w") as f:
        json.dump({"config": vars(args), "metrics": metrics}, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
