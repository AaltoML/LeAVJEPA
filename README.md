# LeAVJEPA: A Minimalist Architecture for Audio-Visual Self-Supervised Learning

Official code for **LeAVJEPA: A Minimalist Architecture for Audio-Visual Self-Supervised Learning**.

**[Paper](https://arxiv.org/abs/2610.06226) · [Project page](https://aaltoml.github.io/LeAVJEPA/)**

Benjamin Robson, Santeri Mentu, Wenshuai Zhao, Arno Solin

ELLIS Institute Finland and Department of Computer Science, Aalto University

LeAVJEPA learns audio, video, and joint audio-video representations with a shared
early-fusion Vision Transformer. Modality-specific local views align with global
views under the LeJEPA objective, with SIGReg preventing representation collapse.
The release includes ViT-B and ViT-L pretraining, downstream fine-tuning, frozen
attentive probes, and audio-video retrieval.

This repository now contains the LeAVJEPA code. The earlier AV-JEPA workshop
version is available in the [repository history](https://github.com/AaltoML/AV-JEPA/tree/3eefa234b841945b5f72111cba4c33aa3944df9f).
The implementation retains the internal names `Echo`, `EchoTrainer`, and
`EchoFineTuner` in `models.py` and the `echo` prefix in training logs and checkpoints.

## Setup

The supplied environment targets Linux with NVIDIA GPUs and CUDA 12.9 wheels
for PyTorch 2.9.1. It includes Python 3.12, Lightning, TorchCodec, torchaudio,
WebDataset, and FFmpeg.

```bash
conda env create -f environment.yml
conda activate leavjepa
```

Datasets and model checkpoints are not included. Pretrain a model with the
scripts below, then supply the resulting checkpoint for downstream evaluation.

## Data

AudioSet and VGGSound use WebDataset tar shards containing one `.mp4` clip
(video with its audio track) per sample, with labels read from CSV files.
Set absolute dataset paths before running scripts:

```bash
export AUDIOSET_ROOT=/path/to/AudioSet
export VGGSOUND_ROOT=/path/to/VGGSound
export ESC50_ROOT=/path/to/ESC-50
export OUTPUT_ROOT=/path/to/outputs
export WANDB_MODE=offline
```

For SLURM jobs, these variables can also be configured in [`jobs/env.sh`](jobs/env.sh).
That script activates the `leavjepa` conda environment, defaults to offline W&B
logging, and creates the output directory. Direct Python runs default to dataset
directories under `./data`; job scripts use the paths from `jobs/env.sh`.

Expected layout:

```text
$AUDIOSET_ROOT/
  shards/data_{000..453}.tar
  shards_256/data_{000..453}.tar
  unbalanced_train_segments.csv
  balanced_train_segments.csv
  eval_segments.csv
$VGGSOUND_ROOT/
  train_tars/vggsound_train_{00..71}.tar
  test_tars/vggsound_test_{00..07}.tar
  train_tars_256/vggsound_train_{00..71}.tar
  test_tars_256/vggsound_test_{00..07}.tar
  train.csv
  test.csv
$ESC50_ROOT/
  audio/*.wav
  meta/esc50.csv
```

Dataset choices are `audioset`, `audioset_256`, `audioset_20k`, `vggsound`, and
`vggsound_256`. The `_256` variants use videos resized to 256 pixels; only the
shards needed by your selected dataset are required. AudioSet-20K uses the
balanced training CSV and the regular AudioSet shards. Edit
[`dataset_config.py`](dataset_config.py) if your shard patterns, sample counts,
or normalization statistics differ.

[`datafiles/`](datafiles/) includes the CAV-MAE AudioSet retrieval subset.
Retrieval reads its `video_id` entries and loads media from your AudioSet shards;
the original media paths in the JSON are unused.

## Pretraining

Submit the supplied AudioSet configurations from `jobs/`. Adjust the SLURM
account, partition, and resource directives for your cluster, and create the log
directory before submitting:

```bash
cd jobs
mkdir -p slurm_logs
sbatch pretrain-audioset-vitb.job
# Or train ViT-L:
sbatch pretrain-audioset-vitl.job
```

Both configurations train on `audioset_256` for 50 epochs with eight GPUs,
two global views, two modality-specific local views, and SIGReg weight 0.05.
The per-GPU batch sizes are 64 for ViT-B and 128 for ViT-L. The jobs use
[`chain.sh`](jobs/chain.sh) to resume checkpoints across SLURM time windows.

To launch the ViT-B configuration directly from the repository root:

```bash
python train.py \
  --dataset audioset_256 --vit_size base \
  --cross_modal --unimodal_token_drop \
  --num_global_views 2 --num_local_views 2 \
  --lambd 0.05 --lr 5e-4 --wd_exclude_norm_embed \
  --epochs 50 --batch_size 64 --num_gpus 8 \
  --num_workers 15 --prefetch_factor 1 \
  --num_frames 16 --frame_size 224 --proj_dim 128 \
  --gradient_checkpointing --no_probe --seed 0 \
  --ckpt_every_n_epochs 1 --checkpoint_dir ./checkpoints
```

`--cross_modal` creates alternating audio-only and video-only local views;
`--unimodal_token_drop` omits the absent modality's tokens. Use `--checkpoint`
to resume pretraining and `--vit_size large` for ViT-L. Training logs to W&B;
`WANDB_MODE=offline` keeps logs local.

## Fine-tuning and frozen evaluation

Run these commands from `jobs/` after configuring your dataset paths and
creating `slurm_logs`. Supply a LeAVJEPA pretraining checkpoint through `CKPT`:

```bash
export CKPT=/path/to/pretrain-vitb.ckpt
sbatch finetune-vggsound-vitb.job
sbatch finetune-audioset20k-vitb.job
sbatch probe-as20k.job
sbatch esc50-frozen.job
sbatch esc50-finetune.job
```

| Task | Entry point | SLURM configuration |
| --- | --- | --- |
| VGGSound fine-tuning | `finetune.py` | `finetune-vggsound-vitb.job`, `finetune-vggsound-vitl.job` |
| AudioSet-20K fine-tuning | `finetune.py` | `finetune-audioset20k-vitb.job` |
| AudioSet-20K frozen attentive probes | `token_probe.py` | `probe-as20k.job` |
| ESC-50 frozen attentive probes | `esc50_probe.py` | `esc50-frozen.job` |
| ESC-50 fine-tuning | `esc50_finetune.py` | `esc50-finetune.job` |

Match the model size to the checkpoint: use `finetune-vggsound-vitl.job` for
ViT-L fine-tuning, and set `VIT=large` for `probe-as20k.job` and the ESC-50 jobs.
For direct Python commands, set `--vit_size large` as appropriate.

The AudioSet-20K probe job extracts frozen tokens, fits attentive heads, and
evaluates audio, video, and joint modalities. It removes cached training tokens
after each modality's evaluation. ESC-50 evaluation uses the dataset's five folds.
The job files contain the full configurations; each Python entry point supports
`--help` for additional options.

## Audio-video retrieval

Zero-shot retrieval evaluates audio-to-video and video-to-audio R@1/5/10 on
the included AudioSet subset. From the repository root:

```bash
python retrieval.py \
  --checkpoint_path /path/to/pretrain.ckpt \
  --num_eval_clips 4 --batch_size 32 --num_workers 16 \
  --output_dir ./outputs/retrieval
```

From `jobs/`, submit `CKPT=/path/to/pretrain.ckpt sbatch retrieval-zeroshot.job`.
The model configuration is loaded from the checkpoint.

[`retrieval_head.py`](retrieval_head.py) separately trains an InfoNCE alignment
head on frozen encoder features. Run it with
`CKPT=/path/to/pretrain.ckpt sbatch retrieval-infonce-head.job` from `jobs/`.
This is a downstream retrieval experiment; LeAVJEPA pretraining uses the LeJEPA
objective.

## Ablations

From `jobs/`, `bash ablation-submit.sh` submits the VGGSound component ablations
defined in `ablation-vggsound.job`. For sensitivity experiments, select a variant
and its value when submitting `sensitivity-vggsound.job`:

```bash
VARIANT=lambda LAMBD=0.1 sbatch sensitivity-vggsound.job
VARIANT=views K=4 sbatch sensitivity-vggsound.job
VARIANT=dual sbatch sensitivity-vggsound.job
```

These jobs use the same dataset and output configuration in `env.sh`.

## Repository layout

| File | Purpose |
| --- | --- |
| `train.py` | Audio-visual self-supervised pretraining |
| `finetune.py` | End-to-end classification fine-tuning |
| `token_probe.py`, `probe_heads.py` | Frozen token extraction and attentive probes |
| `esc50_probe.py`, `esc50_finetune.py` | ESC-50 frozen evaluation and fine-tuning |
| `retrieval.py`, `retrieval_head.py` | Zero-shot retrieval and a learned alignment head |
| `models.py` | Encoder, LeJEPA objective, SIGReg, and Lightning modules |
| `configs.py` | Shared ViT-B/ViT-L, audio, and video configurations |
| `fusion.py`, `transformer.py` | Audio/video embeddings and transformer blocks |
| `data.py`, `dataset_config.py` | Data pipeline and dataset registry |
| `jobs/` | SLURM training and evaluation configurations |
| `datafiles/` | AudioSet retrieval subset metadata |

## Citation

```bibtex
@article{robson2026leavjepa,
  title   = {{LeAVJEPA}: A Minimalist Architecture for Audio-Visual Self-Supervised Learning},
  author  = {Robson, Benjamin and Mentu, Santeri and Zhao, Wenshuai and Solin, Arno},
  journal = {arXiv preprint arXiv:2610.06226},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.06226}
}
```

## License

This code is released under the [MIT license](LICENSE).
