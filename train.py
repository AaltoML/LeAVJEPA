import argparse
import os
from datetime import datetime

import lightning as L
import torch


if os.environ.get("ECHO_DISABLE_DDP_OPTIMIZER", "0") == "1":
    import torch._dynamo

    torch._dynamo.config.optimize_ddp = False

from configs import audio_config, transformer_config, video_config
from data import compute_audioset_pos_weight, get_dataloader
from dataset_config import DATASETS
from models import EchoTrainer

parser = argparse.ArgumentParser(description="Pretrain the audio-visual encoder")

parser.add_argument("--dataset", type=str, default="audioset_256", choices=list(DATASETS.keys()))
parser.add_argument("--num_classes", type=int, default=None,
                    help="Classes for the online probe (default: from the dataset)")
parser.add_argument("--num_frames", type=int, default=16)
parser.add_argument("--frame_size", type=int, default=224)
parser.add_argument("--batch_size", type=int, default=64, help="Per-GPU batch size")
parser.add_argument("--num_workers", type=int, default=11)
parser.add_argument("--num_workers_test", type=int, default=4)
parser.add_argument("--prefetch_factor", type=int, default=3)
parser.add_argument("--num_eval_clips", type=int, default=4)
parser.add_argument("--test_tar_override", type=str, default=None,
                    help="Alternative test shard pattern (needs >= 1 shard per rank)")

parser.add_argument("--num_global_views", type=int, default=2, help="G")
parser.add_argument("--num_local_views", type=int, default=2, help="K")
parser.add_argument("--cross_modal", action="store_true",
                    help="Modality-specific local views: even audio-only, odd video-only")
parser.add_argument("--unimodal_token_drop", action="store_true",
                    help="Encode audio-only / video-only local views without the absent "
                         "modality's tokens (requires --cross_modal)")
parser.add_argument("--modality_drop", type=float, default=0.5,
                    help="Random modality dropout for joint local views (ablations)")
parser.add_argument("--clean_survivor", action="store_true",
                    help="With --modality_drop: keep the surviving modality unmasked")
parser.add_argument("--video_mask", type=float, default=0.80,
                    help="Block-tube mask ratio for joint local views (ablations)")
parser.add_argument("--audio_mask", type=float, default=0.5,
                    help="Max frequency/time mask width, as a fraction of each axis, "
                         "for joint local views (ablations)")
parser.add_argument("--audio_only", action="store_true",
                    help="Audio-only pretraining ablation (video never decoded)")

parser.add_argument("--vit_size", type=str, default="base", choices=["base", "large"])
parser.add_argument("--dual_encoder", action="store_true",
                    help="Separate audio/video encoders (ablation; requires --cross_modal)")
parser.add_argument("--proj_dim", type=int, default=128)
parser.add_argument("--lambd", type=float, default=0.05, help="SIGReg weight lambda")
parser.add_argument("--gradient_checkpointing", action="store_true")

parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--weight_decay", type=float, default=5e-2)
parser.add_argument("--wd_exclude_norm_embed", action="store_true",
                    help="No weight decay on 1-D params and positional/CLS/type embeddings")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--max_steps", type=int, default=-1)
parser.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation")
parser.add_argument("--num_gpus", type=int, default=1)
parser.add_argument("--num_nodes", type=int, default=1)
parser.add_argument("--seed", type=int, default=None)

parser.add_argument("--no_probe", action="store_true", help="Disable the online linear probe")
parser.add_argument("--attentive_probe", action="store_true")
parser.add_argument("--probe_lr", type=float, default=1e-3)
parser.add_argument("--probe_weight_decay", type=float, default=0.0)
parser.add_argument("--eval_every_n_epochs", type=int, default=0,
                    help="Held-out probe evaluation every N epochs (0 = off)")
parser.add_argument("--run_test", action="store_true", help="Probe test evaluation at the end")

parser.add_argument("--checkpoint", type=str, default=None, help="Resume from this checkpoint")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
parser.add_argument("--ckpt_every_n_steps", type=int, default=2000)
parser.add_argument("--ckpt_every_n_epochs", type=int, default=0,
                    help="Save every N epochs instead of every --ckpt_every_n_steps steps")
parser.add_argument("--ckpt_keep_last", type=int, default=-1, choices=[-1, 0, 1],
                    help="-1 keep all, 1 keep newest, 0 none")
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--wandb_project", type=str, default=None,
                    help="Default: echo-<dataset>. Set WANDB_MODE=offline to log locally.")
parser.add_argument("--wandb_id", type=str, default=None,
                    help="Stable run id so chained jobs resume one W&B run")
args = parser.parse_args()

if args.prefetch_factor < 1:
    parser.error("--prefetch_factor must be >= 1")
if args.dual_encoder and not args.cross_modal:
    parser.error("--dual_encoder requires --cross_modal")
if args.unimodal_token_drop and (not args.cross_modal or args.dual_encoder
                                 or args.num_local_views == 0):
    parser.error("--unimodal_token_drop requires --cross_modal, the shared encoder "
                 "and --num_local_views > 0")
if args.no_probe and (args.dual_encoder or args.attentive_probe):
    parser.error("--no_probe is not supported with --dual_encoder / --attentive_probe")
if args.audio_only and (args.cross_modal or args.dual_encoder or args.modality_drop != 0.0):
    parser.error("--audio_only requires --modality_drop 0 and no --cross_modal / --dual_encoder")

if args.seed is not None:
    L.seed_everything(args.seed, workers=True)

a_config = audio_config()
v_config = video_config(args.num_frames, args.frame_size)
t_config = transformer_config(args.vit_size)
freq_mask_param = int(a_config["spectrogram_size"][0] * args.audio_mask)
time_mask_param = int(a_config["spectrogram_size"][1] * args.audio_mask)

ds = DATASETS[args.dataset]
world_size = args.num_gpus * args.num_nodes
print(f"Dataset: {ds.name} ({ds.num_classes} classes, multi_label={ds.multi_label})")

train_loader, test_loader, classes = get_dataloader(
    tar_path=ds.train_tar,
    csv_path=ds.train_csv,
    test_tar_path=args.test_tar_override or ds.test_tar,
    test_csv_path=ds.test_csv,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    num_workers_test=args.num_workers_test,
    prefetch_factor=args.prefetch_factor,
    frame_size=(args.frame_size, args.frame_size),
    num_frames=args.num_frames,
    num_global_views=args.num_global_views,
    num_local_views=args.num_local_views,
    num_eval_clips=args.num_eval_clips,
    train_size=ds.train_size,
    test_size=ds.test_size,
    world_size=world_size,
    video_mask_ratio=args.video_mask,
    freq_mask_param=freq_mask_param,
    time_mask_param=time_mask_param,
    modality_drop_prob=args.modality_drop,
    clean_survivor=args.clean_survivor,
    cross_modal=args.cross_modal,
    csv_format=ds.csv_format,
    spec_mean=ds.spec_mean,
    spec_std=ds.spec_std,
    audio_only=args.audio_only,
)
num_classes = args.num_classes if args.num_classes is not None else len(classes)

pos_weight = compute_audioset_pos_weight(ds.train_csv, classes) if ds.multi_label else None

model = EchoTrainer(
    a_config=a_config,
    v_config=v_config,
    t_config=t_config,
    lr=args.lr,
    weight_decay=args.weight_decay,
    lambd=args.lambd,
    num_views=args.num_local_views,
    batch_size=args.batch_size,
    epochs=args.epochs,
    proj_dim=args.proj_dim,
    num_classes=num_classes,
    probe_lr=args.probe_lr,
    probe_weight_decay=args.probe_weight_decay,
    wd_exclude_norm_embed=args.wd_exclude_norm_embed,
    cross_modal=args.cross_modal,
    unimodal_token_drop=args.unimodal_token_drop,
    gradient_checkpointing=args.gradient_checkpointing,
    attentive_probe=args.attentive_probe,
    dual_encoder=args.dual_encoder,
    multi_label=ds.multi_label,
    total_samples=ds.train_size,
    pos_weight=pos_weight,
    num_global_views=args.num_global_views,
    no_probe=args.no_probe,
)

slurm_id = os.environ.get("SLURM_JOB_ID", "local")
run_dir = f"echo_{args.dataset}/{slurm_id}/{datetime.now().strftime('%d-%m-%H:%M:%S')}"


ckpt_trigger = (
    {"every_n_epochs": args.ckpt_every_n_epochs, "save_on_train_epoch_end": True}
    if args.ckpt_every_n_epochs > 0
    else {"every_n_train_steps": args.ckpt_every_n_steps}
)
checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
    dirpath=f"{args.checkpoint_dir}/{run_dir}",
    filename=f"echo-{args.dataset}" + "-{step}",
    save_top_k=args.ckpt_keep_last,
    monitor=None,
    **ckpt_trigger,
)
logger = L.pytorch.loggers.WandbLogger(
    project=args.wandb_project or f"echo-{args.dataset}",
    name=args.run_name or slurm_id,
    save_dir="runs",
    config=vars(args),
    id=args.wandb_id,
    resume="allow" if args.wandb_id else None,
)

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

periodic_eval = args.eval_every_n_epochs > 0
trainer = L.Trainer(
    max_epochs=args.epochs,
    max_steps=args.max_steps,
    accelerator="gpu",
    devices=args.num_gpus,
    num_nodes=args.num_nodes,
    strategy="ddp" if world_size > 1 else "auto",
    precision="bf16-mixed",
    logger=logger,
    log_every_n_steps=10,
    callbacks=[checkpoint_callback],
    gradient_clip_val=5.0,
    gradient_clip_algorithm="norm",
    accumulate_grad_batches=args.accum_steps,
    check_val_every_n_epoch=args.eval_every_n_epochs if periodic_eval else 1,
    num_sanity_val_steps=2 if periodic_eval else 0,
)

if args.epochs > 0:
    trainer.fit(
        model,
        train_loader,
        val_dataloaders=test_loader if periodic_eval else None,
        ckpt_path=args.checkpoint,
    )

if args.run_test:
    trainer.test(model, dataloaders=test_loader)
