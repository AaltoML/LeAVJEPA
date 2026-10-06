import argparse
import glob
import os
from datetime import datetime

import lightning as L
import torch

from configs import audio_config, transformer_config, video_config
from data import compute_audioset_pos_weight, get_dataloader
from dataset_config import DATASETS
from models import EchoFineTuner, load_pretrained

parser = argparse.ArgumentParser(description="Fine-tune a pretrained encoder")
parser.add_argument("--pretrained_checkpoint", type=str, required=True,
                    help="Pretraining (EchoTrainer) checkpoint")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Fine-tuned checkpoint to load weights from (e.g. for --epochs 0 --run_test)")
parser.add_argument("--resume_from", type=str, default=None,
                    help="Resume a fine-tuning run (optimizer + schedule); 'auto' = newest "
                         "checkpoint under the run directory")
parser.add_argument("--dataset", type=str, default="vggsound", choices=list(DATASETS.keys()))
parser.add_argument("--vit_size", type=str, default="base", choices=["base", "large"])
parser.add_argument("--num_frames", type=int, default=16)
parser.add_argument("--frame_size", type=int, default=224)
parser.add_argument("--gradient_checkpointing", action="store_true")

parser.add_argument("--lr", type=float, default=2e-4, help="Head learning rate")
parser.add_argument("--backbone_lr_scale", type=float, default=0.1,
                    help="Backbone LR = lr * this scale")
parser.add_argument("--weight_decay", type=float, default=5e-2)
parser.add_argument("--warmup_fraction", type=float, default=0.05)
parser.add_argument("--label_smoothing", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=40, help="Per-GPU batch size")
parser.add_argument("--accum_steps", type=int, default=1)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--num_gpus", type=int, default=1)
parser.add_argument("--num_nodes", type=int, default=1)

parser.add_argument("--attentive_probe", action="store_true",
                    help="Also train an attentive head on the patch tokens")
parser.add_argument("--mixup_alpha", type=float, default=0.0)
parser.add_argument("--modality_drop_prob", type=float, default=0.0,
                    help="Per-sample probability of zeroing one modality")
parser.add_argument("--freq_mask_param", type=int, default=0, help="SpecAugment frequency mask")
parser.add_argument("--time_mask_param", type=int, default=0, help="SpecAugment time mask")
parser.add_argument("--random_resized_crop", action="store_true")
parser.add_argument("--rrc_min_scale", type=float, default=0.5)
parser.add_argument("--no_pos_weight", action="store_true",
                    help="Disable BCE pos_weight for multi-label datasets")

parser.add_argument("--num_workers", type=int, default=9)
parser.add_argument("--num_workers_test", type=int, default=2)
parser.add_argument("--num_eval_clips", type=int, default=6)
parser.add_argument("--test_tar_override", type=str, default=None)
parser.add_argument("--val_every_n_epochs", type=int, default=1,
                    help="Validation (= a full test pass) every N epochs; monitoring only")
parser.add_argument("--skip_val", action="store_true")
parser.add_argument("--run_test", action="store_true")

parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--ckpt_every_n_steps", type=int, default=500)
parser.add_argument("--ckpt_keep_last", type=int, default=-1, choices=[-1, 0, 1])
parser.add_argument("--wandb_project", type=str, default="echo-finetune")
args = parser.parse_args()

a_config = audio_config()
v_config = video_config(args.num_frames, args.frame_size)
t_config = transformer_config(args.vit_size)

print(f"Loading pretrained checkpoint: {args.pretrained_checkpoint}")
encoder = load_pretrained(
    args.pretrained_checkpoint,
    a_config=a_config,
    v_config=v_config,
    t_config=t_config,
    gradient_checkpointing=args.gradient_checkpointing,
).encoder

ds = DATASETS[args.dataset]

train_loader, test_loader, classes = get_dataloader(
    tar_path=ds.train_tar,
    csv_path=ds.train_csv,
    test_tar_path=args.test_tar_override or ds.test_tar,
    test_csv_path=ds.test_csv,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    num_workers_test=args.num_workers_test,
    frame_size=(args.frame_size, args.frame_size),
    num_frames=args.num_frames,
    num_global_views=1,
    num_local_views=1,
    num_eval_clips=args.num_eval_clips,
    train_size=ds.train_size,
    test_size=ds.test_size,
    world_size=args.num_gpus * args.num_nodes,
    video_mask_ratio=0.0,
    freq_mask_param=args.freq_mask_param,
    time_mask_param=args.time_mask_param,
    spec_aug_global=(args.freq_mask_param > 0 or args.time_mask_param > 0),
    global_rrc_min_scale=args.rrc_min_scale if args.random_resized_crop else 0.0,
    modality_drop_prob=0.0,
    csv_format=ds.csv_format,
    spec_mean=ds.spec_mean,
    spec_std=ds.spec_std,
)

pos_weight = None
if ds.multi_label and not args.no_pos_weight:
    pos_weight = compute_audioset_pos_weight(ds.train_csv, classes)

model = EchoFineTuner(
    encoder=encoder,
    hidden_size=t_config["hidden_size"],
    num_classes=len(classes),
    lr=args.lr,
    backbone_lr_scale=args.backbone_lr_scale,
    weight_decay=args.weight_decay,
    warmup_fraction=args.warmup_fraction,
    label_smoothing=args.label_smoothing,
    batch_size=args.batch_size,
    total_samples=ds.train_size,
    epochs=args.epochs,
    attentive_probe=args.attentive_probe,
    num_attention_heads=t_config["num_attention_heads"],
    multi_label=ds.multi_label,
    pos_weight=pos_weight,
    mixup_alpha=args.mixup_alpha,
    modality_drop_prob=args.modality_drop_prob,
)
if args.checkpoint:
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state, strict=False)

slurm_id = os.environ.get("SLURM_JOB_ID", "local")
run_name = args.run_name or f"finetune_{args.dataset}/{slurm_id}/{datetime.now().strftime('%d-%m-%H:%M:%S')}"
checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
    dirpath=f"{args.checkpoint_dir}/{run_name}",
    filename="finetune-{step}",
    every_n_train_steps=args.ckpt_every_n_steps,
    save_top_k=args.ckpt_keep_last,
    monitor=None,
)
logger = L.pytorch.loggers.WandbLogger(
    project=args.wandb_project, name=slurm_id, save_dir="runs", config=vars(args)
)

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

total_devices = args.num_gpus * args.num_nodes
trainer = L.Trainer(
    max_epochs=args.epochs,
    accelerator="gpu",
    devices=args.num_gpus,
    num_nodes=args.num_nodes,
    strategy="ddp" if total_devices > 1 else "auto",
    precision="bf16-mixed",
    logger=logger,
    log_every_n_steps=10,
    check_val_every_n_epoch=args.val_every_n_epochs,
    callbacks=[checkpoint_callback],
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
    accumulate_grad_batches=args.accum_steps,
)

resume_path = args.resume_from
if resume_path == "auto":
    candidates = glob.glob(f"{args.checkpoint_dir}/{run_name}/*.ckpt")
    resume_path = max(candidates, key=os.path.getmtime) if candidates else None

if args.epochs > 0:
    trainer.fit(
        model,
        train_loader,
        val_dataloaders=None if args.skip_val else test_loader,
        ckpt_path=resume_path,
    )

if args.run_test:
    trainer.test(model, dataloaders=test_loader)
