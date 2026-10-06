
eval "$(conda shell.bash hook)"
conda activate leavjepa

export AUDIOSET_ROOT="${AUDIOSET_ROOT:-/path/to/AudioSet}"
export VGGSOUND_ROOT="${VGGSOUND_ROOT:-/path/to/VGGSound}"
export ESC50_ROOT="${ESC50_ROOT:-/path/to/ESC-50}"

export OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/../outputs}"
mkdir -p "$OUTPUT_ROOT"

export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
