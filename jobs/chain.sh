CHAIN_RUN="$1"; CHAIN_JOB="$2"
CHAIN_DIR="$OUTPUT_ROOT/checkpoints/$CHAIN_RUN"
mkdir -p "$CHAIN_DIR"
if [ -f "$CHAIN_DIR/DONE" ]; then echo "$CHAIN_RUN already complete"; exit 0; fi

latest=$(find "$CHAIN_DIR" -name '*-step=*.ckpt' 2>/dev/null \
  | awk -F'step=' '{ split($2, a, "."); print a[1], $0 }' | sort -n | tail -1 | cut -d' ' -f2-)
RESUME_FLAG=""
if [ -n "${RESUME_CKPT:-}" ]; then RESUME_FLAG="--checkpoint $RESUME_CKPT"
elif [ -n "$latest" ]; then RESUME_FLAG="--checkpoint $latest"; fi
echo "run=$CHAIN_RUN resume=${RESUME_FLAG:-<fresh start>}"

queue_next() {
  echo "time limit approaching: queueing the next window"
  (cd "$SLURM_SUBMIT_DIR" && sbatch --dependency=afterany:$SLURM_JOB_ID "$CHAIN_JOB")
}
trap queue_next USR1

chain_wait() {
  local pid=$! rc
  wait $pid; rc=$?
  while [ $rc -gt 128 ] && kill -0 $pid 2>/dev/null; do wait $pid; rc=$?; done
  [ $rc -eq 0 ] && touch "$CHAIN_DIR/DONE"
  exit $rc
}
