#!/usr/bin/env bash
set -euo pipefail

GPUS="${1:-0}"
NUM_ITER="${2:-5000}"
SAVE_FREQUENCY="${3:-500}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="parallel_runs/lai_paper_${RUN_ID}"
mkdir -p "$RUN_DIR/shards" "$RUN_DIR/logs" logs results_2tucker_newname

read -r -a GPU_ARRAY <<< "$GPUS"
mapfile -t IMAGES < <(find "$ROOT/datasets/lai/nonuniform" -maxdepth 1 -type f -name "*.png" | sort)

if [ "${#IMAGES[@]}" -eq 0 ]; then
  echo "No PNG images found under datasets/lai/nonuniform" >&2
  exit 1
fi

for gpu in "${GPU_ARRAY[@]}"; do
  mkdir -p "$RUN_DIR/shards/gpu_${gpu}"
done

for i in "${!IMAGES[@]}"; do
  shard=$(( i % ${#GPU_ARRAY[@]} ))
  gpu="${GPU_ARRAY[$shard]}"
  ln -sf "${IMAGES[$i]}" "$RUN_DIR/shards/gpu_${gpu}/$(basename "${IMAGES[$i]}")"
done

PIDS_FILE="logs/parallel_lai_paper_${RUN_ID}.pids"
: > "$PIDS_FILE"

for gpu in "${GPU_ARRAY[@]}"; do
  shard_dir="$RUN_DIR/shards/gpu_${gpu}"
  count="$(find "$shard_dir" -maxdepth 1 -type l -name "*.png" | wc -l)"
  log="$RUN_DIR/logs/gpu_${gpu}.log"
  nohup python mytest_nonuni_2tucker_lr=cos_finetune_paper_consistent_gpu.py \
    --gpu "$gpu" \
    --data_path "$shard_dir" \
    --save_path ./results_2tucker_newname \
    --num_iter "$NUM_ITER" \
    --save_frequency "$SAVE_FREQUENCY" \
    > "$log" 2>&1 &
  pid="$!"
  printf 'gpu=%s pid=%s count=%s shard=%s log=%s num_iter=%s save_frequency=%s\n' \
    "$gpu" "$pid" "$count" "$shard_dir" "$log" "$NUM_ITER" "$SAVE_FREQUENCY" | tee -a "$PIDS_FILE"
done

ln -sfn "$(basename "$PIDS_FILE")" logs/parallel_lai_paper_latest.pids
echo "Run directory: $RUN_DIR"
echo "PID file: $PIDS_FILE"
