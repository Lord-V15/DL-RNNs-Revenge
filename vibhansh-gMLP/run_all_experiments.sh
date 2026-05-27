#!/bin/bash
# Final standardized run: all 9 gMLP cells with matched protocol
# - All tasks: dropout=0.05, seed=42
# - Shakespeare/Copy: 5K steps
# - Induction: 20K steps, LR decay over 50K
#
# GPU strategy (sequential within task, parallel across tasks):
#   GPU 0: reserved for minGRU notebook
#   GPU 1: Shakespeare 256 → 1024 → 2048 (sequential)
#   GPU 2: Copy short → medium → long (sequential)
#   GPU 3: Induction short
#   GPU 4: Induction medium
#   GPU 5: Induction long

set -e
cd "$(dirname "$0")"

SEED=42
OUTPUT_DIR="./results"
mkdir -p $OUTPUT_DIR

echo "=============================================="
echo "Causal gMLP - Final Standardized Run"
echo "9 cells (3 tasks × 3 lengths)"
echo "Protocol: dropout=0.05, seed=42"
echo "          induction: 20K steps / 50K LR decay"
echo "GPU 1: Shakespeare | GPU 2: Copy | GPU 3-5: Induction"
echo "=============================================="
echo ""

# ===== GPU 1: Shakespeare (sequential) =====
(
  echo "[GPU 1] Starting Shakespeare 256..."
  CUDA_VISIBLE_DEVICES=1 python3 train.py \
    --task shakespeare \
    --block_size 256 \
    --seed $SEED \
    --max_steps 5000 \
    --batch_size 64 \
    --dropout 0.05 \
    --output_dir $OUTPUT_DIR \
    --save_checkpoint \
    > "$OUTPUT_DIR/gmlp_shakespeare_256_${SEED}.out" 2>&1

  echo "[GPU 1] Starting Shakespeare 1024..."
  CUDA_VISIBLE_DEVICES=1 python3 train.py \
    --task shakespeare \
    --block_size 1024 \
    --seed $SEED \
    --max_steps 5000 \
    --batch_size 64 \
    --dropout 0.05 \
    --output_dir $OUTPUT_DIR \
    --save_checkpoint \
    > "$OUTPUT_DIR/gmlp_shakespeare_1024_${SEED}.out" 2>&1

  echo "[GPU 1] Starting Shakespeare 2048..."
  CUDA_VISIBLE_DEVICES=1 python3 train.py \
    --task shakespeare \
    --block_size 2048 \
    --seed $SEED \
    --max_steps 5000 \
    --batch_size 32 \
    --dropout 0.05 \
    --output_dir $OUTPUT_DIR \
    --save_checkpoint \
    > "$OUTPUT_DIR/gmlp_shakespeare_2048_${SEED}.out" 2>&1

  echo "[GPU 1] Shakespeare done."
) &

# ===== GPU 2: Copy (sequential) =====
(
  echo "[GPU 2] Starting Copy short..."
  CUDA_VISIBLE_DEVICES=2 python3 train.py \
    --task copy \
    --length short \
    --seed $SEED \
    --max_steps 5000 \
    --batch_size 64 \
    --dropout 0.05 \
    --output_dir $OUTPUT_DIR \
    > "$OUTPUT_DIR/gmlp_copy_short_${SEED}.out" 2>&1

  echo "[GPU 2] Starting Copy medium..."
  CUDA_VISIBLE_DEVICES=2 python3 train.py \
    --task copy \
    --length medium \
    --seed $SEED \
    --max_steps 5000 \
    --batch_size 64 \
    --dropout 0.05 \
    --output_dir $OUTPUT_DIR \
    > "$OUTPUT_DIR/gmlp_copy_medium_${SEED}.out" 2>&1

  echo "[GPU 2] Starting Copy long..."
  CUDA_VISIBLE_DEVICES=2 python3 train.py \
    --task copy \
    --length long \
    --seed $SEED \
    --max_steps 5000 \
    --batch_size 32 \
    --dropout 0.05 \
    --output_dir $OUTPUT_DIR \
    > "$OUTPUT_DIR/gmlp_copy_long_${SEED}.out" 2>&1

  echo "[GPU 2] Copy done."
) &

# ===== GPU 3: Induction short =====
(
  echo "[GPU 3] Starting Induction short..."
  CUDA_VISIBLE_DEVICES=3 python3 train.py \
    --task induction \
    --length short \
    --seed $SEED \
    --max_steps 20000 \
    --lr_decay_steps 50000 \
    --batch_size 64 \
    --dropout 0.05 \
    --output_dir $OUTPUT_DIR \
    > "$OUTPUT_DIR/gmlp_induction_short_${SEED}.out" 2>&1
  echo "[GPU 3] Induction short done."
) &

# ===== GPU 4: Induction medium =====
(
  echo "[GPU 4] Starting Induction medium..."
  CUDA_VISIBLE_DEVICES=4 python3 train.py \
    --task induction \
    --length medium \
    --seed $SEED \
    --max_steps 20000 \
    --lr_decay_steps 50000 \
    --batch_size 64 \
    --dropout 0.05 \
    --output_dir $OUTPUT_DIR \
    > "$OUTPUT_DIR/gmlp_induction_medium_${SEED}.out" 2>&1
  echo "[GPU 4] Induction medium done."
) &

# ===== GPU 5: Induction long =====
(
  echo "[GPU 5] Starting Induction long..."
  CUDA_VISIBLE_DEVICES=5 python3 train.py \
    --task induction \
    --length long \
    --seed $SEED \
    --max_steps 20000 \
    --lr_decay_steps 50000 \
    --batch_size 32 \
    --dropout 0.05 \
    --output_dir $OUTPUT_DIR \
    > "$OUTPUT_DIR/gmlp_induction_long_${SEED}.out" 2>&1
  echo "[GPU 5] Induction long done."
) &

wait
echo ""
echo "=============================================="
echo "All 9 experiments complete!"
echo "Results: $OUTPUT_DIR/*.json"
echo "Logs:    $OUTPUT_DIR/*.out"
echo "=============================================="
