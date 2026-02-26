#!/usr/bin/env bash
# Run all missing benchmarks:
#   - Modal (GPU): gradient attacks for YOLOv8 + YOLO26
#   - Local (CPU): non-gradient attacks for YOLO26 pretrained + finetuned
#
# Existing results (YOLOv8 non-gradient) are NOT overwritten.
#
# Usage:
#   bash scripts/run_all_benchmarks.sh

set -euo pipefail
cd "$(dirname "$0")/.."

# Prevent sleep while benchmarks run
caffeinate -dims -w $$ &

DATA="tracking_ws/data"
VISDRONE="$DATA/VisDrone2019-MOT-val"
RESULTS="$DATA/results"

echo "============================================"
echo " 1/3  Launching gradient benchmarks on Modal"
echo "      (YOLOv8 + YOLO26, parallel A100s)"
echo "============================================"
modal run --detach scripts/modal_benchmark_gradient.py

echo ""
echo "============================================"
echo " 2/3  YOLO26 pretrained — non-gradient (local)"
echo "      Output: $RESULTS/yolo26n_pretrained/"
echo "============================================"
python3 scripts/benchmark.py \
    --sequence-dir "$VISDRONE" \
    --model yolo26n.pt \
    --skip-gradient \
    --device cpu \
    --output "$RESULTS/yolo26n_pretrained"

echo ""
echo "============================================"
echo " 3/3  YOLO26 finetuned — non-gradient (local)"
echo "      Output: $RESULTS/yolo26n_finetuned/"
echo "============================================"
python3 scripts/benchmark.py \
    --sequence-dir "$VISDRONE" \
    --model "$DATA/yolo26n_visdrone_best.pt" \
    --skip-gradient \
    --device cpu \
    --output "$RESULTS/yolo26n_finetuned"

echo ""
echo "============================================"
echo " Local benchmarks done!"
echo ""
echo " When Modal jobs finish, download gradient results:"
echo "   modal volume get visdrone-data /benchmark_gradient/ $DATA/benchmark_gradient/"
echo ""
echo " Final results layout:"
echo "   $RESULTS/pretrained/          — YOLOv8 pretrained (non-gradient) ✓ existing"
echo "   $RESULTS/finetuned/           — YOLOv8 finetuned  (non-gradient) ✓ existing"
echo "   $RESULTS/yolo26n_pretrained/  — YOLO26 pretrained (non-gradient)"
echo "   $RESULTS/yolo26n_finetuned/   — YOLO26 finetuned  (non-gradient)"
echo "   $DATA/benchmark_gradient/     — all gradient results (from Modal)"
echo "============================================"
