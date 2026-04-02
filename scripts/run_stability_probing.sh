#!/bin/bash
# Step 2: Stability probing and filtering
# Usage: bash scripts/run_stability_probing.sh <MODEL> <OUTPUT_DIR> [TAU]

MODEL=${1:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"}
OUTPUT_DIR=${2:-"outputs"}
TAU=${3:-0.8}

echo "=== Step 2a: Re-generate from each boundary prefix (M=10) ==="
python src/filter/probe_behavior_stability.py \
    --model_name_or_path $MODEL \
    --hidden_dir $OUTPUT_DIR/hidden_states \
    --output_dir $OUTPUT_DIR/stability \
    --M 10 \
    --temperature 0.7 \
    --top_p 0.95 \
    --max_new_tokens 128

echo "=== Step 2b: Build stability-filtered vectors (tau=$TAU) ==="
python src/filter/build_vectors_from_behavior_stability.py \
    --stability_dir $OUTPUT_DIR/stability \
    --vector_dir $OUTPUT_DIR/vectors \
    --output_dir $OUTPUT_DIR/vectors_stable \
    --tau $TAU
