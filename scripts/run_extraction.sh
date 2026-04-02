#!/bin/bash
# Step 1: Extract steering vectors from MATH training problems
# Usage: bash scripts/run_extraction.sh <MODEL> <OUTPUT_DIR>

MODEL=${1:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"}
OUTPUT_DIR=${2:-"outputs"}

echo "=== Step 1a: Generate CoT and extract hidden states ==="
python src/extract/vector_generation.py \
    --model_name_or_path $MODEL \
    --dataset MATH \
    --split train \
    --n_samples 100 \
    --max_tokens 8192 \
    --output_dir $OUTPUT_DIR/hidden_states \
    --layer 20

echo "=== Step 1b: Build per-question steering vectors ==="
python src/extract/build_behavior_vectors.py \
    --hidden_dir $OUTPUT_DIR/hidden_states \
    --output_dir $OUTPUT_DIR/vectors

echo "=== Step 1c: Extract question-only hidden states ==="
python src/extract/extract_question_hidden.py \
    --model_name_or_path $MODEL \
    --hidden_dir $OUTPUT_DIR/hidden_states \
    --output_dir $OUTPUT_DIR/question_hidden \
    --layer 20
