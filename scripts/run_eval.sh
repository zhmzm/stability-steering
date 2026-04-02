#!/bin/bash
# Step 4: Evaluate on MATH-500
# Usage: bash scripts/run_eval.sh <MODEL> <VECTOR_PATH> [COEF] [MAX_TOKENS]

MODEL=${1:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"}
VECTOR=${2:-"outputs/vectors_combined/combined.pt"}
COEF=${3:--100}
MAX_TOKENS=${4:-4096}

echo "=== Evaluating on MATH-500 ==="
python src/eval/eval_MATH_vllm_steering.py \
    --model_name_or_path $MODEL \
    --steering_vector $VECTOR \
    --dataset MATH500 \
    --use_chat_format \
    --remove_bos \
    --max_tokens $MAX_TOKENS \
    --coef $COEF \
    --batch_size 8

echo "=== Computing results ==="
python src/eval/get_math_results.py --eval_dir outputs/eval
