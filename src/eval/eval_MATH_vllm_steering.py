#!/usr/bin/env python3
"""Evaluate MATH with vLLM + steering vector injection.

Requires modified vLLM qwen2.py/llama.py with steering support.
Sets steering vector on the model's internal Qwen2Model/LlamaModel before generation.
Forces V0 engine for model_executor API access.

Usage:
    python eval_MATH_vllm_steering.py \
        --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --dataset MATH500 \
        --steering_vector path/to/vector.pt \
        --steering_layer 20 \
        --steering_coef -100 \
        --save_dir results/
"""
import os
os.environ["VLLM_USE_V1"] = "0"

import argparse, json, sys, re
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def load_math_data(dataset, data_path=None, max_examples=None, start=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if data_path:
        with open(data_path) as f:
            data = [json.loads(l) for l in f]
    elif dataset == "MATH500":
        with open(os.path.join(script_dir, "data/MATH/test.jsonl")) as f:
            data = [json.loads(l) for l in f]
    elif dataset in ("GSM", "GSM8K"):
        with open(os.path.join(script_dir, "data/gsm/test.jsonl")) as f:
            data = [json.loads(l) for l in f]
    else:
        with open(os.path.join(script_dir, f"data/MATH/{dataset}.jsonl")) as f:
            data = [json.loads(l) for l in f]

    if start is not None:
        data = data[start:]
    if max_examples is not None:
        data = data[:max_examples]

    test_data = []
    for d in data:
        test_data.append({
            "question": d["problem"],
            "answer": d.get("answer", d.get("solution", "")),
            "gt": d.get("answer", d.get("solution", "")),
        })
    return test_data


def trim_output(text):
    if "</think>" in text:
        idx = text.index("</think>")
        return text[:idx + len("</think>")]
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--tokenizer_name_or_path", default=None)
    parser.add_argument("--dataset", default="MATH500")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--use_chat_format", action="store_true")
    parser.add_argument("--remove_bos", action="store_true", default=True)
    parser.add_argument("--steering_vector", type=str, default=None)
    parser.add_argument("--steering_layer", type=int, default=20)
    parser.add_argument("--steering_coef", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=None, help="ignored, vLLM handles batching")
    parser.add_argument("--steering", action="store_true", default=False)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Load tokenizer
    tok_name = args.tokenizer_name_or_path or args.model_name_or_path
    tok = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Load data
    test_data = load_math_data(args.dataset, args.data_path, args.max_examples, args.start)
    print(f"Loaded {len(test_data)} examples")

    # Build prompts
    prompts = []
    for ex in test_data:
        if args.use_chat_format:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": ex["question"]}],
                tokenize=False, add_generation_prompt=True,
            )
            if args.remove_bos and prompt.startswith(tok.bos_token or ""):
                prompt = prompt[len(tok.bos_token):]
        else:
            prompt = ex["question"] + "\nAnswer: "
        prompts.append(prompt)

    with open(os.path.join(args.save_dir, "example_prompt.txt"), "w") as f:
        f.write(prompts[0])

    # Load vLLM model
    llm = LLM(
        model=args.model_name_or_path,
        tokenizer=tok_name,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        max_model_len=args.max_tokens + 2000,
        tensor_parallel_size=torch.cuda.device_count(),
        enforce_eager=True,  # Disable CUDA graph to ensure steering injection executes
    )

    # Set steering vector via direct model access (requires VLLM_USE_V1=0)
    if args.steering and args.steering_vector:
        steer_vec = torch.load(args.steering_vector, weights_only=True)
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        if hasattr(model, 'model') and hasattr(model.model, 'set_steering'):
            model.model.set_steering(
                vector=steer_vec,
                layer=args.steering_layer,
                coef=args.steering_coef,
            )
            print(f"[STEERING] Set vector at layer {args.steering_layer}, coef={args.steering_coef}")
        else:
            print("[WARNING] Model does not support set_steering. Running without steering.")

    # Generate
    sampling_params = SamplingParams(
        n=1, temperature=0, max_tokens=args.max_tokens,
    )
    outputs = llm.generate(prompts, sampling_params)

    # Collect results
    results = []
    for i, output in enumerate(outputs):
        text = output.outputs[0].text
        text = trim_output(text)
        results.append([text])

    # Save predictions
    predictions = [
        {
            "prompt": p,
            "problem": ex["question"],
            "answer": ex["gt"],
            "solution": ex["gt"],
            "model_generation": out,
        }
        for ex, out, p in zip(test_data, results, prompts)
    ]

    pred_path = os.path.join(args.save_dir, "predictions.jsonl")
    with open(pred_path, "w") as f:
        for row in predictions:
            f.write(json.dumps(row) + "\n")

    # Run evaluation
    eval_script = os.path.join(os.path.dirname(__file__), "eval_math_rule", "evaluate.py")
    if os.path.exists(eval_script):
        import subprocess
        subprocess.run([
            sys.executable, eval_script,
            "--prediction_path", pred_path,
            "--answer_key", "answer",
        ], check=False)

    print(f"[DONE] Saved to {args.save_dir}")


if __name__ == "__main__":
    main()
