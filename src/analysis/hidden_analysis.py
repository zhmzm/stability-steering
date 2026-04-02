# claim_1/hidden_analysis.py
import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_math_data(data_dir, data_path):
    correct, incorrect = [], []
    with open(data_path) as f:
        data = [json.loads(line) for line in f]
    with open(os.path.join(data_dir, "math_eval.jsonl")) as f:
        eval_lines = [json.loads(line) for line in f]

    data = data[:len(eval_lines)]
    for d, e in zip(data, eval_lines):
        local_correct, local_incorrect = [], []
        prompt = e["prompt"]
        assert d["problem"] == e["problem"]
        for o, c in zip(e["model_generation"], e["all_eval"]):
            item = {"prompt": prompt, "response": o, "level": d.get("level", "NA"), "gt": e["answer"]}
            if c:
                local_correct.append(item)
            else:
                local_incorrect.append(item)
        correct.extend(local_correct)
        incorrect.extend(local_incorrect)
    return correct, incorrect

def generate_index(text, tokenizer, split_id, think_only=True):
    # lowercase everything for robust matching
    CHECK_WORDS = [
        "verify","make sure","hold on","think again","'s correct","'s incorrect",
        "let me check","seems right","re-check","double check","double-check","check again","reconsider"
    ]
    CHECK_PREFIX = ["wait"]
    SWITCH_WORDS = [
        "think differently","another way","another approach","another method",
        "another solution","another strategy","another technique","try a different approach",
        "let's switch","change approach","different plan"
    ]
    SWITCH_PREFIX = ["alternatively"]

    tokens = tokenizer.encode(text)
    if think_only:
        think_begin_id = tokenizer.encode("<think>", add_special_tokens=False)[0]
        think_end_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
        if think_begin_id not in tokens:
            return [], [], []

        start = tokens.index(think_begin_id) + 1
        if think_end_id not in tokens[start:]:
            end = len(tokens)
        else:
            end = tokens.index(think_end_id, start)
        think_tokens = tokens[start:end]
    else:
        think_tokens = tokens
        start = 0

    index = [i for i, t in enumerate(think_tokens) if t in split_id] + [len(think_tokens)]
    step_index, check_index, switch_index = [], [], []

    for i in range(len(index) - 1):
        step_index.append(index[i] + start)
        step = think_tokens[index[i] + 1:index[i + 1]]
        step_text = tokenizer.decode(step).strip(" ").strip("\n").lower()
        if any(step_text.startswith(p) for p in CHECK_PREFIX) or any(w in step_text for w in CHECK_WORDS):
            check_index.append(i)
        elif any(step_text.startswith(p) for p in SWITCH_PREFIX) or any(w in step_text for w in SWITCH_WORDS):
            switch_index.append(i)
    return step_index, check_index, switch_index

def generate(model_path, data, save_dir):
    think_only = "deepseek" in model_path.lower()
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    vocab = tokenizer.get_vocab()
    split_id = [vocab[token] for token in vocab.keys() if "ĊĊ" in token]

    prompts = [d["prompt"] + d["response"] for d in data]

    layer_num = model.config.num_hidden_layers + 1
    hidden_dict = [{} for _ in range(layer_num)]

    for k, p in tqdm(enumerate(prompts), total=len(prompts)):
        tokenized = tokenizer([p], return_tensors="pt", padding=True)
        tokenized = {kk: vv.to(model.device) for kk, vv in tokenized.items()}
        with torch.no_grad():
            output = model(**tokenized, output_hidden_states=True)
            hidden_states = [h.detach().cpu() for h in output.hidden_states]

        step_index, check_index, switch_index = generate_index(p, tokenizer, split_id, think_only=think_only)
        step_index = torch.LongTensor(step_index)
        check_index = torch.LongTensor(check_index)
        switch_index = torch.LongTensor(switch_index)

        for i in range(len(hidden_states)):
            h = hidden_states[i][0]           # (seq_len, D)
            step_h = h[step_index] if len(step_index) > 0 else h[:0]  # (S, D) or empty
            hidden_dict[i][k] = {
                "step": step_h,
                "check_index": check_index,
                "switch_index": switch_index
            }

        del hidden_states

    os.makedirs(save_dir, exist_ok=True)
    torch.save(hidden_dict, os.path.join(save_dir, "hidden.pt"))
    json.dump(prompts, open(os.path.join(save_dir, "prompts.json"), "w"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--type", type=str, default="correct", choices=["correct", "incorrect", "mv"])
    args = parser.parse_args()

    correct, incorrect = generate_math_data(data_dir=args.data_dir, data_path=args.data_path)

    if args.type == "mv":
        with open(args.data_path) as f:
            data_lines = [json.loads(l) for l in f]
        with open(os.path.join(args.data_dir, "math_eval.jsonl")) as f:
            eval_lines = [json.loads(l) for l in f]
        data_lines = data_lines[:len(eval_lines)]
        data = []
        for d, e in zip(data_lines, eval_lines):
            mv_idx = e["mv_index"]
            mv_resp = e["model_generation"][mv_idx]
            data.append({
                "prompt": e["prompt"],
                "response": mv_resp,
                "level": d.get("level", "NA"),
                "gt": e["answer"],
            })
        save_dir = os.path.join(args.data_dir, "hidden_mv")
    elif args.type == "correct":
        data = correct
        save_dir = os.path.join(args.data_dir, "hidden_correct")
    else:
        data = incorrect
        save_dir = os.path.join(args.data_dir, "hidden_incorrect")

    if args.start != -1:
        data = data[args.start:]
        if args.sample != -1:
            data = data[:args.sample]
            save_dir = f"{save_dir}_{args.start}_{args.start + args.sample}"
        else:
            save_dir = f"{save_dir}_{args.start}_-1"

    print(save_dir)
    generate(args.model_path, data, save_dir)
