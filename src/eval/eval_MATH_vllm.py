#!/usr/bin/env python3
# claim_1/eval_MATH_vllm.py
import argparse, os, re, json, random, torch, evaluate
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from datasets import load_dataset
from functools import partial
from get_math_results import main as eval_main

os.environ["TOKENIZERS_PARALLELISM"] = "false"
exact_match = evaluate.load("exact_match")


def logit_adjustment(token_ids, logits, adjust_ids, values, max_len=-1):
    if max_len <= 0 or len(token_ids) <= max_len:
        logits[adjust_ids.to(logits.device)] += values
    return logits


def trim_output(output: str) -> str:
    instruction_prefix = "Answer the following question"
    question_prefix = "Question:"
    comment_prefix = "Comment:"
    for prefix in [instruction_prefix, question_prefix, comment_prefix]:
        if prefix in output:
            output = output.split(prefix)[0]
    return output


def extract_box(pred_str: str) -> str:
    ans = pred_str.split("boxed")[-1]
    if len(ans) == 0:
        return ""
    if ans[0] == "{":
        stack, a = 1, ""
        for c in ans[1:]:
            if c == "{":
                stack += 1; a += c
            elif c == "}":
                stack -= 1
                if stack == 0: break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a


def _prompt_injection(style: str, override_path: str | None = None) -> str:
    if override_path:
        try:
            return open(override_path, "r").read().strip()
        except Exception:
            pass
    BUILTIN = {
        "R_plus": (
            "You are solving math problems. Before giving the final answer, "
            "briefly verify your steps in a separate sentence. It is OK to use phrases "
            'like "let\'s check", "verify", "double check", "hold on", or "confirm". '
            "Keep the verification brief and then finish. Put the final numeric answer in \\boxed{}."
        ),
        "R_minus": (
            "Solve the problem in a single, direct pass. Do NOT write reflection or verification language. "
            "Do NOT use words/phrases such as: wait, hmm, let's check, let me check, verify, make sure, "
            "hold on, think again, re-check, double check, check again, reconsider, sanity check, confirm, validate. "
            "Just present the solution and final answer in \\boxed{}."
        ),
        "T_plus": (
            "If you suspect a better method mid-solution, explicitly switch to an alternative approach. "
            'You may use a sentence starting with "alternatively" or phrases like "another approach", '
            '"another way", "different approach", "try a different method", or "switch strategy". '
            "Keep it concise and then finish with the final answer in \\boxed{}."
        ),
        "T_minus": (
            "Stick to your initial approach. Do NOT switch to an alternative approach mid-solution. "
            "Do NOT use words/phrases such as: alternatively, another way, another approach, "
            "different approach, another method, another solution, another strategy, try a different, switch. "
            "Provide a single coherent solution and the final answer in \\boxed{}."
        ),
    }
    return BUILTIN.get(style, "")


def main(args):
    random.seed(42)

    # ---- Load data
    print("Loading data...")
    test_data = []
    if args.data_path:
        with open(args.data_path) as fin:
            for line in fin:
                ex = json.loads(line)
                if "problem" in ex and "solution" in ex:
                    gt = extract_box(ex["solution"])
                    test_data.append({"question": ex["problem"], "answer": ex["solution"], "gt": gt})
                elif "question" in ex and "answer" in ex:
                    answer = ex["answer"].split("####")[1].strip() if "####" in ex["answer"] else ex["answer"].strip()
                    answer = re.sub(r"(\d),(\d)", r"\1\2", answer)
                    rationale = ex["answer"].split("####")[0].strip() if "####" in ex["answer"] else ex["answer"].strip()
                    test_data.append({"question": ex["question"], "answer": rationale, "gt": answer})
                else:
                    raise ValueError(f"Unsupported custom jsonl schema in {args.data_path}")
    elif args.dataset == "MATH500":
        data = load_dataset("HuggingFaceH4/MATH-500", split="test")
        for ex in data:
            gt = extract_box(ex["solution"])
            test_data.append({"question": ex["problem"], "answer": ex["solution"], "gt": gt})
    elif args.dataset == "MATH_train":
        with open("data/MATH/train.jsonl") as fin:
            for line in fin:
                ex = json.loads(line)
                gt = extract_box(ex["solution"])
                test_data.append({"question": ex["problem"], "answer": ex["solution"], "gt": gt})
    elif args.dataset in ["GSM", "GSM_train"]:
        data_path = "data/gsm/train.jsonl" if args.dataset == "GSM_train" else "data/gsm/test.jsonl"
        with open(data_path) as fin:
            for line in fin:
                ex = json.loads(line)
                answer = ex["answer"].split("####")[1].strip()
                answer = re.sub(r"(\d),(\d)", r"\1\2", answer)
                test_data.append({"question": ex["question"], "answer": ex["answer"].split("####")[0].strip(), "gt": answer})
    else:
        raise ValueError("Dataset not supported")

    if args.max_examples and len(test_data) > args.max_examples:
        test_data = test_data[:args.max_examples]

    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Tokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    # ---- Prompts with optional prompt-style injection
    prefix = "Answer the following questions. You should think step-by-step and put your final answer within \\boxed{}.\n"
    sys_txt = _prompt_injection(args.prompt_style, args.prompt_system_txt) if hasattr(args, "prompt_style") else ""
    prompts = []
    for ex in test_data:
        if args.use_chat_format:
            msgs = []
            if sys_txt:
                msgs.append({"role": "system", "content": sys_txt})
            if "gemma" in args.model_name_or_path or "deepseek" in args.model_name_or_path:
                msgs.append({"role": "user", "content": prefix + "Question: " + ex["question"].strip()})
            else:
                msgs.append({"role": "system", "content": prefix})
                msgs.append({"role": "user", "content": "Question: " + ex["question"].strip()})
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            if args.remove_bos and tok.bos_token is not None and prompt.startswith(tok.bos_token):
                prompt = prompt[len(tok.bos_token):]
        else:
            core = prefix + "Question: " + ex["question"].strip() + "\nAnswer: "
            prompt = (sys_txt + "\n\n" + core) if sys_txt else core
        prompts.append(prompt)

    with open(os.path.join(args.save_dir, "example_prompt.txt"), "w") as f:
        f.write(prompts[0])

    # ---- vLLM model
    util = float(os.getenv("VLLM_UTIL", "0.50"))
    extra = int(os.getenv("VLLM_EXTRA_TOK", "512"))
    tp = int(os.getenv("VLLM_TP", "1"))
    model = LLM(
        model=args.model_name_or_path,
        tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path,
        swap_space=16,
        gpu_memory_utilization=util,
        enable_lora=args.peft is not None,
        tensor_parallel_size=tp,
        max_lora_rank=128,
        max_model_len=args.max_tokens + extra,
    )

    if not args.logit_adjustment:
        sampling_params = SamplingParams(n=1, temperature=0, max_tokens=args.max_tokens)
    else:
        vocab = tok.get_vocab()
        ids = [vocab[tok] for tok in vocab.keys() if any(x in tok for x in args.logit_adjustment_tokens)]
        logit_adjustment_tokens = torch.LongTensor(ids).to("cuda")
        process = partial(logit_adjustment, adjust_ids=logit_adjustment_tokens, values=args.logit_adjustment_value, max_len=args.logit_adjustment_max_len)
        sampling_params = SamplingParams(n=1, temperature=0, max_tokens=args.max_tokens, logits_processors=[process])

    if args.peft is not None:
        outs = model.generate(prompts=prompts, sampling_params=sampling_params, lora_request=LoRARequest("lora_path", 1, lora_path=args.peft))
    else:
        outs = model.generate(prompts=prompts, sampling_params=sampling_params)

    # ---- Collect
    result = []
    for o in outs:
        attempts = [ith.text for ith in o.outputs]
        result.append(attempts)
    outputs = [[trim_output(x) for x in attempt_list] for attempt_list in result]

    preds = [
        {
            "prompt": p,
            "problem": ex["question"],
            "answer": ex["gt"],
            "solution": ex["answer"],
            "model_generation": out,
        }
        for ex, out, p in zip(test_data, outputs, prompts)
    ]

    with open(os.path.join(args.save_dir, "predictions.jsonl"), "w") as f:
        for row in preds:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="results/gsm")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--peft", type=str, default=None)
    parser.add_argument("--use_chat_format", action="store_true")
    parser.add_argument("--dataset", type=str, default="MATH")
    parser.add_argument("--remove_bos", action="store_true", default=True)
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--logit_adjustment", action="store_true", default=False)
    parser.add_argument("--logit_adjustment_tokens", type=str, nargs="*", default=[])
    parser.add_argument("--logit_adjustment_value", type=float, default=0.0)
    parser.add_argument("--logit_adjustment_max_len", type=int, default=-1)
    # prompt-only baseline controls
    parser.add_argument("--prompt_style", type=str, default="none",
                        choices=["none", "R_plus", "R_minus", "T_plus", "T_minus"])
    parser.add_argument("--prompt_system_txt", type=str, default=None)
    args = parser.parse_args()

    # save_dir naming
    if args.logit_adjustment:
        name = "_".join(args.logit_adjustment_tokens) + f"_value_{args.logit_adjustment_value}"
        if args.logit_adjustment_max_len > 0:
            name += f"_first{args.logit_adjustment_max_len}"
        args.save_dir = os.path.join(args.save_dir, "logit-adjustment", name)
    else:
        # base path
        os.makedirs(args.save_dir, exist_ok=True)

    if getattr(args, "prompt_style", "none") != "none":
        args.save_dir = os.path.join(args.save_dir, f"prompt_{args.prompt_style}")

    print(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
    eval_main(os.path.join(args.save_dir, "predictions.jsonl"), save=True, k=None, output_dir=args.save_dir)
