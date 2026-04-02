#!/usr/bin/env python3
# mine_vs_seal/eval_MATH_steering.py
import argparse, os, re, json, random, torch, evaluate
from transformers import AutoTokenizer
from modeling_utils.modeling_qwen2 import Qwen2ForCausalLM
from tqdm import trange
from datasets import load_dataset
from get_math_results import main as eval_main
from typing import Dict, Any
import textwrap

os.environ["TOKENIZERS_PARALLELISM"] = "false"
exact_match = evaluate.load("exact_match")


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
    """
    Return the *user-prepended* instruction text.
    If override_path is provided and readable, use its content verbatim.
    Otherwise fall back to built-ins.

    NOTE: We keep legacy keys for back-compat (R_plus/R_minus/T_plus/T_minus),
    and add the exact variants you requested (more_refl/less_refl/more_trans/less_trans).
    """
    if override_path:
        try:
            return open(override_path, "r").read().strip()
        except Exception:
            pass

    BUILTIN = {
        # --- EXACT required variants (prompt-only baseline) ---
        "more_refl": (
            "Think carefully, reflect on each step, double-check your reasoning before giving the final answer."
        ),
        "less_refl": (
            "Answer directly and concisely without extra checking or reflection."
        ),
        "more_trans": (
            "Explore alternative approaches, be creative in your reasoning, and consider different solution paths before deciding on the final answer."
        ),
        "less_trans": (
            "Stick to the most straightforward single solution path without exploring alternatives."
        ),

        # --- Legacy names kept for compatibility ---
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
    elif args.dataset == "GSM":
        data_path = "data/gsm/test.jsonl"
        with open(data_path) as fin:
            for line in fin:
                ex = json.loads(line)
                answer = ex["answer"].split("####")[1].strip()
                answer = re.sub(r"(\d),(\d)", r"\1\2", answer)
                test_data.append({"question": ex["question"], "answer": ex["answer"].split("####")[0].strip(), "gt": answer})
    else:
        raise ValueError("Dataset not supported")

    if args.start:
        test_data = test_data[args.start:]
    if args.max_examples and len(test_data) > args.max_examples:
        test_data = test_data[:args.max_examples]

    # save_dir will be finalized later after path construction
    _original_save_dir = args.save_dir

    # ---- Tokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    # ---- Build prompts (with optional prompt-style PREPENDED to user text)
    prefix = "Answer the following questions. You should think step-by-step and put your final answer within \\boxed{}.\n"

    # This returns the *instruction we will prepend to the user-visible prompt text*.
    injected = _prompt_injection(args.prompt_style, args.prompt_system_txt) if hasattr(args, "prompt_style") else ""
    injected = textwrap.dedent(injected).strip() if injected else ""

    prompts = []
    for ex in test_data:
        core_user = prefix + "Question: " + ex["question"].strip()
        if injected:
            core_user = f"{injected}\n\n{core_user}"

        if args.use_chat_format:
            msgs = [{"role": "user", "content": core_user}]
            # Build final text via chat template but ensure removal of BOS if requested
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            if args.remove_bos and tok.bos_token is not None and prompt.startswith(tok.bos_token):
                prompt = prompt[len(tok.bos_token):]
        else:
            # Non-chat: we prepend directly to the flat prompt
            prompt = core_user + "\nAnswer: "
        prompts.append(prompt)

    with open(os.path.join(args.save_dir, "example_prompt.txt"), "w") as f:
        f.write(prompts[0])

    # ---- Model
    model_lower = args.model_name_or_path.lower()
    # Auto-detect architecture from config
    from transformers import AutoConfig
    _cfg = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    _arch = getattr(_cfg, 'model_type', '').lower()
    # Qwen2 arch (DeepSeek-R1-Distill-Qwen, DeepScaleR, VibeThinker, Nemotron, etc.)
    if _arch == 'qwen2' or (("qwen" in model_lower or "deepseek" in model_lower) and "llama" not in model_lower):
        model = Qwen2ForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto")
        # ---- Steering (optional)
        if args.steering:
            steer_vec = torch.load(args.steering_vector, weights_only=True).to(model.device)
            model.set_steering_flag(
                steering_flag=True,
                steering_layer=args.steering_layer,
                steer_vec=steer_vec,
                steer_coef=args.steering_coef,
                tokenizer=tok,
            )
    elif "llama" in model_lower or "gemma" in model_lower:
        from modeling_utils.steering_wrapper import SteerableCausalLM
        model = SteerableCausalLM.from_pretrained(args.model_name_or_path, device_map="auto")
        if args.steering:
            steer_vec = torch.load(args.steering_vector, weights_only=True).to(model.device)
            model.set_steering(
                steering_flag=True,
                steering_layer=args.steering_layer,
                steer_vec=steer_vec,
                steer_coef=args.steering_coef,
                tokenizer=tok,
            )
    else:
        raise ValueError(f"Model not supported: {args.model_name_or_path}. Supported: qwen, deepseek, llama, gemma")

    # ---- Generate
    outputs = []
    for i in trange(0, len(prompts), args.batch_size):
        if args.steering:
            model.start_new_round()
        batch = prompts[i : i + args.batch_size]
        tokenized = tok(batch, return_tensors="pt", padding=True)
        tokenized = {k: v.to(model.device) for k, v in tokenized.items()}
        with torch.no_grad():
            gens = model.generate(**tokenized, do_sample=False, max_new_tokens=args.max_tokens, use_cache=True)
        prompt_len = tokenized["input_ids"].shape[1]
        decoded = [tok.decode(o[prompt_len:], skip_special_tokens=True) for o in gens]
        outputs.extend(decoded)

    outputs = [[trim_output(o)] for o in outputs]

    # ---- Save predictions
    predictions = [
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
        for row in predictions:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="results/gsm")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--use_chat_format", action="store_true")
    parser.add_argument("--dataset", type=str, default="MATH")
    parser.add_argument("--max_tokens", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--remove_bos", action="store_true", default=True)
    parser.add_argument("--steering", action="store_true", default=False)
    parser.add_argument("--steering_vector", type=str, default=None)
    parser.add_argument("--steering_layer", type=int, default=-1)
    parser.add_argument("--steering_coef", type=float, default=0.0)
    # prompt-only baseline controls (extended)
    parser.add_argument("--prompt_style", type=str, default="none",
                        choices=[
                            "none",
                            # exact required
                            "more_refl", "less_refl", "more_trans", "less_trans",
                            # legacy / back-compat
                            "R_plus", "R_minus", "T_plus", "T_minus"
                        ])
    parser.add_argument("--prompt_system_txt", type=str, default=None,
                        help="Optional path to a .txt file whose content will be PREPENDED to the user prompt (not as a system role).")
    args = parser.parse_args()

    # save_dir naming
    if args.steering:
        parts = args.steering_vector.split("/")[-3:]
        parts[-1] = parts[-1].split(".")[0]
        name = "_".join(parts)
        args.save_dir = os.path.join(args.save_dir, name, f"coef_{args.steering_coef}")
    else:
        # For baseline (no steering), standardize to base_run/base_remove_bos
        args.save_dir = os.path.join(args.save_dir, "base_run")

    if args.remove_bos:
        args.save_dir = os.path.join(args.save_dir, "base_remove_bos")

    # Attach the prompt variant directory only if using a style
    if getattr(args, "prompt_style", "none") != "none":
        args.save_dir = os.path.join(args.save_dir, f"prompt_{args.prompt_style}")

    if args.max_examples or args.start:
        start = 0 if args.start is None else args.start
        end = start + args.max_examples if args.max_examples is not None else -1
        args.save_dir = os.path.join(args.save_dir, f"{start}_{end}")

    print(args.save_dir)
    try:
        os.makedirs(args.save_dir, exist_ok=True)
        _test = os.path.join(args.save_dir, ".write_test")
        open(_test, "w").close()
        os.remove(_test)
    except PermissionError:
        import getpass
        fallback = args.save_dir.replace("/claim_ab/results/", f"/claim_ab/results_{getpass.getuser()}/")
        print(f"[WARN] Permission denied, fallback: {fallback}")
        args.save_dir = fallback
        os.makedirs(args.save_dir, exist_ok=True)
    main(args)
    eval_main(os.path.join(args.save_dir, "predictions.jsonl"), save=True, k=None, output_dir=args.save_dir)
