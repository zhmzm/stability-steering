# MINE: Stability-Guided Steering for Reasoning LLMs

> **Reliable Control-Point Selection for Steering Reasoning in Large Language Models**
>
> Haomin Zhuang, Hojun Yoo, Xiaonan Luo, Kehan Guo, Xiangliang Zhang
>
> University of Notre Dame

Steering vectors offer a training-free way to control reasoning behaviors in LLMs, but existing methods like SEAL detect behavioral boundaries through keyword matching—treating every match as a genuine signal. We find that **93.3% of keyword-detected boundaries are behaviorally unstable**: the model fails to reproduce the detected behavior when re-run from the same prefix. Our method, **stability filtering**, retains only the small fraction of boundaries that consistently reproduce the target behavior, amplifying the steering signal. Combined with content-subspace projection, this achieves **0.784** accuracy on MATH-500 (+5.0 over SEAL), and the vectors transfer across models without re-extraction.

<p align="center">
  <img src="figures/concept_fig.html" width="100%" alt="Concept Figure"/>
</p>

## Main Results

| Model | Baseline | SEAL | Ours | Δ |
|-------|----------|------|------|---|
| DeepSeek-R1-Distill-Qwen-1.5B | 0.608 | 0.734 | **0.784** | +17.6 |
| Nemotron-Research-Reasoning-1.5B | 0.662 | 0.766 | **0.816** | +15.4 |
| DeepScaleR-1.5B-Preview | 0.726 | 0.752 | **0.812** | +8.6 |

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Step 1: Extract steering vectors

```bash
# Generate chain-of-thought responses and extract hidden states
python src/extract/vector_generation.py \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --dataset MATH \
    --split train \
    --max_tokens 8192

# Build per-question steering vectors
python src/extract/build_behavior_vectors.py \
    --hidden_dir outputs/hidden_states \
    --output_dir outputs/vectors
```

### Step 2: Stability probing

```bash
# Re-generate from each boundary prefix ×10
python src/filter/probe_behavior_stability.py \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --hidden_dir outputs/hidden_states \
    --M 10 --temperature 0.7 --top_p 0.95

# Build stability-filtered vectors (τ=0.8)
python src/filter/build_vectors_from_behavior_stability.py \
    --stability_dir outputs/stability \
    --vector_dir outputs/vectors \
    --tau 0.8
```

### Step 3: Content-subspace projection

```bash
# Extract question-only hidden states and compute SVD
python src/project/build_content_subspace.py \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --hidden_dir outputs/hidden_states \
    --K 4

# Combine: stability filtering + content projection
python src/project/combine_vectors.py \
    --stable_vector_dir outputs/vectors_stable \
    --svd_dir outputs/svd \
    --output_dir outputs/vectors_combined
```

### Step 4: Evaluate

```bash
python src/eval/eval_MATH_vllm_steering.py \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --steering_vector outputs/vectors_combined/combined.pt \
    --dataset MATH500 \
    --max_tokens 4096 \
    --coef -100
```

## Project Structure

```
├── src/
│   ├── extract/          # Vector extraction from CoT traces
│   ├── filter/           # Stability probing and filtering
│   ├── project/          # Content-subspace projection (SVD)
│   ├── eval/             # MATH-500 evaluation
│   └── analysis/         # Probing, bootstrap CI, ablations
├── scripts/              # Shell scripts for full pipeline
├── figures/              # Figure generation code
└── requirements.txt
```

## Citation

```bibtex
@article{zhuang2025mine,
  title={Reliable Control-Point Selection for Steering Reasoning in Large Language Models},
  author={Zhuang, Haomin and Yoo, Hojun and Luo, Xiaonan and Guo, Kehan and Zhang, Xiangliang},
  year={2025}
}
```

## Acknowledgments

This codebase is built upon [SEAL](https://github.com/VITA-Group/SEAL) by the VITA Group. We thank the authors for releasing their code and data.
