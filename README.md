# Sparse Attention from Scratch

This repo is my implementation for Task 1 of the Postman AI/ML recruitment task.

The main idea is simple: start from a small, boring implementation of normal attention, make the masking rules explicit, and only then remove the unnecessary score computation. I kept the code fairly direct because the point of the task is to understand where the savings actually come from.

## What is included

- Manual dense scaled dot-product attention (`attention/dense.py`)
- Causal masking (`attention/masks.py`)
- Causal sliding-window attention (`attention/sliding_window.py`)
- Local + global + random block-sparse attention (`attention/block_sparse.py`)
- Correctness harness and pytest tests
- Forward-pass benchmark for sequence lengths 512 to 8192
- A small 2-layer character-level GPT for TinyShakespeare
- Training/evaluation scripts for dense, sliding-window and block-sparse attention
- Benchmark and loss plots
- `WRITEUP.md` with the implementation and experimental discussion

## Important design choice

The dense reference is deliberately written as:

```text
QK^T / sqrt(d)
        -> mask
        -> softmax
        -> P V
```

It does not use `torch.nn.functional.scaled_dot_product_attention`.

For sparse correctness, the oracle uses the **same sparse mask** with the manual dense implementation. A sparse model should not be expected to produce the same output as unrestricted dense attention, because it is intentionally seeing fewer keys. What we are checking is whether the sparse implementation faithfully computes the attention implied by its pattern.

## Install

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\\Scripts\\Activate.ps1

pip install -r requirements.txt
```

## Correctness

Pytest:

```bash
python -m pytest -q
```

Standalone harness:

```bash
python tests/correctness_harness.py
```

The harness prints the maximum absolute difference between each sparse implementation and the dense oracle using the identical mask.

## Benchmark

Default benchmark:

```bash
python benchmarks/benchmark_attention.py
```

The default sweep is:

```text
512, 1024, 2048, 4096, 8192
```

The script reports forward time and peak process memory, and writes:

```text
results/attention_benchmark.json
plots/attention_runtime.png
plots/attention_memory.png
```

For the local CPU run included in this repo I used a deliberately small attention configuration (`1 head`, `head_dim=16`) so the full 8192 case was practical. This is a relative benchmark, not an attempt to claim these numbers represent a T4.

For the submission hardware comparison I would run the same script on a Colab T4, for example:

```bash
python benchmarks/benchmark_attention.py --device cuda --dtype float16
```

## TinyShakespeare experiment

The character-level dataset is fetched from Karpathy's TinyShakespeare file when needed:

```text
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Run the three-model comparison with:

```bash
python gpt/train.py --device cuda --steps 2000
```

The script trains the same 2-layer GPT configuration with:

- dense attention
- sliding-window attention
- block-sparse attention

and stores checkpoints plus validation-loss history under `results/gpt/`.

Plot the comparison with:

```bash
python gpt/evaluate_quality.py results/gpt/quality_results.json
```

### Suggested Colab run

A free T4 is a much better place to run the final TinyShakespeare experiment. The exact command above is intentionally simple so the comparison can be reproduced without editing the model code.

## NaN handling

A fully masked row becomes:

```text
[-inf, -inf, ..., -inf]
```

After exponentiation that is all zeros, so softmax would be `0 / 0`, which is NaN. `safe_softmax()` explicitly detects this case and returns an all-zero probability row. This is important at sparse boundaries, especially when causal and sparse masks are combined.

## References

- Zaheer et al., *Big Bird: Transformers for Longer Sequences*: https://arxiv.org/abs/2007.14062
- Karpathy, TinyShakespeare dataset: https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt
