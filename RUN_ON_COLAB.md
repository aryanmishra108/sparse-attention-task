# Final Colab run

The repository is set up so the last hardware-dependent experiment can be run without changing the implementation.

## 1. Check the GPU

```python
!nvidia-smi
```

A T4 is enough for the intended run.

## 2. Install dependencies

```bash
!pip install -r requirements.txt
```

## 3. Correctness first

```bash
!python -m pytest -q
!python tests/correctness_harness.py
```

## 4. Benchmark

```bash
!python benchmarks/benchmark_attention.py --device cuda --dtype float16
```

This writes the runtime/memory JSON and plots.

## 5. TinyShakespeare quality comparison

```bash
!python gpt/train.py --device cuda --steps 2000 --outdir results/gpt
!python gpt/evaluate_quality.py results/gpt/quality_results.json
```

The training script downloads the same TinyShakespeare text used by the reference character-level examples if it is not already present.

## 6. What to copy into WRITEUP.md

The training script prints a final validation loss for each of:

```text
dense
sliding_window
block_sparse
```

The important comparison is not only the lowest loss. Look at how much loss increases relative to dense attention and compare that with the speed/memory gain from the benchmark.

Do not change seeds or model hyperparameters between the three runs when reporting the comparison.
