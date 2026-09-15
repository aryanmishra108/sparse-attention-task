#!/usr/bin/env bash
set -euo pipefail

python -m pytest -q
python tests/correctness_harness.py
python benchmarks/benchmark_attention.py
