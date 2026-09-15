import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attention.block_sparse import BlockSparseConfig, block_sparse_attention
from attention.dense import dense_attention
from attention.sliding_window import sliding_window_attention


def hardware_string():
    pieces = [f'OS={platform.platform()}', f'CPU={platform.processor() or platform.machine()}']
    if torch.cuda.is_available():
        pieces.append(f'GPU={torch.cuda.get_device_name(0)}')
        pieces.append(f'CUDA={torch.version.cuda}')
    else:
        pieces.append('GPU=none')
    return '; '.join(pieces)


def process_peak_memory_mb():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == 'darwin':
        return value / (1024 * 1024)
    return value / 1024


def build_inputs(n, heads, dim, device, dtype):
    torch.manual_seed(1234)
    q = torch.randn(1, heads, n, dim, device=device, dtype=dtype)
    k = torch.randn(1, heads, n, dim, device=device, dtype=dtype)
    v = torch.randn(1, heads, n, dim, device=device, dtype=dtype)
    return q, k, v


def run_one(pattern, n, args):
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    dtype = torch.float16 if args.dtype == 'float16' else torch.float32
    if device.type == 'cpu' and dtype == torch.float16:
        dtype = torch.float32

    q, k, v = build_inputs(n, args.heads, args.dim, device, dtype)
    cfg = BlockSparseConfig(
        block_size=args.block_size,
        local_blocks=args.local_blocks,
        random_blocks=args.random_blocks,
        global_tokens=args.global_tokens,
        seed=17,
    )
    causal = torch.tril(torch.ones(n, n, dtype=torch.bool, device=device))
    fns = {
        'dense': lambda: dense_attention(q, k, v, causal),
        'sliding_window': lambda: sliding_window_attention(q, k, v, min(args.window, n)),
        'block_sparse': lambda: block_sparse_attention(q, k, v, cfg),
    }
    fn = fns[pattern]

    for _ in range(args.warmup):
        _ = fn()
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(args.repeats):
        _ = fn()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    seconds = (time.perf_counter() - t0) / args.repeats

    if device.type == 'cuda':
        memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        memory_mb = process_peak_memory_mb()

    return {
        'pattern': pattern,
        'sequence_length': n,
        'seconds': seconds,
        'peak_memory_mb': memory_mb,
        'device': str(device),
        'dtype': str(dtype),
    }


def master(args):
    rows = []
    for n in args.lengths:
        for pattern in ['dense', 'sliding_window', 'block_sparse']:
            cmd = [
                sys.executable, __file__, '--worker', pattern,
                '--n', str(n), '--heads', str(args.heads), '--dim', str(args.dim),
                '--window', str(args.window), '--block-size', str(args.block_size),
                '--local-blocks', str(args.local_blocks), '--random-blocks', str(args.random_blocks),
                '--global-tokens', str(args.global_tokens), '--warmup', str(args.warmup),
                '--repeats', str(args.repeats), '--device', args.device or '', '--dtype', args.dtype,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                if 'out of memory' in (proc.stderr + proc.stdout).lower():
                    row = {'pattern': pattern, 'sequence_length': n, 'seconds': None,
                           'peak_memory_mb': None, 'error': 'out_of_memory'}
                    print(f'{pattern:16s} N={n:5d} OUT OF MEMORY')
                    rows.append(row)
                    continue
                print(proc.stdout)
                print(proc.stderr, file=sys.stderr)
                raise RuntimeError(f'benchmark worker failed for {pattern} N={n}')
            row = json.loads(proc.stdout.strip().splitlines()[-1])
            rows.append(row)
            print(f"{pattern:16s} N={n:5d} time={row['seconds']:8.4f}s peak={row['peak_memory_mb']:9.1f} MB")

    output = ROOT / 'results' / 'attention_benchmark.json'
    output.write_text(json.dumps({'hardware': hardware_string(), 'args': vars(args), 'results': rows}, indent=2))
    plot_results(rows, ROOT / 'plots')


def plot_results(rows, outdir):
    outdir.mkdir(exist_ok=True)
    for metric, ylabel, filename in [
        ('seconds', 'Forward time (s)', 'attention_runtime.png'),
        ('peak_memory_mb', 'Peak process memory (MB)', 'attention_memory.png'),
    ]:
        plt.figure(figsize=(7, 4.5))
        for pattern in ['dense', 'sliding_window', 'block_sparse']:
            xs = [r['sequence_length'] for r in rows if r['pattern'] == pattern and r.get(metric) is not None]
            ys = [r[metric] for r in rows if r['pattern'] == pattern and r.get(metric) is not None]
            if xs:
                plt.plot(xs, ys, marker='o', label=pattern)
        plt.xlabel('Sequence length')
        plt.ylabel(ylabel)
        plt.xscale('log', base=2)
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / filename, dpi=160)
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=['dense', 'sliding_window', 'block_sparse'])
    parser.add_argument('--n', type=int)
    parser.add_argument('--lengths', nargs='+', type=int, default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--window', type=int, default=128)
    parser.add_argument('--block-size', type=int, default=64)
    parser.add_argument('--local-blocks', type=int, default=1)
    parser.add_argument('--random-blocks', type=int, default=1)
    parser.add_argument('--global-tokens', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--device', default=None)
    parser.add_argument('--dtype', choices=['float32', 'float16'], default='float32')
    args = parser.parse_args()

    if args.worker:
        row = run_one(args.worker, args.n, args)
        print(json.dumps(row))
    else:
        master(args)
