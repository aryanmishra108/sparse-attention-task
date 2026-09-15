import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from gpt.data import CharDataset, download_tiny_shakespeare
from gpt.model import GPTConfig, TinyGPT
from attention.block_sparse import BlockSparseConfig


@torch.no_grad()
def estimate_loss(model, dataset, batch_size, block_size, eval_iters, generator):
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = []
        for _ in range(eval_iters):
            x, y = dataset.get_batch(split, batch_size, block_size, generator)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def train_one(pattern, args, dataset, outdir):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model_cfg = GPTConfig(
        vocab_size=dataset.vocab_size,
        block_size=args.block_size,
        n_layer=2,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        attention_pattern=pattern,
        window_size=args.window_size,
        block_sparse=BlockSparseConfig(
            block_size=args.sparse_block_size,
            local_blocks=args.local_blocks,
            random_blocks=args.random_blocks,
            global_tokens=args.global_tokens,
            seed=args.sparse_seed,
        ),
    )
    model = TinyGPT(model_cfg).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Same batch stream for all attention patterns makes the comparison easier
    # to reproduce. Each pattern starts from the same initialization too.
    generator = torch.Generator(device='cpu').manual_seed(args.seed + 1000)
    eval_generator = torch.Generator(device='cpu').manual_seed(args.seed + 2000)

    history = []
    started = time.perf_counter()
    for step in range(args.steps):
        if step % args.eval_interval == 0 or step == args.steps - 1:
            metrics = estimate_loss(
                model, dataset, args.batch_size, args.block_size,
                args.eval_iters, eval_generator
            )
            metrics['step'] = step
            history.append(metrics)
            print(f'[{pattern}] step {step:5d} train={metrics["train"]:.4f} val={metrics["val"]:.4f}')

        x, y = dataset.get_batch('train', args.batch_size, args.block_size, generator)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    elapsed = time.perf_counter() - started
    checkpoint = {
        'model': model.state_dict(),
        'config': asdict(model_cfg),
        'pattern': pattern,
        'history': history,
        'elapsed_seconds': elapsed,
        'args': vars(args),
    }
    ckpt_path = outdir / f'{pattern}.pt'
    torch.save(checkpoint, ckpt_path)
    return {
        'pattern': pattern,
        'elapsed_seconds': elapsed,
        'history': history,
        'checkpoint': str(ckpt_path),
    }


def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data)
    download_tiny_shakespeare(data_path)
    dataset = CharDataset(data_path, args.device)

    results = []
    for pattern in ['dense', 'sliding_window', 'block_sparse']:
        results.append(train_one(pattern, args, dataset, outdir))

    (outdir / 'quality_results.json').write_text(json.dumps({
        'dataset': str(data_path),
        'vocab_size': dataset.vocab_size,
        'results': results,
    }, indent=2))

    print('\nFinal validation losses:')
    for result in results:
        final = result['history'][-1]
        print(f"  {result['pattern']:16s} {final['val']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/tinyshakespeare/input.txt')
    parser.add_argument('--outdir', default='results/gpt')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--eval-interval', type=int, default=200)
    parser.add_argument('--eval-iters', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--block-size', type=int, default=128)
    parser.add_argument('--n-head', type=int, default=4)
    parser.add_argument('--n-embd', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--window-size', type=int, default=32)
    parser.add_argument('--sparse-block-size', type=int, default=32)
    parser.add_argument('--local-blocks', type=int, default=1)
    parser.add_argument('--random-blocks', type=int, default=1)
    parser.add_argument('--global-tokens', type=int, default=4)
    parser.add_argument('--sparse-seed', type=int, default=17)
    parser.add_argument('--seed', type=int, default=1337)
    args = parser.parse_args()
    main(args)
