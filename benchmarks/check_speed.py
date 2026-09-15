"""Small local benchmark useful while developing the kernels."""

from benchmark_attention import run
import argparse

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--device', default=None)
    args = p.parse_args()
    run(argparse.Namespace(
        lengths=[256, 512, 1024], heads=2, dim=32,
        window=64, block_size=32, local_blocks=1,
        random_blocks=1, global_tokens=8,
        warmup=1, repeats=2, device=args.device, dtype='float32'
    ))
