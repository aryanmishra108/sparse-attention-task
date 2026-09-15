import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main(path):
    data = json.loads(Path(path).read_text())
    plt.figure(figsize=(7, 4.5))
    for result in data['results']:
        history = result['history']
        xs = [x['step'] for x in history]
        ys = [x['val'] for x in history]
        plt.plot(xs, ys, marker='o', label=result['pattern'])
    plt.xlabel('Training step')
    plt.ylabel('Validation loss')
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out = Path(path).parent / 'quality_validation_loss.png'
    plt.savefig(out, dpi=160)
    print(f'Wrote {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='?', default='results/gpt/quality_results.json')
    main(parser.parse_args().path)
