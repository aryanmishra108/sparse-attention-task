from pathlib import Path
from urllib.request import urlopen

import torch

DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'


def download_tiny_shakespeare(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path

    print(f'Downloading TinyShakespeare to {path} ...')
    with urlopen(DATA_URL, timeout=30) as response:
        text = response.read().decode('utf-8')
    path.write_text(text, encoding='utf-8')
    return path


class CharDataset:
    def __init__(self, text_path: str | Path, device: torch.device | str = 'cpu'):
        self.text_path = Path(text_path)
        self.device = torch.device(device)
        text = self.text_path.read_text(encoding='utf-8')
        if len(text) < 1000:
            raise ValueError('TinyShakespeare file looks too small')
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        ids = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        split = int(0.9 * len(ids))
        self.train = ids[:split]
        self.val = ids[split:]

    @property
    def vocab_size(self):
        return len(self.stoi)

    def encode(self, text: str):
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return ''.join(self.itos[int(i)] for i in ids)

    def get_batch(self, split, batch_size, block_size, generator=None):
        data = self.train if split == 'train' else self.val
        max_start = len(data) - block_size - 1
        if max_start <= 0:
            raise ValueError('block_size is too large for the dataset')
        ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        return x.to(self.device), y.to(self.device)
