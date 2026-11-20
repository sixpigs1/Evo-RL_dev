"""Read model and optimizer safetensors on CPU; reject contaminated checkpoints."""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def scan(checkpoint):
    rows = []
    for relative in ['pretrained_model/model.safetensors', 'training_state/optimizer_state.safetensors']:
        row = {'checkpoint': str(checkpoint), 'file': relative, 'tensors': 0, 'bad': [], 'dtypes': {}}
        with safe_open(str(checkpoint / relative), framework='pt', device='cpu') as tensors:
            for name in tensors.keys():
                tensor = tensors.get_tensor(name)
                row['tensors'] += 1
                dtype = str(tensor.dtype)
                row['dtypes'][dtype] = row['dtypes'].get(dtype, 0) + 1
                if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                    row['bad'].append(name)
        rows.append(row)
        print(json.dumps(row), flush=True)
    return not any(row['bad'] for row in rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoints', nargs='+', type=Path)
    args = p.parse_args()
    torch.set_num_threads(4)
    valid = [scan(checkpoint) for checkpoint in args.checkpoints]
    raise SystemExit(0 if all(valid) else 1)
