"""Materialise the published Inkling 2 Mini checkpoint into a plain BF16 safetensors file.

The published checkpoint ships NVFP4 and MXFP8 payloads and `transformers` has no quantiser for
its `modelopt` layout, so the reference implementation cannot load it directly. This tool
dequantises it once with the same `expert_matrix` the converter uses -- one authority for the
mathematics -- and writes an ordinary BF16 checkpoint plus a matching index.

    python tests/inkling2/materialize_bf16.py --source /path/to/checkpoint --out /path/to/bf16

The output is a reference input for `dump_visits.py`, not a product artifact.
"""

import json, os, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.convert.sources.safetensors import SafetensorsSource
from tools.convert.inkling2 import (expert_matrix, e4m3_table, _E2M1,
                                    MXFP8_LAYERS, NVFP4_LAYERS)

SRC = Path(os.environ['INKLING2_SOURCE'])
OUT = Path(os.environ['INKLING2_BF16']) / 'model-00001-of-00001.safetensors'

def main() -> None:
    index = json.loads((SRC / 'model.safetensors.index.json').read_text())['weight_map']
    names = sorted(index)
    header = {'__metadata__': {'format': 'pt'}}
    offset = 0
    plan = []
    for name in names:
        # Quantisation metadata is consumed by the dequantiser, not stored in a BF16 checkpoint.
        if 'mlp.experts.' in name and name.endswith(('_scale', '_amax', '_outer_scale_denominator')):
            continue
        info = SafetensorsSource(str(SRC / 'model-00001-of-00001.safetensors')).describe(name)
        shape = list(info.shape)
        if 'mlp.experts.' in name and name.endswith(('w13_weight', 'w2_weight')):
            layer = int(name.split('.')[3])
            cols = info.shape[2]
            shape = [info.shape[0], info.shape[1], cols * (1 if layer in MXFP8_LAYERS else 2)]
            dtype = 'BF16'
            nbytes = int(np.prod(shape)) * 2
        elif info.dtype in ('BF16', 'F32'):
            dtype = info.dtype
            nbytes = info.bytes
        else:
            raise SystemExit(f'unexpected dtype {info.dtype} for {name}')
        header[name] = {'dtype': dtype, 'shape': shape,
                        'data_offsets': [offset, offset + nbytes]}
        plan.append((name, info, shape, dtype, offset, nbytes))
        offset += nbytes
    text = json.dumps(header, separators=(',', ':')).encode()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'wb') as f:
        f.write(len(text).to_bytes(8, 'little'))
        f.write(text)
        start = f.tell()
        f.truncate(start + offset)
        src = SafetensorsSource(str(SRC / 'model-00001-of-00001.safetensors'))
        began = time.time()
        for i, (name, info, shape, dtype, at, nbytes) in enumerate(plan):
            if 'mlp.experts.' in name and name.endswith(('w13_weight', 'w2_weight')):
                layer = int(name.split('.')[3])
                weight = 'w13_weight' if 'w13' in name else 'w2_weight'
                rows, k = shape[1], shape[2]
                block = torch.zeros((info.shape[0], rows, k), dtype=torch.bfloat16)
                for e in range(info.shape[0]):
                    block[e] = expert_matrix(src, layer, weight, e)
                f.seek(start + at)
                f.write(block.view(torch.uint8).numpy().tobytes())
                del block
            elif info.dtype in ('BF16', 'F32'):
                f.seek(start + at)
                flat = src.read_flat(name)
                # numpy has no bfloat16; copy the stored 16-bit words unchanged.
                f.write(flat.view(torch.uint8).numpy().tobytes()
                        if info.dtype == 'BF16' else flat.numpy().tobytes())
            if i % 100 == 0:
                print(f'[{i}/{len(plan)}] {name} {nbytes/1e6:.0f} MB', flush=True)
        print(f'done in {time.time()-began:.0f}s, {offset/1e9:.1f} GB', flush=True)

main()
