"""Reference MLP sub-block of one layer for one token, computed from the checkpoint itself.

Inputs come from the oracle dumps (visit_in_0, visit_in_1, layer1_attn), so no 52 GB model load
is needed: only the layer's own tensors are read through a lazy safetensors handle.
"""
import sys
import numpy as np
import torch
from safetensors import safe_open

SHARD = '/workspaces/models/inkling2/model-00001-of-00001.safetensors'
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)

def e4m3_table():
    t = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e, m = (b >> 3) & 0xF, b & 0x7
        t[b] = s * (m / 8.0) * 2.0 ** -6 if e == 0 else s * (1 + m / 8.0) * 2.0 ** (e - 7)
    return t

E4M3 = e4m3_table()

def dequant(codes, scales, amax, denom, rows, cols_bytes):
    packed = codes.reshape(rows, cols_bytes)
    lo, hi = (packed & 0xF).astype(np.int64), (packed >> 4).astype(np.int64)
    vals = np.empty((rows, cols_bytes * 2), dtype=np.float32)
    vals[:, 0::2] = E2M1[lo & 7] * np.where(lo & 8, -1, 1)
    vals[:, 1::2] = E2M1[hi & 7] * np.where(hi & 8, -1, 1)
    block = E4M3[scales.reshape(rows, cols_bytes // 8).astype(np.int64)]
    return vals * np.repeat(block, 16, axis=1) * (amax / denom)

def rms(x):
    return x / np.sqrt((x.astype(np.float64) ** 2).mean() + 1e-6)

layer = int(sys.argv[1])
bf = lambda p: (np.fromfile(p, dtype='<u2').astype(np.uint32) << 16).view(np.float32)

h_prev = np.fromfile('/tmp/oracle1/visit_in_0.bin', dtype=np.float32)          # block state
h_in = np.fromfile(f'/tmp/oracle1/visit_in_{layer}.bin', dtype=np.float32)     # layer input
attn = np.fromfile(f'/tmp/stages/layer{layer}_attn.bin', dtype=np.float32)     # post-attention conv

with safe_open(SHARD, framework='pt') as f:
    get = lambda n: f.get_tensor(f'model.llm.layers.{layer}.{n}')
    query = f.get_tensor(f'model.llm.attn_res.mlp_queries.{layer}').float().numpy().reshape(1, -1)
    norm_w = get('mlp_norm.weight').float().numpy()
    gate_w = get('mlp.gate.weight').float().numpy()
    gate_b = get('mlp.gate.bias').float().numpy()
    w13_amax = get('mlp.experts.w13_weight_amax').float().numpy()
    w13_den = get('mlp.experts.w13_weight_outer_scale_denominator').float().numpy()
    w2_amax = get('mlp.experts.w2_weight_amax').float().numpy()
    w2_den = get('mlp.experts.w2_weight_outer_scale_denominator').float().numpy()
    w13_codes = get('mlp.experts.w13_weight').view(-1).numpy() if False else None

    # residual and the block read
    residual = h_in.astype(np.float64) + attn.astype(np.float64)
    stack = np.stack([h_prev, residual])
    normed = stack / np.sqrt((stack ** 2).mean(axis=1, keepdims=True) + 1e-6)
    logits = (query.astype(np.float64) @ normed.T) / np.sqrt(2304)
    p = np.exp(logits - logits.max()); p /= p.sum()
    mlp_in = (p @ normed)[0]
    print('mlp_in first3', np.round(mlp_in[:3], 6))

    x = rms(mlp_in) * norm_w
    s = 1.0 / (1.0 + np.exp(-(gate_w.astype(np.float64) @ x)))
    sel = np.argsort(-(s + gate_b))[:8]
    w = s[sel] / s[sel].sum() * 8.0
    import json
    json.dump({'indices': [int(i) for i in sel], 'weights': [float(v) for v in w]},
              open(f'/tmp/ref_route_{layer}.json', 'w'))
    print('experts', sel.tolist())
    print('weights', np.round(w, 5).tolist())
    denom = float(w13_den.reshape(-1)[0])
    acc = np.zeros(2304, dtype=np.float64)
    for j, e in enumerate(sel):
        c13 = f.get_tensor('model.llm.layers.%d.mlp.experts.w13_weight' % layer)[e].view(torch.uint8).numpy()
        s13 = f.get_tensor('model.llm.layers.%d.mlp.experts.w13_weight_scale' % layer)[e].view(torch.uint8).numpy()
        c2 = f.get_tensor('model.llm.layers.%d.mlp.experts.w2_weight' % layer)[e].view(torch.uint8).numpy()
        s2 = f.get_tensor('model.llm.layers.%d.mlp.experts.w2_weight_scale' % layer)[e].view(torch.uint8).numpy()
        w13 = dequant(c13, s13, float(w13_amax[e]), denom, 2304, 1152)
        w2 = dequant(c2, s2, float(w2_amax[e]), denom, 2304, 576)
        up = w13 @ x
        gate_part, up_part = up[:1152], up[1152:]
        act = np.tanh(gate_part / 4.0) * 4.0 * (1/(1+np.exp(-gate_part))) * np.tanh(up_part / 25.0) * 25.0
        acc += w[j] * (w2 @ act)
    conv_w = get('mlp_sconv.weight').float().numpy().reshape(2304, 4)[:, 3]
    mlp_out = (acc * conv_w + acc) / np.sqrt(5.0)
    out = residual + mlp_out
    np.save('/tmp/ref_mlp_out.npy', out.astype(np.float32))
    np.save('/tmp/ref_mlp_acc.npy', acc.astype(np.float32))
    print('moe out first3', np.round(acc[:3], 6))
    print('layer out first3', np.round(out[:3], 6))
