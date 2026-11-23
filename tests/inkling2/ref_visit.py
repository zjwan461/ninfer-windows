"""Reference for one layer visit, including a loop pass's cross-pass messages.

Reads the checkpoint's own tensors and the oracle's recorded hidden states, so it needs neither
the 52 GB reference model nor a GPU. Writes the same stage fixtures as ref_layer.py.

    python tests/inkling2/ref_visit.py 6 1     # layer 6, second pass
"""
import sys
import numpy as np
from safetensors import safe_open

SHARD = '/workspaces/models/inkling2/model-00001-of-00001.safetensors'
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
FULL = {5, 11, 17, 23}
CORE = range(6, 18)
EPS = 1e-6
QALPHA, QFLOOR = 0.1, 8192.0
QK_CLIP, REL_CLIP = 22.627416997969522, 8.0

def e4m3_table():
    t = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e, m = (b >> 3) & 0xF, b & 0x7
        t[b] = s * (m / 8.0) * 2.0 ** -6 if e == 0 else s * (1 + m / 8.0) * 2.0 ** (e - 7)
    return t

E4M3 = e4m3_table()

def rms(x, w=None):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + EPS) * (w if w is not None else 1.0)
    return x / np.sqrt((x ** 2).mean() + EPS) * (w if w is not None else 1.0)

def read_block(queries, states):
    normed = states / np.sqrt((states ** 2).mean(axis=1, keepdims=True) + EPS)
    logits = (queries @ normed.T) / np.sqrt(2304)
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    return p @ normed

def clip(x, scale, threshold, tau=1.0):
    v = x.astype(np.float64) * scale.astype(np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return (v * np.minimum(threshold / (n + 1e-6), 1.0) * tau).astype(np.float32)

def merged_queries(f, loop_index):
    """The reference merges (query, mlp) and (key, value) into [query, key, value, mlp]."""
    cp = f.get_tensor('model.llm.middle_cycle_cross_pass_queries').float().numpy()
    kv = f.get_tensor('model.llm.middle_cycle_cross_pass_kv_queries').float().numpy()
    cp = cp.reshape(-1, 2, 2304)
    kv = kv.reshape(-1, 2, 2304)
    merged = np.concatenate([cp[loop_index, :1], kv[loop_index], cp[loop_index, 1:]], axis=0)
    return merged  # [query, key, value, mlp]

layer = int(sys.argv[1])
loop_pass = int(sys.argv[2]) if len(sys.argv) > 2 else 0
sliding = layer not in FULL
boundary = (layer % 6) == 0
in_core = 6 <= layer < 18
loop_index = layer - 6

# The visits this one depends on, from the oracle's own recording.
def visit_in(index):
    return np.fromfile(f'/tmp/oracle1/visit_in_{index}.bin', dtype=np.float32).astype(np.float64)

# Visit index of a layer visit: 6 prelude, then three passes of the core, then the coda.
def visit_of(target_layer, target_pass):
    if target_layer < 6:
        return target_layer
    if target_layer < 18:
        return 6 + target_pass * 12 + (target_layer - 6)
    # The coda runs after the core's last pass, so its visits start at 42.
    return 42 + (target_layer - 18)

h_in = visit_in(visit_of(layer, loop_pass))
if in_core:
    stack = [visit_in(0)] if loop_pass == 0 else [visit_in(0)]
    prev_h = visit_in(visit_of(17, loop_pass - 1) + 1) if loop_pass > 0 else None
    prev_block = visit_in(12) if loop_pass > 0 else None
else:
    # After the core's last pass the block stack holds the pass-2 boundary entries; the coda's own
    # first layer (18) is a boundary as well, so later coda layers see its entry too.
    stack = [visit_in(0), visit_in(visit_of(6, 2)), visit_in(visit_of(12, 2))]
    if layer > 18:
        stack.append(visit_in(42))
    prev_h = prev_block = None

with safe_open(SHARD, framework='pt') as f:
    g = lambda n: f.get_tensor(f'model.llm.layers.{layer}.{n}').float().numpy()

    # ---- block read for the attention input ------------------------------------------------
    nq = 1 if sliding else 3
    aq = f.get_tensor(f'model.llm.attn_res.attn_queries.{layer - 1}').float().numpy().reshape(-1, 2304)[:nq]
    streams = read_block(aq, np.stack(stack + [h_in]))
    attn_in = streams[0]
    k_base = streams[1] if nq == 3 else streams[0]
    v_base = streams[2] if nq == 3 else streams[0]

    attn_in_pre = attn_in.copy()

    # ---- cross-pass messages ---------------------------------------------------------------
    if loop_pass > 0 and in_core:
        mq = merged_queries(f, loop_index)
        msgs = read_block(mq, np.stack([prev_block, prev_h]))
        # The checkpoint names these controllers.{attn,k,v,mlp}; the reference renames them to
        # q/k/v/mlp_message_gate, with gain = message_gate_proj and forget = forget_gate_proj.
        ctrl = {'q': 'controllers.attn', 'k': 'controllers.k', 'v': 'controllers.v',
                'mlp': 'controllers.mlp'}
        def gate(x, m, which):
            base = ctrl[which]
            hh = g(f'{base}.down_weight').astype(np.float64) @ np.concatenate([rms(x), rms(m)])
            keep = 1 - np.tanh(g(f'{base}.forget_weight').astype(np.float64) @ hh)
            add = np.tanh(g(f'{base}.gain_weight').astype(np.float64) @ hh)
            return keep * x + add * m
        np.ascontiguousarray(msgs.reshape(-1), dtype=np.float32).tofile(
            f'/tmp/stages/layer{layer}_message.bin')
        np.ascontiguousarray(mq.reshape(-1), dtype=np.float32).tofile(
            f'/tmp/stages/layer{layer}_message_query.bin')
        attn_in = gate(attn_in, msgs[0], 'q')
        k_base = gate(k_base, msgs[1], 'k')
        v_base = gate(v_base, msgs[2], 'v')

    if loop_pass > 0 and in_core:
        w = lambda name, arr: np.ascontiguousarray(np.asarray(arr, dtype=np.float32).reshape(-1)).tofile(
            f'/tmp/stages/layer{layer}_{name}.bin')
        w('q_in', attn_in)
        w('k_in', k_base)
        w('v_in', v_base)

    # ---- norms, projections, attention ------------------------------------------------------
    if sliding:
        norm_w = g('attn_norm.weight')
        qn, kn, vn = rms(attn_in, norm_w), rms(k_base, norm_w), rms(v_base, norm_w)
    else:
        qn = rms(attn_in, g('q_input_norm.weight'))
        kn = rms(k_base, g('k_input_norm.weight'))
        vn = rms(v_base, g('v_input_norm.weight'))

    tau = 1.0 + QALPHA * np.log(max(1.0 / QFLOOR, 1.0))
    wq, wk, wv = g('attn.wq_du.weight'), g('attn.wk_dv.weight'), g('attn.wv_dv.weight')
    qv = clip((wq.astype(np.float64) @ qn).reshape(24, 128), g('attn.q_l2_clip.scale'), QK_CLIP,
              1.0 if sliding else tau)

    base_k = (wk.astype(np.float64) @ kn).reshape(1024)
    if sliding:
        conv = g('attn.k_sconv.weight').reshape(1024, 4)
        kv = (conv[:, 3] * base_k + base_k) / np.sqrt(5.0)
    else:
        static = g('attn.k_sconv.static_w').reshape(4, 1024)
        dyn = (g('attn.k_sconv.weight_proj').astype(np.float64) @ kn).reshape(8, 4)
        taps = static[3] + np.repeat(dyn[:, 3], 128)
        kv = (taps * base_k + base_k) / 3.0
    kk = clip(kv.reshape(8, 128), np.ones((8, 128), dtype=np.float32), QK_CLIP)

    base_v = (wv.astype(np.float64) @ vn).reshape(1024)
    if sliding:
        conv = g('attn.v_sconv.weight').reshape(1024, 4)
        vv = (conv[:, 3] * base_v + base_v) / np.sqrt(5.0)
    else:
        static = g('attn.v_sconv.static_w').reshape(4, 1024)
        dyn = (g('attn.v_sconv.weight_proj').astype(np.float64) @ vn).reshape(8, 4)
        taps = static[3] + np.repeat(dyn[:, 3], 128)
        vv = (taps * base_v + base_v) / 3.0
    vv = rms(vv.reshape(8, 128))

    rel = clip((g('attn.wr_du.weight').astype(np.float64) @ qn).reshape(24, 16),
               g('attn.rel_l2_clip.scale'), REL_CLIP, 1.0 if sliding else tau)
    bias = (rel @ g('attn.rel_logits_proj.proj').astype(np.float64))[:, 0]
    sink = g('attn.attn_sink_proj.weight').astype(np.float64) @ qn
    gate_w = 1.0 / (1.0 + np.exp(-(g('attn.gate_attn.weight').astype(np.float64) @ qn)))

    kk3 = np.repeat(kk, 3, axis=0)
    vv3 = np.repeat(vv, 3, axis=0)
    logits = (qv * kk3).sum(axis=1) / np.sqrt(128) + bias
    combined = np.stack([logits, sink], axis=1)
    p = np.exp(combined - combined.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    attn = ((p[:, 0, None] * vv3).reshape(-1) * gate_w).reshape(1, 3072)
    proj_out = (g('attn.wo_ud.weight').astype(np.float64) @ attn.reshape(-1))
    sconv = g('attn_sconv.weight').reshape(2304, 4)
    conv_out = (sconv[:, 3] * proj_out + proj_out) / np.sqrt(5.0)

    # ---- residual and MLP ------------------------------------------------------------------
    if boundary:
        residual = conv_out
        stack_mlp = stack + [h_in, residual]
    else:
        residual = h_in + conv_out
        stack_mlp = stack + [residual]
    mq_in = f.get_tensor(f'model.llm.attn_res.mlp_queries.{layer}').float().numpy().reshape(1, 2304)
    mlp_in = read_block(mq_in, np.stack(stack_mlp))[0]
    if loop_pass > 0 and in_core:
        base = 'controllers.mlp'
        hh = (g(f'{base}.down_weight').astype(np.float64)
              @ np.concatenate([rms(mlp_in), rms(msgs[3])]))
        mlp_in = ((1 - np.tanh(g(f'{base}.forget_weight').astype(np.float64) @ hh)) * mlp_in
                  + np.tanh(g(f'{base}.gain_weight').astype(np.float64) @ hh) * msgs[3])
    x = rms(mlp_in) * g('mlp_norm.weight')

    s = 1.0 / (1.0 + np.exp(-(g('mlp.gate.weight').astype(np.float64) @ x)))
    sel = np.argsort(-(s + g('mlp.gate.bias').astype(np.float64)))[:8]
    w = s[sel] / s[sel].sum() * 8.0
    print('experts', sel.tolist())

    order = np.where(np.arange(2304) < 1152, np.arange(2304) * 2, (np.arange(2304) - 1152) * 2 + 1)
    mxfp8 = layer in (20, 21, 22, 23)
    if not mxfp8:
        denom = float(g('mlp.experts.w13_weight_outer_scale_denominator').reshape(-1)[0])
    acc = np.zeros(2304, dtype=np.float64)
    for j, e in enumerate(sel):
        if mxfp8:
            # Layers 20-23 keep the checkpoint's MXFP8 experts; the artifact requantises them.
            import torch
            from tools.convert.inkling2 import dequantize_mxfp8
            prefix = f'model.llm.layers.{layer}.mlp.experts'
            def mxfp(name):
                packed = f.get_tensor(f'{prefix}.{name}')
                packed = packed[e].view(torch.uint8).numpy()
                scales = f.get_tensor(f'{prefix}.{name}_scale')[e].view(torch.uint8).numpy()
                rows = f.get_tensor(f'{prefix}.{name}').shape[1]
                values = dequantize_mxfp8(torch.from_numpy(packed), torch.from_numpy(scales)).numpy()
                return values
            w13 = mxfp('w13_weight').astype(np.float64)[order]
            w2 = mxfp('w2_weight').astype(np.float64)
            up = w13 @ x
            gp, upp = up[:1152], up[1152:]
            act = np.tanh(gp / 4.0) * 4.0 * (1 / (1 + np.exp(-gp))) * np.tanh(upp / 25.0) * 25.0
            acc += w[j] * (w2 @ act)
            continue

        def deq(codes, scales, rows, cols_bytes):
            packed = codes.reshape(rows, cols_bytes)
            lo, hi = (packed & 0xF).astype(np.int64), (packed >> 4).astype(np.int64)
            vals = np.empty((rows, cols_bytes * 2), dtype=np.float32)
            vals[:, 0::2] = E2M1[lo & 7] * np.where(lo & 8, -1, 1)
            vals[:, 1::2] = E2M1[hi & 7] * np.where(hi & 8, -1, 1)
            block = E4M3[scales.reshape(rows, cols_bytes // 8).astype(np.int64)]
            return vals * np.repeat(block, 16, axis=1)
        c13 = f.get_tensor(f'model.llm.layers.{layer}.mlp.experts.w13_weight')[e].view(__import__('torch').uint8).numpy()
        s13 = f.get_tensor(f'model.llm.layers.{layer}.mlp.experts.w13_weight_scale')[e].view(__import__('torch').uint8).numpy()
        c2 = f.get_tensor(f'model.llm.layers.{layer}.mlp.experts.w2_weight')[e].view(__import__('torch').uint8).numpy()
        s2 = f.get_tensor(f'model.llm.layers.{layer}.mlp.experts.w2_weight_scale')[e].view(__import__('torch').uint8).numpy()
        amax13 = float(g('mlp.experts.w13_weight_amax')[e])
        amax2 = float(g('mlp.experts.w2_weight_amax')[e])
        w13 = deq(c13, s13, 2304, 1152)[order] * (amax13 / denom)
        w2 = deq(c2, s2, 2304, 576) * (amax2 / denom)
        up = w13.astype(np.float64) @ x
        gp, upp = up[:1152], up[1152:]
        act = np.tanh(gp / 4.0) * 4.0 * (1 / (1 + np.exp(-gp))) * np.tanh(upp / 25.0) * 25.0
        acc += w[j] * (w2.astype(np.float64) @ act)
    conv_w = g('mlp_sconv.weight').reshape(2304, 4)[:, 3]
    mlp_out = (conv_w * acc + acc) / np.sqrt(5.0)
    out = residual + mlp_out

    def write(name, arr):
        np.ascontiguousarray(np.asarray(arr, dtype=np.float32).reshape(-1)).tofile(
            f'/tmp/stages/layer{layer}_{name}.bin')
    write('attn_in_pre', attn_in_pre)
    write('ln', rms(attn_in, g('attn_norm.weight')) if sliding else rms(attn_in, g('q_input_norm.weight')))
    write('q', qv)
    write('k', kk)
    write('v', vv)
    write('sink', sink)
    write('attnpre', proj_out)
    write('attn', conv_out)
    write('mlp_in', x)
    write('mlp_pre', acc)
    write('mlp', mlp_out)
    write('out', out)
    print('wrote layer', layer, 'pass', loop_pass, 'stages')
