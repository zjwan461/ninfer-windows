"""Reference attention sub-block for one layer and one token, computed from the checkpoint.

This is the reference mathematics written straight from the model source, not a call into it: it
reads only the layer's own tensors through a lazy safetensors handle, so producing fixtures costs
seconds instead of the forty minutes a 52 GB BF16 model load takes on this machine.

Writes the same stage fixtures dump_layer0.py writes, so tests can compare against the checkpoint
without loading the 52 GB reference model.
"""
import sys
import numpy as np
from safetensors import safe_open

SHARD = '/workspaces/models/inkling2/model-00001-of-00001.safetensors'
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
FULL = {5, 11, 17, 23}
EPS = 1e-6
QFLOOR, QALPHA = 8192.0, 0.1
QK_CLIP = 22.627416997969522
REL_CLIP = 8.0

def e4m3_table():
    t = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e, m = (b >> 3) & 0xF, b & 0x7
        t[b] = s * (m / 8.0) * 2.0 ** -6 if e == 0 else s * (1 + m / 8.0) * 2.0 ** (e - 7)
    return t

E4M3 = e4m3_table()

def rms(x, w=None):
    """Row-wise when a head axis is present: the reference normalises each head separately."""
    x = x.astype(np.float64)
    if x.ndim == 2:
        y = x / np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + EPS)
    else:
        y = x / np.sqrt((x ** 2).mean() + EPS)
    return y * (w if w is not None else 1.0)

def read_block(query, stack, eps=EPS):
    normed = stack / np.sqrt((stack.astype(np.float64) ** 2).mean(axis=1, keepdims=True) + eps)
    logits = (query.astype(np.float64) @ normed.T) / np.sqrt(2304)
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    return p @ normed

def clip(x, scale, threshold, tau=1.0):
    v = x.astype(np.float64) * scale.astype(np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    v = v * np.minimum(threshold / (n + 1e-6), 1.0)
    return (v * tau).astype(np.float32)

layer = int(sys.argv[1])
sliding = layer not in FULL
bf = lambda p: (np.fromfile(p, dtype='<u2').astype(np.uint32) << 16).view(np.float32)

h_in = np.fromfile(f'/tmp/oracle1/visit_in_{layer}.bin', dtype=np.float32)
stack = np.stack([np.fromfile('/tmp/oracle1/visit_in_0.bin', dtype=np.float32), h_in])

with safe_open(SHARD, framework='pt') as f:
    g = lambda n: f.get_tensor(f'model.llm.layers.{layer}.{n}').float().numpy()
    if layer == 0:
        attn_in = h_in.astype(np.float64)
    else:
        nq = 1 if sliding else 3
        q = f.get_tensor(f'model.llm.attn_res.attn_queries.{layer - 1}').float().numpy().reshape(-1, 2304)[:nq]
        streams = read_block(q, stack)
        attn_in = streams[0]
        k_base = streams[1] if nq == 3 else streams[0]
        v_base = streams[2] if nq == 3 else streams[0]

    if sliding:
        norm_w = g('attn_norm.weight')
        qn, kn, vn = rms(attn_in, norm_w), rms(k_base, norm_w), rms(v_base, norm_w)
    else:
        qn = rms(attn_in, g('q_input_norm.weight'))
        kn = rms(k_base, g('k_input_norm.weight'))
        vn = rms(v_base, g('v_input_norm.weight'))

    tau = 1.0 + QALPHA * np.log(max((0 + 1) / QFLOOR, 1.0))
    wq, wk, wv = g('attn.wq_du.weight'), g('attn.wk_dv.weight'), g('attn.wv_dv.weight')
    qv = (wq.astype(np.float64) @ qn).reshape(24, 128)
    qv = clip(qv, g('attn.q_l2_clip.scale'), QK_CLIP, 1.0 if sliding else tau)

    kv = (wk.astype(np.float64) @ kn).reshape(8, 128)
    if sliding:
        conv = g('attn.k_sconv.weight').reshape(1024, 4)
        kv = ((conv[:, 3:4] * 1.0) * (wk.astype(np.float64) @ kn).reshape(1024, 1) +
              (wk.astype(np.float64) @ kn).reshape(1024, 1)) / np.sqrt(5.0)
    else:
        proj = g('attn.k_sconv.weight_proj')  # [32, 2304]
        static = g('attn.k_sconv.static_w').reshape(4, 1024)
        dyn = (proj.astype(np.float64) @ kn).reshape(8, 4)
        base = (wk.astype(np.float64) @ kn).reshape(1024, 1)
        # tap j=3 is the current token; the dynamic part is shared by the 128 channels of a head
        taps = static[3, :] + np.repeat(dyn[:, 3], 128)
        kv = (taps[:, None] * base + base) / 3.0
    kk = clip(kv.reshape(8, 128), np.ones((8, 128), dtype=np.float32), QK_CLIP).reshape(8, 128)

    vv = (wv.astype(np.float64) @ vn).reshape(8, 128)
    if sliding:
        conv = g('attn.v_sconv.weight').reshape(1024, 4)
        vv = ((conv[:, 3:4] * 1.0) * (wv.astype(np.float64) @ vn).reshape(1024, 1) +
              (wv.astype(np.float64) @ vn).reshape(1024, 1)) / np.sqrt(5.0)
        vv = vv.reshape(8, 128)
    else:
        wvv = g('attn.v_sconv.weight_proj')
        static_v = g('attn.v_sconv.static_w').reshape(4, 1024)
        dyn_v = (wvv.astype(np.float64) @ vn).reshape(8, 4)
        np.ascontiguousarray(dyn_v.reshape(-1), dtype=np.float32).tofile(
            f'/tmp/stages/layer{layer}_dynv.bin')
        base_v = (wv.astype(np.float64) @ vn).reshape(1024, 1)
        taps_v = static_v[3, :] + np.repeat(dyn_v[:, 3], 128)
        vv = ((taps_v[:, None] * base_v + base_v) / 3.0).reshape(8, 128)
        vconv = vv.copy()
    vv = rms(vv)

    rel = (g('attn.wr_du.weight').astype(np.float64) @ qn).reshape(24, 16)
    rel = clip(rel, g('attn.rel_l2_clip.scale'), REL_CLIP, 1.0 if sliding else tau)
    proj_rel = g('attn.rel_logits_proj.proj').astype(np.float64)  # [16, 1024]
    bias = (rel @ proj_rel)[:, 0]
    sink = (g('attn.attn_sink_proj.weight').astype(np.float64) @ qn)
    gate = 1.0 / (1.0 + np.exp(-(g('attn.gate_attn.weight').astype(np.float64) @ qn)))

    kk_unrepeated, vv_unrepeated = kk, vv
    kk = np.repeat(kk, 3, axis=0)
    vv = np.repeat(vv, 3, axis=0)
    logits = (qv * kk).sum(axis=1) / np.sqrt(128) + bias
    # One sink logit per head: the softmax runs over [key, sink] and the key keeps the rest.
    combined = np.stack([logits, sink], axis=1)
    p = np.exp(combined - combined.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    # Per head: the key keeps the weight the sink does not, applied to that head's value vector.
    out = ((p[:, 0, None] * vv).reshape(-1) * gate).reshape(1, 3072)
    attn = out
    proj_out = (g('attn.wo_ud.weight').astype(np.float64) @ attn.reshape(-1)).reshape(1, 2304)
    # tap 3 is the current token; a [2304,1] column here would broadcast into a 2304x2304 matrix
    sconv = g('attn_sconv.weight').reshape(2304, 4)
    conv_out = (sconv[:, 3] * proj_out.reshape(-1) + proj_out.reshape(-1)) / np.sqrt(5.0)

    w = lambda name, arr: np.ascontiguousarray(arr.reshape(-1), dtype=np.float32).tofile(
        f'/tmp/stages/layer{layer}_{name}.bin')
    if not sliding:
        w2 = lambda name, arr: np.ascontiguousarray(arr.reshape(-1), dtype=np.float32).tofile(
            f'/tmp/stages/layer{layer}_{name}.bin')
        w2('kbase', (wk.astype(np.float64) @ kn).reshape(-1))
        w2('dynk', (g('attn.k_sconv.weight_proj').astype(np.float64) @ kn).reshape(-1))
        w2('vbase', (wv.astype(np.float64) @ vn).reshape(-1))
        w2('vconv', vconv.reshape(-1))
    w('ln', rms(attn_in, g('attn_norm.weight')) * 1.0 if sliding else qn)
    w('q', qv.reshape(-1))
    w('k', kk_unrepeated.reshape(-1))
    w('v', vv_unrepeated.reshape(-1))
    w('sink', sink)
    w('attnpre', proj_out.reshape(-1))
    w('attn', conv_out.reshape(-1))
    print('wrote layer', layer, 'stages; tau', tau)
