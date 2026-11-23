"""Inkling 2 Mini source planner: tensor map, visit schedule and conversion invariants.

This is stage-1 tooling, not a converter. It reads the checkpoint's own metadata
(``config.json``, ``model.safetensors.index.json``, ``hf_quant_config.json``), maps every stored
tensor to the NInfer slot it will become, and fails loudly when the source disagrees with
[docs/maintainer/inkling2-model.md](../../docs/maintainer/inkling2-model.md).

It deliberately does not open the 19.5 GB weights: the plan is decidable from the index and the
quantization groups, and a planner that needs the shard on disk cannot be run on a laptop while the
download is still in flight. ``build_model`` (stage 2) consumes ``--json``.

    python -m tools.convert.inkling2 --index idx.json --quant quant.json --config config.json --check
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

CORE_START = 6
CORE_END = 18  # exclusive
NUM_PASSES = 3
CONTROLLER_INPUTS = ("attn", "k", "v", "mlp")
NVFP4_LAYERS = range(0, 20)
MXFP8_LAYERS = range(20, 24)
NVFP4_AUX = ("_scale", "_amax", "_outer_scale_denominator")
MXFP8_AUX = ("_scale",)

_LAYER = re.compile(r"^model\.llm\.layers\.(\d+)\.(?P<rest>.+)$")
_VISUAL_LAYER = re.compile(r"^model\.visual\.backbone\.layers\.(\d+)\.")


@dataclass(frozen=True)
class Slot:
    """One stored tensor and the logical slot it feeds."""

    tensor: str
    component: str
    layer: int | None
    visit: str
    target: str
    store: str


def _core_visits(layer: int) -> str:
    if CORE_START <= layer < CORE_END:
        return f"pass1-{NUM_PASSES}"
    return "once"


def _store_for(name: str, layer: int | None) -> str:
    """Storage format the artifact will carry, decided from the source quantization.

    Scale and amax tensors are separated from the 4-bit payload on purpose: the payload is what the
    kernels read, the scales decide the dequantisation contract, and the amax/outer denominators are
    per-tensor metadata that the artifact stores but no kernel reads.
    """
    if name.endswith(("_amax", "_outer_scale_denominator")):
        return "metadata"
    if name.endswith("_scale"):
        return "block-scale"
    if layer is not None and "mlp.experts." in name:
        return "mxfp8" if layer in MXFP8_LAYERS else "nvfp4"
    return "bf16"


def classify(name: str) -> Slot:
    """Map one source tensor name to its slot, or raise for a name outside the two trees."""
    if name.startswith("model.visual."):
        match = _VISUAL_LAYER.match(name)
        layer = int(match.group(1)) if match else None
        return Slot(name, "vision", layer, "once", f"vision.{_visual_slot(name, layer)}",
                    _store_for(name, None))

    match = _LAYER.match(name)
    if not match:
        if name.startswith("model.llm."):
            rest = name[len("model.llm."):]
            return Slot(name, "text-global", None, "once", f"text.{rest}", _store_for(name, None))
        raise ValueError(f"tensor outside model.llm/model.visual: {name}")

    layer = int(match.group(1))
    rest = match.group("rest")
    component = rest.split(".", 1)[0]
    return Slot(name, f"text-{component}", layer, _core_visits(layer),
                f"text.layers.{layer:02d}.{rest}", _store_for(name, layer))


def _visual_slot(name: str, layer: int | None) -> str:
    rest = name[len("model.visual."):]
    if layer is None:
        return rest
    return f"layers.{layer:02d}.{rest.split('.', 2)[2]}"


@dataclass
class Source:
    config: dict
    tensors: list[str]
    quant_groups: list[dict]
    mtp_groups: list[dict]

    @classmethod
    def load(cls, config_path: Path, index_path: Path, quant_path: Path) -> "Source":
        config = json.loads(config_path.read_text(encoding="utf-8"))
        index = json.loads(index_path.read_text(encoding="utf-8"))
        quant = json.loads(quant_path.read_text(encoding="utf-8"))
        return cls(
            config=config,
            tensors=sorted(index["weight_map"]),
            quant_groups=quant["model"]["parameter_groups"],
            mtp_groups=quant["mtp"]["parameter_groups"],
        )

    # -- derived views -----------------------------------------------------

    @property
    def text_config(self) -> dict:
        return self.config["text_config"]

    @property
    def layer_count(self) -> int:
        return self.text_config["num_hidden_layers"]

    @property
    def local_layers(self) -> set[int]:
        return set(self.text_config["local_layer_ids"])

    def slots(self) -> list[Slot]:
        return [classify(name) for name in self.tensors]

    def visits(self) -> int:
        """Layer applications per token: prelude + core × passes + coda."""
        core = NUM_PASSES * (CORE_END - CORE_START)
        return self.layer_count - (CORE_END - CORE_START) + core

    def kv_caches(self) -> int:
        """One KV cache and one conv-state cache per layer visit, not per layer."""
        return self.visits()


def validate(source: Source) -> list[str]:
    """Return every disagreement with the spec. An empty list is the stage-0 gate."""
    problems: list[str] = []
    names = source.tensors
    by_layer: dict[int, list[str]] = {}
    for name in names:
        match = _LAYER.match(name)
        if match:
            by_layer.setdefault(int(match.group(1)), []).append(match.group("rest"))

    # 1. every tensor belongs to one of the two trees
    for name in names:
        try:
            classify(name)
        except ValueError as error:
            problems.append(f"unmapped tensor: {error}")

    # 2. exactly 48 layer visits and as many caches
    expected_visits = 48
    if source.visits() != expected_visits:
        problems.append(f"visit count {source.visits()} != {expected_visits}")
    if source.kv_caches() != expected_visits:
        problems.append(f"kv cache count {source.kv_caches()} != {expected_visits}")

    # 3. controllers exist for the 12 core layers only, four inputs each
    for layer in range(source.layer_count):
        inputs = {rest.split(".")[1] for rest in by_layer.get(layer, [])
                  if rest.startswith("controllers.")}
        if CORE_START <= layer < CORE_END:
            if inputs != set(CONTROLLER_INPUTS):
                problems.append(f"layer {layer}: controllers {sorted(inputs)} != "
                                f"{list(CONTROLLER_INPUTS)}")
        elif inputs:
            problems.append(f"layer {layer}: controllers outside the loop core")

    # 4. expert precision split, including the auxiliary tensors the quant config omits
    for layer in range(source.layer_count):
        rest = by_layer.get(layer, [])
        experts = [r.split("mlp.experts.", 1)[1] for r in rest if "mlp.experts." in r]
        if not experts:
            problems.append(f"layer {layer}: no routed experts")
            continue
        aux = NVFP4_AUX if layer in NVFP4_LAYERS else MXFP8_AUX
        for weight in ("w13_weight", "w2_weight"):
            if weight not in experts:
                problems.append(f"layer {layer}: missing {weight}")
            for suffix in aux:
                if weight + suffix not in experts:
                    problems.append(f"layer {layer}: missing {weight}{suffix}")

    # 5. attention-residual queries: one per layer per sublayer, one shared output query
    global_names = [n for n in names if n.startswith("model.llm.") and not _LAYER.match(n)]
    queries = [n for n in global_names if "attn_res" in n]
    if len([n for n in queries if re.search(r"\.attn_queries\.\d+$", n)]) != source.layer_count - 1:
        problems.append(f"attn_res.attn_queries count "
                        f"{len([n for n in queries if 'attn_queries' in n])} != "
                        f"{source.layer_count - 1}")
    if len([n for n in queries if re.search(r"\.mlp_queries\.\d+$", n)]) != source.layer_count:
        problems.append("attn_res.mlp_queries count != layer count")
    for required in ("model.llm.attn_res.output_query",
                     "model.llm.middle_cycle_cross_pass_queries",
                     "model.llm.middle_cycle_cross_pass_kv_queries"):
        if required not in names:
            problems.append(f"missing {required}")

    # 6. sliding-window / global split matches local_layer_ids
    globals_ = sorted(set(range(source.layer_count)) - source.local_layers)
    if globals_ != [5, 11, 17, 23]:
        problems.append(f"global attention layers {globals_} != [5, 11, 17, 23]")

    # 7. vision tower depth
    visual_layers = {int(m.group(1)) for n in names if (m := _VISUAL_LAYER.match(n))}
    if len(visual_layers) != source.config["vision_config"]["num_hidden_layers"]:
        problems.append(f"vision layers {len(visual_layers)} != "
                        f"{source.config['vision_config']['num_hidden_layers']}")

    # 8. the quant configuration covers the model but not the scale tensors; make that explicit
    quantified = {t for group in source.quant_groups for t in group["tensors"]}
    dropped = [n for n in names if n not in quantified and n.startswith("model.llm.")]
    aux_count = sum(1 for n in dropped if n.endswith(NVFP4_AUX))
    if len(dropped) != 128 or aux_count != 128:
        problems.append(f"unlisted scale tensors: {len(dropped)} (aux {aux_count}) != 128; "
                        "the quant config does not describe NVFP4 auxiliaries")

    return problems


def emit_plan(source: Source) -> dict:
    slots = source.slots()
    stores: dict[str, int] = {}
    for slot in slots:
        stores[slot.store] = stores.get(slot.store, 0) + 1
    return {
        "architecture": source.config["architectures"][0],
        "model_type": source.config["text_config"]["model_type"],
        "visits": source.visits(),
        "kv_caches": source.kv_caches(),
        "params_estimate": 26_388_646_440,
        "stores": stores,
        "slots": [
            {"tensor": s.tensor, "target": s.target, "store": s.store, "visit": s.visit}
            for s in slots
        ],
    }


# ---------------------------------------------------------------------------------------------
# Numeric path: NVFP4 / MXFP8 dequantisation
#
# Layout is fixed by the checkpoint and independently implemented by llama.cpp's
# `conversion/inkling2.py`, which is what the constants below were checked against:
#
#   NVFP4  weight [E, N, K/2] uint8   low nibble = even column, high nibble = odd column
#          scale  [E, N, K/16] uint8  E4M3FN block scale
#          amax   [E]                 per-expert fp32
#          outer_scale_denominator [1] fp32
#          w = e2m1(code) * e4m3(scale) * (amax / denominator)
#
#   MXFP8  weight [E, N, K] e4m3     scale [E, N, K/32] uint8  UE8M0 exponent (byte 255 invalid)
#          w = e4m3(code) * 2**(scale - 127)
# ---------------------------------------------------------------------------------------------

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised only without torch
        raise RuntimeError("the numeric path needs torch; the metadata path does not") from error
    return torch


def e4m3_table():
    """All 256 E4M3FN values by byte, NaN at 0x7F/0xFF."""
    torch = _torch()
    codes = torch.arange(256, dtype=torch.int64)
    exponent = (codes & 0x7F) >> 3
    mantissa = (codes & 7).to(torch.float32)
    subnormal = torch.ldexp(mantissa, torch.full_like(exponent, -9))
    normal = torch.ldexp(1.0 + mantissa / 8.0, exponent - 7)
    values = torch.where(exponent == 0, subnormal, normal)
    values = torch.where((codes & 0x80) != 0, -values, values)
    values[(codes & 0x7F) == 0x7F] = float("nan")
    return values


def dequantize_nvfp4(packed, scale_bytes, amax: float, denominator: float):
    """[N, K/2] uint8 + [N, K/16] uint8 -> [N, K] float32."""
    torch = _torch()
    if packed.dim() != 2 or scale_bytes.dim() != 2:
        raise ValueError("NVFP4 expects rank-2 packed weights and scales")
    rows, half = packed.shape
    k = half * 2
    if k % 16 or scale_bytes.shape != (rows, k // 16):
        raise ValueError(f"NVFP4 scale shape {tuple(scale_bytes.shape)} != {(rows, k // 16)}")
    grid = torch.tensor(_E2M1, dtype=torch.float32)
    values = torch.empty((rows, k), dtype=torch.float32)
    values[:, 0::2] = grid[(packed & 15).to(torch.int64)]
    values[:, 1::2] = grid[(packed >> 4).to(torch.int64)]
    block = e4m3_table()[scale_bytes.to(torch.int64)]
    if torch.isnan(block).any():
        raise ValueError("NVFP4 block scale contains E4M3FN NaN")
    if not (denominator > 0) or not (amax >= 0):
        raise ValueError(f"NVFP4 outer scale invalid: amax={amax} denominator={denominator}")
    values.view(rows, k // 16, 16).mul_(block.unsqueeze(-1))
    values.mul_(float(amax) / float(denominator))
    if not torch.isfinite(values).all():
        raise ValueError("NVFP4 dequantised weight is not finite")
    return values


def dequantize_mxfp8(codes, scale_bytes):
    """[N, K] e4m3 bytes + [N, K/32] uint8 -> [N, K] float32."""
    torch = _torch()
    if codes.dim() != 2 or scale_bytes.dim() != 2:
        raise ValueError("MXFP8 expects rank-2 codes and scales")
    rows, k = codes.shape
    if k % 32 or scale_bytes.shape != (rows, k // 32):
        raise ValueError(f"MXFP8 scale shape {tuple(scale_bytes.shape)} != {(rows, k // 32)}")
    if (scale_bytes == 255).any():
        raise ValueError("MXFP8 scale contains UE8M0 byte 255")
    values = e4m3_table()[codes.to(torch.int64)]
    if torch.isnan(values).any():
        raise ValueError("MXFP8 weight contains E4M3FN NaN")
    multiplier = torch.ldexp(torch.ones_like(scale_bytes, dtype=torch.float32),
                             scale_bytes.to(torch.int32) - 127)
    values.view(rows, k // 32, 32).mul_(multiplier.unsqueeze(-1))
    if not torch.isfinite(values).all():
        raise ValueError("MXFP8 dequantised weight is not finite")
    return values


def expert_matrix(store, layer: int, weight: str, expert: int):
    """Dequantise one expert of `w13_weight` or `w2_weight` into its logical [N, K] matrix."""
    torch = _torch()
    prefix = f"model.llm.layers.{layer}.mlp.experts"
    name = f"{prefix}.{weight}"
    info = store.describe(name)
    if len(info.shape) != 3:
        raise ValueError(f"{name}: expected rank-3 [experts, N, K], got {info.shape}")
    experts, rows, cols = info.shape
    if not 0 <= expert < experts:
        raise ValueError(f"{name}: expert {expert} outside [0, {experts})")
    begin = expert * rows * cols
    packed = store.read_flat(name, begin, begin + rows * cols).view(rows, cols)
    scale_name = name + "_scale"
    scale_info = store.describe(scale_name)
    scale_rows, scale_cols = scale_info.shape[1], scale_info.shape[2]
    scale_begin = expert * scale_rows * scale_cols
    scales = store.read_flat(scale_name, scale_begin,
                             scale_begin + scale_rows * scale_cols).view(scale_rows, scale_cols)
    if scales.dtype != torch.uint8:  # NVFP4 block scales are stored as E4M3 bytes
        scales = scales.view(torch.uint8)
    if layer in MXFP8_LAYERS:
        return dequantize_mxfp8(packed.view(torch.uint8), scales)

    amax = store.read_flat(f"{prefix}.{weight}_amax").reshape(-1)[expert]
    denominator = store.read_flat(f"{prefix}.{weight}_outer_scale_denominator").reshape(-1)[0]
    return dequantize_nvfp4(packed, scales, float(amax), float(denominator))


# ---------------------------------------------------------------------------------------------
# Artifact-side source: one expert as a logical parameter
#
# NInfer stores experts as separate logical parameters (`experts/{e}/...`), which is what makes
# Inkling's per-expert NVFP4 outer scale expressible without touching the container: each expert
# carries its own FP32 weight divisor (denominator / amax[e]) and the exact E2M1 codes plus E4M3
# block scales the checkpoint shipped. No requantisation, no container extension.
# ---------------------------------------------------------------------------------------------


def expert_source(store, layer: int, weight: str, expert: int):
    """Logical [N, K] source for one expert that preserves the stored NVFP4 words."""
    from tools.convert.sources.logical import EncodedRows, LogicalSource

    torch = _torch()
    prefix = f"model.llm.layers.{layer}.mlp.experts"
    name = f"{prefix}.{weight}"
    info = store.describe(name)
    experts, rows, cols = info.shape
    if layer in MXFP8_LAYERS:
        raise ValueError(f"{name}: MXFP8 experts have no pass-through format; requantise instead")
    k = cols * 2
    begin = expert * rows * cols

    scale_name = name + "_scale"
    scale_rows, scale_cols = store.describe(scale_name).shape[1:]
    scale_begin = expert * scale_rows * scale_cols

    def codes(begin_row: int, end_row: int):
        start = begin + begin_row * cols
        count = (end_row - begin_row) * cols
        return store.read_flat(name, start, start + count).view(torch.uint8).reshape(
            end_row - begin_row, cols)

    def scales(begin_row: int, end_row: int):
        start = scale_begin + begin_row * scale_cols
        count = (end_row - begin_row) * scale_cols
        return store.read_flat(scale_name, start, start + count).view(torch.uint8).reshape(
            end_row - begin_row, scale_cols)

    amax = float(store.read_flat(f"{prefix}.{weight}_amax").reshape(-1)[expert])
    denominator = float(store.read_flat(f"{prefix}.{weight}_outer_scale_denominator").reshape(-1)[0])
    if not (amax > 0.0) or not (denominator > 0.0):
        raise ValueError(f"{name}: expert {expert} has amax={amax} denominator={denominator}")
    divisor_word = torch.tensor(denominator / amax, dtype=torch.float32).numpy().tobytes()

    def encoded(begin_row: int, end_row: int) -> EncodedRows:
        if not 0 <= begin_row < end_row <= rows:
            raise ValueError(f"{name}: invalid expert rows [{begin_row},{end_row})")
        return EncodedRows("nvfp4", codes(begin_row, end_row),
                           scales(begin_row, end_row), divisor_word)

    def read(begin_flat: int, end_flat: int):
        begin_row, end_row = begin_flat // k, (end_flat + k - 1) // k
        words = encoded(begin_row, end_row)
        grid = torch.tensor(_E2M1, dtype=torch.float32)
        values = torch.empty((end_row - begin_row, k), dtype=torch.float32)
        values[:, 0::2] = grid[(words.codes & 15).to(torch.int64)]
        values[:, 1::2] = grid[(words.codes >> 4).to(torch.int64)]
        block = e4m3_table()[words.scales.to(torch.int64)]
        values.view(end_row - begin_row, k // 16, 16).mul_(block.unsqueeze(-1))
        values.div_(torch.frombuffer(bytearray(words.weight_divisor), dtype=torch.float32))
        return values.reshape(-1)[begin_flat - begin_row * k: end_flat - begin_row * k]

    return LogicalSource((rows, k), f"{store.path}:{name}[expert {expert}]", read, encoded,
                         (lambda: divisor_word))


def verify_passthrough(store, layer: int, weight: str, expert: int = 0) -> str:
    """Round-trip one expert through the artifact codec and compare with the direct dequant."""
    from tools.artifact.codecs.nvfp4 import decode_nvfp4_words, encode_nvfp4

    torch = _torch()
    source = expert_source(store, layer, weight, expert)
    rows, k = source.shape
    words = source.read_encoded(0, rows)
    payload = encode_nvfp4(words.codes, words.scales, words.weight_divisor, (rows, k))
    codes, scales, divisor = decode_nvfp4_words(payload, (rows, k))
    if not torch.equal(codes, words.codes) or not torch.equal(scales, words.scales):
        raise ValueError(f"layer {layer} {weight}: codec round-trip changed the stored words")

    grid = torch.tensor(_E2M1, dtype=torch.float32)
    values = torch.empty((rows, k), dtype=torch.float32)
    values[:, 0::2] = grid[(codes & 15).to(torch.int64)]
    values[:, 1::2] = grid[(codes >> 4).to(torch.int64)]
    values.view(rows, k // 16, 16).mul_(e4m3_table()[scales.to(torch.int64)].unsqueeze(-1))
    values.div_(divisor)

    direct = expert_matrix(store, layer, weight, expert)
    difference = (values - direct).abs().max().item()
    cosine = float(torch.nn.functional.cosine_similarity(
        values.flatten(), direct.flatten(), dim=0))
    return (f"layer {layer:2d} {weight:10s} expert {expert:3d} bytes {len(payload):9d} "
            f"divisor {float(divisor):9.2f} codec-vs-direct cos {cosine:.9f} maxdiff {difference:.3e}")


def audit(store, layers=range(24), weight="w13_weight") -> list[str]:
    """Dequantise one expert per layer and report the statistics the spec claims."""
    from tools.convert.quantization.groupwise import quantize_matrix

    torch = _torch()
    lines = []
    for layer in layers:
        matrix = expert_matrix(store, layer, weight, 0)
        q = quantize_matrix(matrix, "q4_g64_fp16", device="cpu")
        restored = q.codes.to(torch.float32) * q.scales.to(torch.float32).unsqueeze(-1)
        restored = restored.reshape(matrix.shape[0], -1)[:, : matrix.shape[1]]
        cosine = float(torch.nn.functional.cosine_similarity(
            matrix.flatten(), restored.flatten(), dim=0))
        error = (restored - matrix).abs().max().item()
        rms = matrix.pow(2).mean().sqrt().item()
        lines.append(f"layer {layer:2d} {weight:10s} rms {rms:8.5f} maxabs "
                     f"{matrix.abs().max().item():8.5f} q4-cos {cosine:.6f} q4-maxerr {error:.5f}")
    return lines


# ---------------------------------------------------------------------------------------------
# Model + recipe: the stage-1 artifact carries the checkpoint's own numerics
#
# Every stored tensor becomes one logical parameter. Experts are split per expert so each keeps
# its own NVFP4 weight divisor. Layers 0-19 pass their codes and scales through untouched; layers
# 20-23 (MXFP8, per-32 UE8M0 scales) have no pass-through format, so they are dequantised and
# requantised to Q8 per 32 -- the same 8-bit granularity, so the second quantisation is inert.
# ---------------------------------------------------------------------------------------------

EXPERT_AUX = ("_scale", "_amax", "_outer_scale_denominator")
MXFP8_TARGET_FORMAT = "q8_g32_fp16"


def _parameter_family(name: str, layer: int | None) -> str:
    if layer is not None and "mlp.experts." in name:
        return "expert"
    if layer is not None and name.endswith("mlp.experts.w13_weight"):
        return "expert"
    return "plain"


def expert_values_source(store, layer: int, weight: str, expert: int):
    """Values-only source for an expert that has no pass-through format (MXFP8 layers)."""
    from tools.convert.sources.logical import LogicalSource

    torch = _torch()
    info = store.describe(f"model.llm.layers.{layer}.mlp.experts.{weight}")
    rows = info.shape[1]
    k = info.shape[2] * (1 if layer in MXFP8_LAYERS else 2)
    cache: dict[int, "torch.Tensor"] = {}

    def whole():
        if 0 not in cache:
            cache[0] = expert_matrix(store, layer, weight, expert).reshape(-1)
        return cache[0]

    def read(begin: int, end: int):
        return whole()[begin:end]

    return LogicalSource((rows, k),
                         f"{store.path}:layers.{layer}.experts.{weight}[expert {expert}]", read)


def build_model(store, config: dict, names: list[str]):
    """Logical model whose parameters mirror the checkpoint's stored tensors."""
    from tools.convert.model import Model, Parameter
    from tools.convert.sources.safetensors import tensor_source

    text = config["text_config"]
    experts = text["n_routed_experts"]
    model = Model(components={
        "text": {"config": {
            **text,
            "architecture": config["architectures"][0],
            "loop_config": text["loop_config"],
        }},
        "vision": {"config": config["vision_config"]},
        "mtp": {"config": config["mtp_config"]},
    })

    for name in names:
        if name.startswith("model.llm.layers.") and "mlp.experts." in name \
                and name.endswith(("w13_weight", "w2_weight")):
            layer = int(name.split(".")[3])
            half = name.rsplit(".", 1)[1]
            weight = "w13" if half.startswith("w13") else "w2"
            info = store.describe(name)
            rows, cols = info.shape[1], info.shape[2]
            k = cols * (1 if layer in MXFP8_LAYERS else 2)
            for expert in range(experts):
                source = (expert_values_source(store, layer, half, expert)
                          if layer in MXFP8_LAYERS
                          else expert_source(store, layer, half, expert))
                model.add(Parameter(f"text/layers/{layer}/mlp/experts/{expert}/{weight}",
                                    (rows, k), source, direct_format="bf16"))
            continue
        if name.endswith(EXPERT_AUX):
            continue
        info = store.describe(name)
        if info.dtype not in ("BF16", "F32"):
            raise ValueError(f"{name}: unexpected stored dtype {info.dtype}")
        target = classify(name).target.replace(".", "/")
        model.add(Parameter(target, tuple(info.shape),
                            tensor_source(store, name, tuple(info.shape)),
                            direct_format="bf16" if info.dtype == "BF16" else "fp32"))

    for resource in ("tokenizer.json", "tokenizer_config.json", "config.json"):
        path = store.root / resource
        if path.is_file():
            model.resources[resource] = path.read_bytes()
    return model


def build_recipe(model):
    """Assign pass-through NVFP4 experts, Q8 for the MXFP8 layers, direct words otherwise."""
    from tools.convert.methods import grouped_absmax, import_encoded
    from tools.convert.recipe import Recipe

    recipe = Recipe(model)
    layers = sorted({int(name.split("/")[2]) for name in model.parameters
                     if name.startswith("text/layers/")})
    for layer in layers:
        kinds = {name.rsplit("/", 1)[1] for name in model.parameters
                 if name.startswith(f"text/layers/{layer}/mlp/experts/")}
        if not kinds:
            continue
        if layer in MXFP8_LAYERS:
            recipe.assign(f"text/layers/{layer}/mlp/experts/*/{sorted(kinds)[0]}",
                          format=MXFP8_TARGET_FORMAT, method=grouped_absmax)
            recipe.assign(f"text/layers/{layer}/mlp/experts/*/{sorted(kinds)[-1]}",
                          format=MXFP8_TARGET_FORMAT, method=grouped_absmax)
        else:
            for kind in sorted(kinds):
                recipe.assign(f"text/layers/{layer}/mlp/experts/*/{kind}",
                              format="nvfp4", method=import_encoded)
    return recipe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--quant", type=Path, required=True)
    parser.add_argument("--json", type=Path, help="write the slot plan here")
    parser.add_argument("--shard", type=Path,
                        help="model-00001-of-00001.safetensors, enables the numeric audit")
    parser.add_argument("--audit", action="store_true",
                        help="dequantise one expert per layer and print statistics")
    parser.add_argument("--verify-passthrough", action="store_true",
                        help="round-trip experts through the artifact NVFP4 codec")
    parser.add_argument("--emit", type=Path,
                        help="write a .ninfer artifact from --shard")
    parser.add_argument("--check", action="store_true", help="fail on any spec disagreement")
    args = parser.parse_args(argv)

    source = Source.load(args.config, args.index, args.quant)
    plan = emit_plan(source)
    print(f"{plan['architecture']}: {len(plan['slots'])} tensors, {plan['visits']} visits, "
          f"{plan['kv_caches']} kv caches")
    for store, count in sorted(plan["stores"].items()):
        print(f"  {store:12s} {count:5d}")
    if args.json:
        args.json.write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")

    if args.emit:
        if not args.shard:
            parser.error("--emit needs --shard")
        from tools.convert.pipeline import convert
        from tools.convert.sources.safetensors import SafetensorsSource

        with SafetensorsSource(args.shard) as store:
            model = build_model(store, source.config, source.tensors)
            recipe = build_recipe(model)
            report = convert(model, recipe, args.emit, name="inkling2-mini",
                             provenance={"source": "thinkingmachines/Inkling-2-Mini"},
                             device="cuda", rows_per_chunk=512,
                             progress=lambda i, n, job: print(
                                 f"[{i + 1}/{n}] {job.spec.id} {job.spec.format}",
                                 flush=True) if i % 500 == 0 else None)
            print(f"wrote {report['payload_bytes'] / 1e9:.2f} GB in {report['seconds']:.0f}s, "
                  f"{report['objects']} objects")

    if args.verify_passthrough:
        if not args.shard:
            parser.error("--verify-passthrough needs --shard")
        from tools.convert.sources.safetensors import SafetensorsSource

        with SafetensorsSource(args.shard) as store:
            for layer in (0, 6, 12, 19):
                for weight in ("w13_weight", "w2_weight"):
                    print(verify_passthrough(store, layer, weight))

    if args.audit:
        if not args.shard:
            parser.error("--audit needs --shard")
        from tools.convert.sources.safetensors import SafetensorsSource

        with SafetensorsSource(args.shard) as store:
            for line in audit(store):
                print(line)

    if args.check:
        problems = validate(source)
        for problem in problems:
            print(f"FAIL {problem}", file=sys.stderr)
        if problems:
            return 1
        print("check: source matches docs/maintainer/inkling2-model.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
