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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--quant", type=Path, required=True)
    parser.add_argument("--json", type=Path, help="write the slot plan here")
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
