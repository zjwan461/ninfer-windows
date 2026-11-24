"""Dump the reference model's per-layer intermediates for every visit.

The stage fixtures built from the checkpoint are only as good as the reading of the model source
they were written from: when a stage disagrees, the fixture and the kernel can be wrong together
and agree. This script hooks the model's own submodules instead, so the comparison has an
independent side.

    python tests/inkling2/dump_stages.py --out /tmp/stages_model --token-id 1000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMultimodalLM, AutoTokenizer


def text_stack(model):
    for _, module in model.named_modules():
        layers = getattr(module, "layers", None)
        if isinstance(layers, torch.nn.ModuleList) and len(layers) == 24:
            return module, layers
    raise SystemExit("could not find the 24-layer text stack")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/workspaces/models/inkling2-bf16")
    parser.add_argument("--out", type=Path, default=Path("/tmp/stages_model"))
    parser.add_argument("--token-id", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device, low_cpu_mem_usage=True
    ).eval()
    text, layers = text_stack(model)

    order: list[str] = []
    pending: dict[str, list[np.ndarray]] = {}

    def record(name: str, value) -> None:
        if isinstance(value, tuple):
            value = value[0]
        array = value.detach().float().cpu().numpy()[0]
        pending.setdefault(name, []).append(np.ascontiguousarray(array.reshape(-1), np.float32))

    def hook(name: str):
        def inner(module, inputs, output):
            record(name, output)
        return inner

    def in_hook(name: str):
        def inner(module, inputs):
            record(name, inputs[0])
        return inner

    handles = []
    for index, layer in enumerate(layers):
        tag = f"layer{index}"
        for attribute, name in (
            ("attention_input", "attn_in"),
            ("self_attn", "attn_raw"),
            ("post_attention_convolution", "attn_conv"),
            ("mlp_input", "mlp_in"),
            ("post_attention_layernorm", "mlp_normed"),
            ("mlp", "moe"),
            ("post_mlp_convolution", "mlp_conv"),
        ):
            module = getattr(layer, attribute, None)
            if module is not None:
                handles.append(module.register_forward_hook(hook(f"{tag}.{name}")))
        if getattr(layer, "attention_input", None) is not None:
            # The block stack itself, so the read can be checked against the states the model
            # actually used rather than against the ones this script would assemble.
            def block_hook(module, args, tag=tag):
                # A pre-hook that returns anything replaces the inputs; this one only records.
                pending.setdefault(f"{tag}.block_states", []).append(
                    np.ascontiguousarray(args[0].detach().float().cpu().numpy()[0].reshape(-1),
                                         np.float32))
                pending.setdefault(f"{tag}.partial", []).append(
                    np.ascontiguousarray(args[1].detach().float().cpu().numpy()[0].reshape(-1),
                                         np.float32))
                return None

            handles.append(layer.attention_input.register_forward_pre_hook(block_hook))
        attention = getattr(layer, "self_attn", None)
        if attention is not None:
            for attribute in ("q_proj", "k_proj", "v_proj", "sink_proj", "gate_proj", "o_proj"):
                module = getattr(attention, attribute, None)
                if module is not None:
                    handles.append(module.register_forward_hook(hook(f"{tag}.{attribute}")))
            linear = getattr(getattr(attention, "o_proj", None), "linear", None)
            if linear is not None:
                # The gated attention output, i.e. what the output projection consumes.
                handles.append(linear.register_forward_pre_hook(in_hook(f"{tag}.attn_gated")))
        for attribute, name in (("input_layernorm", "ln"), ("q_input_layernorm", "q_norm"),
                                ("k_input_layernorm", "k_norm"), ("v_input_layernorm", "v_norm")):
            module = getattr(layer, attribute, None)
            if module is not None:
                handles.append(module.register_forward_hook(hook(f"{tag}.{name}")))
        handles.append(layer.register_forward_hook(hook(f"{tag}.out")))
        handles.append(layer.register_forward_pre_hook(in_hook(f"{tag}.in")))
        handles.append(layer.mlp.gate.register_forward_hook(
            lambda module, inputs, output, tag=tag: pending.setdefault(f"{tag}.route", []).append(
                np.ascontiguousarray(output[2].detach().cpu().numpy()[0].astype(np.int64)))))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids = torch.tensor([[args.token_id]], dtype=torch.long, device=args.device)
    print("prompt token:", tokenizer.decode([args.token_id]))
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
    for handle in handles:
        handle.remove()

    counts = {len(v) for v in pending.values()}
    print("records per stage:", counts)
    # Every stage records one entry per visit; write them indexed by visit order per layer.
    for name, values in pending.items():
        for index, array in enumerate(values):
            np.ascontiguousarray(array, np.float32).tofile(args.out / f"{name}_{index:02d}.bin")
    print("wrote", len(pending), "stage series to", args.out)


if __name__ == "__main__":
    main()
