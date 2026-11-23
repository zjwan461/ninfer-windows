"""Dump the inputs and attention-subblock outputs of layer 0 for one token.

The C++ parity test replays the same token through NInfer's ops, so this writes raw float32
buffers instead of a framework-specific format:

    layer0_in.bin       [tokens, hidden]  hidden state entering layer 0
    layer0_attn.bin     [tokens, hidden]  output of post_attention_convolution
    layer0_out.bin      [tokens, hidden]  output of the whole layer
    meta.json           token ids and shapes

    python tests/inkling2/dump_layer0.py --model <bf16 dir> --out <dir> --token-id 1234
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMultimodalLM

LAYER_COUNT = 24


def find_text_stack(model) -> torch.nn.ModuleList:
    for _, module in model.named_modules():
        stack = getattr(module, "layers", None)
        if isinstance(stack, torch.nn.ModuleList) and len(stack) == LAYER_COUNT:
            return stack
    raise SystemExit("no 24-layer text stack found")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--token-id", type=int, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model = AutoModelForMultimodalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True
    ).eval()
    layers = find_text_stack(model)

    captured: dict[str, np.ndarray] = {}

    def capture(name):
        def hook(module, inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[name] = tensor.detach().float().cpu().numpy()
        return hook

    def capture_input(module, args):
        # A pre-hook must return None (or replacement args); returning the captured array would be
        # read as a replacement argument list, which is how this first failed.
        captured["layer0_in"] = args[0].detach().float().cpu().numpy()

    handles = [
        layers[0].register_forward_pre_hook(capture_input),
        layers[0].input_layernorm.register_forward_hook(capture("layer0_ln")),
        layers[0].self_attn.q_proj.register_forward_hook(capture("layer0_q")),
        layers[0].self_attn.k_proj.register_forward_hook(capture("layer0_k")),
        layers[0].self_attn.v_proj.register_forward_hook(capture("layer0_v")),
        layers[0].self_attn.sink_proj.register_forward_hook(capture("layer0_sink")),
        layers[0].self_attn.relative_position_bias.register_forward_hook(
            capture("layer0_relbias")),
        layers[0].self_attn.register_forward_hook(capture("layer0_attnpre")),
        layers[0].post_attention_convolution.register_forward_hook(capture("layer0_attn")),
        layers[0].register_forward_hook(capture("layer0_out")),
    ]

    ids = torch.tensor([[args.token_id]], dtype=torch.long, device="cuda")
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    for handle in handles:
        handle.remove()

    for name in ("layer0_in", "layer0_ln", "layer0_q", "layer0_k", "layer0_v", "layer0_sink",
                 "layer0_relbias", "layer0_attnpre", "layer0_attn", "layer0_out"):
        if name not in captured:
            raise SystemExit(f"hook {name} never fired")
        array = captured[name].reshape(1, -1).astype(np.float32)
        print(f"  {name} raw shape {captured[name].shape}")
        (args.out / f"{name}.bin").write_bytes(array.tobytes())
        print(f"{name}: {array.shape}")
    (args.out / "meta.json").write_text(json.dumps(
        {"token_ids": [args.token_id], "hidden": 2304, "layers": 1}, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
