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
    parser.add_argument("--layer", type=int, default=0, help="which decoder layer to capture")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model = AutoModelForMultimodalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True
    ).eval()
    layers = find_text_stack(model)
    layer0 = layers[args.layer]

    captured: dict[str, np.ndarray] = {}
    tag = f"layer{args.layer}"

    def capture(name):
        def hook(module, inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[name] = tensor.detach().float().cpu().numpy()
        return hook

    def capture_input(module, args):
        # A pre-hook must return None (or replacement args); returning the captured array would be
        # read as a replacement argument list, which is how this first failed.
        captured[f"{tag}_in"] = args[0].detach().float().cpu().numpy()

    handles = [
        layer0.register_forward_pre_hook(capture_input),
        layer0.input_layernorm.register_forward_hook(capture(f"{tag}_ln")),
        *([layer0.attention_input.register_forward_hook(
            lambda m, i, o: captured.__setitem__(
                f"{tag}_attn_input",
                np.reshape((o[0] if isinstance(o, tuple) else o).detach().float().cpu().numpy(),
                           -1)))] if getattr(layer0, "attention_input", None) is not None else []),
        layer0.self_attn.q_proj.register_forward_hook(capture(f"{tag}_q")),
        layer0.self_attn.k_proj.register_forward_hook(capture(f"{tag}_k")),
        layer0.self_attn.v_proj.register_forward_hook(capture(f"{tag}_v")),
        layer0.self_attn.sink_proj.register_forward_hook(capture(f"{tag}_sink")),
        layer0.self_attn.relative_position_bias.register_forward_hook(
            capture(f"{tag}_relbias")),
        layer0.self_attn.register_forward_hook(capture(f"{tag}_attnpre")),
        layer0.post_attention_convolution.register_forward_hook(capture(f"{tag}_attn")),
        layer0.post_attention_layernorm.register_forward_hook(capture(f"{tag}_mlp_in")),
        layer0.mlp.register_forward_hook(capture(f"{tag}_mlp_pre")),
        layer0.post_mlp_convolution.register_forward_hook(capture(f"{tag}_mlp")),
        layer0.register_forward_hook(capture(f"{tag}_out")),
    ]

    ids = torch.tensor([[args.token_id]], dtype=torch.long, device="cuda")
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    for handle in handles:
        handle.remove()

    for name in (f"{tag}_in", f"{tag}_ln", f"{tag}_q", f"{tag}_k", f"{tag}_v", f"{tag}_sink",
                 f"{tag}_relbias", f"{tag}_attnpre", f"{tag}_attn", f"{tag}_out",
                 f"{tag}_attn_input", f"{tag}_mlp_in", f"{tag}_mlp_pre", f"{tag}_mlp"):
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
