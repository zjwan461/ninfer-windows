"""Dump the MLP sub-block of layer 0 for one token.

    layer0_mlp_in.bin     [hidden]     post_attention_layernorm output
    layer0_mlp_pre.bin    [hidden]     MoE output before post_mlp_convolution
    layer0_mlp.bin        [hidden]     after post_mlp_convolution
    layer0_router.json    selected experts and their weights
    expert_<e>_w13.bin    [2*inter, hidden]  BF16, dequantised
    expert_<e>_w2.bin     [hidden, inter]    BF16, dequantised

Only the experts the router selects are written, which keeps the fixture at a few hundred MB
instead of the whole 26B model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMultimodalLM

LAYER_COUNT = 24


def find_text_stack(model):
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
        args.model, dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True).eval()
    layers = find_text_stack(model)
    layer = layers[0]

    captured = {}

    def capture(name):
        def hook(module, inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[name] = tensor.detach().float().cpu().numpy()
        return hook

    router_state = {}

    def capture_router(module, inputs, output):
        # (logits, weights, indices)
        router_state["weights"] = output[1].detach().float().cpu().numpy()
        router_state["indices"] = output[2].detach().cpu().numpy()

    handles = [
        layer.post_attention_layernorm.register_forward_hook(capture("mlp_in")),
        layer.mlp.gate.register_forward_hook(capture_router),
        layer.mlp.register_forward_hook(capture("mlp_pre")),
        layer.post_mlp_convolution.register_forward_hook(capture("mlp")),
    ]
    ids = torch.tensor([[args.token_id]], dtype=torch.long, device="cuda")
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    for handle in handles:
        handle.remove()

    for name in ("mlp_in", "mlp_pre", "mlp"):
        array = captured[name].reshape(1, -1).astype(np.float32)
        (args.out / f"layer0_{name}.bin").write_bytes(array.tobytes())
        print(f"layer0_{name}: {array.shape}")

    indices = [int(i) for i in router_state["indices"][0]]
    weights = [float(w) for w in router_state["weights"][0]]
    print("selected", indices)
    print("weights", [round(w, 5) for w in weights])

    experts = layer.mlp.experts
    gate_up = experts.gate_up_proj.detach()
    down = experts.down_proj.detach()
    for slot, expert in enumerate(indices):
        w13 = gate_up[expert].to(torch.bfloat16).contiguous()
        w2 = down[expert].to(torch.bfloat16).contiguous()
        (args.out / f"expert_{slot}_w13.bin").write_bytes(w13.cpu().view(torch.uint8).numpy().tobytes())
        (args.out / f"expert_{slot}_w2.bin").write_bytes(w2.cpu().view(torch.uint8).numpy().tobytes())
    (args.out / "layer0_router.json").write_text(json.dumps(
        {"indices": indices, "weights": weights, "hidden": 2304, "inter": 1152}, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
