"""Dump the 48 layer-visit hidden states of Inkling 2 Mini through the reference implementation.

The reference is `transformers`' own `Inkling2LoopForConditionalGeneration`, not NInfer, and the
weights come from `materialize_bf16.py` -- so the numbers this produces are the checkpoint's
mathematics, which is what the NInfer kernels are measured against.

    python tests/inkling2/dump_visits.py --model /path/to/inkling2-bf16 --out oracle.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMultimodalLM, AutoTokenizer

LAYER_COUNT = 24
VISITS = 48


def find_text_stack(model) -> torch.nn.ModuleList:
    """The text decoder is the only 24-layer ModuleList in the model."""
    for name, module in model.named_modules():
        stack = getattr(module, "layers", None)
        if isinstance(stack, torch.nn.ModuleList) and len(stack) == LAYER_COUNT:
            print(f"text stack: {name} ({type(module).__name__})")
            return stack
    raise SystemExit("no 24-layer text stack found")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True, help="BF16 reference checkpoint")
    parser.add_argument("--out", type=Path, required=True, help=".npz with the 48 visits")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model = AutoModelForMultimodalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device, low_cpu_mem_usage=True
    ).eval()
    print(f"vram {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

    visits: list[tuple[int, np.ndarray]] = []

    def hook_for(index: int):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            visits.append((index, hidden.detach().float().cpu().numpy()))
        return hook

    handles = [layer.register_forward_hook(hook_for(i))
               for i, layer in enumerate(find_text_stack(model))]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(args.device)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
    for handle in handles:
        handle.remove()

    if len(visits) != VISITS:
        raise SystemExit(f"expected {VISITS} layer visits, captured {len(visits)}")

    logits = out.logits[:, -1, :].float().cpu().numpy()
    top = np.argsort(-logits[0])[:5]
    print("prompt:", args.prompt)
    print("top5:", [(tokenizer.decode([int(t)]), round(float(logits[0][t]), 2)) for t in top])
    for index, (layer, hidden) in enumerate(visits):
        print(f"  visit {index:2d} layer {layer:2d} std {hidden.std():.5f}")

    np.savez(args.out,
             hidden=np.stack([v[1][0] for v in visits]),
             layers=np.array([v[0] for v in visits]),
             logits=logits, ids=ids.cpu().numpy())
    print("saved", args.out)


if __name__ == "__main__":
    main()
