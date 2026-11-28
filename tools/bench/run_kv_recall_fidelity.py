"""Compare greedy continuations of two Serve endpoints at one fixed long prompt.

This is the fidelity gate for a windowed KV view: the same prompt, the same greedy
decoding settings, two endpoints, byte-for-byte comparison of the generated text.
A recall budget that drops a block changes the answer silently -- the model still
returns a well-formed list, one fact short -- so the verdict has to be exact text
plus an explicit count of recovered markers, never a judgement of the prose.

Both endpoints are black boxes: this client speaks the public OpenAI chat protocol
and never calls Engine.

    python3 tools/bench/run_kv_recall_fidelity.py \
        --baseline-url http://127.0.0.1:18200/v1 \
        --candidate-url http://127.0.0.1:18201/v1 \
        --corpus eval/corpora/perplexity-1m/data \
        --chars 640000 --probe both --out profiles/bench/kv/fidelity.json
"""

import argparse
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

# Four facts planted at fixed depths, far enough apart that a window smaller than the
# gap between two of them must lose one.
NEEDLES = [
    (0.05, "ALPHA", "ALPHA=7134-MAGENTA"),
    (0.35, "BETA", "BETA=2291-ORANGE"),
    (0.65, "GAMMA", "GAMMA=5077-TURQUOISE"),
    (0.995, "DELTA", "DELTA=8642-VIOLET"),
]

MARKER = re.compile(r"\b(ALPHA|BETA|GAMMA|DELTA)\s*=\s*([0-9]{4}-[A-Z]+)\b")

PROBES = {
    "codes": (
        "List the four codes that were marked 'Remember it', one per line, "
        "in the order they appear. Use the form NAME=CODE. Do not explain."
    ),
    "first_lines": (
        "Quote verbatim the first three lines of the document above, exactly as "
        "written, with no preamble."
    ),
}


def load_corpus(directory: pathlib.Path) -> str:
    files = sorted(directory.rglob("*.txt"))
    if not files:
        raise SystemExit(f"no corpus text under {directory}")
    return files[0].read_text(encoding="utf-8", errors="replace")


def build_prompt(corpus_text: str, target_chars: int) -> str:
    repeats = target_chars // len(corpus_text) + 1
    body = (corpus_text * repeats)[:target_chars]
    out, previous = [], 0
    for depth, _name, text in NEEDLES:
        position = int(depth * target_chars)
        out.append(body[previous:position])
        out.append("\n" + text + ". Remember it.\n")
        previous = position
    out.append(body[previous:])
    return "".join(out)


def ask(base_url: str, model: str, prompt: str, max_tokens: int, timeout: float):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:800]
        raise SystemExit(f"{base_url}: HTTP {exc.code}: {detail}")
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = message.get("content") or choice.get("text") or ""
    return text, time.time() - started


def recovered(text: str) -> list[str]:
    return sorted({f"{name}={code}" for name, code in MARKER.findall(text)})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-url", required=True, help="reference endpoint, including /v1")
    parser.add_argument("--candidate-url", required=True, help="endpoint under test")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--corpus", type=pathlib.Path, required=True,
                        help="directory searched for *.txt; the first file is repeated")
    parser.add_argument("--chars", type=int, default=640000, help="prompt characters")
    parser.add_argument("--probe", choices=["codes", "first_lines", "both"], default="both")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--expect-codes", type=int, default=0,
                        help="also require this many markers in a codes probe")
    parser.add_argument("--out", type=pathlib.Path, help="write the verdict as JSON here")
    args = parser.parse_args(argv)

    prompt = build_prompt(load_corpus(args.corpus), args.chars)
    probes = ["codes", "first_lines"] if args.probe == "both" else [args.probe]

    report = {"chars": len(prompt), "probes": {}, "passed": True}
    for probe in probes:
        question = prompt + "\n\n" + PROBES[probe] + "\n"
        baseline, baseline_seconds = ask(args.baseline_url, args.model, question,
                                         args.max_tokens, args.timeout)
        candidate, candidate_seconds = ask(args.candidate_url, args.model, question,
                                           args.max_tokens, args.timeout)
        entry = {
            "baseline_seconds": round(baseline_seconds, 2),
            "candidate_seconds": round(candidate_seconds, 2),
            "baseline": baseline,
            "candidate": candidate,
            "identical": baseline == candidate,
        }
        if probe == "codes":
            entry["baseline_codes"] = recovered(baseline)
            entry["candidate_codes"] = recovered(candidate)
            if args.expect_codes:
                entry["expected_codes"] = args.expect_codes
                if len(entry["candidate_codes"]) < args.expect_codes:
                    entry["identical"] = False
        report["probes"][probe] = entry
        report["passed"] = report["passed"] and entry["identical"]
        verdict = "identical" if entry["identical"] else "DIFFERENT"
        print(f"{probe}: {verdict} "
              f"(baseline {entry['baseline_seconds']}s, candidate {entry['candidate_seconds']}s)")
        if not entry["identical"]:
            print(f"  baseline : {baseline[:400]!r}")
            print(f"  candidate: {candidate[:400]!r}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    print("PASS" if report["passed"] else "FAIL")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())