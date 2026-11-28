#!/usr/bin/env python3
"""Gate the compacted-view swap on a byte-identical answer at a named retention ratio.

``run_kv_recall_fidelity.py`` compares two endpoints that are already running. This
script owns the whole cycle instead, because the two configurations cannot run at the
same time on one card: a 262,144-token context reserves most of the device, so the
baseline and the candidate are served in sequence, one after the other.

The claim being gated is the one `docs/maintainer/long-context-workspace.md` section
11.2 makes: a compacted view answers exactly like the full view as long as the
operator keeps enough of the history, and it degrades visibly when the retention ratio
is too low. So the gate reports the ratio it ran at and refuses to pass when the
answers differ, which makes the second half of that claim a check rather than a note.

Usage::

    python tools/bench/run_kv_recall_gate.py \
        --exe build-4080s/apps/ninfer-serve.exe \
        --artifact E:/models/qwen3_8_27b_a8.ninfer/qwen3_8_27b_a8.ninfer \
        --corpus eval/corpora/perplexity-1m/data/ninfer \
        --chars 700000 --max-context 262144 --budget 131072

Exit code 0 means the compacted view reproduced the full view byte for byte; 1 means it
did not; 2 means a server could not be started or a request failed.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

MARKER = re.compile(r"\b([A-Z]{4,6})\s*=\s*([0-9]{4}-[A-Z]+)\b")

FIXTURE = [("ALPHA", "7134-MAGENTA"), ("BETA", "2291-ORANGE"),
           ("GAMMA", "5077-TURQUOISE"), ("DELTA", "8642-VIOLET"),
           ("EPSILON", "2208-CRIMSON"), ("ZETA", "6165-OBSIDIAN"),
           ("ETA", "4429-AMBER"), ("THETA", "9073-SIENNA")]

DEFAULT_DEPTHS = "0.05,0.35,0.65,0.95"

QUESTION = (
    "\n\nList every fact of the form NAME=CODE that appears in the document above, "
    "one per line, in the order they appear. If a fact is absent, omit it.\n"
)

APPLIED = re.compile(r"kv recall: applied view (\d+) tokens, rope delta (\d+)")


def load_corpus(directory: pathlib.Path) -> str:
    files = sorted(directory.glob("*.txt"))
    if not files:
        raise SystemExit(f"no *.txt under {directory}")
    return files[0].read_text(encoding="utf-8", errors="replace")


def build_prompt(corpus_text: str, target_chars: int, depths) -> str:
    """Repeat the corpus until it covers the request, then plant the four markers."""
    body = corpus_text.strip()
    repeats = max(1, target_chars // max(1, len(body)))
    text = (body + "\n\n") * repeats
    text = text[:target_chars]
    if len(depths) > len(FIXTURE):
        raise SystemExit(f"at most {len(FIXTURE)} depths")
    for index, depth in enumerate(depths):
        name, code = FIXTURE[index]
        position = max(0, min(len(text), int(len(text) * depth)))
        text = text[:position] + f"\n\n{name}={code} is the access code.\n\n" + text[position:]
    return text


def free_port(start: int) -> int:
    for port in range(start, start + 40):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise SystemExit("no free port")


def start_server(exe: pathlib.Path, artifact: pathlib.Path, port: int, max_context: int,
                 kv_dtype: str, budget: int | None, log: pathlib.Path) -> subprocess.Popen:
    environment = dict(os.environ)
    environment.pop("NINFER_KV_RECALL", None)
    environment.pop("NINFER_KV_RECALL_APPLY", None)
    environment.pop("NINFER_KV_RECALL_BUDGET", None)
    if budget is not None:
        environment["NINFER_KV_RECALL"] = "1"
        environment["NINFER_KV_RECALL_APPLY"] = "1"
        environment["NINFER_KV_RECALL_BUDGET"] = str(budget)
    command = [str(exe), str(artifact), "--max-context", str(max_context),
               "--kv-dtype", kv_dtype, "--no-thinking", "--port", str(port)]
    handle = log.open("w", encoding="utf-8", errors="replace")
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=handle,
                               env=environment, cwd=str(exe.parent))
    process._log_handle = handle  # type: ignore[attr-defined]
    deadline = time.time() + 600.0
    while time.time() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"server exited with {process.returncode}; see {log}")
        if "listening on" in log.read_text(encoding="utf-8", errors="replace"):
            return process
        time.sleep(1.0)
    process.kill()
    raise SystemExit(f"server did not listen within 600 s; see {log}")


def stop_server(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)
    handle = getattr(process, "_log_handle", None)
    if handle is not None:
        handle.close()


def ask(base_url: str, model: str, prompt: str, max_tokens: int, timeout: float) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
    }).encode("utf-8")
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=payload,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise SystemExit(f"HTTP {error.code}: {error.read().decode('utf-8', 'replace')}")
    return body["choices"][0]["message"]["content"] or ""


def served(exe: pathlib.Path, artifact: pathlib.Path, prompt: str, model: str, max_tokens: int,
           timeout: float, max_context: int, kv_dtype: str, budget: int | None,
           log: pathlib.Path, label: str) -> dict:
    port = free_port(18300)
    process = start_server(exe, artifact, port, max_context, kv_dtype, budget, log)
    seconds = 0.0
    try:
        started = time.time()
        answer = ask(f"http://127.0.0.1:{port}/v1", model, prompt, max_tokens, timeout)
        seconds = time.time() - started
    finally:
        stop_server(process)
    text = log.read_text(encoding="utf-8", errors="replace")
    applied = APPLIED.search(text)
    log_text = log.read_text(encoding="utf-8", errors="replace")
    failure = None
    for line in log_text.splitlines():
        if "was refused" in line:
            failure = line.split("refused:", 1)[-1].strip() or line.strip()
    entry = {
        "label": label,
        "budget": budget,
        "port": port,
        "seconds": round(seconds, 2),
        "answer": answer,
        "log": log.read_text(encoding="utf-8", errors="replace"),
        "refusal": (REFUSED.search(log.read_text(encoding="utf-8", errors="replace")) or
                    [None])[0] if False else failure,
        "codes": sorted({f"{n}={c}" for n, c in MARKER.findall(answer)}),
        "view_tokens": int(applied.group(1)) if applied else None,
        "rope_delta": int(applied.group(2)) if applied else None,
        "log": str(log),
    }
    print(f"{label}: {seconds:.1f}s view={entry['view_tokens']} codes={len(entry['codes'])}")
    return entry


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", type=pathlib.Path, required=True)
    parser.add_argument("--artifact", type=pathlib.Path, required=True)
    parser.add_argument("--corpus", type=pathlib.Path, required=True)
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--chars", type=int, default=700000)
    parser.add_argument("--depths", default=DEFAULT_DEPTHS,
                        help="comma separated marker depths in (0,1)")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--max-context", type=int, default=262144)
    parser.add_argument("--kv-dtype", default="int8")
    parser.add_argument("--budget", type=int, required=True,
                        help="compacted-view budget in tokens; the view keeps this much")
    parser.add_argument("--baseline-budget", type=int, default=None,
                        help="budget for the reference run; omit for no view at all")
    parser.add_argument("--expect-codes", type=int, default=0,
                        help="markers the candidate must recover; 0 means every marker the baseline recovered")
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args(argv)
    depths = [float(value) for value in args.depths.split(",") if value.strip()]
    if not depths:
        raise SystemExit("no depths given")

    prompt = build_prompt(load_corpus(args.corpus), args.chars, depths) + QUESTION
    work = pathlib.Path(os.environ.get("TEMP", ".")) / "kv_recall_gate"
    work.mkdir(parents=True, exist_ok=True)

    baseline = served(args.exe, args.artifact, prompt, args.model, args.max_tokens,
                      args.timeout, args.max_context, args.kv_dtype, args.baseline_budget,
                      work / "baseline.log", "baseline")
    candidate = served(args.exe, args.artifact, prompt, args.model, args.max_tokens,
                       args.timeout, args.max_context, args.kv_dtype, args.budget,
                       work / "candidate.log", "candidate")

    history = candidate["view_tokens"] + (candidate["rope_delta"] or 0) \
        if candidate["view_tokens"] else None
    retention = (candidate["view_tokens"] / history) if history else None
    refused = "was refused" in candidate["log"]
    same = baseline["answer"] == candidate["answer"]
    if args.expect_codes:
        expected = len(depths)
    else:
        expected = len(baseline["codes"])
    # Marker-level verdict: every planted fact the baseline reported must come back
    # with the same value. Wording may differ; an invented value is a failure.
    planted = {FIXTURE[index][0]: FIXTURE[index][1] for index in range(len(depths))}
    baseline_codes = dict(item.split("=", 1) for item in baseline["codes"])
    candidate_codes = dict(item.split("=", 1) for item in candidate["codes"])
    checked = {name: planted[name] for name in baseline_codes if name in planted}
    wrong = {name: candidate_codes.get(name) for name, value in checked.items()
             if candidate_codes.get(name) != value}
    recovered = len(checked) - len(wrong)
    enough = recovered == len(checked) and bool(checked)
    report = {
        "chars": len(prompt),
        "budget": args.budget,
        "history_tokens": history,
        "retention": round(retention, 4) if retention else None,
        "identical": same,
        "refused": refused,
        "markers_checked": checked,
        "markers_wrong": wrong,
        "markers_expected": expected,
        "baseline": baseline,
        "candidate": candidate,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")

    if not same:
        print(f"  baseline : {baseline['answer'][:400]!r}")
        print(f"  candidate: {candidate['answer'][:400]!r}")
    for name, value in wrong.items():
        print(f"  marker {name}: planted {checked[name]}, answered {value}")
    print(f"retention {report['retention']}: "
          f"{'identical wording' if same else 'different wording'}, "
          f"markers {recovered}/{len(checked)}"
          f" (baseline reported {len(baseline_codes)} of {expected} planted)")
    if refused:
        print(f"  swap refused: {candidate['refusal']}")
    passed = enough and not refused
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())