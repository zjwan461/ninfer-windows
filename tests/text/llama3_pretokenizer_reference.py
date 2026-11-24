"""Reference chunking for a Llama-3 style pre-tokeniser.

Inkling 2's tokenizer.json splits with a seven-alternative regex before BPE. The engine has to
reproduce that split exactly, and the only trustworthy oracle is the pattern itself, evaluated by a
regex engine that supports Unicode properties. This script produces test vectors from it: a list of
input strings in, the chunks out.

    python tests/text/llama3_pretokenizer_reference.py --out tests/text/fixtures/llama3_split.json

The C++ side reads that file and asserts its own scanner produces the same chunks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

# The cases are chosen for the alternatives that are easy to get wrong rather than for coverage of
# the alphabet: a leading space attaches to the following word, digits cap at three, contractions
# stay with their word, letter case decides which of the two word alternatives applies, and
# trailing whitespace is captured on its own.
CASES = [
    "hello world",
    "Hello World",
    "  leading spaces",
    "trailing   ",
    "don't stop",
    "I'll've",
    "MACD2000",
    "2024",
    "1234567",
    "a1b2",
    "中文分词器测试",
    "Приве́т мир",
    "مرحبا بالعالم",
    "日本語のテキスト",
    "code: x = 1;",
    "line\nbreak\r\nwindows",
    "\n\n\n",
    "   \t  ",
    "MiXeD CaSe wOrDs",
    "URL https://example.com/path/",
    "ünïcödé",
    "e\u0301 combining",
    "🇨🇳 emoji",
    "  ",
    "",
]


def chunks(text: str) -> list[str]:
    import regex

    compiled = regex.compile(PATTERN)
    return [match.group(0) for match in compiled.finditer(text)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("tests/text/fixtures/llama3_split.json"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    vectors = [{"text": text, "chunks": chunks(text)} for text in CASES]
    args.out.write_text(json.dumps(vectors, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    total = sum(len(v["chunks"]) for v in vectors)
    print(f"wrote {len(vectors)} vectors, {total} chunks, to {args.out}")


if __name__ == "__main__":
    main()
