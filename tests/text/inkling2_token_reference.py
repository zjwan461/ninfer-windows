"""Token ids for a set of strings, produced by the library that owns the file.

    python tests/text/inkling2_token_reference.py --out tests/text/fixtures/inkling2_tokens.json

The C++ encoder is checked against these, so the tokenizer.json is the authority rather than any
reading of it. The cases cover what actually differs between tokenizers: leading spaces,
contractions, digit runs, non-Latin scripts, combining marks, emoji and the special tokens.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CASES = [
    "Hello, world!",
    "hello world",
    "  leading spaces",
    "don't stop",
    "I'll've",
    "2024 and 1234567",
    "中文分词器测试",
    "日本語のテキスト",
    "Приве́т мир",
    "مرحبا بالعالم",
    "e\u0301 combining",
    "🇨🇳 emoji",
    "def f(x):\n    return x + 1\n",
    "  <|endoftext|>  ",
    "<|message_user|>hi<|message_model|>",
    "",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path,
                        default=Path("/workspaces/models/inkling2/tokenizer.json"))
    parser.add_argument("--out", type=Path, default=Path("tests/text/fixtures/inkling2_tokens.json"))
    args = parser.parse_args()

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    vectors = [{"text": text, "ids": tokenizer.encode(text).ids} for text in CASES]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(vectors, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    total = sum(len(v["ids"]) for v in vectors)
    print(f"wrote {len(vectors)} vectors, {total} ids, to {args.out}")


if __name__ == "__main__":
    main()
