#!/usr/bin/env python3
"""Reproduce pinned Qwen3.5 tokenizer/template qualification fixtures.

This uses only the pinned text/config artifacts. It does not download weights.
The tokenizer.json fast profile is the normative tokenizer reference here;
Transformers 5.14.1's Qwen2Tokenizer class is audited separately because its
hard-coded regex diverges from the serialized profile and tokenizer_config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

PINNED_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
CHAT_TEMPLATE_SHA256 = "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
ARTIFACTS = {
    "config.json": (3126, "d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05"),
    "tokenizer.json": (12807982, "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"),
    "tokenizer_config.json": (16710, "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8"),
    "chat_template.jinja": (7756, CHAT_TEMPLATE_SHA256),
    "vocab.json": (6722759, "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"),
    "merges.txt": (3353259, "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"),
}
REFERENCE_VERSIONS = {"transformers": "5.14.1", "jinja2": "3.1.6", "tokenizers": "0.22.2"}


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"qwen35 tokenizer qualification: {message}")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_identity(source_dir: Path) -> dict:
    records = {}
    for name, (size, digest) in ARTIFACTS.items():
        path = source_dir / name
        data = path.read_bytes()
        if len(data) != size or sha256(data) != digest:
            fail(f"pinned artifact identity mismatch: {path}")
        records[name] = data
    return records


def add_config_only_tokens(tokenizer, tokenizer_config: dict) -> int:
    from tokenizers import AddedToken

    initial = tokenizer.get_vocab_size(with_added_tokens=True)
    additions = []
    for key, item in sorted(tokenizer_config["added_tokens_decoder"].items(), key=lambda pair: int(pair[0])):
        if int(key) < initial:
            continue
        additions.append(
            AddedToken(
                item["content"],
                single_word=item["single_word"],
                lstrip=item["lstrip"],
                rstrip=item["rstrip"],
                normalized=item["normalized"],
                special=item["special"],
            )
        )
    added = tokenizer.add_tokens(additions)
    if added != 7 or tokenizer.get_vocab_size(with_added_tokens=True) != 248077:
        fail("config-only token insertion did not produce the pinned 248077-token table")
    return added


def tokenizer_text_cases(source_dir: Path, old_vectors_path: Path) -> tuple[dict, int, list[tuple[str, str]]]:
    from tokenizers import Tokenizer

    tokenizer_config = json.loads((source_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(source_dir / "tokenizer.json"))
    add_config_only_tokens(tokenizer, tokenizer_config)
    markers = [
        item["content"]
        for _, item in sorted(
            tokenizer_config["added_tokens_decoder"].items(), key=lambda pair: int(pair[0])
        )
    ]
    fixed = [
        ("empty", ""),
        ("ascii", "Hello, world!"),
        ("utf8", "café déjà vu — Ελληνικά Русский العربية 中文 日本語 한국어"),
        ("nfc-composed", "café Å é"),
        ("nfd-decomposed", "cafe\u0301 A\u030a e\u0301"),
        ("combining-stack", "a\u0301\u0327\u0308"),
        ("emoji", "🙂 🧑🏽‍💻 👨‍👩‍👧‍👦 🏳️‍🌈"),
        ("zero-width", "a\u200bb\u200cc\u200dd\ufeff"),
        ("whitespace", "  a   b\t\tend  "),
        ("newlines", "line1\r\nline2\nline3\rline4"),
        ("long-whitespace", " " * 40 + "x" + "\t" * 17 + "y" + "\n" * 13),
        ("markdown", "# Heading\n\n- item **bold** `code`\n> quote\n"),
        ("html", '<root attr="a&b"><child>text</child></root>'),
        ("json", '{"z": [1, true, null], "nested": {"π": "雪"}}'),
        ("yaml", "name: qwen\nitems:\n  - one\n  - two\n"),
        ("shell", '#!/bin/sh\nset -euo pipefail\nprintf "hello\\n"\n'),
        ("rust", 'fn main() { println!("π = {}", 3.14); }'),
        ("go", 'package main\nimport "fmt"\nfunc main() { fmt.Println("hello") }'),
        ("c", '#include <stdio.h>\nint main(void) { puts("hello"); return 0; }'),
        ("cpp", '#include <iostream>\nint main(){ std::cout << "hi"; }'),
        ("python", 'def f(x: dict[str, int]) -> bool:\n    return x.get("ok", 0) > 0\n'),
        ("escaped", '\\\\path\\t\n"double" \'single\' `tick`'),
        ("tool-markers", "<tool_call><function=bash><parameter=command>echo hi</parameter></function></tool_call>"),
        ("thinking", "<think>reason π</think> final"),
        ("all-added-tokens", "|".join(markers)),
        ("nul", "before\x00after"),
        ("long-uninterrupted", "Qwen35TokenizationBoundary_" * 2048),
    ]
    rng = random.Random(0x359B262)
    chars = list("abcXYZ019_-.,;:!?/\\\n\r\t ") + [
        "é", "e\u0301", "Å", "A\u030a", "中", "日", "ع", "λ", "π", "🙂",
        "👩\u200d💻", "\u200b", "\u200c", "\u200d", "\ufeff", "\u0301", "\u0327",
    ]
    cases = list(fixed)
    for index in range(96):
        text = "".join(rng.choice(chars) for _ in range(rng.randint(1, 110)))
        cases.append((f"random-{index:03}", text))

    vectors = []
    for name, text in cases:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        tokens = [tokenizer.id_to_token(token_id) for token_id in ids]
        if any(token is None for token in tokens):
            fail(f"reference tokenizer emitted an unmapped token in {name}")
        vectors.append(
            {
                "name": name,
                "text": text,
                "ids": ids,
                "tokens": tokens,
                "decoded": tokenizer.decode(ids, skip_special_tokens=False),
            }
        )

    old = json.loads(old_vectors_path.read_text(encoding="utf-8"))
    all_count = len(vectors) + len(old["vectors"])
    fixture = {
        "fixture_format": "vbuf-ml.qwen35-tokenizer-vectors.v1",
        "source_revision": PINNED_REVISION,
        "tokenizer_json_sha256": ARTIFACTS["tokenizer.json"][1],
        "tokenizer_config_sha256": ARTIFACTS["tokenizer_config.json"][1],
        "reference": REFERENCE_VERSIONS,
        "reference_profile": "tokenizers.Tokenizer.from_file(tokenizer.json) plus tokenizer_config-only AddedToken entries",
        "vectors": vectors,
    }
    return fixture, all_count, cases


def chat_cases(source_dir: Path, source_files: dict) -> dict:
    from transformers import AutoTokenizer
    from tokenizers import Tokenizer

    hf = AutoTokenizer.from_pretrained(str(source_dir), local_files_only=True, trust_remote_code=False)
    if hf.chat_template.encode("utf-8") != source_files["chat_template.jinja"]:
        fail("Transformers-loaded chat template differs from pinned chat_template.jinja")
    fast = Tokenizer.from_file(str(source_dir / "tokenizer.json"))
    tokenizer_config = json.loads(source_files["tokenizer_config.json"])
    add_config_only_tokens(fast, tokenizer_config)

    cases = [
        {
            "name": "multiturn-system-generation",
            "messages": [
                {"role": "system", "content": "You are concise and exact."},
                {"role": "user", "content": "Hi."},
                {"role": "assistant", "content": "Hello."},
                {"role": "user", "content": "What is 2 + 2?"},
            ],
            "tools": None,
            "options": {"add_generation_prompt": True, "enable_thinking": False},
            "completion_reserve": 64,
        },
        {
            "name": "text-content-blocks-and-nfc",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Café"}, {"type": "text", "text": "— please normalize and answer."}]},
                {"role": "assistant", "reasoning_content": "Check spelling and Unicode.", "content": "<think>ignored content</think>42"},
            ],
            "tools": None,
            "options": {"add_generation_prompt": False},
            "completion_reserve": 32,
        },
        {
            "name": "tool-schema-and-roundtrip",
            "messages": [
                {"role": "system", "content": "Use the lookup tool when needed."},
                {"role": "user", "content": "Find the price for π widgets."},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": {"query": "π widgets", "limit": 3, "enabled": True, "score": 1.5, "metadata": {"locale": "fr-CA", "tags": ["é", "雪"]}, "note": None}}}]},
                {"role": "tool", "tool_call_id": "call-1", "name": "lookup", "content": "{\"price\": 7}"},
                {"role": "assistant", "content": "The price is 7."},
            ],
            "tools": [{"type": "function", "function": {"name": "lookup", "description": "Find a product by query.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search phrase"}, "limit": {"type": "integer", "default": 3}}, "required": ["query"], "additionalProperties": False}}}],
            "options": {"add_generation_prompt": True, "enable_thinking": True},
            "completion_reserve": 128,
        },
        {
            "name": "consecutive-tool-results",
            "messages": [
                {"role": "user", "content": "Read two values."},
                {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "first", "arguments": {"key": "a"}}}, {"function": {"name": "second", "arguments": {"key": "b"}}}]},
                {"role": "tool", "name": "first", "content": "one"},
                {"role": "tool", "name": "second", "content": "two"},
                {"role": "assistant", "content": "Combined: one and two."},
                {"role": "user", "content": "Summarize the values."},
            ],
            "tools": None,
            "options": {"add_generation_prompt": True},
            "completion_reserve": 64,
        },
        {
            "name": "assistant-reasoning-extraction",
            "messages": [
                {"role": "user", "content": "Explain in one sentence."},
                {"role": "assistant", "content": "<think>private reasoning\nline</think>\nPublic answer."},
                {"role": "user", "content": "Now answer briefly."},
            ],
            "tools": None,
            "options": {"add_generation_prompt": True, "enable_thinking": False},
            "completion_reserve": 16,
        },
        {
            "name": "special-markers-text-only",
            "messages": [{"role": "user", "content": "Literal markers: <|vision_start|> <|image_pad|> <|video_pad|> <|audio|> <|text_audio|> <|im_end|>"}],
            "tools": None,
            "options": {"add_generation_prompt": True},
            "completion_reserve": 32,
        },
        {
            "name": "ordered-tool-schema",
            "messages": [{"role": "user", "content": "Run the command."}],
            "tools": [{"type": "function", "function": {"name": "run", "description": "Order-sensitive JSON fields: 雪, π, true.", "parameters": {"type": "object", "properties": {"z-last": {"type": "string"}, "a-first": {"type": "boolean"}, "middle": {"type": "number"}}, "required": ["z-last", "a-first"]}}}],
            "options": {"add_generation_prompt": False},
            "completion_reserve": 0,
        },
        {
            "name": "tool-python-scalar-stringification",
            "messages": [
                {"role": "user", "content": "Preserve JSON scalar values."},
                {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "echo", "arguments": {"integer": 7, "one": 1.0, "negative_zero": -0.0, "small": 1e-7, "threshold": 0.0001, "large": 1e16, "null_value": None, "truth": True}}}]},
            ],
            "tools": None,
            "options": {"add_generation_prompt": True},
            "completion_reserve": 8,
        },
    ]
    for case in cases:
        options = case["options"]
        kwargs = {"tokenize": False, "add_generation_prompt": options.get("add_generation_prompt", False)}
        if "enable_thinking" in options:
            kwargs["enable_thinking"] = options["enable_thinking"]
        if "add_vision_id" in options:
            kwargs["add_vision_id"] = options["add_vision_id"]
        rendered = hf.apply_chat_template(case["messages"], tools=case.get("tools"), **kwargs)
        ids = fast.encode(rendered, add_special_tokens=False).ids
        if case["name"] == "ordered-tool-schema":
            context_limit = len(ids) + case["completion_reserve"]
        else:
            context_limit = 262144
        case.update(
            expected_rendered=rendered,
            expected_ids=ids,
            expected_tokens=[fast.id_to_token(token_id) for token_id in ids],
            expected_decoded=fast.decode(ids, skip_special_tokens=False),
            context_limit=context_limit,
            expected_remaining=context_limit - len(ids) - case["completion_reserve"],
        )

    invalid_cases = [
        {"name": "reject-image-block", "messages": [{"role": "user", "content": [{"type": "image", "image_url": "not-loaded"}]}], "expected_error": "non-text image/video content"},
        {"name": "reject-nontext-part", "messages": [{"role": "user", "content": [{"type": "audio", "data": "not-loaded"}]}], "expected_error": "non-text audio or multimodal content"},
        {"name": "reject-audio-with-text-field", "messages": [{"role": "user", "content": [{"type": "audio", "audio": "not-loaded", "text": "misleading"}]}], "expected_error": "non-text audio or multimodal content"},
        {"name": "reject-no-user-query", "messages": [{"role": "tool", "content": "<tool_response>only</tool_response>"}], "expected_error": "No user query found in messages."},
        {"name": "reject-empty-messages", "messages": [], "expected_error": "at least one message"},
    ]
    return {
        "fixture_format": "vbuf-ml.qwen35-chat-vectors.v1",
        "source_revision": PINNED_REVISION,
        "reference": REFERENCE_VERSIONS,
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "tokenizer_reference_profile": "tokenizers tokenizer.json plus the seven tokenizer_config-only AddedToken entries",
        "cases": cases,
        "invalid_cases": invalid_cases,
    }


def encode_json(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=False) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cli", type=Path, required=True, help="built vbuf-ml-qwen35-tokenizer executable")
    parser.add_argument("--fixture-dir", type=Path, default=Path("rust/vbuf-ml/tests/fixtures"))
    parser.add_argument("--write-fixtures", action="store_true", help="write regenerated expected vectors")
    parser.add_argument("--old-vectors", type=Path, default=Path("rust/vbuf-ml/tests/fixtures/qwen35-tokenizer-vectors.json"))
    args = parser.parse_args()

    import jinja2
    import tokenizers
    import transformers

    actual_versions = {
        "transformers": transformers.__version__,
        "jinja2": jinja2.__version__,
        "tokenizers": tokenizers.__version__,
    }
    if actual_versions != REFERENCE_VERSIONS:
        fail(f"reference versions differ: expected {REFERENCE_VERSIONS}, got {actual_versions}")

    source_dir = args.source_dir.resolve()
    source_files = source_identity(source_dir)
    text_fixture, full_vector_count, _ = tokenizer_text_cases(source_dir, args.old_vectors)
    chat_fixture = chat_cases(source_dir, source_files)
    output_paths = {
        "qwen35-9b-tokenizer-vectors.json": encode_json(text_fixture),
        "qwen35-9b-chat-vectors.json": encode_json(chat_fixture),
    }
    for name, expected in output_paths.items():
        path = args.fixture_dir / name
        if args.write_fixtures:
            path.write_bytes(expected)
        elif not path.is_file() or path.read_bytes() != expected:
            fail(f"regenerated fixture differs: {path}; use --write-fixtures after review")

    def run(*command: str) -> None:
        subprocess.run([str(args.cli), *command], check=True)

    run("verify", "--sidecar", str(args.sidecar), "--manifest", str(args.manifest))
    run("check-vectors", "--sidecar", str(args.sidecar), "--manifest", str(args.manifest), "--vectors", str(args.old_vectors))
    run("check-vectors", "--sidecar", str(args.sidecar), "--manifest", str(args.manifest), "--vectors", str(args.fixture_dir / "qwen35-9b-tokenizer-vectors.json"))
    run("check-chat-vectors", "--sidecar", str(args.sidecar), "--manifest", str(args.manifest), "--fixture", str(args.fixture_dir / "qwen35-9b-chat-vectors.json"))

    # AutoTokenizer resolves Qwen2Tokenizer, whose Transformers 5.14.1 class
    # hard-codes a regex that ignores the pinned `pretokenize_regex` field.
    # Count this compatibility difference; do not silently use it as the
    # tokenization oracle for tokenizer.json.
    from transformers import AutoTokenizer
    from tokenizers import Tokenizer

    hf = AutoTokenizer.from_pretrained(str(source_dir), local_files_only=True, trust_remote_code=False)
    fast = Tokenizer.from_file(str(source_dir / "tokenizer.json"))
    config = json.loads(source_files["tokenizer_config.json"])
    add_config_only_tokens(fast, config)
    old = json.loads(args.old_vectors.read_text(encoding="utf-8"))
    combined = text_fixture["vectors"] + old["vectors"]
    divergences = []
    for vector in combined:
        hf_ids = hf.encode(vector["text"], add_special_tokens=False)
        fast_ids = fast.encode(vector["text"], add_special_tokens=False).ids
        if hf_ids != fast_ids:
            divergences.append(vector.get("name", "legacy-vector"))
    print(
        json.dumps(
            {
                "status": "PASS",
                "source_revision": PINNED_REVISION,
                "reference_versions": actual_versions,
                "new_tokenizer_vectors": len(text_fixture["vectors"]),
                "preserved_previous_vectors": len(old["vectors"]),
                "total_tokenizer_vectors": full_vector_count,
                "chat_cases": len(chat_fixture["cases"]),
                "chat_rejection_cases": len(chat_fixture["invalid_cases"]),
                "transformers_class": f"{type(hf).__module__}.{type(hf).__name__}",
                "transformers_vs_tokenizer_json_id_divergences": len(divergences),
                "divergence_examples": divergences[:8],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
