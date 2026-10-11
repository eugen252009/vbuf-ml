---
date: 2026-10-11
title: Qwen3.5-9B tokenizer and text-chat qualification
---

# Qwen3.5-9B tokenizer and text-chat qualification

**Status:** tokenizer-only sidecar conversion, pinned text-only chat-template rendering, and local parity tests pass. This does **not** qualify model execution, generation, server admission, or production use. No weights were downloaded and no inference was run.

## Identity and artifacts

- Hugging Face repository: `Qwen/Qwen3.5-9B`
- Pinned revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- Converter implementation: `9f4d87612efbe4d728fdbd909ec071c8a70f7419`
- Local source directory: `/home/eugen/.cache/vbuf-agent-qualification/qwen35-tokenizer-chat-source-20261011/source`
- Committed sidecar fixture: `rust/vbuf-ml/tests/fixtures/qwen35-9b-tokenizer.vbuf`
- Sidecar: 7,086,064 bytes, SHA-256 `ef248fae2f794714a46f5e815d5622555b42f6ae75d116cf002f9436ecea2522`
- Manifest: `rust/vbuf-ml/tests/fixtures/qwen35-9b-tokenizer.manifest.json`; it records the six source paths, sizes, hashes, acquisition method, converter commit, profile identities, counts, and EOS/PAD roles.

The local source directory contains the six pinned text/config artifacts and a checksum ledger; it contains no model weights. Source identity is checked before conversion:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `config.json` | 3,126 | `d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05` |
| `tokenizer.json` | 12,807,982 | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |
| `tokenizer_config.json` | 16,710 | `316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8` |
| `chat_template.jinja` | 7,756 | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` |
| `vocab.json` | 6,722,759 | `ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003` |
| `merges.txt` | 3,353,259 | `a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d` |

A second conversion from the stable local source directory was byte-compared with the committed sidecar fixture; the files matched exactly. Conversion from a copied source with a modified `tokenizer_config.json` exited with identity error 2 and emitted no sidecar.

The vBuf v0.6 envelope has model/tokenizer metadata and an empty TensorDirectory. This adds no generic wire-format rule and carries no weights.

## Vocabulary and decoding contract

The pinned table has 248,044 base BPE entries and 33 added tokens. The runtime classifies IDs explicitly:

| ID range, inclusive | Classification | Count |
| --- | --- | ---: |
| `0–248043` | Base BPE entries | 248,044 |
| `248044–248069` | Added in `tokenizer.json` | 26 |
| `248070–248076` | Added only in `tokenizer_config.json` | 7 |
| `248077–248319` | Model vocabulary rows with no token string | 243 |
| `248320+` | Outside the pinned model vocabulary | — |

All 33 added tokens and all 21 special-token flags are checked. The pinned NFC normalizer, Qwen3.5 pre-tokenizer profile, byte-level decoder, merge inventory, and token metadata are validated fail-closed. The manifest and runtime preserve distinct roles: model-config EOS is 248044 (`<|endoftext|>`), tokenizer-config EOS is 248046 (`<|im_end|>`), and tokenizer PAD is 248044. No stopping policy is inferred from the EOS difference.

Decoding uses the pinned `errors=replace` behavior, including streaming UTF-8 boundaries. The detailed Qwen3.5 decoder omits unmapped model rows and out-of-vocabulary IDs from text while reporting their IDs and classifications. That is a decoder behavior only; generation behavior for those IDs remains unresolved.

## Tokenizer parity

The qualification oracle is the serialized `tokenizer.json` loaded by `tokenizers` 0.22.2, with the seven config-only `AddedToken` records applied from the pinned tokenizer config. The Rust implementation matches **441/441** vectors for IDs, token strings, and decoded text (318 preserved vectors plus 123 expanded vectors). Coverage includes NFC/NFD and combining marks, multilingual text, emoji and zero-width characters, whitespace/newlines, NUL, markup and source snippets, all 33 added-token strings, deterministic mixed-Unicode samples, and a 53-KiB uninterrupted input.

A known upstream API distinction is retained rather than hidden: under Transformers 5.14.1, `AutoTokenizer` resolves to `Qwen2Tokenizer`, whose hard-coded regex differs from the explicit serialized tokenizer/config profile on 19/441 audit inputs. vBuf follows the pinned serialized tokenizer and config regex. Thus the qualification establishes parity with that pinned serialized profile, **not** with the legacy `Qwen2Tokenizer` class on those 19 inputs.

## Text-only chat-template parity

The original 7,756-byte template is stored byte-for-byte and checked by SHA-256 before rendering. The isolated Rust renderer uses the pinned template and JSON inputs; it does not compile caller-supplied templates or execute Python callables. The API rejects image, video, audio, and other non-text content, including an audio block that also supplies a misleading `text` field.

Against `transformers` 5.14.1 / Jinja 3.1.6, **8/8** rendered cases match output bytes exactly, then match token IDs, token strings, decoded text, and context accounting. Cases cover multi-turn prompts, assistant generation prompts, reasoning, ordered tool schemas and arguments, tool-call/tool-response cycles, consecutive tool responses, Unicode, and scalar formatting. Five invalid-input cases fail closed. Context admission counts actual rendered prompt token IDs and a caller-supplied completion reserve; this does not select a generation limit or execute a model.

## Verification

Reproduction from the repository root:

```bash
python3 scripts/qualify_qwen35_tokenizer.py \
  --source-dir /home/eugen/.cache/vbuf-agent-qualification/qwen35-tokenizer-chat-source-20261011/source \
  --sidecar rust/vbuf-ml/tests/fixtures/qwen35-9b-tokenizer.vbuf \
  --manifest rust/vbuf-ml/tests/fixtures/qwen35-9b-tokenizer.manifest.json \
  --cli rust/target/debug/vbuf-ml-qwen35-tokenizer

cargo test --offline --manifest-path rust/vbuf-ml/Cargo.toml
cargo clippy --offline --manifest-path rust/vbuf-ml/Cargo.toml --all-targets --all-features
```

Results: the qualification script passed sidecar verification, both tokenizer-vector sets, and all chat acceptance/rejection vectors. The complete `vbuf-ml` crate test suite passed, including negative checks for sidecar truncation/corruption, manifest mismatch, unsupported vBuf/tokenizer schema versions, malformed token-offset maps, text decoding, ID-range classification, and chat-template identity. Clippy exited successfully; it reports existing warnings in unrelated FFI, importer, quantization, and legacy tokenizer code, but no warnings in the new Qwen3.5 modules. Direct `rustfmt --check` on the changed Rust files, Python bytecode compilation, and `git diff --check` passed.

The repository-wide `cargo fmt --all -- --check` still reports pre-existing formatting drift in `rust/vbuf-ml/src/consumer_ffi.rs` and `rust/vbuf-runtime/src/bin/vbuf-runtime-step32e-block.rs`; neither file was changed for this work.

## Qualification boundary

Qwen3.5-9B remains experimental and isolated. There was no weight acquisition, inference, generation, numerical comparison, hybrid DeltaNet/full-attention execution, recurrent/KV lifecycle qualification, capacity trial, Pi tool-use trial, or server/production admission. Tokenizer and template parity are necessary input-side evidence, not evidence for those gates. The Qwen3-8B and Qwen3-14B paths and their existing admission decisions were not changed.
