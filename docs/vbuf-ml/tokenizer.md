# vBuf-ML tokenizer metadata

Tokenizer metadata is an optional, independently loadable semantic index. It is
separate from model metadata and TensorDirectory. vBuf-ML stores portable data
and algorithm identities. The Rust runtime has tokenizer views for qualified
profiles; storage metadata itself does not execute tokenization or render chat
templates.

## Qualified kinds

`VocabularyOnly` (kind `1`) remains valid and unchanged. It provides a direct
vocabulary view with implicit ordinal token IDs.

`Gpt2BpeQwen2` (kind `2`) is the first qualified richer profile. It means:

```text
TokenizerModelIdentity = GPT2_BPE (profile ID 1)
PreTokenizerIdentity    = QWEN2    (profile ID 1)
```

These are profile-local identities mapped to the pinned consumer. They are not
llama.cpp enum values and do not embed executable tokenizer logic. Kind `2`
may also be paired with the experimental Qwen3.5 pre-tokenizer and NFC
normalizer described below; that combination is distinct from the Qwen2 profile.

## Composition

Both kinds reference canonical values:

```text
TokenTextBytes → opaque UTF-8 byte pool
TokenOffsets   → u64 array with vocabulary_count + 1 values
TokenScores    → optional f32/f64 array
TokenTypes     → optional unsigned integer array
TokenFlags     → optional u8 flags per vocabulary ordinal
SpecialTokenIds → optional u32 list
special IDs    → optional unsigned scalar values
```

The richer kind additionally references:

```text
MergeLeftIds   → u32 vocabulary-ordinal array
MergeRightIds  → u32 vocabulary-ordinal array
TokenizerModelIdentity → unsigned scalar, value 1
PreTokenizerIdentity   → unsigned scalar, value 1
AddBos         → unsigned scalar, 0 or 1
ChatTemplate   → optional opaque UTF-8 bytes
NormalizerIdentity → optional unsigned scalar, value 1 = Unicode NFC
```

Role IDs are `9 = MergeLeftIds`, `10 = MergeRightIds`, `11 =
TokenizerModelIdentity`, `12 = PreTokenizerIdentity`, `13 = AddBos`, `14 =
ChatTemplate`, and `18 = NormalizerIdentity`. The normalizer role is an optional
vBuf-ML semantic extension; the canonical v0.6 wire format is unchanged. Older
readers may skip the optional role.

The merge array ordinal is the BPE rank. No rank sorting is performed. The
runtime may construct a string-keyed lookup from these ordinal IDs during
initialization; that lookup is derived runtime state, not wire state.

## Pinned llama.cpp mapping

At commit
`4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c`, `llama_vocab::impl::load`:

- requires `tokenizer.ggml.model`;
- maps `gpt2` to `LLAMA_VOCAB_TYPE_BPE`;
- loads `tokenizer.ggml.merges` in source order;
- splits each merge at the first ASCII space found at or after byte index 1;
- maps `qwen2` to the Qwen2 pre-tokenizer implementation and sets
  `clean_spaces = false`;
- loads vocabulary ordinals directly from `tokenizer.ggml.tokens`;
- treats scores and token types as optional parallel arrays;
- loads special IDs and reads `tokenizer.ggml.add_bos_token`, defaulting BOS
  insertion to true when the key is absent.

The exact source locations are recorded in
`benchmark-results/vbuf-ml-step19b/upstream-provenance.json`.

## Merge decision

The real Qwen3 tables contain 151,387 merges in both artifacts. Every source
merge is valid under the pinned split rule, both components resolve against the
151,936-entry vocabulary, and there are no duplicate pairs.

Two representations were measured:

1. original UTF-8 merge strings plus a u64 offset table;
2. two u32 arrays of vocabulary ordinals.

The numeric representation is selected. It is approximately 1.21 MiB versus
approximately 2.34 MiB for strings plus offsets, reuses the existing token
ordinal namespace, and preserves source order exactly. A future adapter must
resolve IDs through the exact vocabulary when constructing the pinned
string-keyed `bpe_ranks` map. Duplicate pairs, if encountered in another
artifact, must retain the pinned `emplace` first-rank behavior rather than be
silently reordered or deduplicated.

## Special tokens and flags

The Qwen3 artifacts contain:

```text
BOS = 151643
EOS = 151645
PAD = 151643
UNK = absent
add_bos = false
```

IDs are checked against vocabulary size. An absent UNK remains absent; no
fallback is invented. `add_bos` is portable because the pinned consumer reads it
and it changes raw input token sequences.

`add_eos_token`, `add_space_prefix`, and related flags are absent from these
artifacts and are not added to this first profile. For the pinned Qwen2 branch,
`clean_spaces = false` is a consumer behavior selected by the pre-tokenizer
identity, not a new executable wire field.

## Chat template

`tokenizer.chat_template` is preserved as exact optional UTF-8 bytes. The two
research artifacts contain different template bytes (4,100 versus 4,168 bytes),
and each is preserved without normalization. It is not parsed or executed. The pinned `llama_model_chat_template` accessor retrieves
it for chat construction; raw vocabulary loading and raw tokenization do not
require template rendering.

Therefore:

```text
raw inference tokenizer: READY
chat template storage: READY
chat execution/parity: deferred to consumer/runtime
```

## Experimental Qwen3.5 runtime tokenizer

`PreTokenizerIdentity = 4` selects the pinned Qwen3.5 byte-level BPE
pre-tokenizer. It requires tokenizer kind `2` and `NormalizerIdentity = 1`
(Unicode NFC); the parser rejects missing or mismatched identities. The Rust runtime implements the
pinned Qwen3.5 regex behavior, including Unicode combining marks, one-digit
numeric pieces, and the whitespace split behavior. It also treats token-flag
bit 0 as an added-token marker (bit 1 remains the special-token marker), so
non-special control strings such as `<tool_call>` stay atomic and decode as
literal token text. This interpretation is scoped to Qwen3.5; existing
Qwen2/GPT-2 profiles continue to use their prior special-ID handling.

A reduced test table derived from the exact pinned tokenizer files passed 318
input vectors against Hugging Face `tokenizers` 0.22.2, comparing original
model token IDs, token strings, and decoded text. The set includes Unicode
normalization, numeric and whitespace boundaries, tool markers, and all 26
added tokens. This qualifies only those tokenizer vectors; the complete
Qwen3.5 vocabulary has not been converted into a vBuf-ML sidecar or connected
to inference.

Tokenizer/model generation admission remains blocked. The text config uses EOS
ID 248044 (`<|endoftext|>`), while tokenizer config names ID 248046
(`<|im_end|>`); the pinned model has no generation config. The model has 248320
embedding rows, while the tokenizer provides strings only through ID 248069,
leaving 250 output rows without token strings. Neither termination policy nor
those generated IDs are guessed by this implementation.

## Validation and loading

Richer profiles require model identity, pre-tokenizer identity, both merge arrays,
and `AddBos`. Merge arrays must be bounded u32 arrays of equal length, every ID
must be within the vocabulary, and chat-template bytes must be bounded UTF-8.
Unknown required identities fail closed. VocabularyOnly remains accepted without
any richer roles.

The tokenizer control region remains bounded to 4096 bytes. Large merge arrays
and chat-template data are separate canonical ranges. Model open does not parse
or touch tokenizer regions. Tokenizer initialization resolves and validates the
vocabulary and merge ranges; chat data is only accessed when requested.

No tokenizer execution methods are present on storage-layer types.
