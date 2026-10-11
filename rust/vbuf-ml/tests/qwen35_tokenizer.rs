use serde::Deserialize;
use serde_json::Value as JsonValue;
use std::collections::HashSet;
use vbuf_core::v06::parse_v06;
use vbuf_ml::qwen35_chat::{
    Qwen35ChatOptions, account_qwen35_context, render_and_account_qwen35_chat, render_qwen35_chat,
};
use vbuf_ml::qwen35_tokenizer::{
    Qwen35TokenIdClass, Qwen35TokenizerManifest, classify_qwen35_token_id, load_and_verify_sidecar,
    sha256_hex, validate_manifest, verify_sidecar_identity,
};
use vbuf_ml::{Gpt2ByteLevelTokenizer, Qwen35SkippedTokenKind, TokenizerMetadata};

const SIDECAR: &[u8] = include_bytes!("fixtures/qwen35-9b-tokenizer.vbuf");
const MANIFEST_BYTES: &[u8] = include_bytes!("fixtures/qwen35-9b-tokenizer.manifest.json");
const OLD_VECTORS: &[u8] = include_bytes!("fixtures/qwen35-tokenizer-vectors.json");
const VECTORS: &[u8] = include_bytes!("fixtures/qwen35-9b-tokenizer-vectors.json");
const CHAT_VECTORS: &[u8] = include_bytes!("fixtures/qwen35-9b-chat-vectors.json");
const EXPECTED_SIDECAR_SHA256: &str =
    "ef248fae2f794714a46f5e815d5622555b42f6ae75d116cf002f9436ecea2522";

#[derive(Deserialize)]
struct VectorFixture {
    vectors: Vec<TextVector>,
}

#[derive(Deserialize)]
struct TextVector {
    #[serde(default)]
    name: String,
    text: String,
    ids: Vec<u32>,
    tokens: Vec<String>,
    decoded: String,
}

#[derive(Deserialize)]
struct ChatFixture {
    cases: Vec<ChatVector>,
    invalid_cases: Vec<InvalidChatVector>,
}

#[derive(Deserialize)]
struct ChatOptions {
    #[serde(default)]
    add_generation_prompt: bool,
    #[serde(default)]
    add_vision_id: bool,
    enable_thinking: Option<bool>,
}

impl From<ChatOptions> for Qwen35ChatOptions {
    fn from(value: ChatOptions) -> Self {
        Self {
            add_generation_prompt: value.add_generation_prompt,
            add_vision_id: value.add_vision_id,
            enable_thinking: value.enable_thinking,
        }
    }
}

#[derive(Deserialize)]
struct ChatVector {
    name: String,
    messages: Vec<JsonValue>,
    tools: Option<Vec<JsonValue>>,
    options: ChatOptions,
    expected_rendered: String,
    expected_ids: Vec<u32>,
    expected_tokens: Vec<String>,
    expected_decoded: String,
    context_limit: usize,
    completion_reserve: usize,
    expected_remaining: usize,
}

#[derive(Deserialize)]
struct InvalidChatVector {
    name: String,
    messages: Vec<JsonValue>,
    expected_error: String,
}

fn open_fixture() -> (Qwen35TokenizerManifest, TokenizerMetadata<'static>) {
    let manifest: Qwen35TokenizerManifest = serde_json::from_slice(MANIFEST_BYTES).unwrap();
    let (_bootstrap, metadata) = load_and_verify_sidecar(SIDECAR, &manifest).unwrap();
    (manifest, metadata)
}

#[test]
fn complete_sidecar_identity_and_vocabulary_classification_are_pinned() {
    assert_eq!(sha256_hex(SIDECAR), EXPECTED_SIDECAR_SHA256);
    let (manifest, metadata) = open_fixture();
    assert_eq!(manifest.repository, "Qwen/Qwen3.5-9B");
    assert_eq!(
        manifest.revision,
        "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    );
    assert_eq!(manifest.base_vocabulary_entries, 248_044);
    assert_eq!(manifest.added_tokens, 33);
    assert_eq!(manifest.special_tokens, 21);
    assert_eq!(manifest.explicit_token_count, 248_077);
    assert_eq!(manifest.model_embedding_rows, 248_320);
    assert_eq!(manifest.unassigned_model_rows, 243);
    assert_eq!(manifest.merges, 247_587);
    assert_eq!(manifest.model_config_eos_id, 248_044);
    assert_eq!(manifest.tokenizer_eos_id, 248_046);
    assert_eq!(manifest.tokenizer_pad_id, 248_044);
    assert_eq!(manifest.source_artifacts.len(), 6);
    assert_eq!(metadata.token_count(), 248_077);
    assert_eq!(metadata.merge_count(), 247_587);
    assert_eq!(metadata.add_bos(), Some(false));
    assert_eq!(
        metadata.normalizer(),
        Some(vbuf_ml::TokenizerNormalizer::UnicodeNfc)
    );
    assert_eq!(
        metadata.pre_tokenizer(),
        Some(vbuf_ml::PreTokenizer::Qwen3_5)
    );

    assert!((0..248_044).all(|id| metadata.token_flags(id).unwrap_or_default() & 1 == 0));
    assert!((248_044..248_077).all(|id| metadata.token_flags(id).unwrap_or_default() & 1 != 0));
    assert!((248_044..248_070).all(|id| metadata.token_flags(id).unwrap_or_default() & 1 != 0));
    assert!((248_070..248_077).all(|id| metadata.token_flags(id).unwrap_or_default() & 1 != 0));
    let special_ids = metadata.special_ids().unwrap();
    assert_eq!(special_ids.len(), 21);
    assert_eq!(
        special_ids.iter().copied().collect::<HashSet<_>>().len(),
        21
    );
    assert_eq!(
        (0..metadata.token_count())
            .filter(|id| metadata.token_flags(*id).unwrap_or_default() & 2 != 0)
            .count(),
        21
    );
    assert_eq!(metadata.token_text(248_044), Some("<|endoftext|>"));
    assert_eq!(metadata.token_text(248_046), Some("<|im_end|>"));
    assert_eq!(metadata.token_text(248_070), Some("<|audio_start|>"));
    assert_eq!(metadata.token_text(248_076), Some("<|audio_pad|>"));
    assert_eq!(metadata.token_text(248_077), None);
    assert_eq!(metadata.token_text(248_319), None);
    assert_eq!(metadata.token_text(248_320), None);
}

#[test]
fn full_sidecar_matches_previous_and_expanded_hugging_face_vectors() {
    let (_manifest, metadata) = open_fixture();
    let runtime = Gpt2ByteLevelTokenizer::build(&metadata).unwrap();
    let mut count = 0usize;
    for fixture_bytes in [OLD_VECTORS, VECTORS] {
        let fixture: VectorFixture = serde_json::from_slice(fixture_bytes).unwrap();
        for vector in fixture.vectors {
            let ids = runtime
                .encode(&vector.text)
                .unwrap_or_else(|error| panic!("vector {} encode failed: {error}", vector.name));
            assert_eq!(ids, vector.ids, "token IDs for {}", vector.name);
            let tokens: Vec<String> = ids
                .iter()
                .map(|id| metadata.token_text(u64::from(*id)).unwrap().to_owned())
                .collect();
            assert_eq!(tokens, vector.tokens, "token strings for {}", vector.name);
            assert_eq!(
                runtime.decode(&ids).unwrap(),
                vector.decoded,
                "decode for {}",
                vector.name
            );
            count += 1;
        }
    }
    assert_eq!(count, 441);
}

#[test]
fn pinned_chat_template_matches_multiturn_tool_and_context_reference_vectors() {
    let (_manifest, metadata) = open_fixture();
    let fixture: ChatFixture = serde_json::from_slice(CHAT_VECTORS).unwrap();
    let runtime = Gpt2ByteLevelTokenizer::build(&metadata).unwrap();
    for case in fixture.cases {
        let rendered = render_and_account_qwen35_chat(
            &metadata,
            &case.messages,
            case.tools.as_deref(),
            case.options.into(),
            case.context_limit,
            case.completion_reserve,
        )
        .unwrap_or_else(|error| panic!("chat case {} failed: {error}", case.name));
        assert_eq!(
            rendered.prompt.as_bytes(),
            case.expected_rendered.as_bytes(),
            "rendered bytes: {}",
            case.name
        );
        assert_eq!(
            rendered.token_ids, case.expected_ids,
            "token IDs: {}",
            case.name
        );
        let token_strings: Vec<String> = rendered
            .token_ids
            .iter()
            .map(|id| metadata.token_text(u64::from(*id)).unwrap().to_owned())
            .collect();
        assert_eq!(
            token_strings, case.expected_tokens,
            "token strings: {}",
            case.name
        );
        assert_eq!(
            runtime.decode(&rendered.token_ids).unwrap(),
            case.expected_decoded,
            "decoded prompt: {}",
            case.name
        );
        assert_eq!(
            rendered.context.prompt_tokens,
            case.expected_ids.len(),
            "prompt count: {}",
            case.name
        );
        assert_eq!(
            rendered.context.remaining, case.expected_remaining,
            "remaining context: {}",
            case.name
        );
    }
    for case in fixture.invalid_cases {
        let error = render_qwen35_chat(
            &metadata,
            &case.messages,
            None,
            Qwen35ChatOptions::default(),
        )
        .unwrap_err();
        assert!(
            error.contains(&case.expected_error),
            "{}: {error}",
            case.name
        );
    }
}

#[test]
fn qwen35_decoder_replacement_skip_and_unassigned_row_behavior_are_explicit() {
    let (_manifest, metadata) = open_fixture();
    let runtime = Gpt2ByteLevelTokenizer::build(&metadata).unwrap();
    let c3 = (0..metadata.token_count())
        .find(|id| metadata.token_text(*id) == Some("Ã"))
        .unwrap() as u32;
    let a9 = (0..metadata.token_count())
        .find(|id| metadata.token_text(*id) == Some("©"))
        .unwrap() as u32;
    assert_eq!(runtime.decode(&[c3]).unwrap(), "�");
    assert_eq!(runtime.decode_qwen35(&[c3], false).unwrap().text, "�");
    assert_eq!(runtime.decode_qwen35(&[c3, a9], false).unwrap().text, "é");

    let mut stream = runtime.streaming_decoder_qwen35(false).unwrap();
    assert_eq!(stream.push(&[c3]).unwrap().text, "");
    assert_eq!(stream.push(&[a9]).unwrap().text, "é");
    assert_eq!(stream.finish(), "");
    let mut incomplete = runtime.streaming_decoder_qwen35(false).unwrap();
    assert_eq!(incomplete.push(&[c3]).unwrap().text, "");
    assert_eq!(incomplete.finish(), "�");

    let special = runtime.decode_qwen35(&[248_044, 248_046], false).unwrap();
    assert_eq!(special.text, "<|endoftext|><|im_end|>");
    assert_eq!(
        runtime
            .decode_qwen35(&[248_044, 248_046], true)
            .unwrap()
            .text,
        ""
    );
    let unknown = runtime.decode_qwen35(&[248_077, 248_320], false).unwrap();
    assert_eq!(unknown.text, "");
    assert_eq!(unknown.skipped_tokens.len(), 2);
    assert_eq!(
        unknown.skipped_tokens[0].kind,
        Qwen35SkippedTokenKind::UnassignedModelVocabularyRow
    );
    assert_eq!(
        unknown.skipped_tokens[1].kind,
        Qwen35SkippedTokenKind::OutsideModelVocabulary
    );
}

#[test]
fn sidecar_hash_rejects_truncation_corruption_and_manifest_mismatch() {
    let manifest: Qwen35TokenizerManifest = serde_json::from_slice(MANIFEST_BYTES).unwrap();
    assert!(load_and_verify_sidecar(&SIDECAR[..SIDECAR.len() - 1], &manifest).is_err());
    let mut corrupted = SIDECAR.to_vec();
    let corruption_offset = corrupted.len() - 32;
    corrupted[corruption_offset] ^= 1;
    assert!(load_and_verify_sidecar(&corrupted, &manifest).is_err());
    let (_bootstrap, metadata) = load_and_verify_sidecar(SIDECAR, &manifest).unwrap();
    let mut altered_manifest = manifest.clone();
    altered_manifest.sidecar_sha256 = "00".repeat(32);
    assert!(verify_sidecar_identity(SIDECAR, &altered_manifest, &metadata).is_err());
}

#[test]
fn vocabulary_id_ranges_are_classified_without_guessing_unassigned_rows() {
    assert_eq!(classify_qwen35_token_id(0), Qwen35TokenIdClass::BaseBpe);
    assert_eq!(
        classify_qwen35_token_id(248_043),
        Qwen35TokenIdClass::BaseBpe
    );
    assert_eq!(
        classify_qwen35_token_id(248_044),
        Qwen35TokenIdClass::TokenizerJsonAdded
    );
    assert_eq!(
        classify_qwen35_token_id(248_069),
        Qwen35TokenIdClass::TokenizerJsonAdded
    );
    assert_eq!(
        classify_qwen35_token_id(248_070),
        Qwen35TokenIdClass::TokenizerConfigOnlyAdded
    );
    assert_eq!(
        classify_qwen35_token_id(248_076),
        Qwen35TokenIdClass::TokenizerConfigOnlyAdded
    );
    assert_eq!(
        classify_qwen35_token_id(248_077),
        Qwen35TokenIdClass::UnassignedModelVocabularyRow
    );
    assert_eq!(
        classify_qwen35_token_id(248_319),
        Qwen35TokenIdClass::UnassignedModelVocabularyRow
    );
    assert_eq!(
        classify_qwen35_token_id(248_320),
        Qwen35TokenIdClass::OutsideModelVocabulary
    );
}

#[test]
fn unsupported_wire_schema_and_corrupted_token_map_fail_closed() {
    let manifest: Qwen35TokenizerManifest = serde_json::from_slice(MANIFEST_BYTES).unwrap();
    let validated = parse_v06(SIDECAR).unwrap();

    let mut bad_wire_version = SIDECAR.to_vec();
    bad_wire_version[4..8].copy_from_slice(&0x0007_0000u32.to_le_bytes());
    assert!(load_and_verify_sidecar(&bad_wire_version, &manifest).is_err());

    let tokenizer_control = validated.block(11, 0).unwrap();
    let tokenizer_header = usize::try_from(tokenizer_control.payload_start).unwrap();
    let mut bad_tokenizer_version = SIDECAR.to_vec();
    bad_tokenizer_version[tokenizer_header + 8..tokenizer_header + 10]
        .copy_from_slice(&2u16.to_le_bytes());
    assert!(load_and_verify_sidecar(&bad_tokenizer_version, &manifest).is_err());

    let offsets = validated.block(201, 0).unwrap();
    let last_offset = usize::try_from(offsets.payload_end - 8).unwrap();
    let mut bad_token_map = SIDECAR.to_vec();
    bad_token_map[last_offset..last_offset + 8].copy_from_slice(&0u64.to_le_bytes());
    assert!(load_and_verify_sidecar(&bad_token_map, &manifest).is_err());

    let mut unsupported_manifest = manifest;
    unsupported_manifest.tokenizer_schema_version = 2;
    assert!(validate_manifest(&unsupported_manifest).is_err());
}

#[test]
fn context_accounting_uses_token_ids_not_rendered_bytes() {
    let exact = account_qwen35_context(262_136, 8, 262_144).unwrap();
    assert_eq!(exact.total_reserved, 262_144);
    assert_eq!(exact.remaining, 0);
    assert!(account_qwen35_context(262_137, 8, 262_144).is_err());
}
