//! Pinned Qwen3.5-9B tokenizer sidecar conversion and identity checks.
//!
//! This module accepts only the six text/configuration artifacts at the pinned
//! Hugging Face revision. It produces a tokenizer-only canonical vBuf envelope;
//! it does not contain weights and is not a model-execution artifact.

use crate::bootstrap::{
    BOOTSTRAP_KEY_ID, Bootstrap, BootstrapEntry, encode_payload as encode_bootstrap,
};
use crate::metadata::{MetadataEntry, ModelMetadata, encode_payload as encode_model_metadata};
use crate::region_roles::RegionRole;
use crate::runtime_tokenizer::Gpt2ByteLevelTokenizer;
use crate::tensor_directory::{TensorDirectory, encode_payload as encode_tensor_directory};
use crate::tokenizer::{
    PreTokenizer, TokenizerEntry, TokenizerKind, TokenizerMetadata, TokenizerModel,
    TokenizerNormalizer, encode_payload as encode_tokenizer_metadata,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::io::Cursor;
use std::path::Path;
use vbuf_core::v06::parse_v06;
use vbuf_core::writer::{BlockOptions, VBufV06Writer};

pub const QWEN35_REPOSITORY: &str = "Qwen/Qwen3.5-9B";
pub const QWEN35_REVISION: &str = "c202236235762e1c871ad0ccb60c8ee5ba337b9a";
pub const QWEN35_CONFIG_SHA256: &str =
    "d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05";
pub const QWEN35_TOKENIZER_JSON_SHA256: &str =
    "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42";
pub const QWEN35_TOKENIZER_CONFIG_SHA256: &str =
    "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8";
pub const QWEN35_CHAT_TEMPLATE_SHA256: &str =
    "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715";
pub const QWEN35_VOCAB_SHA256: &str =
    "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003";
pub const QWEN35_MERGES_SHA256: &str =
    "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d";
pub const QWEN35_BASE_VOCAB_SIZE: usize = 248_044;
pub const QWEN35_EXPLICIT_TOKEN_COUNT: usize = 248_077;
pub const QWEN35_MODEL_VOCAB_SIZE: usize = 248_320;
pub const QWEN35_JSON_ADDED_TOKEN_COUNT: usize = 26;
pub const QWEN35_CONFIG_ONLY_ADDED_TOKEN_COUNT: usize = 7;
pub const QWEN35_ADDED_TOKEN_COUNT: usize = 33;
pub const QWEN35_SPECIAL_TOKEN_COUNT: usize = 21;
pub const QWEN35_MERGE_COUNT: usize = 247_587;
pub const QWEN35_MODEL_EOS_ID: u32 = 248_044;
pub const QWEN35_TOKENIZER_EOS_ID: u32 = 248_046;
pub const QWEN35_TOKENIZER_PAD_ID: u32 = 248_044;

/// Pinned digest of the deterministic tokenizer-only vBuf conversion. A change
/// to conversion output requires an explicit identity and parity requalification.
pub const QWEN35_SIDECAR_SHA256: &str =
    "ef248fae2f794714a46f5e815d5622555b42f6ae75d116cf002f9436ecea2522";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen35TokenIdClass {
    BaseBpe,
    TokenizerJsonAdded,
    TokenizerConfigOnlyAdded,
    UnassignedModelVocabularyRow,
    OutsideModelVocabulary,
}

pub const fn classify_qwen35_token_id(id: u32) -> Qwen35TokenIdClass {
    if id < QWEN35_BASE_VOCAB_SIZE as u32 {
        Qwen35TokenIdClass::BaseBpe
    } else if id < (QWEN35_BASE_VOCAB_SIZE + QWEN35_JSON_ADDED_TOKEN_COUNT) as u32 {
        Qwen35TokenIdClass::TokenizerJsonAdded
    } else if id < QWEN35_EXPLICIT_TOKEN_COUNT as u32 {
        Qwen35TokenIdClass::TokenizerConfigOnlyAdded
    } else if id < QWEN35_MODEL_VOCAB_SIZE as u32 {
        Qwen35TokenIdClass::UnassignedModelVocabularyRow
    } else {
        Qwen35TokenIdClass::OutsideModelVocabulary
    }
}

const CONFIG_NAME: &str = "config.json";
const TOKENIZER_NAME: &str = "tokenizer.json";
const TOKENIZER_CONFIG_NAME: &str = "tokenizer_config.json";
const TEMPLATE_NAME: &str = "chat_template.jinja";
const VOCAB_NAME: &str = "vocab.json";
const MERGES_NAME: &str = "merges.txt";
const MAX_SOURCE_FILE_BYTES: u64 = 32 * 1024 * 1024;
const ACQUISITION_METHOD: &str =
    "HTTPS GET from the Hugging Face resolve endpoint at the pinned revision";

const QWEN35_PRETOKENIZER_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct SourceArtifact {
    pub file: String,
    pub revision: String,
    pub size_bytes: u64,
    pub sha256: String,
    pub local_source_path: String,
    pub acquisition_method: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Qwen35TokenizerManifest {
    pub manifest_format: String,
    pub repository: String,
    pub revision: String,
    pub vbuf_wire_format: String,
    pub tokenizer_schema_version: u16,
    pub tokenizer_kind: String,
    pub tokenizer_model: String,
    pub pre_tokenizer: String,
    pub normalizer: String,
    pub decoder: String,
    pub decoder_errors: String,
    pub source_artifacts: Vec<SourceArtifact>,
    pub converter: String,
    pub converter_version: String,
    pub converter_revision: String,
    pub sidecar_size_bytes: u64,
    pub sidecar_sha256: String,
    pub base_vocabulary_entries: usize,
    pub tokenizer_json_added_tokens: usize,
    pub tokenizer_config_only_added_tokens: usize,
    pub added_tokens: usize,
    pub special_tokens: usize,
    pub explicit_token_count: usize,
    pub model_embedding_rows: usize,
    pub unassigned_model_rows: usize,
    pub merges: usize,
    pub model_config_eos_id: u32,
    pub tokenizer_eos_id: u32,
    pub tokenizer_pad_id: u32,
    pub chat_template_sha256: String,
}

#[derive(Debug)]
pub struct ConvertedQwen35Sidecar {
    pub sidecar: Vec<u8>,
    pub manifest: Qwen35TokenizerManifest,
}

#[derive(Debug, Deserialize)]
struct TokenizerJson {
    model: BpeModel,
    added_tokens: Vec<JsonAddedToken>,
    normalizer: Value,
    pre_tokenizer: Value,
    decoder: Value,
    post_processor: Value,
}

#[derive(Debug, Deserialize)]
struct BpeModel {
    #[serde(rename = "type")]
    model_type: String,
    dropout: Option<Value>,
    unk_token: Option<String>,
    continuing_subword_prefix: String,
    end_of_word_suffix: String,
    fuse_unk: bool,
    byte_fallback: bool,
    ignore_merges: bool,
    vocab: BTreeMap<String, u32>,
    merges: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
struct JsonAddedToken {
    id: u32,
    content: String,
    single_word: bool,
    lstrip: bool,
    rstrip: bool,
    normalized: bool,
    special: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
struct ConfigAddedToken {
    content: String,
    single_word: bool,
    lstrip: bool,
    rstrip: bool,
    normalized: bool,
    special: bool,
}

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    tokenizer_class: String,
    add_bos_token: bool,
    clean_up_tokenization_spaces: bool,
    errors: String,
    pretokenize_regex: String,
    eos_token: Option<Value>,
    pad_token: Option<Value>,
    bos_token: Option<Value>,
    unk_token: Option<Value>,
    chat_template: String,
    added_tokens_decoder: BTreeMap<String, ConfigAddedToken>,
}

/// Convert the exact pinned source files into a deterministic vBuf-ML tokenizer
/// envelope. The output contains a real Qwen3.5 text-profile ModelMetadata
/// block and an empty TensorDirectory, not fake model weights.
pub fn convert_pinned_directory(
    source_dir: &Path,
    converter_revision: &str,
) -> Result<ConvertedQwen35Sidecar, String> {
    let source_dir = source_dir
        .canonicalize()
        .map_err(|error| format!("cannot resolve source directory: {error}"))?;
    let names = [
        CONFIG_NAME,
        TOKENIZER_NAME,
        TOKENIZER_CONFIG_NAME,
        TEMPLATE_NAME,
        VOCAB_NAME,
        MERGES_NAME,
    ];
    let expected = [
        QWEN35_CONFIG_SHA256,
        QWEN35_TOKENIZER_JSON_SHA256,
        QWEN35_TOKENIZER_CONFIG_SHA256,
        QWEN35_CHAT_TEMPLATE_SHA256,
        QWEN35_VOCAB_SHA256,
        QWEN35_MERGES_SHA256,
    ];
    let mut raw = BTreeMap::new();
    let mut artifacts = Vec::with_capacity(names.len());
    for (name, expected_hash) in names.iter().zip(expected) {
        let path = source_dir.join(name);
        let metadata = fs::metadata(&path)
            .map_err(|error| format!("cannot stat pinned source {}: {error}", path.display()))?;
        if !metadata.is_file() || metadata.len() > MAX_SOURCE_FILE_BYTES {
            return Err(format!(
                "pinned source {} is not a bounded regular file",
                path.display()
            ));
        }
        let bytes = fs::read(&path)
            .map_err(|error| format!("cannot read pinned source {}: {error}", path.display()))?;
        let actual_hash = sha256_hex(&bytes);
        if actual_hash != expected_hash {
            return Err(format!(
                "pinned source identity mismatch for {name}: expected {expected_hash}, got {actual_hash}"
            ));
        }
        artifacts.push(SourceArtifact {
            file: (*name).to_owned(),
            revision: QWEN35_REVISION.to_owned(),
            size_bytes: u64::try_from(bytes.len()).map_err(|_| "source size exceeds u64")?,
            sha256: actual_hash,
            local_source_path: path.display().to_string(),
            acquisition_method: ACQUISITION_METHOD.to_owned(),
        });
        raw.insert(*name, bytes);
    }

    let config: Value = parse_json(required_source(&raw, CONFIG_NAME)?, CONFIG_NAME)?;
    let tokenizer_json: TokenizerJson =
        serde_json::from_slice(required_source(&raw, TOKENIZER_NAME)?)
            .map_err(|error| format!("invalid pinned tokenizer.json: {error}"))?;
    let tokenizer_config: TokenizerConfig =
        serde_json::from_slice(required_source(&raw, TOKENIZER_CONFIG_NAME)?)
            .map_err(|error| format!("invalid pinned tokenizer_config.json: {error}"))?;
    let template_bytes = required_source(&raw, TEMPLATE_NAME)?;
    let vocab_file: BTreeMap<String, u32> =
        serde_json::from_slice(required_source(&raw, VOCAB_NAME)?)
            .map_err(|error| format!("invalid pinned vocab.json: {error}"))?;
    let merges_file = std::str::from_utf8(required_source(&raw, MERGES_NAME)?)
        .map_err(|_| "pinned merges.txt is not UTF-8")?;

    let (model_vocab_size, model_eos) = validate_model_config(&config)?;
    validate_tokenizer_config(&tokenizer_json, &tokenizer_config, template_bytes)?;
    if vocab_file != tokenizer_json.model.vocab {
        return Err("vocab.json differs from tokenizer.json model.vocab".to_owned());
    }
    validate_merge_sources(&tokenizer_json.model.merges, merges_file)?;

    let inventory = build_inventory(&tokenizer_json, &tokenizer_config, &vocab_file)?;
    let tokenizer_eos = resolve_config_token(
        tokenizer_config.eos_token.as_ref(),
        &inventory.by_id,
        "tokenizer eos_token",
    )?;
    let tokenizer_pad = resolve_config_token(
        tokenizer_config.pad_token.as_ref(),
        &inventory.by_id,
        "tokenizer pad_token",
    )?;
    if tokenizer_eos != QWEN35_TOKENIZER_EOS_ID
        || tokenizer_pad != QWEN35_TOKENIZER_PAD_ID
        || model_eos != QWEN35_MODEL_EOS_ID
        || model_vocab_size != QWEN35_MODEL_VOCAB_SIZE
    {
        return Err("pinned model/tokenizer vocabulary and EOS identities changed".to_owned());
    }

    let sidecar = build_vbuf_sidecar(
        &inventory,
        &tokenizer_json.model.merges,
        template_bytes,
        tokenizer_eos,
        tokenizer_pad,
    )?;
    let manifest = Qwen35TokenizerManifest {
        manifest_format: "vbuf-ml.qwen35-tokenizer-manifest.v1".to_owned(),
        repository: QWEN35_REPOSITORY.to_owned(),
        revision: QWEN35_REVISION.to_owned(),
        vbuf_wire_format: "vBuf v0.6".to_owned(),
        tokenizer_schema_version: 1,
        tokenizer_kind: "Gpt2BpeQwen2/profile-kind-2".to_owned(),
        tokenizer_model: "GPT-2 byte-level BPE".to_owned(),
        pre_tokenizer: "Qwen3.5 pinned regex profile 4".to_owned(),
        normalizer: "NFC".to_owned(),
        decoder: "ByteLevel".to_owned(),
        decoder_errors: tokenizer_config.errors,
        source_artifacts: artifacts,
        converter: "vbuf-ml::qwen35_tokenizer::convert_pinned_directory".to_owned(),
        converter_version: env!("CARGO_PKG_VERSION").to_owned(),
        converter_revision: converter_revision.to_owned(),
        sidecar_size_bytes: u64::try_from(sidecar.len()).map_err(|_| "sidecar size exceeds u64")?,
        sidecar_sha256: sha256_hex(&sidecar),
        base_vocabulary_entries: inventory.base_vocab_entries,
        tokenizer_json_added_tokens: inventory.json_added_tokens,
        tokenizer_config_only_added_tokens: inventory.config_only_added_tokens,
        added_tokens: inventory.added_tokens,
        special_tokens: inventory.special_tokens,
        explicit_token_count: inventory.tokens.len(),
        model_embedding_rows: model_vocab_size,
        unassigned_model_rows: model_vocab_size - inventory.tokens.len(),
        merges: tokenizer_json.model.merges.len(),
        model_config_eos_id: model_eos,
        tokenizer_eos_id: tokenizer_eos,
        tokenizer_pad_id: tokenizer_pad,
        chat_template_sha256: sha256_hex(template_bytes),
    };
    validate_manifest(&manifest)?;
    if !QWEN35_SIDECAR_SHA256.is_empty() && manifest.sidecar_sha256 != QWEN35_SIDECAR_SHA256 {
        return Err(format!(
            "pinned Qwen3.5 conversion changed: expected {}, got {}",
            QWEN35_SIDECAR_SHA256, manifest.sidecar_sha256
        ));
    }
    Ok(ConvertedQwen35Sidecar { sidecar, manifest })
}

/// Verify the sidecar hash, pinned source manifest, normal vBuf structure, and
/// tokenizer semantic identity before constructing the existing runtime view.
pub fn verify_sidecar_identity(
    sidecar_bytes: &[u8],
    manifest: &Qwen35TokenizerManifest,
    metadata: &TokenizerMetadata<'_>,
) -> Result<(), String> {
    validate_manifest(manifest)?;
    let digest = sha256_hex(sidecar_bytes);
    if digest != manifest.sidecar_sha256 {
        return Err(format!(
            "sidecar SHA-256 mismatch: manifest {}, actual {digest}",
            manifest.sidecar_sha256
        ));
    }
    if u64::try_from(sidecar_bytes.len()).ok() != Some(manifest.sidecar_size_bytes) {
        return Err("sidecar byte length differs from manifest".to_owned());
    }
    if !QWEN35_SIDECAR_SHA256.is_empty() && digest != QWEN35_SIDECAR_SHA256 {
        return Err(format!(
            "sidecar is not the qualified pinned conversion: expected {}, got {digest}",
            QWEN35_SIDECAR_SHA256
        ));
    }
    if metadata.kind != TokenizerKind::Gpt2BpeQwen2
        || metadata.model() != Some(TokenizerModel::Gpt2Bpe)
        || metadata.pre_tokenizer() != Some(PreTokenizer::Qwen3_5)
        || metadata.normalizer() != Some(TokenizerNormalizer::UnicodeNfc)
        || metadata.add_bos() != Some(false)
        || metadata.token_count() != QWEN35_EXPLICIT_TOKEN_COUNT as u64
        || metadata.merge_count() != QWEN35_MERGE_COUNT as u64
        || metadata
            .chat_template()
            .map(str::as_bytes)
            .map(sha256_hex)
            .as_deref()
            != Some(QWEN35_CHAT_TEMPLATE_SHA256)
        || !metadata
            .specials()
            .contains(&(crate::SpecialToken::Eos, u64::from(QWEN35_TOKENIZER_EOS_ID)))
        || !metadata
            .specials()
            .contains(&(crate::SpecialToken::Pad, u64::from(QWEN35_TOKENIZER_PAD_ID)))
    {
        return Err("sidecar tokenizer profile does not match pinned Qwen3.5 identity".to_owned());
    }
    let special_ids = metadata.special_ids().unwrap_or_default();
    if special_ids.len() != QWEN35_SPECIAL_TOKEN_COUNT
        || (0..metadata.token_count())
            .filter(|id| {
                metadata
                    .token_flags(*id)
                    .is_some_and(|flags| flags & 1 != 0)
            })
            .count()
            != QWEN35_ADDED_TOKEN_COUNT
        || (248_044..248_070)
            .filter(|id| {
                metadata
                    .token_flags(*id)
                    .is_some_and(|flags| flags & 1 != 0)
            })
            .count()
            != QWEN35_JSON_ADDED_TOKEN_COUNT
        || (248_070..248_077)
            .filter(|id| {
                metadata
                    .token_flags(*id)
                    .is_some_and(|flags| flags & 1 != 0)
            })
            .count()
            != QWEN35_CONFIG_ONLY_ADDED_TOKEN_COUNT
        || (0..metadata.token_count())
            .filter(|id| {
                metadata
                    .token_flags(*id)
                    .is_some_and(|flags| flags & 2 != 0)
            })
            .count()
            != QWEN35_SPECIAL_TOKEN_COUNT
    {
        return Err("sidecar added/special-token flags do not match pinned counts".to_owned());
    }
    if metadata.token_text(248_044) != Some("<|endoftext|>")
        || metadata.token_text(248_046) != Some("<|im_end|>")
        || metadata.token_text(248_058) != Some("<tool_call>")
        || metadata.token_text(248_066) != Some("<tool_response>")
        || metadata.token_text(248_068) != Some("<think>")
        || metadata.token_text(248_070) != Some("<|audio_start|>")
        || metadata.token_text(248_076) != Some("<|audio_pad|>")
    {
        return Err("pinned special-token string-to-ID inventory mismatch".to_owned());
    }
    if manifest.base_vocabulary_entries != QWEN35_BASE_VOCAB_SIZE
        || manifest.tokenizer_json_added_tokens != QWEN35_JSON_ADDED_TOKEN_COUNT
        || manifest.tokenizer_config_only_added_tokens != QWEN35_CONFIG_ONLY_ADDED_TOKEN_COUNT
        || manifest.added_tokens != QWEN35_ADDED_TOKEN_COUNT
        || manifest.special_tokens != QWEN35_SPECIAL_TOKEN_COUNT
        || manifest.explicit_token_count != QWEN35_EXPLICIT_TOKEN_COUNT
        || manifest.model_embedding_rows != QWEN35_MODEL_VOCAB_SIZE
        || manifest.unassigned_model_rows != 243
        || manifest.merges != QWEN35_MERGE_COUNT
        || manifest.model_config_eos_id != QWEN35_MODEL_EOS_ID
        || manifest.tokenizer_eos_id != QWEN35_TOKENIZER_EOS_ID
        || manifest.tokenizer_pad_id != QWEN35_TOKENIZER_PAD_ID
    {
        return Err("sidecar manifest cardinality or special-token identity mismatch".to_owned());
    }
    Ok(())
}

pub fn validate_manifest(manifest: &Qwen35TokenizerManifest) -> Result<(), String> {
    if manifest.converter_revision.len() != 40
        || !manifest
            .converter_revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        || manifest.manifest_format != "vbuf-ml.qwen35-tokenizer-manifest.v1"
        || manifest.repository != QWEN35_REPOSITORY
        || manifest.revision != QWEN35_REVISION
        || manifest.vbuf_wire_format != "vBuf v0.6"
        || manifest.tokenizer_schema_version != 1
        || manifest.tokenizer_kind != "Gpt2BpeQwen2/profile-kind-2"
        || manifest.tokenizer_model != "GPT-2 byte-level BPE"
        || manifest.pre_tokenizer != "Qwen3.5 pinned regex profile 4"
        || manifest.normalizer != "NFC"
        || manifest.decoder != "ByteLevel"
        || manifest.decoder_errors != "replace"
        || manifest.chat_template_sha256 != QWEN35_CHAT_TEMPLATE_SHA256
        || manifest.base_vocabulary_entries != QWEN35_BASE_VOCAB_SIZE
        || manifest.tokenizer_json_added_tokens != QWEN35_JSON_ADDED_TOKEN_COUNT
        || manifest.tokenizer_config_only_added_tokens != QWEN35_CONFIG_ONLY_ADDED_TOKEN_COUNT
        || manifest.added_tokens != QWEN35_ADDED_TOKEN_COUNT
        || manifest.special_tokens != QWEN35_SPECIAL_TOKEN_COUNT
        || manifest.explicit_token_count != QWEN35_EXPLICIT_TOKEN_COUNT
        || manifest.model_embedding_rows != QWEN35_MODEL_VOCAB_SIZE
        || manifest.unassigned_model_rows != 243
        || manifest.merges != QWEN35_MERGE_COUNT
        || manifest.model_config_eos_id != QWEN35_MODEL_EOS_ID
        || manifest.tokenizer_eos_id != QWEN35_TOKENIZER_EOS_ID
        || manifest.tokenizer_pad_id != QWEN35_TOKENIZER_PAD_ID
        || manifest.sidecar_size_bytes == 0
        || manifest.sidecar_sha256.len() != 64
        || !manifest
            .sidecar_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
        || manifest.converter.is_empty()
        || manifest.converter_version.is_empty()
        || manifest.converter_revision.trim().is_empty()
    {
        return Err("unsupported or mismatched Qwen3.5 tokenizer manifest identity".to_owned());
    }
    let expected: BTreeMap<&str, (&str, u64)> = [
        (CONFIG_NAME, (QWEN35_CONFIG_SHA256, 3_126)),
        (TOKENIZER_NAME, (QWEN35_TOKENIZER_JSON_SHA256, 12_807_982)),
        (
            TOKENIZER_CONFIG_NAME,
            (QWEN35_TOKENIZER_CONFIG_SHA256, 16_710),
        ),
        (TEMPLATE_NAME, (QWEN35_CHAT_TEMPLATE_SHA256, 7_756)),
        (VOCAB_NAME, (QWEN35_VOCAB_SHA256, 6_722_759)),
        (MERGES_NAME, (QWEN35_MERGES_SHA256, 3_353_259)),
    ]
    .into_iter()
    .collect();
    let mut seen = BTreeSet::new();
    for artifact in &manifest.source_artifacts {
        if artifact.revision != QWEN35_REVISION
            || expected.get(artifact.file.as_str()).copied()
                != Some((artifact.sha256.as_str(), artifact.size_bytes))
            || artifact.local_source_path.is_empty()
            || artifact.acquisition_method != ACQUISITION_METHOD
            || !seen.insert(artifact.file.as_str())
        {
            return Err(format!(
                "manifest source provenance is missing, duplicate, or mismatched for {}",
                artifact.file
            ));
        }
    }
    if seen.len() != expected.len() {
        return Err("manifest does not enumerate all six pinned source artifacts".to_owned());
    }
    Ok(())
}

#[derive(Debug)]
struct TokenInventory {
    tokens: Vec<String>,
    flags: Vec<u8>,
    special_ids: Vec<u32>,
    by_id: Vec<String>,
    base_vocab_entries: usize,
    json_added_tokens: usize,
    config_only_added_tokens: usize,
    added_tokens: usize,
    special_tokens: usize,
}

fn build_inventory(
    tokenizer_json: &TokenizerJson,
    tokenizer_config: &TokenizerConfig,
    vocab_file: &BTreeMap<String, u32>,
) -> Result<TokenInventory, String> {
    let vocab = &tokenizer_json.model.vocab;
    if vocab.len() != QWEN35_BASE_VOCAB_SIZE || vocab_file.len() != vocab.len() {
        return Err("pinned base vocabulary cardinality changed".to_owned());
    }
    let mut by_id = vec![None; QWEN35_MODEL_VOCAB_SIZE];
    for (token, id) in vocab {
        let index = usize::try_from(*id).map_err(|_| "base token ID exceeds host width")?;
        if index >= QWEN35_BASE_VOCAB_SIZE || by_id[index].replace(token.clone()).is_some() {
            return Err(format!("invalid or duplicate base token ID {id}"));
        }
    }

    let mut added_by_id = BTreeMap::<u32, ConfigAddedToken>::new();
    for (key, token) in &tokenizer_config.added_tokens_decoder {
        let id = key
            .parse::<u32>()
            .map_err(|_| format!("invalid added-token ID key {key:?}"))?;
        if added_by_id.insert(id, token.clone()).is_some() {
            return Err(format!("duplicate added-token ID {id}"));
        }
    }
    let mut json_by_id = BTreeMap::new();
    for token in &tokenizer_json.added_tokens {
        if json_by_id.insert(token.id, token).is_some() {
            return Err(format!(
                "duplicate tokenizer.json added-token ID {}",
                token.id
            ));
        }
        let from_config = added_by_id.get(&token.id).ok_or_else(|| {
            format!(
                "tokenizer.json added token {} is absent from tokenizer_config.json",
                token.id
            )
        })?;
        if from_config.content != token.content
            || from_config.single_word != token.single_word
            || from_config.lstrip != token.lstrip
            || from_config.rstrip != token.rstrip
            || from_config.normalized != token.normalized
            || from_config.special != token.special
        {
            return Err(format!("added-token metadata differs at ID {}", token.id));
        }
    }
    if json_by_id.len() != QWEN35_JSON_ADDED_TOKEN_COUNT
        || added_by_id.len() != QWEN35_ADDED_TOKEN_COUNT
    {
        return Err("pinned added-token cardinality changed".to_owned());
    }
    if json_by_id.keys().any(|id| !(248_044..248_070).contains(id))
        || added_by_id
            .keys()
            .filter(|id| !json_by_id.contains_key(id))
            .any(|id| !(248_070..248_077).contains(id))
    {
        return Err("JSON/config-only added-token ID classification changed".to_owned());
    }
    for (id, token) in &added_by_id {
        let index = usize::try_from(*id).map_err(|_| "added-token ID exceeds host width")?;
        if !(QWEN35_BASE_VOCAB_SIZE..QWEN35_EXPLICIT_TOKEN_COUNT).contains(&index) {
            return Err(format!(
                "added token ID {id} is outside the pinned contiguous range"
            ));
        }
        for (field, enabled) in [
            ("single_word", token.single_word),
            ("lstrip", token.lstrip),
            ("rstrip", token.rstrip),
            ("normalized", token.normalized),
        ] {
            if enabled {
                return Err(format!(
                    "unsupported pinned added-token option {field} at ID {id}"
                ));
            }
        }
        if by_id[index].replace(token.content.clone()).is_some() {
            return Err(format!("added token ID {id} collides with a base token"));
        }
    }
    if by_id[..QWEN35_EXPLICIT_TOKEN_COUNT]
        .iter()
        .any(Option::is_none)
    {
        return Err("one or more explicit tokenizer IDs are missing".to_owned());
    }
    if by_id[QWEN35_EXPLICIT_TOKEN_COUNT..]
        .iter()
        .any(Option::is_some)
    {
        return Err("tokenizer source unexpectedly defines IDs in padded model rows".to_owned());
    }
    let tokens: Vec<String> = by_id[..QWEN35_EXPLICIT_TOKEN_COUNT]
        .iter()
        .enumerate()
        .map(|(id, token)| {
            token
                .clone()
                .ok_or_else(|| format!("explicit token {id} is absent after validation"))
        })
        .collect::<Result<_, _>>()?;
    let mut flags = vec![0u8; tokens.len()];
    let mut special_ids = Vec::new();
    for (id, token) in &added_by_id {
        let index = usize::try_from(*id).map_err(|_| "added-token ID exceeds host width")?;
        flags[index] = 1 | (u8::from(token.special) << 1);
        if token.special {
            special_ids.push(*id);
        }
    }
    if special_ids.len() != QWEN35_SPECIAL_TOKEN_COUNT {
        return Err(format!(
            "expected 21 special tokens, found {}",
            special_ids.len()
        ));
    }
    Ok(TokenInventory {
        tokens: tokens.clone(),
        flags,
        special_ids,
        by_id: tokens,
        base_vocab_entries: vocab.len(),
        json_added_tokens: json_by_id.len(),
        config_only_added_tokens: added_by_id.len() - json_by_id.len(),
        added_tokens: added_by_id.len(),
        special_tokens: QWEN35_SPECIAL_TOKEN_COUNT,
    })
}

fn validate_model_config(config: &Value) -> Result<(usize, u32), String> {
    if config.get("model_type").and_then(Value::as_str) != Some("qwen3_5")
        || !config
            .get("architectures")
            .and_then(Value::as_array)
            .is_some_and(|items| {
                items
                    .iter()
                    .any(|v| v.as_str() == Some("Qwen3_5ForConditionalGeneration"))
            })
    {
        return Err("config.json does not identify the pinned Qwen3.5 architecture".to_owned());
    }
    let text = config
        .get("text_config")
        .and_then(Value::as_object)
        .ok_or("config.json has no text_config")?;
    let vocab = json_u64(text.get("vocab_size"), "text_config.vocab_size")?;
    let eos = json_u64(text.get("eos_token_id"), "text_config.eos_token_id")?;
    Ok((
        usize::try_from(vocab).map_err(|_| "model vocabulary exceeds host range")?,
        u32::try_from(eos).map_err(|_| "model EOS exceeds u32")?,
    ))
}

fn validate_tokenizer_config(
    tokenizer: &TokenizerJson,
    config: &TokenizerConfig,
    template_bytes: &[u8],
) -> Result<(), String> {
    if tokenizer.model.model_type != "BPE"
        || tokenizer
            .model
            .dropout
            .as_ref()
            .is_some_and(|value| !value.is_null())
        || tokenizer.model.unk_token.is_some()
        || !tokenizer.model.continuing_subword_prefix.is_empty()
        || !tokenizer.model.end_of_word_suffix.is_empty()
        || tokenizer.model.fuse_unk
        || tokenizer.model.byte_fallback
        || tokenizer.model.ignore_merges
    {
        return Err("unsupported or changed pinned BPE model options".to_owned());
    }
    if tokenizer.normalizer != serde_json::json!({"type":"NFC"}) {
        return Err("pinned tokenizer normalizer is not exactly NFC".to_owned());
    }
    validate_qwen35_pretokenizer(&tokenizer.pre_tokenizer)?;
    validate_bytelevel_decoder(&tokenizer.decoder)?;
    validate_bytelevel_decoder(&tokenizer.post_processor)?;
    if config.tokenizer_class != "Qwen2Tokenizer"
        || config.add_bos_token
        || config.clean_up_tokenization_spaces
        || config.errors != "replace"
        || config.pretokenize_regex != QWEN35_PRETOKENIZER_REGEX
    {
        return Err("pinned tokenizer_config behavior changed".to_owned());
    }
    if config.chat_template.as_bytes() != template_bytes
        || sha256_hex(template_bytes) != QWEN35_CHAT_TEMPLATE_SHA256
    {
        return Err("chat_template.jinja and tokenizer_config template differ".to_owned());
    }
    if config.bos_token.is_some() || config.unk_token.is_some() {
        return Err("pinned tokenizer unexpectedly defines BOS or UNK".to_owned());
    }
    Ok(())
}

fn validate_qwen35_pretokenizer(value: &Value) -> Result<(), String> {
    let sequence = value
        .as_object()
        .filter(|map| map.get("type").and_then(Value::as_str) == Some("Sequence"))
        .and_then(|map| map.get("pretokenizers"))
        .and_then(Value::as_array)
        .ok_or("pinned pre-tokenizer is not a sequence")?;
    if sequence.len() != 2 {
        return Err("pinned pre-tokenizer sequence length changed".to_owned());
    }
    let split = &sequence[0];
    let pattern_ok = split.get("type").and_then(Value::as_str) == Some("Split")
        && split
            .get("pattern")
            .and_then(|v| v.get("Regex"))
            .and_then(Value::as_str)
            == Some(QWEN35_PRETOKENIZER_REGEX)
        && split.get("behavior").and_then(Value::as_str) == Some("Isolated")
        && split.get("invert").and_then(Value::as_bool) == Some(false);
    if !pattern_ok {
        return Err("pinned Qwen3.5 regex split behavior changed".to_owned());
    }
    validate_bytelevel_decoder(&sequence[1])
}

fn validate_bytelevel_decoder(value: &Value) -> Result<(), String> {
    if value.get("type").and_then(Value::as_str) != Some("ByteLevel")
        || value.get("add_prefix_space").and_then(Value::as_bool) != Some(false)
        || value.get("trim_offsets").and_then(Value::as_bool) != Some(false)
        || value.get("use_regex").and_then(Value::as_bool) != Some(false)
    {
        return Err("pinned ByteLevel decoder/pre-tokenizer options changed".to_owned());
    }
    Ok(())
}

fn validate_merge_sources(json_merges: &[String], merges_file: &str) -> Result<(), String> {
    if json_merges.len() != QWEN35_MERGE_COUNT {
        return Err(format!(
            "expected {} BPE merges, found {}",
            QWEN35_MERGE_COUNT,
            json_merges.len()
        ));
    }
    let lines: Vec<&str> = merges_file.lines().collect();
    if lines.len() != json_merges.len() {
        return Err("merges.txt and tokenizer.json merge counts differ".to_owned());
    }
    for (rank, (json, line)) in json_merges.iter().zip(lines).enumerate() {
        if json != line {
            return Err(format!("merge source differs at rank {rank}"));
        }
        if json.split_once(' ').is_none() {
            return Err(format!("merge entry {rank} has no component separator"));
        }
    }
    Ok(())
}

fn build_vbuf_sidecar(
    inventory: &TokenInventory,
    merge_strings: &[String],
    chat_template: &[u8],
    tokenizer_eos: u32,
    tokenizer_pad: u32,
) -> Result<Vec<u8>, String> {
    let mut token_ids = HashMap::<&str, u32>::with_capacity(inventory.tokens.len());
    for (index, token) in inventory.tokens.iter().enumerate() {
        if token.is_empty() || token.contains('\0') || token.len() > 4096 {
            return Err(format!("token text at ID {index} violates sidecar bounds"));
        }
        if token_ids
            .insert(
                token,
                u32::try_from(index).map_err(|_| "token ID exceeds u32")?,
            )
            .is_some()
        {
            return Err(format!("duplicate token string at ID {index}"));
        }
    }
    let mut token_text = Vec::new();
    let mut offsets = Vec::with_capacity(inventory.tokens.len() + 1);
    offsets.push(0u64);
    for token in &inventory.tokens {
        token_text.extend_from_slice(token.as_bytes());
        offsets.push(u64::try_from(token_text.len()).map_err(|_| "token pool exceeds u64")?);
    }
    let mut merge_left = Vec::with_capacity(merge_strings.len());
    let mut merge_right = Vec::with_capacity(merge_strings.len());
    let mut pairs = HashMap::<(u32, u32), usize>::with_capacity(merge_strings.len());
    for (rank, merge) in merge_strings.iter().enumerate() {
        let (left, right) = merge
            .split_once(' ')
            .ok_or_else(|| format!("merge at rank {rank} has no separator"))?;
        if left.is_empty() || right.is_empty() || right.contains(' ') {
            return Err(format!("merge at rank {rank} is not a two-token pair"));
        }
        let left_id = *token_ids
            .get(left)
            .ok_or_else(|| format!("merge-left token missing at rank {rank}"))?;
        let right_id = *token_ids
            .get(right)
            .ok_or_else(|| format!("merge-right token missing at rank {rank}"))?;
        if pairs.insert((left_id, right_id), rank).is_some() {
            return Err(format!("duplicate BPE merge pair at rank {rank}"));
        }
        merge_left.push(left_id);
        merge_right.push(right_id);
    }
    let mut special_ids = inventory.special_ids.clone();
    special_ids.sort_unstable();
    let metadata_entries = [
        TokenizerEntry::new(1, true, 200, 0),
        TokenizerEntry::new(2, true, 201, 0),
        TokenizerEntry::new(6, false, 209, 0),
        TokenizerEntry::new(8, false, 210, 0),
        TokenizerEntry::new(9, true, 202, 0),
        TokenizerEntry::new(10, true, 203, 0),
        TokenizerEntry::new(11, true, 204, 0),
        TokenizerEntry::new(12, true, 205, 0),
        TokenizerEntry::new(13, true, 206, 0),
        TokenizerEntry::new(14, false, 211, 0),
        TokenizerEntry::new(15, false, 207, 0),
        TokenizerEntry::new(16, false, 208, 0),
        TokenizerEntry::new(17, false, 212, 0),
        TokenizerEntry::new(18, false, 213, 0),
    ];
    let tokenizer_control =
        encode_tokenizer_metadata(TokenizerKind::Gpt2BpeQwen2, &metadata_entries)
            .map_err(|error| error.to_string())?;

    let model_entries = [
        MetadataEntry::new(1, true, 100, 0),
        MetadataEntry::new(2, true, 101, 0),
        MetadataEntry::new(3, true, 102, 0),
        MetadataEntry::new(4, true, 103, 0),
        MetadataEntry::new(5, true, 104, 0),
    ];
    let model_control = encode_model_metadata(&model_entries).map_err(|error| error.to_string())?;
    let tensor_control = encode_tensor_directory(&[]).map_err(|error| error.to_string())?;
    let bootstrap = encode_bootstrap(&[
        BootstrapEntry::new(RegionRole::TensorDirectory as u16, true, 12, 0),
        BootstrapEntry::new(RegionRole::ModelMetadata as u16, true, 10, 0),
        BootstrapEntry::new(RegionRole::TokenizerMetadata as u16, false, 11, 0),
    ])
    .map_err(|error| error.to_string())?;

    let mut writer = VBufV06Writer::new_known_size(Cursor::new(Vec::new()), 3)
        .map_err(|error| error.to_string())?;
    writer
        .write_opaque(BlockOptions::array(BOOTSTRAP_KEY_ID), &bootstrap)
        .map_err(|error| error.to_string())?;
    writer
        .write_opaque(BlockOptions::array(10), &model_control)
        .map_err(|error| error.to_string())?;
    writer
        .write_opaque(BlockOptions::array(12), &tensor_control)
        .map_err(|error| error.to_string())?;
    writer
        .write_opaque(BlockOptions::array(11), &tokenizer_control)
        .map_err(|error| error.to_string())?;
    writer
        .write_opaque(BlockOptions::array(100), b"qwen3_5")
        .map_err(|error| error.to_string())?;
    for (key, value) in [(101u16, 262_144u64), (102, 4096), (103, 32), (104, 16)] {
        writer
            .write_u64(BlockOptions::scalar(key), &[value])
            .map_err(|error| error.to_string())?;
    }
    writer
        .write_opaque(BlockOptions::array(200), &token_text)
        .map_err(|error| error.to_string())?;
    writer
        .write_u64(BlockOptions::array(201), &offsets)
        .map_err(|error| error.to_string())?;
    writer
        .write_u32(BlockOptions::array(202), &merge_left)
        .map_err(|error| error.to_string())?;
    writer
        .write_u32(BlockOptions::array(203), &merge_right)
        .map_err(|error| error.to_string())?;
    writer
        .write_u64(BlockOptions::scalar(204), &[TokenizerModel::Gpt2Bpe as u64])
        .map_err(|error| error.to_string())?;
    writer
        .write_u64(BlockOptions::scalar(205), &[PreTokenizer::Qwen3_5 as u64])
        .map_err(|error| error.to_string())?;
    writer
        .write_u8(BlockOptions::scalar(206), &[0])
        .map_err(|error| error.to_string())?;
    writer
        .write_u8(BlockOptions::array(207), &inventory.flags)
        .map_err(|error| error.to_string())?;
    writer
        .write_u32(BlockOptions::array(208), &special_ids)
        .map_err(|error| error.to_string())?;
    writer
        .write_u32(BlockOptions::scalar(209), &[tokenizer_eos])
        .map_err(|error| error.to_string())?;
    writer
        .write_u32(BlockOptions::scalar(210), &[tokenizer_pad])
        .map_err(|error| error.to_string())?;
    writer
        .write_opaque(BlockOptions::array(211), chat_template)
        .map_err(|error| error.to_string())?;
    writer
        .write_u8(BlockOptions::scalar(212), &[0])
        .map_err(|error| error.to_string())?;
    writer
        .write_u64(
            BlockOptions::scalar(213),
            &[TokenizerNormalizer::UnicodeNfc as u64],
        )
        .map_err(|error| error.to_string())?;
    let bytes = writer
        .finish()
        .map_err(|error| error.to_string())?
        .into_inner();
    Ok(bytes)
}

pub fn load_and_verify_sidecar<'a>(
    sidecar: &'a [u8],
    manifest: &Qwen35TokenizerManifest,
) -> Result<(crate::bootstrap::Bootstrap<'a>, TokenizerMetadata<'a>), String> {
    let validated =
        parse_v06(sidecar).map_err(|error| format!("invalid canonical v0.6 sidecar: {error}"))?;
    let bootstrap = Bootstrap::discover(&validated).map_err(|error| error.to_string())?;
    let model = ModelMetadata::parse(&validated, &bootstrap).map_err(|error| error.to_string())?;
    let directory =
        TensorDirectory::parse(&validated, &bootstrap).map_err(|error| error.to_string())?;
    if model.architecture() != Some("qwen3_5") || !directory.tensors().is_empty() {
        return Err(
            "tokenizer-only envelope model identity or empty tensor directory is invalid"
                .to_owned(),
        );
    }
    let metadata =
        TokenizerMetadata::parse(&validated, &bootstrap).map_err(|error| error.to_string())?;
    verify_sidecar_identity(sidecar, manifest, &metadata)?;
    Gpt2ByteLevelTokenizer::build(&metadata).map_err(|error| error.to_string())?;
    // The returned metadata borrows `sidecar`; the local `validated` owns no
    // bytes and only supplies ranges, so the borrowed tokenizer ranges remain
    // valid for the caller's sidecar lifetime.
    Ok((bootstrap, metadata))
}

fn resolve_config_token(
    token: Option<&Value>,
    tokens: &[String],
    label: &str,
) -> Result<u32, String> {
    let value = token.ok_or_else(|| format!("{label} is absent"))?;
    let content = if let Some(value) = value.as_str() {
        value.to_owned()
    } else if let Some(object) = value.as_object() {
        object
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{label} object has no content"))?
            .to_owned()
    } else {
        return Err(format!("{label} has unsupported representation"));
    };
    tokens
        .iter()
        .position(|token| token == &content)
        .and_then(|id| u32::try_from(id).ok())
        .ok_or_else(|| format!("{label} {content:?} has no explicit token ID"))
}

fn required_source<'a>(
    sources: &'a BTreeMap<&str, Vec<u8>>,
    name: &str,
) -> Result<&'a [u8], String> {
    sources
        .get(name)
        .map(Vec::as_slice)
        .ok_or_else(|| format!("required pinned source {name} is absent"))
}

fn parse_json(bytes: &[u8], name: &str) -> Result<Value, String> {
    serde_json::from_slice(bytes).map_err(|error| format!("invalid pinned {name}: {error}"))
}

fn json_u64(value: Option<&Value>, name: &str) -> Result<u64, String> {
    value
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("pinned config field {name} is not an unsigned integer"))
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

pub fn read_manifest(path: &Path) -> Result<Qwen35TokenizerManifest, String> {
    let bytes = fs::read(path)
        .map_err(|error| format!("cannot read manifest {}: {error}", path.display()))?;
    let manifest: Qwen35TokenizerManifest = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid sidecar manifest: {error}"))?;
    validate_manifest(&manifest)?;
    Ok(manifest)
}

pub fn write_new_file(path: &Path, bytes: &[u8]) -> Result<(), String> {
    use std::io::Write as _;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("cannot create {}: {error}", path.display()))?;
    file.write_all(bytes)
        .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    file.sync_all()
        .map_err(|error| format!("cannot sync {}: {error}", path.display()))
}
