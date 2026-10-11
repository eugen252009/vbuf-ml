use std::collections::HashSet;
use std::io::Cursor;

use serde::Deserialize;
use vbuf_core::v06::parse_v06;
use vbuf_core::writer::{BlockOptions, VBufV06Writer};
use vbuf_ml::bootstrap::{
    BOOTSTRAP_KEY_ID, Bootstrap, BootstrapEntry, encode_payload as encode_bootstrap,
};
use vbuf_ml::tokenizer::{
    PreTokenizer, TokenizerEntry, TokenizerKind, TokenizerModel, TokenizerNormalizer,
    encode_payload,
};
use vbuf_ml::{Gpt2ByteLevelTokenizer, MlErrorCode, SpecialToken, TokenizerMetadata};

fn tokenizer_file(
    payload: &[u8],
    pool: &[u8],
    offsets: &[u64],
    scores: &[f32],
    types: &[u8],
    eos: u32,
) -> Vec<u8> {
    let mut writer = VBufV06Writer::new_known_size(Cursor::new(Vec::new()), 3).unwrap();
    writer
        .write_opaque(
            BlockOptions::array(BOOTSTRAP_KEY_ID),
            &encode_bootstrap(&[
                BootstrapEntry::new(1, true, 12, 0),
                BootstrapEntry::new(2, true, 10, 0),
                BootstrapEntry::new(3, false, 11, 0),
            ])
            .unwrap(),
        )
        .unwrap();
    writer
        .write_opaque(BlockOptions::array(10), b"model")
        .unwrap();
    writer
        .write_opaque(BlockOptions::array(11), payload)
        .unwrap();
    writer
        .write_opaque(BlockOptions::array(12), b"directory")
        .unwrap();
    writer.write_opaque(BlockOptions::array(30), pool).unwrap();
    writer.write_u64(BlockOptions::array(31), offsets).unwrap();
    writer.write_f32(BlockOptions::array(32), scores).unwrap();
    writer.write_u8(BlockOptions::array(33), types).unwrap();
    writer.write_u32(BlockOptions::scalar(34), &[0]).unwrap();
    writer.write_u32(BlockOptions::scalar(35), &[eos]).unwrap();
    writer.write_u32(BlockOptions::array(36), &[0, 1]).unwrap();
    writer.write_u32(BlockOptions::array(37), &[1, 0]).unwrap();
    writer.write_u8(BlockOptions::scalar(38), &[1]).unwrap();
    writer.write_u8(BlockOptions::scalar(39), &[1]).unwrap();
    writer.write_u8(BlockOptions::scalar(40), &[0]).unwrap();
    writer
        .write_opaque(BlockOptions::array(41), b"{{ messages }}")
        .unwrap();
    writer.finish().unwrap().into_inner()
}

fn gpt2_payload() -> Vec<u8> {
    encode_payload(
        TokenizerKind::Gpt2BpeQwen2,
        &[
            TokenizerEntry::new(1, true, 30, 0),
            TokenizerEntry::new(2, true, 31, 0),
            TokenizerEntry::new(4, false, 33, 0),
            TokenizerEntry::new(5, false, 34, 0),
            TokenizerEntry::new(6, false, 35, 0),
            TokenizerEntry::new(9, true, 36, 0),
            TokenizerEntry::new(10, true, 37, 0),
            TokenizerEntry::new(11, true, 38, 0),
            TokenizerEntry::new(12, true, 39, 0),
            TokenizerEntry::new(13, true, 40, 0),
            TokenizerEntry::new(14, false, 41, 0),
        ],
    )
    .unwrap()
}

fn valid_payload() -> Vec<u8> {
    encode_payload(
        TokenizerKind::VocabularyOnly,
        &[
            TokenizerEntry::new(1, true, 30, 0),
            TokenizerEntry::new(2, true, 31, 0),
            TokenizerEntry::new(3, false, 32, 0),
            TokenizerEntry::new(4, false, 33, 0),
            TokenizerEntry::new(5, false, 34, 0),
            TokenizerEntry::new(6, false, 35, 0),
        ],
    )
    .unwrap()
}

fn parse_tokenizer(
    payload: &[u8],
    pool: &[u8],
    offsets: &[u64],
    scores: &[f32],
    types: &[u8],
    eos: u32,
) -> Result<TokenizerMetadata<'static>, MlErrorCode> {
    let bytes: &'static [u8] =
        Box::leak(tokenizer_file(payload, pool, offsets, scores, types, eos).into_boxed_slice());
    let validated = parse_v06(bytes).map_err(|error| MlErrorCode::Canonical(error.code))?;
    let bootstrap = Bootstrap::discover(&validated).map_err(|error| error.code)?;
    TokenizerMetadata::parse(&validated, &bootstrap).map_err(|error| error.code)
}

#[test]
fn canonical_tokenizer_fixture_is_self_contained() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/minimal-tokenizer.vbuf");
    let bytes: &'static [u8] = Box::leak(std::fs::read(path).unwrap().into_boxed_slice());
    let validated = parse_v06(bytes).unwrap();
    let bootstrap = Bootstrap::discover(&validated).unwrap();
    let tokenizer = TokenizerMetadata::parse(&validated, &bootstrap).unwrap();
    assert_eq!(tokenizer.token_text(1), Some("world"));
}

#[test]
fn tokenizer_views_are_direct_canonical_ranges() {
    let tokenizer = parse_tokenizer(
        &valid_payload(),
        b"helloworld",
        &[0, 5, 10],
        &[0.5, -0.25],
        &[1, 2],
        1,
    )
    .unwrap();
    assert_eq!(tokenizer.kind, TokenizerKind::VocabularyOnly);
    assert_eq!(tokenizer.token_count(), 2);
    assert_eq!(tokenizer.token_text(0), Some("hello"));
    assert_eq!(tokenizer.token_text(1), Some("world"));
    assert_eq!(tokenizer.score(0), Some(0.5));
    assert_eq!(tokenizer.token_type(1), Some(2));
    assert!(tokenizer.specials().contains(&(SpecialToken::Bos, 0)));
    assert!(tokenizer.specials().contains(&(SpecialToken::Eos, 1)));
}

#[test]
fn gpt2_qwen2_descriptor_exposes_storage_without_execution() {
    let tokenizer = parse_tokenizer(
        &gpt2_payload(),
        b"helloworld",
        &[0, 5, 10],
        &[0.5, -0.25],
        &[1, 2],
        1,
    )
    .unwrap();
    assert_eq!(tokenizer.kind, TokenizerKind::Gpt2BpeQwen2);
    assert_eq!(tokenizer.model(), Some(TokenizerModel::Gpt2Bpe));
    assert_eq!(tokenizer.pre_tokenizer(), Some(PreTokenizer::Qwen2));
    assert_eq!(tokenizer.add_bos(), Some(false));
    assert_eq!(tokenizer.merge_count(), 2);
    assert_eq!(tokenizer.merge_pair(0), Some((0, 1)));
    assert_eq!(tokenizer.merge_pair(1), Some((1, 0)));
    assert_eq!(tokenizer.chat_template(), Some("{{ messages }}"));
}

#[test]
fn tokenizer_parallel_arrays_and_special_ids_are_checked() {
    assert_eq!(
        parse_tokenizer(
            &valid_payload(),
            b"helloworld",
            &[0, 5, 10],
            &[0.5],
            &[1, 2],
            1
        )
        .unwrap_err(),
        MlErrorCode::TokenizerArrayLengthMismatch
    );
    assert_eq!(
        parse_tokenizer(
            &valid_payload(),
            b"helloworld",
            &[0, 5, 10],
            &[0.5, -0.25],
            &[1, 2],
            2
        )
        .unwrap_err(),
        MlErrorCode::InvalidSpecialTokenId
    );
    assert_eq!(
        parse_tokenizer(
            &valid_payload(),
            b"hello\xff",
            &[0, 5, 6],
            &[0.5, -0.25],
            &[1, 2],
            1
        )
        .unwrap_err(),
        MlErrorCode::InvalidTokenText
    );
    assert_eq!(
        parse_tokenizer(
            &valid_payload(),
            b"helloworld",
            &[0, 6, 5],
            &[0.5, -0.25],
            &[1, 2],
            1
        )
        .unwrap_err(),
        MlErrorCode::InvalidTokenText
    );
}

#[test]
fn tokenizer_header_and_role_errors_fail_closed() {
    let mut unknown_kind = valid_payload();
    unknown_kind[12] = 99;
    assert_eq!(
        parse_tokenizer(
            &unknown_kind,
            b"helloworld",
            &[0, 5, 10],
            &[0.5, -0.25],
            &[1, 2],
            1
        )
        .unwrap_err(),
        MlErrorCode::UnsupportedTokenizerKind
    );

    let mut truncated = valid_payload();
    truncated.pop();
    assert_eq!(
        parse_tokenizer(
            &truncated,
            b"helloworld",
            &[0, 5, 10],
            &[0.5, -0.25],
            &[1, 2],
            1
        )
        .unwrap_err(),
        MlErrorCode::MalformedTokenizerMetadata
    );

    let duplicate = [
        TokenizerEntry::new(1, true, 30, 0),
        TokenizerEntry::new(1, true, 30, 0),
    ];
    assert_eq!(
        encode_payload(TokenizerKind::VocabularyOnly, &duplicate)
            .unwrap_err()
            .code,
        MlErrorCode::DuplicateTokenizerRole
    );
}

#[derive(Deserialize)]
struct Qwen35TokenizerFixture {
    tokenizer_json_sha256: String,
    tokenizer_config_sha256: String,
    reference_library: String,
    reference_version: String,
    model_config_eos_id: u32,
    tokenizer_config_eos_id: u32,
    unmapped_embedding_rows: u32,
    tokens: Vec<String>,
    original_ids: Vec<u32>,
    source_vocab_size: u32,
    model_embedding_vocab_size: u32,
    merges: Vec<[u32; 2]>,
    added_token_ids: Vec<u32>,
    special_token_ids: Vec<u32>,
    vectors: Vec<Qwen35TokenizerVector>,
}

#[derive(Deserialize)]
struct Qwen35TokenizerVector {
    text: String,
    ids: Vec<u32>,
    tokens: Vec<String>,
    decoded: String,
}

fn qwen35_tokenizer_file(
    fixture: &Qwen35TokenizerFixture,
    include_nfc_normalizer: bool,
) -> Vec<u8> {
    let mut writer = VBufV06Writer::new_known_size(Cursor::new(Vec::new()), 3).unwrap();
    let mut entries = vec![
        TokenizerEntry::new(1, true, 20, 0),
        TokenizerEntry::new(2, true, 21, 0),
        TokenizerEntry::new(9, true, 22, 0),
        TokenizerEntry::new(10, true, 23, 0),
        TokenizerEntry::new(11, true, 24, 0),
        TokenizerEntry::new(12, true, 25, 0),
        TokenizerEntry::new(13, true, 26, 0),
        TokenizerEntry::new(15, false, 27, 0),
        TokenizerEntry::new(16, false, 28, 0),
    ];
    if include_nfc_normalizer {
        entries.push(TokenizerEntry::new(18, false, 29, 0));
    }
    let payload = encode_payload(TokenizerKind::Gpt2BpeQwen2, &entries).unwrap();
    writer
        .write_opaque(
            BlockOptions::array(BOOTSTRAP_KEY_ID),
            &encode_bootstrap(&[
                BootstrapEntry::new(1, true, 12, 0),
                BootstrapEntry::new(2, true, 10, 0),
                BootstrapEntry::new(3, false, 11, 0),
            ])
            .unwrap(),
        )
        .unwrap();
    writer
        .write_opaque(BlockOptions::array(10), b"model")
        .unwrap();
    writer
        .write_opaque(BlockOptions::array(11), &payload)
        .unwrap();
    writer
        .write_opaque(BlockOptions::array(12), b"directory")
        .unwrap();

    let mut text_pool = Vec::new();
    let mut offsets = vec![0u64];
    let mut flags = vec![0u8; fixture.tokens.len()];
    let added: HashSet<usize> = fixture
        .added_token_ids
        .iter()
        .map(|id| usize::try_from(*id).unwrap())
        .collect();
    let special: HashSet<usize> = fixture
        .special_token_ids
        .iter()
        .map(|id| usize::try_from(*id).unwrap())
        .collect();
    for (index, token) in fixture.tokens.iter().enumerate() {
        text_pool.extend_from_slice(token.as_bytes());
        offsets.push(u64::try_from(text_pool.len()).unwrap());
        if added.contains(&index) {
            flags[index] |= 1;
        }
        if special.contains(&index) {
            flags[index] |= 2;
        }
    }
    let mut left = Vec::with_capacity(fixture.merges.len());
    let mut right = Vec::with_capacity(fixture.merges.len());
    for pair in &fixture.merges {
        left.push(pair[0]);
        right.push(pair[1]);
    }
    writer
        .write_opaque(BlockOptions::array(20), &text_pool)
        .unwrap();
    writer.write_u64(BlockOptions::array(21), &offsets).unwrap();
    writer.write_u32(BlockOptions::array(22), &left).unwrap();
    writer.write_u32(BlockOptions::array(23), &right).unwrap();
    writer
        .write_u64(BlockOptions::scalar(24), &[TokenizerModel::Gpt2Bpe as u64])
        .unwrap();
    writer
        .write_u64(BlockOptions::scalar(25), &[PreTokenizer::Qwen3_5 as u64])
        .unwrap();
    writer.write_u8(BlockOptions::scalar(26), &[0]).unwrap();
    writer.write_u8(BlockOptions::array(27), &flags).unwrap();
    writer
        .write_u32(BlockOptions::array(28), &fixture.special_token_ids)
        .unwrap();
    if include_nfc_normalizer {
        writer
            .write_u64(
                BlockOptions::scalar(29),
                &[TokenizerNormalizer::UnicodeNfc as u64],
            )
            .unwrap();
    }
    writer.finish().unwrap().into_inner()
}

fn parse_qwen35_tokenizer(bytes: Vec<u8>) -> Result<TokenizerMetadata<'static>, MlErrorCode> {
    let bytes: &'static [u8] = Box::leak(bytes.into_boxed_slice());
    let validated = parse_v06(bytes).map_err(|error| MlErrorCode::Canonical(error.code))?;
    let bootstrap = Bootstrap::discover(&validated).map_err(|error| error.code)?;
    TokenizerMetadata::parse(&validated, &bootstrap).map_err(|error| error.code)
}

#[test]
fn qwen35_tokenizer_matches_pinned_hugging_face_vectors() {
    let fixture: Qwen35TokenizerFixture =
        serde_json::from_slice(include_bytes!("fixtures/qwen35-tokenizer-vectors.json")).unwrap();
    assert_eq!(
        fixture.tokenizer_json_sha256,
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
    );
    assert_eq!(
        fixture.tokenizer_config_sha256,
        "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8"
    );
    assert_eq!(fixture.reference_library, "tokenizers");
    assert_eq!(fixture.reference_version, "0.22.2");
    assert_eq!(fixture.model_config_eos_id, 248_044);
    assert_eq!(fixture.tokenizer_config_eos_id, 248_046);
    assert_eq!(fixture.unmapped_embedding_rows, 250);
    assert_ne!(fixture.model_config_eos_id, fixture.tokenizer_config_eos_id);
    assert_eq!(fixture.source_vocab_size, 248_044);
    assert_eq!(fixture.model_embedding_vocab_size, 248_320);
    assert_eq!(fixture.added_token_ids.len(), 26);
    assert_eq!(fixture.special_token_ids.len(), 14);
    assert_eq!(fixture.tokens.len(), fixture.original_ids.len());
    let metadata = parse_qwen35_tokenizer(qwen35_tokenizer_file(&fixture, true)).unwrap();
    assert_eq!(metadata.pre_tokenizer(), Some(PreTokenizer::Qwen3_5));
    assert_eq!(metadata.normalizer(), Some(TokenizerNormalizer::UnicodeNfc));
    let tokenizer = Gpt2ByteLevelTokenizer::build(&metadata).unwrap();
    for vector in &fixture.vectors {
        let ids = tokenizer.encode(&vector.text).unwrap();
        let tokens: Vec<&str> = ids
            .iter()
            .map(|id| metadata.token_text(u64::from(*id)).unwrap())
            .collect();
        assert_eq!(tokens, vector.tokens, "input {:?}", vector.text);
        let original_ids: Vec<u32> = ids
            .iter()
            .map(|id| fixture.original_ids[*id as usize])
            .collect();
        assert_eq!(original_ids, vector.ids, "input {:?}", vector.text);
        assert_eq!(tokenizer.decode(&ids).unwrap(), vector.decoded);
    }
}

#[test]
fn qwen35_pre_tokenizer_without_nfc_metadata_is_rejected() {
    let fixture: Qwen35TokenizerFixture =
        serde_json::from_slice(include_bytes!("fixtures/qwen35-tokenizer-vectors.json")).unwrap();
    assert_eq!(
        parse_qwen35_tokenizer(qwen35_tokenizer_file(&fixture, false)).unwrap_err(),
        MlErrorCode::InvalidTokenizerIdentity
    );
}
