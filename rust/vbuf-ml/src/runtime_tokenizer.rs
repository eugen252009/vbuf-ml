//! Runtime-local tokenizer indexes over validated canonical tokenizer views.
//!
//! These indexes are deliberately not part of the vBuf representation. They
//! borrow token bytes from `TokenizerMetadata` and own only lookup-table
//! structure. The backing mapping must outlive the indexes.

use crate::{
    MlError, MlErrorCode, PreTokenizer, TokenizerKind, TokenizerMetadata, TokenizerNormalizer,
};
use regex::Regex;
use std::collections::{HashMap, HashSet};
use unicode_normalization::UnicodeNormalization;

#[derive(Debug)]
pub struct TokenIndex<'a> {
    entries: HashMap<&'a [u8], u32>,
}

impl<'a> TokenIndex<'a> {
    pub fn build(tokenizer: &'a TokenizerMetadata<'a>) -> Result<Self, MlError> {
        let count = usize::try_from(tokenizer.token_count()).map_err(|_| {
            MlError::new(
                MlErrorCode::TokenizerArrayLengthMismatch,
                "token count exceeds host range",
            )
        })?;
        let mut entries = HashMap::with_capacity(count);
        for index in 0..tokenizer.token_count() {
            let bytes = tokenizer.token_bytes(index).ok_or_else(|| {
                MlError::new(
                    MlErrorCode::InvalidTokenText,
                    "validated token span is unavailable",
                )
            })?;
            // Match llama's last-ordinal-wins behavior for duplicate text.
            entries.insert(
                bytes,
                u32::try_from(index).map_err(|_| {
                    MlError::new(
                        MlErrorCode::TokenizerArrayLengthMismatch,
                        "token ID exceeds runtime index width",
                    )
                })?,
            );
        }
        Ok(Self { entries })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn lookup(&self, bytes: &[u8]) -> Option<u32> {
        self.entries.get(bytes).copied()
    }
    pub fn retained_key_bytes(&self) -> usize {
        self.entries.keys().map(|key| key.len()).sum()
    }
}

#[derive(Debug)]
pub struct MergeRankIndex {
    entries: HashMap<u64, u32>,
}

impl MergeRankIndex {
    pub fn build(tokenizer: &TokenizerMetadata<'_>) -> Result<Self, MlError> {
        let count = usize::try_from(tokenizer.merge_count()).map_err(|_| {
            MlError::new(
                MlErrorCode::InvalidMergeTable,
                "merge count exceeds host range",
            )
        })?;
        let mut entries = HashMap::with_capacity(count);
        for rank in 0..tokenizer.merge_count() {
            let (left, right) = tokenizer.merge_pair(rank).ok_or_else(|| {
                MlError::new(
                    MlErrorCode::InvalidMergeTable,
                    "validated merge pair is unavailable",
                )
            })?;
            let key = pack_pair(left, right)?;
            entries.insert(
                key,
                u32::try_from(rank).map_err(|_| {
                    MlError::new(
                        MlErrorCode::InvalidMergeTable,
                        "merge rank exceeds runtime index width",
                    )
                })?,
            );
        }
        Ok(Self { entries })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn lookup(&self, left: u64, right: u64) -> Option<u32> {
        pack_pair(left, right)
            .ok()
            .and_then(|key| self.entries.get(&key).copied())
    }
    pub fn retained_bytes(&self) -> usize {
        self.entries.len() * std::mem::size_of::<(u64, u32)>()
    }
}

fn pack_pair(left: u64, right: u64) -> Result<u64, MlError> {
    if left > u64::from(u32::MAX) || right > u64::from(u32::MAX) {
        return Err(MlError::new(
            MlErrorCode::InvalidMergeTable,
            "merge ID exceeds packed runtime index width",
        ));
    }
    Ok((left << 32) | right)
}

#[derive(Debug)]
pub struct RuntimeTokenizerIndexes<'a> {
    pub token: TokenIndex<'a>,
    pub merge: MergeRankIndex,
}

/// Runtime GPT-2 byte-level BPE view over persistent tokenizer ranges.
///
/// The regex and lookup maps are runtime-only; vocabulary, merge ranks, and
/// special-token identities remain borrowed from the canonical sidecar.
#[derive(Debug)]
pub struct Gpt2ByteLevelTokenizer<'a> {
    metadata: &'a TokenizerMetadata<'a>,
    indexes: RuntimeTokenizerIndexes<'a>,
    pretokenizer: Regex,
    added_tokens: Vec<(u32, &'a str)>,
    literal_token_ids: HashSet<u32>,
    normalizer: Option<TokenizerNormalizer>,
    qwen35_pretokenizer: bool,
}

impl<'a> Gpt2ByteLevelTokenizer<'a> {
    pub fn build(metadata: &'a TokenizerMetadata<'a>) -> Result<Self, MlError> {
        let supported_profile = matches!(
            (
                metadata.kind,
                metadata.pre_tokenizer(),
                metadata.normalizer()
            ),
            (
                TokenizerKind::Gpt2BpeByteLevel,
                Some(PreTokenizer::Gpt2ByteLevel),
                None
            ) | (TokenizerKind::Gpt2BpeQwen2, Some(PreTokenizer::Qwen2), None)
                | (
                    TokenizerKind::Gpt2BpeQwen2,
                    Some(PreTokenizer::Qwen3_5),
                    Some(TokenizerNormalizer::UnicodeNfc)
                )
        );
        if !supported_profile || metadata.model().is_none() {
            return Err(MlError::new(
                MlErrorCode::UnsupportedTokenizerKind,
                "tokenizer is not the persistent GPT-2 byte-level profile",
            ));
        }
        let pattern = match metadata.pre_tokenizer() {
            Some(PreTokenizer::Qwen3_5) => {
                r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+"
            }
            Some(PreTokenizer::Qwen2 | PreTokenizer::Gpt2ByteLevel) => {
                r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+"
            }
            _ => unreachable!("unsupported pre-tokenizer was rejected above"),
        };
        let pretokenizer = Regex::new(pattern).map_err(|_| {
            MlError::new(
                MlErrorCode::InvalidTokenizerIdentity,
                "GPT-2 regex is invalid",
            )
        })?;
        let qwen35_pretokenizer = metadata.pre_tokenizer() == Some(PreTokenizer::Qwen3_5);
        let special_ids: HashSet<u32> = metadata
            .special_ids()
            .unwrap_or_default()
            .into_iter()
            .filter_map(|id| u32::try_from(id).ok())
            .collect();
        let mut added_tokens = Vec::new();
        let mut literal_token_ids = HashSet::new();
        for index in 0..metadata.token_count() {
            let id = u32::try_from(index).map_err(|_| {
                MlError::new(
                    MlErrorCode::TokenizerArrayLengthMismatch,
                    "token ID exceeds runtime index width",
                )
            })?;
            let is_added = qwen35_pretokenizer
                && metadata
                    .token_flags(index)
                    .is_some_and(|flags| flags & 1 != 0);
            if (is_added || special_ids.contains(&id))
                && let Some(text) = metadata.token_text(index).filter(|text| !text.is_empty())
            {
                added_tokens.push((id, text));
                literal_token_ids.insert(id);
            }
        }
        added_tokens.sort_by_key(|entry| std::cmp::Reverse(entry.1.len()));
        Ok(Self {
            metadata,
            indexes: RuntimeTokenizerIndexes::build(metadata)?,
            pretokenizer,
            added_tokens,
            literal_token_ids,
            normalizer: metadata.normalizer(),
            qwen35_pretokenizer,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, MlError> {
        let mut output = Vec::new();
        let mut cursor = 0usize;
        if self.qwen35_pretokenizer {
            while cursor < text.len() {
                if let Some((id, length)) = self.added_token_at(text, cursor) {
                    output.push(id);
                    cursor += length;
                    continue;
                }
                let end = self
                    .next_added_token_start(text, cursor)
                    .unwrap_or(text.len());
                let plain = &text[cursor..end];
                let normalized = match self.normalizer {
                    Some(TokenizerNormalizer::UnicodeNfc) => plain.nfc().collect::<String>(),
                    None => plain.to_owned(),
                };
                self.encode_ordinary(&normalized, &mut output)?;
                cursor = end;
            }
        } else {
            while cursor < text.len() {
                if let Some((id, length)) = self.added_token_at(text, cursor) {
                    output.push(id);
                    cursor += length;
                    continue;
                }
                let piece = self.pretokenizer.find_at(text, cursor).ok_or_else(|| {
                    MlError::new(
                        MlErrorCode::InvalidTokenText,
                        "GPT-2 pre-tokenization made no progress",
                    )
                })?;
                if piece.start() != cursor {
                    return Err(MlError::new(
                        MlErrorCode::InvalidTokenText,
                        "GPT-2 pre-tokenizer left an uncovered input span",
                    ));
                }
                self.encode_piece(piece.as_str(), &mut output)?;
                cursor = piece.end();
            }
        }
        if self.metadata.add_bos() == Some(true)
            && let Some((_, bos)) = self
                .metadata
                .specials()
                .iter()
                .find(|(kind, _)| matches!(kind, crate::SpecialToken::Bos))
        {
            output.insert(
                0,
                u32::try_from(*bos).map_err(|_| {
                    MlError::new(
                        MlErrorCode::InvalidSpecialTokenId,
                        "BOS ID exceeds runtime width",
                    )
                })?,
            );
        }
        Ok(output)
    }

    pub fn decode_bytes(&self, tokens: &[u32]) -> Result<Vec<u8>, MlError> {
        let mut bytes = Vec::new();
        for token in tokens {
            let index = u64::from(*token);
            let text = self.metadata.token_text(index).ok_or_else(|| {
                MlError::new(
                    MlErrorCode::InvalidSpecialTokenId,
                    "token ID is outside vocabulary",
                )
            })?;
            if self.literal_token_ids.contains(token) {
                bytes.extend_from_slice(text.as_bytes());
            } else {
                for character in text.chars() {
                    bytes.push(unicode_to_byte(character).ok_or_else(|| {
                        MlError::new(
                            MlErrorCode::InvalidTokenText,
                            "token is not GPT-2 byte-level text",
                        )
                    })?);
                }
            }
        }
        Ok(bytes)
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String, MlError> {
        String::from_utf8(self.decode_bytes(tokens)?)
            .map_err(|_| MlError::new(MlErrorCode::InvalidTokenText, "decoded bytes are not UTF-8"))
    }

    fn added_token_at(&self, text: &str, cursor: usize) -> Option<(u32, usize)> {
        let remaining = text.get(cursor..)?;
        self.added_tokens
            .iter()
            .find(|(_, token)| remaining.starts_with(token))
            .map(|(id, token)| (*id, token.len()))
    }

    fn next_added_token_start(&self, text: &str, cursor: usize) -> Option<usize> {
        let remaining = text.get(cursor..)?;
        self.added_tokens
            .iter()
            .filter_map(|(_, token)| remaining.find(token).map(|offset| cursor + offset))
            .min()
    }

    fn encode_ordinary(&self, text: &str, output: &mut Vec<u32>) -> Result<(), MlError> {
        let mut cursor = 0usize;
        while cursor < text.len() {
            let piece = self.pretokenizer.find_at(text, cursor);
            let piece_end = match piece {
                Some(piece) if piece.start() == cursor => piece.end(),
                _ if self.qwen35_pretokenizer
                    && text[cursor..]
                        .chars()
                        .next()
                        .is_some_and(char::is_whitespace) =>
                {
                    self.qwen35_whitespace_end(text, cursor)
                }
                _ => {
                    return Err(MlError::new(
                        MlErrorCode::InvalidTokenText,
                        "GPT-2 pre-tokenizer left an uncovered input span",
                    ));
                }
            };
            self.encode_piece(&text[cursor..piece_end], output)?;
            cursor = piece_end;
        }
        Ok(())
    }

    fn encode_piece(&self, piece: &str, output: &mut Vec<u32>) -> Result<(), MlError> {
        let mut symbols = Vec::new();
        for byte in piece.as_bytes() {
            let symbol = byte_to_unicode(*byte);
            let mut symbol_bytes = [0u8; 4];
            let symbol_bytes = symbol.encode_utf8(&mut symbol_bytes).as_bytes();
            let id = self.indexes.token.lookup(symbol_bytes).ok_or_else(|| {
                MlError::new(MlErrorCode::InvalidTokenText, "byte token is absent")
            })?;
            symbols.push(id);
        }
        if !self.metadata.ignore_merges().unwrap_or(false) {
            symbols = self.merge_symbols(symbols)?;
        }
        output.extend(symbols);
        Ok(())
    }

    fn qwen35_whitespace_end(&self, text: &str, start: usize) -> usize {
        let mut end = start;
        let mut last_char_start = start;
        let mut count = 0usize;
        for (offset, character) in text[start..].char_indices() {
            if !character.is_whitespace() {
                break;
            }
            last_char_start = start + offset;
            end = start + offset + character.len_utf8();
            count += 1;
        }
        if end < text.len() && count > 1 {
            last_char_start
        } else {
            end
        }
    }

    fn merge_symbols(&self, mut symbols: Vec<u32>) -> Result<Vec<u32>, MlError> {
        while symbols.len() > 1 {
            let Some((position, _rank)) = symbols
                .windows(2)
                .enumerate()
                .filter_map(|(position, pair)| {
                    self.indexes
                        .merge
                        .lookup(u64::from(pair[0]), u64::from(pair[1]))
                        .map(|rank| (position, rank))
                })
                .min_by_key(|(_, rank)| *rank)
            else {
                break;
            };
            let left = self
                .metadata
                .token_bytes(u64::from(symbols[position]))
                .ok_or_else(|| {
                    MlError::new(MlErrorCode::InvalidTokenText, "merge-left token is absent")
                })?;
            let right = self
                .metadata
                .token_bytes(u64::from(symbols[position + 1]))
                .ok_or_else(|| {
                    MlError::new(MlErrorCode::InvalidTokenText, "merge-right token is absent")
                })?;
            let mut merged = Vec::with_capacity(left.len() + right.len());
            merged.extend_from_slice(left);
            merged.extend_from_slice(right);
            let id = self.indexes.token.lookup(&merged).ok_or_else(|| {
                MlError::new(MlErrorCode::InvalidTokenText, "merged token is absent")
            })?;
            symbols.splice(position..=position + 1, [id]);
        }
        Ok(symbols)
    }
}

fn byte_to_unicode(byte: u8) -> char {
    let mut extra = 0u32;
    for value in 0..=255u8 {
        if is_direct_byte(value) {
            if value == byte {
                return char::from(value);
            }
        } else {
            if value == byte {
                return char::from_u32(256 + extra).unwrap();
            }
            extra += 1;
        }
    }
    unreachable!()
}

fn unicode_to_byte(character: char) -> Option<u8> {
    let code = character as u32;
    if code <= u32::from(u8::MAX) && is_direct_byte(code as u8) {
        return Some(code as u8);
    }
    let mut extra = 0u32;
    for value in 0..=255u8 {
        if !is_direct_byte(value) {
            if 256 + extra == code {
                return Some(value);
            }
            extra += 1;
        }
    }
    None
}

fn is_direct_byte(value: u8) -> bool {
    (b'!'..=b'~').contains(&value)
        || (0xa1..=0xac).contains(&value)
        || (0xae..=0xff).contains(&value)
}

impl<'a> RuntimeTokenizerIndexes<'a> {
    pub fn build(tokenizer: &'a TokenizerMetadata<'a>) -> Result<Self, MlError> {
        Ok(Self {
            token: TokenIndex::build(tokenizer)?,
            merge: MergeRankIndex::build(tokenizer)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::pack_pair;

    #[test]
    fn packed_pair_is_bounded_and_ordinal_safe() {
        assert_eq!(pack_pair(1, 2).unwrap(), (1u64 << 32) | 2);
        assert!(pack_pair(u64::from(u32::MAX) + 1, 0).is_err());
    }
}
