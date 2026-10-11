//! Source-level profile and bounded-planning contracts for Qwen3.5-9B text inference.
//!
//! This module does not load model weights, convert a checkpoint, or admit a
//! model to the GGML runtime. It records the exact hybrid text architecture
//! and provides overflow-checked context, prefill, and state-memory planning
//! so those contracts can be tested without changing the qualified Qwen3 path.

use serde::Deserialize;
use std::fmt;

pub const QWEN35_9B_CONTEXT_TOKENS: u32 = 262_144;

const EXPECTED_ARCHITECTURE: &str = "Qwen3_5ForConditionalGeneration";
const EXPECTED_MODEL_TYPE: &str = "qwen3_5";
const EXPECTED_TEXT_MODEL_TYPE: &str = "qwen3_5_text";
const EXPECTED_LAYER_COUNT: usize = 32;
const EXPECTED_FULL_ATTENTION_INTERVAL: u32 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen35LayerKind {
    LinearAttention,
    FullAttention,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen35TextProfile {
    context_tokens: u32,
    hidden_size: u32,
    layer_kinds: Vec<Qwen35LayerKind>,
    attention_heads: u32,
    kv_heads: u32,
    head_dimension: u32,
    intermediate_size: u32,
    vocabulary_size: u32,
    eos_token_id: u32,
    linear_key_heads: u32,
    linear_value_heads: u32,
    linear_key_head_dimension: u32,
    linear_value_head_dimension: u32,
    linear_convolution_kernel: u32,
    rope_theta: u64,
    partial_rotary_factor: f64,
    mrope_sections: [u32; 3],
    mtp_hidden_layers: u32,
    contains_vision_tower: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Qwen35PlanError {
    InvalidConfig(String),
    UnsupportedProfile(&'static str),
    ContextExceeded { requested: u64, capacity: u32 },
    InvalidPrompt,
    InvalidChunkSize,
    ArithmeticOverflow,
    InvalidElementSize,
}

impl fmt::Display for Qwen35PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(detail) => write!(f, "invalid Qwen3.5 config: {detail}"),
            Self::UnsupportedProfile(detail) => {
                write!(f, "unsupported Qwen3.5 text profile: {detail}")
            }
            Self::ContextExceeded {
                requested,
                capacity,
            } => write!(
                f,
                "requested context ({requested}) exceeds configured capacity ({capacity})"
            ),
            Self::InvalidPrompt => write!(f, "prompt and completion reserve must be nonzero"),
            Self::InvalidChunkSize => write!(f, "prefill chunk size must be nonzero"),
            Self::ArithmeticOverflow => write!(f, "Qwen3.5 planning arithmetic overflow"),
            Self::InvalidElementSize => write!(f, "state element sizes must be nonzero"),
        }
    }
}

impl std::error::Error for Qwen35PlanError {}

#[derive(Deserialize)]
struct HfConfig {
    model_type: String,
    architectures: Vec<String>,
    tie_word_embeddings: bool,
    image_token_id: u32,
    video_token_id: u32,
    vision_start_token_id: u32,
    vision_end_token_id: u32,
    text_config: HfTextConfig,
    vision_config: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct HfTextConfig {
    model_type: String,
    attention_bias: bool,
    attention_dropout: f64,
    attn_output_gate: bool,
    hidden_act: String,
    rms_norm_eps: f64,
    use_cache: bool,
    mtp_num_hidden_layers: u32,
    mtp_use_dedicated_embeddings: bool,
    max_position_embeddings: u32,
    hidden_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    intermediate_size: u32,
    vocab_size: u32,
    eos_token_id: u32,
    full_attention_interval: u32,
    layer_types: Vec<String>,
    linear_num_key_heads: u32,
    linear_num_value_heads: u32,
    linear_key_head_dim: u32,
    linear_value_head_dim: u32,
    linear_conv_kernel_dim: u32,
    dtype: String,
    mamba_ssm_dtype: String,
    rope_parameters: HfRopeParameters,
}

#[derive(Deserialize)]
struct HfRopeParameters {
    rope_type: String,
    rope_theta: u64,
    partial_rotary_factor: f64,
    mrope_interleaved: bool,
    mrope_section: [u32; 3],
}

impl Qwen35TextProfile {
    /// Parse and validate the exact Qwen3.5-9B text-profile dimensions from an
    /// HF config.json. A valid profile is not proof that a weight artifact or
    /// tokenizer is present or identity-verified.
    pub fn from_hf_config(config_json: &[u8]) -> Result<Self, Qwen35PlanError> {
        let config: HfConfig = serde_json::from_slice(config_json)
            .map_err(|error| Qwen35PlanError::InvalidConfig(error.to_string()))?;
        if config.model_type != EXPECTED_MODEL_TYPE
            || !config
                .architectures
                .iter()
                .any(|architecture| architecture == EXPECTED_ARCHITECTURE)
            || config.text_config.model_type != EXPECTED_TEXT_MODEL_TYPE
        {
            return Err(Qwen35PlanError::UnsupportedProfile(
                "HF architecture identity does not match Qwen3.5 text",
            ));
        }
        if config.tie_word_embeddings
            || config.image_token_id != 248_056
            || config.video_token_id != 248_057
            || config.vision_start_token_id != 248_053
            || config.vision_end_token_id != 248_054
            || config
                .vision_config
                .as_ref()
                .and_then(|vision| vision.get("model_type"))
                .and_then(serde_json::Value::as_str)
                != Some(EXPECTED_MODEL_TYPE)
        {
            return Err(Qwen35PlanError::UnsupportedProfile(
                "output embedding or multimodal token/config identity differs from the pinned profile",
            ));
        }
        let text = config.text_config;
        if text.dtype != "bfloat16" || text.mamba_ssm_dtype != "float32" {
            return Err(Qwen35PlanError::UnsupportedProfile(
                "weight or recurrent-state dtype differs from the pinned profile",
            ));
        }
        if text.attention_bias
            || text.attention_dropout != 0.0
            || !text.attn_output_gate
            || text.hidden_act != "silu"
            || text.rms_norm_eps != 1e-6
            || !text.use_cache
            || text.mtp_num_hidden_layers != 1
            || text.mtp_use_dedicated_embeddings
        {
            return Err(Qwen35PlanError::UnsupportedProfile(
                "attention, normalization, or MTP semantics differ from the pinned profile",
            ));
        }
        if text.max_position_embeddings != QWEN35_9B_CONTEXT_TOKENS
            || text.hidden_size != 4096
            || text.num_hidden_layers != EXPECTED_LAYER_COUNT as u32
            || text.num_attention_heads != 16
            || text.num_key_value_heads != 4
            || text.head_dim != 256
            || text.intermediate_size != 12_288
            || text.vocab_size != 248_320
            || text.eos_token_id != 248_044
            || text.full_attention_interval != EXPECTED_FULL_ATTENTION_INTERVAL
            || text.linear_num_key_heads != 16
            || text.linear_num_value_heads != 32
            || text.linear_key_head_dim != 128
            || text.linear_value_head_dim != 128
            || text.linear_conv_kernel_dim != 4
        {
            return Err(Qwen35PlanError::UnsupportedProfile(
                "one or more text architecture dimensions differ from Qwen3.5-9B",
            ));
        }
        if text.rope_parameters.rope_type != "default"
            || text.rope_parameters.rope_theta != 10_000_000
            || text.rope_parameters.partial_rotary_factor != 0.25
            || !text.rope_parameters.mrope_interleaved
            || text.rope_parameters.mrope_section != [11, 11, 10]
        {
            return Err(Qwen35PlanError::UnsupportedProfile(
                "RoPE/MRoPE parameters differ from Qwen3.5-9B",
            ));
        }
        if text.layer_types.len() != EXPECTED_LAYER_COUNT {
            return Err(Qwen35PlanError::UnsupportedProfile(
                "layer_types length differs from the 32-layer text stack",
            ));
        }
        let mut layer_kinds = Vec::with_capacity(text.layer_types.len());
        for (index, layer_type) in text.layer_types.iter().enumerate() {
            let expected_full_attention =
                (index + 1) % EXPECTED_FULL_ATTENTION_INTERVAL as usize == 0;
            let expected = if expected_full_attention {
                "full_attention"
            } else {
                "linear_attention"
            };
            if layer_type != expected {
                return Err(Qwen35PlanError::UnsupportedProfile(
                    "hybrid layer schedule is not three linear layers followed by full attention",
                ));
            }
            layer_kinds.push(if expected_full_attention {
                Qwen35LayerKind::FullAttention
            } else {
                Qwen35LayerKind::LinearAttention
            });
        }

        Ok(Self {
            context_tokens: text.max_position_embeddings,
            hidden_size: text.hidden_size,
            layer_kinds,
            attention_heads: text.num_attention_heads,
            kv_heads: text.num_key_value_heads,
            head_dimension: text.head_dim,
            intermediate_size: text.intermediate_size,
            vocabulary_size: text.vocab_size,
            eos_token_id: text.eos_token_id,
            linear_key_heads: text.linear_num_key_heads,
            linear_value_heads: text.linear_num_value_heads,
            linear_key_head_dimension: text.linear_key_head_dim,
            linear_value_head_dimension: text.linear_value_head_dim,
            linear_convolution_kernel: text.linear_conv_kernel_dim,
            rope_theta: text.rope_parameters.rope_theta,
            partial_rotary_factor: text.rope_parameters.partial_rotary_factor,
            mrope_sections: text.rope_parameters.mrope_section,
            mtp_hidden_layers: text.mtp_num_hidden_layers,
            contains_vision_tower: config.vision_config.is_some(),
        })
    }

    pub fn context_tokens(&self) -> u32 {
        self.context_tokens
    }

    pub fn hidden_size(&self) -> u32 {
        self.hidden_size
    }

    pub fn layer_kinds(&self) -> &[Qwen35LayerKind] {
        &self.layer_kinds
    }

    pub fn attention_heads(&self) -> u32 {
        self.attention_heads
    }

    pub fn kv_heads(&self) -> u32 {
        self.kv_heads
    }

    pub fn head_dimension(&self) -> u32 {
        self.head_dimension
    }

    pub fn intermediate_size(&self) -> u32 {
        self.intermediate_size
    }

    pub fn vocabulary_size(&self) -> u32 {
        self.vocabulary_size
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    pub fn linear_attention_head_dimensions(&self) -> (u32, u32, u32, u32) {
        (
            self.linear_key_heads,
            self.linear_key_head_dimension,
            self.linear_value_heads,
            self.linear_value_head_dimension,
        )
    }

    pub fn linear_value_head_dimension(&self) -> u32 {
        self.linear_value_head_dimension
    }

    pub fn linear_convolution_kernel(&self) -> u32 {
        self.linear_convolution_kernel
    }

    pub fn rope_parameters(&self) -> (u64, f64, [u32; 3]) {
        (
            self.rope_theta,
            self.partial_rotary_factor,
            self.mrope_sections,
        )
    }

    pub fn mtp_hidden_layers(&self) -> u32 {
        self.mtp_hidden_layers
    }

    pub fn contains_vision_tower(&self) -> bool {
        self.contains_vision_tower
    }

    pub fn linear_attention_layers(&self) -> usize {
        self.layer_kinds
            .iter()
            .filter(|kind| **kind == Qwen35LayerKind::LinearAttention)
            .count()
    }

    pub fn full_attention_layers(&self) -> usize {
        self.layer_kinds
            .iter()
            .filter(|kind| **kind == Qwen35LayerKind::FullAttention)
            .count()
    }

    /// Check the total reserved context, not just the prompt length.
    pub fn admit_request(
        &self,
        prompt_tokens: u32,
        completion_reserve: u32,
    ) -> Result<u32, Qwen35PlanError> {
        if prompt_tokens == 0 || completion_reserve == 0 {
            return Err(Qwen35PlanError::InvalidPrompt);
        }
        let total = prompt_tokens
            .checked_add(completion_reserve)
            .ok_or(Qwen35PlanError::ArithmeticOverflow)?;
        if total > self.context_tokens {
            return Err(Qwen35PlanError::ContextExceeded {
                requested: u64::from(total),
                capacity: self.context_tokens,
            });
        }
        Ok(total)
    }

    /// Plan bounded prompt chunks. This plans token ranges only; it does not
    /// execute model layers or claim that chunk activations fit a device.
    pub fn plan_prefill_chunks(
        &self,
        prompt_tokens: u32,
        completion_reserve: u32,
        max_chunk_tokens: u32,
    ) -> Result<Vec<std::ops::Range<u32>>, Qwen35PlanError> {
        self.admit_request(prompt_tokens, completion_reserve)?;
        if max_chunk_tokens == 0 {
            return Err(Qwen35PlanError::InvalidChunkSize);
        }
        let mut chunks = Vec::new();
        let mut start = 0u32;
        while start < prompt_tokens {
            let end = start + max_chunk_tokens.min(prompt_tokens - start);
            chunks.push(start..end);
            start = end;
        }
        Ok(chunks)
    }

    /// Estimate persistent attention and recurrent state at the requested
    /// reserved capacity. The result excludes weights, activations, allocator
    /// overhead, CUDA context, and backend scratch.
    pub fn estimate_state_bytes(
        &self,
        context_tokens: u32,
        kv_element_bytes: u32,
        recurrent_element_bytes: u32,
        convolution_element_bytes: u32,
    ) -> Result<Qwen35StateMemoryEstimate, Qwen35PlanError> {
        if context_tokens > self.context_tokens {
            return Err(Qwen35PlanError::ContextExceeded {
                requested: u64::from(context_tokens),
                capacity: self.context_tokens,
            });
        }
        if kv_element_bytes == 0 || recurrent_element_bytes == 0 || convolution_element_bytes == 0 {
            return Err(Qwen35PlanError::InvalidElementSize);
        }
        let full_layers = u64::try_from(self.full_attention_layers())
            .map_err(|_| Qwen35PlanError::ArithmeticOverflow)?;
        let linear_layers = u64::try_from(self.linear_attention_layers())
            .map_err(|_| Qwen35PlanError::ArithmeticOverflow)?;
        let kv_cache_bytes = checked_product(&[
            u64::from(context_tokens),
            full_layers,
            2,
            u64::from(self.kv_heads),
            u64::from(self.head_dimension),
            u64::from(kv_element_bytes),
        ])?;
        let recurrent_state_bytes = checked_product(&[
            linear_layers,
            u64::from(self.linear_value_heads),
            u64::from(self.linear_value_head_dimension),
            u64::from(self.linear_key_head_dimension),
            u64::from(recurrent_element_bytes),
        ])?;
        let key_channels = u64::from(self.linear_key_heads)
            .checked_mul(u64::from(self.linear_key_head_dimension))
            .ok_or(Qwen35PlanError::ArithmeticOverflow)?;
        let value_channels = u64::from(self.linear_value_heads)
            .checked_mul(u64::from(self.linear_value_head_dimension))
            .ok_or(Qwen35PlanError::ArithmeticOverflow)?;
        let convolution_channels = key_channels
            .checked_mul(2)
            .and_then(|keys| keys.checked_add(value_channels))
            .ok_or(Qwen35PlanError::ArithmeticOverflow)?;
        let convolution_state_bytes = checked_product(&[
            linear_layers,
            u64::from(self.linear_convolution_kernel - 1),
            convolution_channels,
            u64::from(convolution_element_bytes),
        ])?;
        let total_bytes = kv_cache_bytes
            .checked_add(recurrent_state_bytes)
            .and_then(|bytes| bytes.checked_add(convolution_state_bytes))
            .ok_or(Qwen35PlanError::ArithmeticOverflow)?;
        Ok(Qwen35StateMemoryEstimate {
            kv_cache_bytes,
            recurrent_state_bytes,
            convolution_state_bytes,
            total_bytes,
        })
    }

    /// Evaluate one device's free-memory budget. Multiple device capacities
    /// must not be summed here; placement is a separate runtime decision.
    pub fn fits_single_device(
        &self,
        free_bytes: u64,
        weights_bytes: u64,
        state_bytes: u64,
        scratch_bytes: u64,
        safety_margin_bytes: u64,
    ) -> bool {
        weights_bytes
            .checked_add(state_bytes)
            .and_then(|bytes| bytes.checked_add(scratch_bytes))
            .and_then(|bytes| bytes.checked_add(safety_margin_bytes))
            .is_some_and(|required| required <= free_bytes)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen35StateMemoryEstimate {
    pub kv_cache_bytes: u64,
    pub recurrent_state_bytes: u64,
    pub convolution_state_bytes: u64,
    pub total_bytes: u64,
}

fn checked_product(values: &[u64]) -> Result<u64, Qwen35PlanError> {
    values
        .iter()
        .try_fold(1u64, |product, value| product.checked_mul(*value))
        .ok_or(Qwen35PlanError::ArithmeticOverflow)
}

#[cfg(test)]
mod tests {
    use super::{QWEN35_9B_CONTEXT_TOKENS, Qwen35LayerKind, Qwen35PlanError, Qwen35TextProfile};
    use sha2::{Digest, Sha256};

    const CONFIG: &str = r#"{
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "tie_word_embeddings": false,
        "image_token_id": 248056,
        "video_token_id": 248057,
        "vision_start_token_id": 248053,
        "vision_end_token_id": 248054,
        "text_config": {
            "model_type": "qwen3_5_text",
            "attention_bias": false,
            "attention_dropout": 0.0,
            "attn_output_gate": true,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-06,
            "use_cache": true,
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": false,
            "max_position_embeddings": 262144,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "intermediate_size": 12288,
            "vocab_size": 248320,
            "eos_token_id": 248044,
            "full_attention_interval": 4,
            "layer_types": [
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention"
            ],
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "dtype": "bfloat16",
            "mamba_ssm_dtype": "float32",
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
                "mrope_interleaved": true,
                "mrope_section": [11, 11, 10]
            }
        },
        "vision_config": {"model_type": "qwen3_5"}
    }"#;

    fn profile() -> Qwen35TextProfile {
        Qwen35TextProfile::from_hf_config(CONFIG.as_bytes()).unwrap()
    }

    #[test]
    fn parses_the_pinned_hugging_face_config_fixture() {
        let bytes = include_bytes!("../tests/fixtures/qwen35-9b-config.json");
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            "d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05"
        );
        let profile = Qwen35TextProfile::from_hf_config(bytes).unwrap();
        assert_eq!(profile.linear_attention_layers(), 24);
        assert_eq!(profile.full_attention_layers(), 8);
    }

    #[test]
    fn parses_the_qwen35_hybrid_schedule_and_text_profile() {
        let profile = profile();
        assert_eq!(profile.context_tokens(), QWEN35_9B_CONTEXT_TOKENS);
        assert_eq!(profile.layer_kinds().len(), 32);
        assert_eq!(profile.linear_attention_layers(), 24);
        assert_eq!(profile.full_attention_layers(), 8);
        assert_eq!(profile.layer_kinds()[3], Qwen35LayerKind::FullAttention);
        assert_eq!(profile.layer_kinds()[4], Qwen35LayerKind::LinearAttention);
        assert!(profile.contains_vision_tower());
    }

    #[test]
    fn context_admission_includes_reserved_completion_tokens() {
        let profile = profile();
        assert_eq!(
            profile.admit_request(253_952, 8_192).unwrap(),
            QWEN35_9B_CONTEXT_TOKENS
        );
        assert!(matches!(
            profile.admit_request(253_953, 8_192),
            Err(Qwen35PlanError::ContextExceeded { .. })
        ));
        assert!(matches!(
            profile.admit_request(u32::MAX, 1),
            Err(Qwen35PlanError::ArithmeticOverflow)
        ));
        assert!(matches!(
            profile.admit_request(0, 1),
            Err(Qwen35PlanError::InvalidPrompt)
        ));
    }

    #[test]
    fn prefill_chunks_are_bounded_and_cover_the_prompt_once() {
        let chunks = profile().plan_prefill_chunks(10_001, 4_096, 4_096).unwrap();
        assert_eq!(chunks, vec![0..4_096, 4_096..8_192, 8_192..10_001]);
        assert!(chunks.iter().all(|chunk| chunk.len() <= 4_096));
        assert!(profile().plan_prefill_chunks(10, 1, 0).is_err());
        let full_capacity_chunks = profile()
            .plan_prefill_chunks(253_952, 8_192, 4_096)
            .unwrap();
        assert_eq!(full_capacity_chunks.len(), 62);
        assert_eq!(full_capacity_chunks.last(), Some(&(249_856..253_952)));
    }

    #[test]
    fn state_estimate_separates_context_growth_from_recurrent_state() {
        let profile = profile();
        let estimate = profile
            .estimate_state_bytes(QWEN35_9B_CONTEXT_TOKENS, 2, 4, 4)
            .unwrap();
        assert_eq!(estimate.kv_cache_bytes, 8 * 1024 * 1024 * 1024);
        assert_eq!(estimate.recurrent_state_bytes, 48 * 1024 * 1024);
        assert_eq!(estimate.convolution_state_bytes, 2_359_296);
        assert_eq!(
            estimate.total_bytes,
            8 * 1024 * 1024 * 1024 + 48 * 1024 * 1024 + 2_359_296
        );
        assert!(matches!(
            profile.estimate_state_bytes(262_145, 2, 4, 4),
            Err(Qwen35PlanError::ContextExceeded { .. })
        ));
        assert!(matches!(
            profile.estimate_state_bytes(QWEN35_9B_CONTEXT_TOKENS, u32::MAX, u32::MAX, u32::MAX,),
            Err(Qwen35PlanError::ArithmeticOverflow)
        ));
        assert!(profile.fits_single_device(
            estimate.total_bytes + 1,
            0,
            estimate.total_bytes,
            0,
            1
        ));
        assert!(!profile.fits_single_device(estimate.total_bytes, 0, estimate.total_bytes, 0, 1));
        assert!(!profile.fits_single_device(u64::MAX, u64::MAX, 1, 0, 0));
    }

    #[test]
    fn rejects_wrong_architecture_hybrid_schedule_and_dtype() {
        let wrong_model = CONFIG.replace("\"qwen3_5\"", "\"qwen3\"");
        assert!(matches!(
            Qwen35TextProfile::from_hf_config(wrong_model.as_bytes()),
            Err(Qwen35PlanError::UnsupportedProfile(_))
        ));
        let wrong_schedule = CONFIG.replacen(
            "\"linear_attention\", \"linear_attention\", \"linear_attention\", \"full_attention\"",
            "\"full_attention\", \"linear_attention\", \"linear_attention\", \"full_attention\"",
            1,
        );
        assert!(matches!(
            Qwen35TextProfile::from_hf_config(wrong_schedule.as_bytes()),
            Err(Qwen35PlanError::UnsupportedProfile(_))
        ));
        let wrong_dtype = CONFIG.replace("\"float32\"", "\"bfloat16\"");
        assert!(matches!(
            Qwen35TextProfile::from_hf_config(wrong_dtype.as_bytes()),
            Err(Qwen35PlanError::UnsupportedProfile(_))
        ));
    }
}
