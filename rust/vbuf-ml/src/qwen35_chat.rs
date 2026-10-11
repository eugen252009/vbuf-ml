//! Text-only renderer for the pinned Qwen3.5-9B Jinja chat template.
//!
//! The template bytes are loaded from the validated tokenizer sidecar and
//! identity-checked before rendering. Inputs are JSON chat messages and JSON
//! tool schemas; image/video content and Python callable tools are deliberately
//! outside this text-only profile.

use crate::qwen35_text::QWEN35_9B_CONTEXT_TOKENS;
use crate::qwen35_tokenizer::{QWEN35_CHAT_TEMPLATE_SHA256, sha256_hex};
use crate::runtime_tokenizer::Gpt2ByteLevelTokenizer;
use crate::tokenizer::{PreTokenizer, TokenizerKind, TokenizerMetadata, TokenizerNormalizer};
use minijinja::value::ValueKind;
use minijinja::{Environment, Error, ErrorKind, State, Value};
use serde::Serialize;
use serde_json::{Map, Value as JsonValue};

const MAX_TEXT_CHAT_INPUT_BYTES: usize = 4 * 1024 * 1024;
const MAX_TEXT_CHAT_MESSAGES: usize = 4096;
const MAX_TEXT_CHAT_TOOLS: usize = 256;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Qwen35ChatOptions {
    pub add_generation_prompt: bool,
    pub add_vision_id: bool,
    /// `None` follows the template's default thinking-enabled behavior.
    pub enable_thinking: Option<bool>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct Qwen35ContextAccounting {
    pub prompt_tokens: usize,
    pub completion_reserve: usize,
    pub total_reserved: usize,
    pub context_limit: usize,
    pub remaining: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RenderedQwen35Chat {
    pub prompt: String,
    pub token_ids: Vec<u32>,
    pub context: Qwen35ContextAccounting,
}

/// Render the pinned text-only template against a validated Qwen3.5 tokenizer
/// metadata view. The exact source template hash is checked at every call.
pub fn render_qwen35_chat(
    tokenizer: &TokenizerMetadata<'_>,
    messages: &[JsonValue],
    tools: Option<&[JsonValue]>,
    options: Qwen35ChatOptions,
) -> Result<String, String> {
    if tokenizer.kind != TokenizerKind::Gpt2BpeQwen2
        || tokenizer.pre_tokenizer() != Some(PreTokenizer::Qwen3_5)
        || tokenizer.normalizer() != Some(TokenizerNormalizer::UnicodeNfc)
    {
        return Err("chat template requires the pinned Qwen3.5 tokenizer profile".to_owned());
    }
    let template = tokenizer
        .chat_template()
        .ok_or("validated tokenizer sidecar has no UTF-8 chat template")?;
    render_pinned_qwen35_chat(template, messages, tools, options)
}

/// Render a pre-extracted template only when its bytes match the pinned source.
pub fn render_pinned_qwen35_chat(
    template: &str,
    messages: &[JsonValue],
    tools: Option<&[JsonValue]>,
    options: Qwen35ChatOptions,
) -> Result<String, String> {
    if sha256_hex(template.as_bytes()) != QWEN35_CHAT_TEMPLATE_SHA256 {
        return Err("chat template source identity mismatch".to_owned());
    }
    validate_text_inputs(messages, tools)?;

    let mut env = Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.add_function(
        "raise_exception",
        |message: String| -> Result<String, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, message))
        },
    );
    env.add_filter("tojson", python_tojson);
    env.add_filter("string", python_string);
    env.set_unknown_method_callback(qwen35_string_method);
    env.add_template("qwen35", template)
        .map_err(|error| format!("cannot compile pinned Qwen3.5 chat template: {error}"))?;

    let mut context = Map::new();
    context.insert("messages".to_owned(), JsonValue::Array(messages.to_vec()));
    if let Some(tools) = tools {
        context.insert("tools".to_owned(), JsonValue::Array(tools.to_vec()));
    }
    context.insert(
        "add_generation_prompt".to_owned(),
        JsonValue::Bool(options.add_generation_prompt),
    );
    context.insert(
        "add_vision_id".to_owned(),
        JsonValue::Bool(options.add_vision_id),
    );
    if let Some(enable_thinking) = options.enable_thinking {
        context.insert(
            "enable_thinking".to_owned(),
            JsonValue::Bool(enable_thinking),
        );
    }
    let context = Value::from_serialize(&context);
    env.get_template("qwen35")
        .map_err(|error| format!("cannot load pinned Qwen3.5 chat template: {error}"))?
        .render(context)
        .map_err(|error| format!("Qwen3.5 chat rendering failed: {error}"))
}

/// Render and tokenize a complete prompt, then reserve the caller's requested
/// decode budget against the configured context. This counts actual tokenizer
/// IDs, including template and tool-serialization tokens; it does not infer a
/// generation length or perform inference.
pub fn render_and_account_qwen35_chat(
    tokenizer: &TokenizerMetadata<'_>,
    messages: &[JsonValue],
    tools: Option<&[JsonValue]>,
    options: Qwen35ChatOptions,
    context_limit: usize,
    completion_reserve: usize,
) -> Result<RenderedQwen35Chat, String> {
    account_qwen35_context(0, completion_reserve, context_limit)?;
    let prompt = render_qwen35_chat(tokenizer, messages, tools, options)?;
    let runtime = Gpt2ByteLevelTokenizer::build(tokenizer).map_err(|error| error.to_string())?;
    let token_ids = runtime.encode(&prompt).map_err(|error| error.to_string())?;
    let context = account_qwen35_context(token_ids.len(), completion_reserve, context_limit)?;
    Ok(RenderedQwen35Chat {
        prompt,
        token_ids,
        context,
    })
}

pub fn account_qwen35_context(
    prompt_tokens: usize,
    completion_reserve: usize,
    context_limit: usize,
) -> Result<Qwen35ContextAccounting, String> {
    if context_limit == 0 || context_limit > QWEN35_9B_CONTEXT_TOKENS as usize {
        return Err(format!(
            "context limit must be in 1..={QWEN35_9B_CONTEXT_TOKENS} tokens"
        ));
    }
    let total_reserved = prompt_tokens
        .checked_add(completion_reserve)
        .ok_or("prompt and completion token count overflow")?;
    if total_reserved > context_limit {
        return Err(format!(
            "prompt plus completion reserve ({total_reserved}) exceeds context limit ({context_limit})"
        ));
    }
    Ok(Qwen35ContextAccounting {
        prompt_tokens,
        completion_reserve,
        total_reserved,
        context_limit,
        remaining: context_limit - total_reserved,
    })
}

fn validate_text_inputs(messages: &[JsonValue], tools: Option<&[JsonValue]>) -> Result<(), String> {
    if messages.len() > MAX_TEXT_CHAT_MESSAGES {
        return Err(format!("chat exceeds {MAX_TEXT_CHAT_MESSAGES} messages"));
    }
    if tools.is_some_and(|tools| tools.len() > MAX_TEXT_CHAT_TOOLS) {
        return Err(format!("chat exceeds {MAX_TEXT_CHAT_TOOLS} tool schemas"));
    }
    let input_bytes = serde_json::to_vec(&(messages, tools))
        .map_err(|error| format!("cannot measure chat input: {error}"))?
        .len();
    if input_bytes > MAX_TEXT_CHAT_INPUT_BYTES {
        return Err(format!(
            "text chat JSON input exceeds {MAX_TEXT_CHAT_INPUT_BYTES} bytes"
        ));
    }
    if messages.is_empty() {
        return Err("chat requires at least one message".to_owned());
    }
    for (index, message) in messages.iter().enumerate() {
        let object = message
            .as_object()
            .ok_or_else(|| format!("message {index} must be a JSON object"))?;
        let role = object
            .get("role")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| format!("message {index} has no string role"))?;
        if !matches!(role, "system" | "user" | "assistant" | "tool") {
            return Err(format!("message {index} has unsupported role {role:?}"));
        }
        if let Some(content) = object.get("content") {
            validate_text_content(content, index)?;
        }
        if let Some(tool_calls) = object.get("tool_calls")
            && !tool_calls.is_null()
            && !tool_calls.is_array()
        {
            return Err(format!("message {index} tool_calls must be a JSON array"));
        }
    }
    if let Some(tools) = tools {
        for (index, tool) in tools.iter().enumerate() {
            if !tool.is_object() {
                return Err(format!("tool schema {index} must be a JSON object"));
            }
        }
    }
    Ok(())
}

fn validate_text_content(content: &JsonValue, message_index: usize) -> Result<(), String> {
    match content {
        JsonValue::Null | JsonValue::String(_) => Ok(()),
        JsonValue::Array(parts) => {
            for (part_index, part) in parts.iter().enumerate() {
                let object = part.as_object().ok_or_else(|| {
                    format!("message {message_index} content part {part_index} must be an object")
                })?;
                let part_type = object.get("type").and_then(JsonValue::as_str);
                if object.contains_key("image")
                    || object.contains_key("image_url")
                    || part_type == Some("image")
                    || object.contains_key("video")
                    || part_type == Some("video")
                {
                    return Err(format!(
                        "message {message_index} contains non-text image/video content"
                    ));
                }
                if ["audio", "audio_url", "input_audio", "frames"]
                    .iter()
                    .any(|key| object.contains_key(*key))
                    || part_type.is_some_and(|kind| kind != "text")
                {
                    return Err(format!(
                        "message {message_index} contains non-text audio or multimodal content"
                    ));
                }
                if object.get("text").and_then(JsonValue::as_str).is_none() {
                    return Err(format!(
                        "message {message_index} content part {part_index} is not a text block"
                    ));
                }
            }
            Ok(())
        }
        _ => Err(format!(
            "message {message_index} content must be text, null, or a text-block array"
        )),
    }
}

fn qwen35_string_method(
    _state: &State,
    value: &Value,
    method: &str,
    args: &[Value],
) -> Result<Value, Error> {
    if value.kind() != ValueKind::String {
        return Err(Error::from(ErrorKind::UnknownMethod));
    }
    let source = value.as_str().unwrap_or_default();
    let argument = || {
        args.first().and_then(Value::as_str).ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidOperation,
                "string method needs a string argument",
            )
        })
    };
    match method {
        "startswith" => Ok(Value::from(source.starts_with(argument()?))),
        "endswith" => Ok(Value::from(source.ends_with(argument()?))),
        "split" => {
            let separator = argument()?;
            let pieces: Vec<String> = source.split(separator).map(str::to_owned).collect();
            Ok(Value::from_serialize(pieces))
        }
        "rstrip" => {
            let chars = argument()?;
            Ok(Value::from(
                source.trim_end_matches(|character| chars.contains(character)),
            ))
        }
        "lstrip" => {
            let chars = argument()?;
            Ok(Value::from(
                source.trim_start_matches(|character| chars.contains(character)),
            ))
        }
        _ => Err(Error::from(ErrorKind::UnknownMethod)),
    }
}

fn python_tojson(value: Value) -> Result<String, Error> {
    let value = serde_json::to_value(value)
        .map_err(|error| Error::new(ErrorKind::InvalidOperation, error.to_string()))?;
    let compact = serde_json::to_string(&value)
        .map_err(|error| Error::new(ErrorKind::InvalidOperation, error.to_string()))?;
    Ok(python_json_spacing(&compact))
}

fn python_string(value: Value) -> Result<String, Error> {
    match value.kind() {
        ValueKind::Undefined => Ok("".to_owned()),
        ValueKind::None => Ok("None".to_owned()),
        ValueKind::Bool => Ok(if value.is_true() { "True" } else { "False" }.to_owned()),
        ValueKind::String => Ok(value.as_str().unwrap_or_default().to_owned()),
        ValueKind::Number => {
            let json = serde_json::to_value(value)
                .map_err(|error| Error::new(ErrorKind::InvalidOperation, error.to_string()))?;
            let JsonValue::Number(number) = json else {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    "expected JSON number",
                ));
            };
            Ok(python_number_string(&number))
        }
        _ => Err(Error::new(
            ErrorKind::InvalidOperation,
            "Qwen3.5 template scalar stringification received a non-scalar value",
        )),
    }
}

fn python_number_string(number: &serde_json::Number) -> String {
    if number.is_i64() || number.is_u64() {
        return number.to_string();
    }
    let Some(value) = number.as_f64() else {
        return number.to_string();
    };
    let mut result = format!("{value:?}");
    if let Some(exponent_index) = result.find('e') {
        let mantissa = result[..exponent_index].to_owned();
        let exponent = result[exponent_index + 1..].parse::<i32>().unwrap_or(0);
        // Python repr uses fixed notation for exponents in [-4, 15].
        if (-4..16).contains(&exponent) {
            result = expand_exponent(&mantissa, exponent);
        } else {
            let sign = if exponent < 0 { '-' } else { '+' };
            let digits = exponent.unsigned_abs();
            result = format!("{mantissa}e{sign}{digits:02}");
        }
    } else if !result.contains('.') {
        result.push_str(".0");
    }
    result
}

fn expand_exponent(mantissa: &str, exponent: i32) -> String {
    let negative = mantissa.starts_with('-');
    let unsigned = mantissa.strip_prefix('-').unwrap_or(mantissa);
    let (integer, fraction) = unsigned.split_once('.').unwrap_or((unsigned, ""));
    let digits = format!("{integer}{fraction}");
    let decimal_position = i32::try_from(integer.len()).unwrap_or(0) + exponent;
    let mut expanded = if decimal_position <= 0 {
        format!("0.{}{}", "0".repeat((-decimal_position) as usize), digits)
    } else if usize::try_from(decimal_position).unwrap_or(usize::MAX) >= digits.len() {
        format!(
            "{}{}",
            digits,
            "0".repeat(decimal_position as usize - digits.len())
        )
    } else {
        let split = usize::try_from(decimal_position).unwrap_or(0);
        format!("{}.{}", &digits[..split], &digits[split..])
    };
    if !expanded.contains('.') {
        expanded.push_str(".0");
    }
    if negative {
        expanded.insert(0, '-');
    }
    expanded
}

fn python_json_spacing(compact: &str) -> String {
    let mut output = String::with_capacity(compact.len().saturating_add(compact.len() / 8));
    let mut in_string = false;
    let mut escaped = false;
    for character in compact.chars() {
        output.push(character);
        if in_string {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                in_string = false;
            }
        } else if character == '"' {
            in_string = true;
        } else if matches!(character, ',' | ':') {
            output.push(' ');
        }
    }
    output
}
