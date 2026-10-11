use serde::Deserialize;
use serde_json::{Value as JsonValue, json};
use std::env;
use std::fs;
use std::path::PathBuf;
use vbuf_ml::Gpt2ByteLevelTokenizer;
use vbuf_ml::qwen35_chat::{
    Qwen35ChatOptions, account_qwen35_context, render_and_account_qwen35_chat,
};
use vbuf_ml::qwen35_tokenizer::{
    convert_pinned_directory, load_and_verify_sidecar, read_manifest, sha256_hex, write_new_file,
};

#[derive(Deserialize)]
struct VectorFile {
    vectors: Vec<TokenizerVector>,
}

#[derive(Deserialize)]
struct TokenizerVector {
    #[serde(default)]
    name: String,
    text: String,
    ids: Vec<u32>,
    tokens: Vec<String>,
    decoded: String,
}

#[derive(Clone, Deserialize, Default)]
struct ChatOptionsFile {
    #[serde(default)]
    add_generation_prompt: bool,
    #[serde(default)]
    add_vision_id: bool,
    enable_thinking: Option<bool>,
}

impl From<ChatOptionsFile> for Qwen35ChatOptions {
    fn from(value: ChatOptionsFile) -> Self {
        Self {
            add_generation_prompt: value.add_generation_prompt,
            add_vision_id: value.add_vision_id,
            enable_thinking: value.enable_thinking,
        }
    }
}

#[derive(Deserialize)]
struct ChatInput {
    messages: Vec<JsonValue>,
    #[serde(default)]
    tools: Option<Vec<JsonValue>>,
    #[serde(default)]
    options: ChatOptionsFile,
    context_limit: Option<usize>,
    completion_reserve: Option<usize>,
}

#[derive(Deserialize)]
struct ChatVectorFile {
    cases: Vec<ChatVector>,
    #[serde(default)]
    invalid_cases: Vec<InvalidChatVector>,
}

#[derive(Deserialize)]
struct ChatVector {
    name: String,
    messages: Vec<JsonValue>,
    #[serde(default)]
    tools: Option<Vec<JsonValue>>,
    #[serde(default)]
    options: ChatOptionsFile,
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

fn main() {
    if let Err(error) = run() {
        eprintln!("vbuf-ml-qwen35-tokenizer: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let command = args.next().ok_or(usage())?;
    let options = parse_options(args.collect())?;
    match command.as_str() {
        "convert" => convert(&options),
        "verify" => verify(&options),
        "check-vectors" => check_vectors(&options),
        "render-chat" => render_chat(&options),
        "check-chat-vectors" => check_chat_vectors(&options),
        _ => Err(usage()),
    }
}

fn convert(options: &std::collections::BTreeMap<String, String>) -> Result<(), String> {
    let source_dir = required_path(options, "source-dir")?;
    let output = required_path(options, "output")?;
    let manifest_path = required_path(options, "manifest")?;
    let converter_revision = options
        .get("tool-revision")
        .filter(|value| {
            value.len() == 40
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        })
        .ok_or("--tool-revision must be a full lowercase 40-character Git commit SHA")?;
    let converted = convert_pinned_directory(&source_dir, converter_revision)?;
    write_new_file(&output, &converted.sidecar)?;
    let manifest_bytes = serde_json::to_vec_pretty(&converted.manifest)
        .map_err(|error| format!("cannot serialize sidecar manifest: {error}"))?;
    write_new_file(&manifest_path, &[manifest_bytes.as_slice(), b"\n"].concat())?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "sidecar": output,
            "manifest": manifest_path,
            "size_bytes": converted.manifest.sidecar_size_bytes,
            "sha256": converted.manifest.sidecar_sha256,
            "tokens": converted.manifest.explicit_token_count,
            "tokenizer_json_added_tokens": converted.manifest.tokenizer_json_added_tokens,
            "tokenizer_config_only_added_tokens": converted.manifest.tokenizer_config_only_added_tokens,
            "merges": converted.manifest.merges,
        }))
        .map_err(|error| error.to_string())?
    );
    Ok(())
}

fn verify(options: &std::collections::BTreeMap<String, String>) -> Result<(), String> {
    let sidecar_path = required_path(options, "sidecar")?;
    let manifest_path = required_path(options, "manifest")?;
    let sidecar = fs::read(&sidecar_path)
        .map_err(|error| format!("cannot read sidecar {}: {error}", sidecar_path.display()))?;
    let manifest = read_manifest(&manifest_path)?;
    let (_bootstrap, metadata) = load_and_verify_sidecar(&sidecar, &manifest)?;
    let _runtime = Gpt2ByteLevelTokenizer::build(&metadata).map_err(|error| error.to_string())?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "status": "PASS",
            "sidecar_sha256": sha256_hex(&sidecar),
            "token_count": metadata.token_count(),
            "merge_count": metadata.merge_count(),
            "normalizer": format!("{:?}", metadata.normalizer()),
            "pre_tokenizer": format!("{:?}", metadata.pre_tokenizer()),
            "chat_template_bytes": metadata.chat_template().map_or(0, str::len),
        }))
        .map_err(|error| error.to_string())?
    );
    Ok(())
}

fn check_vectors(options: &std::collections::BTreeMap<String, String>) -> Result<(), String> {
    let sidecar_path = required_path(options, "sidecar")?;
    let manifest_path = required_path(options, "manifest")?;
    let vectors_path = required_path(options, "vectors")?;
    let sidecar = fs::read(&sidecar_path)
        .map_err(|error| format!("cannot read sidecar {}: {error}", sidecar_path.display()))?;
    let manifest = read_manifest(&manifest_path)?;
    let (_bootstrap, metadata) = load_and_verify_sidecar(&sidecar, &manifest)?;
    let runtime = Gpt2ByteLevelTokenizer::build(&metadata).map_err(|error| error.to_string())?;
    let fixture: VectorFile = serde_json::from_slice(
        &fs::read(&vectors_path)
            .map_err(|error| format!("cannot read vectors {}: {error}", vectors_path.display()))?,
    )
    .map_err(|error| format!("invalid vector fixture: {error}"))?;
    for (index, vector) in fixture.vectors.iter().enumerate() {
        let ids = runtime
            .encode(&vector.text)
            .map_err(|error| format!("vector {index} encode failed: {error}"))?;
        if ids != vector.ids {
            let first = ids
                .iter()
                .zip(&vector.ids)
                .position(|(actual, expected)| actual != expected)
                .unwrap_or(ids.len().min(vector.ids.len()));
            let start = first.saturating_sub(4);
            let end = first.saturating_add(5).min(ids.len().max(vector.ids.len()));
            return Err(format!(
                "vector {index} ({}) token-ID mismatch at {first}: actual {}, expected {}; actual window {:?}, expected window {:?}",
                vector.name,
                ids.get(first)
                    .map_or_else(|| "<end>".to_owned(), |id| id.to_string()),
                vector
                    .ids
                    .get(first)
                    .map_or_else(|| "<end>".to_owned(), |id| id.to_string()),
                ids.get(start..end).unwrap_or(&[]),
                vector.ids.get(start..end).unwrap_or(&[])
            ));
        }
        let token_strings: Vec<String> = ids
            .iter()
            .map(|id| {
                metadata
                    .token_text(u64::from(*id))
                    .map(str::to_owned)
                    .ok_or_else(|| format!("vector {index} emitted missing token ID {id}"))
            })
            .collect::<Result<_, _>>()?;
        if token_strings != vector.tokens {
            return Err(format!("vector {index} token-string mismatch"));
        }
        let decoded = runtime
            .decode(&ids)
            .map_err(|error| format!("vector {index} decode failed: {error}"))?;
        if decoded != vector.decoded {
            return Err(format!("vector {index} decoded-text mismatch"));
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "status": "PASS",
            "vectors": fixture.vectors.len(),
            "sidecar_sha256": sha256_hex(&sidecar),
        }))
        .map_err(|error| error.to_string())?
    );
    Ok(())
}

fn render_chat(options: &std::collections::BTreeMap<String, String>) -> Result<(), String> {
    let sidecar_path = required_path(options, "sidecar")?;
    let manifest_path = required_path(options, "manifest")?;
    let input_path = required_path(options, "input")?;
    let sidecar = fs::read(&sidecar_path)
        .map_err(|error| format!("cannot read sidecar {}: {error}", sidecar_path.display()))?;
    let manifest = read_manifest(&manifest_path)?;
    let (_bootstrap, metadata) = load_and_verify_sidecar(&sidecar, &manifest)?;
    let input: ChatInput =
        serde_json::from_slice(&fs::read(&input_path).map_err(|error| {
            format!("cannot read chat input {}: {error}", input_path.display())
        })?)
        .map_err(|error| format!("invalid chat input: {error}"))?;
    let context_limit = input.context_limit.unwrap_or(262_144);
    let completion_reserve = input.completion_reserve.unwrap_or(0);
    let rendered = render_and_account_qwen35_chat(
        &metadata,
        &input.messages,
        input.tools.as_deref(),
        input.options.into(),
        context_limit,
        completion_reserve,
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "rendered": rendered.prompt,
            "token_ids": rendered.token_ids,
            "context": rendered.context,
        }))
        .map_err(|error| error.to_string())?
    );
    Ok(())
}

fn check_chat_vectors(options: &std::collections::BTreeMap<String, String>) -> Result<(), String> {
    let sidecar_path = required_path(options, "sidecar")?;
    let manifest_path = required_path(options, "manifest")?;
    let fixture_path = required_path(options, "fixture")?;
    let sidecar = fs::read(&sidecar_path)
        .map_err(|error| format!("cannot read sidecar {}: {error}", sidecar_path.display()))?;
    let manifest = read_manifest(&manifest_path)?;
    let (_bootstrap, metadata) = load_and_verify_sidecar(&sidecar, &manifest)?;
    let fixture: ChatVectorFile =
        serde_json::from_slice(&fs::read(&fixture_path).map_err(|error| {
            format!(
                "cannot read chat fixture {}: {error}",
                fixture_path.display()
            )
        })?)
        .map_err(|error| format!("invalid chat fixture: {error}"))?;
    let runtime = Gpt2ByteLevelTokenizer::build(&metadata).map_err(|error| error.to_string())?;
    for case in &fixture.cases {
        let rendered = render_and_account_qwen35_chat(
            &metadata,
            &case.messages,
            case.tools.as_deref(),
            case.options.clone().into(),
            case.context_limit,
            case.completion_reserve,
        )
        .map_err(|error| format!("chat case {} failed: {error}", case.name))?;
        if rendered.prompt.as_bytes() != case.expected_rendered.as_bytes() {
            return Err(format!("chat case {} rendered bytes differ", case.name));
        }
        if rendered.token_ids != case.expected_ids {
            let first = rendered
                .token_ids
                .iter()
                .zip(&case.expected_ids)
                .position(|(actual, expected)| actual != expected)
                .unwrap_or(rendered.token_ids.len().min(case.expected_ids.len()));
            return Err(format!(
                "chat case {} token IDs differ at {first}: actual {:?}, expected {:?}",
                case.name,
                rendered.token_ids.get(first),
                case.expected_ids.get(first)
            ));
        }
        let token_strings: Vec<String> = rendered
            .token_ids
            .iter()
            .map(|id| {
                metadata
                    .token_text(u64::from(*id))
                    .map(str::to_owned)
                    .ok_or_else(|| format!("chat case {} emitted unmapped ID {id}", case.name))
            })
            .collect::<Result<_, _>>()?;
        if token_strings != case.expected_tokens {
            return Err(format!("chat case {} token strings differ", case.name));
        }
        let decoded = runtime
            .decode(&rendered.token_ids)
            .map_err(|error| format!("chat case {} decode failed: {error}", case.name))?;
        if decoded != case.expected_decoded {
            return Err(format!(
                "chat case {} decoded prompt differs: actual {:?}, expected {:?}",
                case.name, decoded, case.expected_decoded
            ));
        }
        let expected_remaining = account_qwen35_context(
            case.expected_ids.len(),
            case.completion_reserve,
            case.context_limit,
        )?
        .remaining;
        if rendered.context.prompt_tokens != case.expected_ids.len()
            || rendered.context.remaining != expected_remaining
            || rendered.context.remaining != case.expected_remaining
        {
            return Err(format!(
                "chat case {} context accounting differs",
                case.name
            ));
        }
    }
    for case in &fixture.invalid_cases {
        let result = render_qwen35_chat_for_invalid_case(&metadata, &case.messages);
        match result {
            Err(error) if error.contains(&case.expected_error) => {}
            Err(error) => {
                return Err(format!(
                    "invalid chat case {} failed with unexpected error: {error}",
                    case.name
                ));
            }
            Ok(_) => return Err(format!("invalid chat case {} was accepted", case.name)),
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "status": "PASS",
            "cases": fixture.cases.len(),
            "invalid_cases": fixture.invalid_cases.len(),
            "sidecar_sha256": sha256_hex(&sidecar),
        }))
        .map_err(|error| error.to_string())?
    );
    Ok(())
}

fn render_qwen35_chat_for_invalid_case(
    metadata: &vbuf_ml::TokenizerMetadata<'_>,
    messages: &[JsonValue],
) -> Result<String, String> {
    vbuf_ml::qwen35_chat::render_qwen35_chat(metadata, messages, None, Qwen35ChatOptions::default())
}

fn parse_options(args: Vec<String>) -> Result<std::collections::BTreeMap<String, String>, String> {
    let mut options = std::collections::BTreeMap::new();
    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        let key = arg
            .strip_prefix("--")
            .ok_or_else(|| format!("unexpected positional argument {arg:?}"))?;
        let value = iter
            .next()
            .ok_or_else(|| format!("missing value for --{key}"))?;
        if options.insert(key.to_owned(), value).is_some() {
            return Err(format!("duplicate option --{key}"));
        }
    }
    Ok(options)
}

fn required_path(
    options: &std::collections::BTreeMap<String, String>,
    key: &str,
) -> Result<PathBuf, String> {
    options
        .get(key)
        .map(PathBuf::from)
        .ok_or_else(|| format!("missing --{key}\n{}", usage()))
}

fn usage() -> String {
    "usage:\n  vbuf-ml-qwen35-tokenizer convert --source-dir DIR --output FILE --manifest FILE --tool-revision GIT_SHA\n  vbuf-ml-qwen35-tokenizer verify --sidecar FILE --manifest FILE\n  vbuf-ml-qwen35-tokenizer check-vectors --sidecar FILE --manifest FILE --vectors FILE\n  vbuf-ml-qwen35-tokenizer render-chat --sidecar FILE --manifest FILE --input JSON\n  vbuf-ml-qwen35-tokenizer check-chat-vectors --sidecar FILE --manifest FILE --fixture JSON".to_owned()
}
