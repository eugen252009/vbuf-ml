use serde_json::json;
use vbuf_ml::qwen35_chat::{Qwen35ChatOptions, account_qwen35_context, render_pinned_qwen35_chat};
use vbuf_ml::qwen35_tokenizer::{QWEN35_CHAT_TEMPLATE_SHA256, sha256_hex};

const TEMPLATE: &str = include_str!("fixtures/qwen35-9b-chat-template.jinja");

#[test]
fn pinned_template_identity_is_expected() {
    assert_eq!(sha256_hex(TEMPLATE.as_bytes()), QWEN35_CHAT_TEMPLATE_SHA256);
}

#[test]
fn context_accounting_accepts_exact_boundary_and_rejects_overflow() {
    let exact = account_qwen35_context(262_136, 8, 262_144).unwrap();
    assert_eq!(exact.total_reserved, 262_144);
    assert_eq!(exact.remaining, 0);

    assert!(account_qwen35_context(262_137, 8, 262_144).is_err());
    assert!(account_qwen35_context(1, 1, 262_145).is_err());
    assert!(account_qwen35_context(usize::MAX, 1, 262_144).is_err());
}

#[test]
fn template_rejects_untrusted_template_and_non_text_content() {
    let messages = [json!({"role":"user", "content":"hello"})];
    assert!(
        render_pinned_qwen35_chat(
            "{{ messages }}",
            &messages,
            None,
            Qwen35ChatOptions::default(),
        )
        .is_err()
    );

    let image = [json!({
        "role":"user",
        "content":[{"type":"image", "image_url":"not-loaded"}]
    })];
    let error = render_pinned_qwen35_chat(TEMPLATE, &image, None, Qwen35ChatOptions::default())
        .unwrap_err();
    assert!(error.contains("non-text image/video content"));
}
