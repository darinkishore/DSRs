/*
Script to inspect prediction history from the persistent database.

Run with:
```
cargo run --example 07-inspect-history
```
*/

use anyhow::Result;
use dspy_rs::{
    ChatAdapter, LM, Predict, PredictionDb, Signature, configure, init_tracing, session_id,
};

#[derive(Signature, Clone, Debug)]
struct QA {
    #[input]
    question: String,

    #[output]
    answer: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing()?;

    let lm = LM::builder()
        .model("openai:gpt-4o-mini".to_string())
        .build()
        .await?;
    configure(lm, ChatAdapter);

    let predictor = Predict::<QA>::new();
    let output = predictor
        .call(QAInput {
            question: "What is the capital of France?".to_string(),
        })
        .await?
        .into_inner();
    println!("prediction: {:?}", output.answer);

    // Query the prediction database for recent history
    if let Some(db) = PredictionDb::global() {
        let recent = db.query_recent(5).unwrap_or_default();
        println!("\n--- Recent predictions ({}) ---", recent.len());
        for rec in &recent {
            println!(
                "  [{}] {} | {} | {} tokens | {}ms",
                rec.status, rec.signature_name, rec.model_name, rec.total_tokens, rec.duration_ms,
            );
        }

        // Query just this session
        let session = db.query_by_session(session_id()).unwrap_or_default();
        println!(
            "\n--- This session ({}) ---\n  {} prediction(s)",
            session_id(),
            session.len()
        );

        // Token usage summary
        let usage = db.token_usage_by_model().unwrap_or_default();
        println!("\n--- Token usage by model ---");
        for (model, prompt, completion, total) in &usage {
            println!(
                "  {model}: {total} total ({prompt} prompt + {completion} completion)"
            );
        }
    }

    Ok(())
}
