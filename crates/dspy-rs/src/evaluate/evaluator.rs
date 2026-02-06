use crate::core::Module;
use crate::data::{example::Example, prediction::Prediction};
use futures::stream::{self, StreamExt};

#[allow(async_fn_in_trait)]
pub trait Evaluator: Module {
    const MAX_CONCURRENCY: usize = 32;
    const DISPLAY_PROGRESS: bool = true;

    async fn metric(&self, example: &Example, prediction: &Prediction) -> f32;

    async fn evaluate(&self, examples: Vec<Example>) -> f32 {
        let total = examples.len();

        let span = tracing::info_span!("evaluate", examples = total);
        let _enter = span.enter();

        let predictions = self
            .batch(
                examples.clone(),
                Self::MAX_CONCURRENCY,
                Self::DISPLAY_PROGRESS,
            )
            .await
            .unwrap();

        // Pair examples with predictions and evaluate with controlled concurrency
        let metrics: Vec<f32> = stream::iter(examples.iter().zip(predictions.iter()).enumerate())
            .map(|(_, (example, prediction))| {
                let prediction = prediction.clone();
                async move { self.metric(example, &prediction).await }
            })
            .buffer_unordered(Self::MAX_CONCURRENCY)
            .collect()
            .await;

        let score = metrics.iter().sum::<f32>() / total as f32;
        tracing::info!(score = score, examples = total, "evaluation complete");
        score
    }
}
