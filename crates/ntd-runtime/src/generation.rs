#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use ntd_ir::{Graph, ValueId};

use crate::{
    CognitiveSignals, ExecutionError, ExecutionProvider, GraphExecutor, Tensor, TensorError,
    TensorResolver, TextTokenizer, TokenizerError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistributionKind {
    Logits,
    Probabilities,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SamplingMode {
    Greedy,
    Stochastic {
        temperature: f32,
        top_k: Option<usize>,
        seed: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub context_limit: usize,
    pub eos_token: Option<u32>,
    pub distribution: DistributionKind,
    pub sampling: SamplingMode,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 32,
            context_limit: 2048,
            eos_token: None,
            distribution: DistributionKind::Logits,
            sampling: SamplingMode::Greedy,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationResult {
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub all_tokens: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationControl {
    Continue,
    Cancel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationFinishReason {
    MaxTokens,
    Eos,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamingGenerationResult {
    pub generation: GenerationResult,
    pub finish_reason: GenerationFinishReason,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelInferenceSignals {
    pub normalized_entropy: f32,
    pub top_probability: f32,
    pub top_margin: f32,
}

impl ModelInferenceSignals {
    pub fn cognitive_signals(
        self,
        prompt_tokens: usize,
        context_limit: usize,
        prior_failure: bool,
    ) -> CognitiveSignals {
        let context_pressure = if context_limit == 0 {
            1.0
        } else {
            (prompt_tokens as f32 / context_limit as f32).clamp(0.0, 1.0)
        };
        let uncertainty = (self.normalized_entropy * 0.75
            + (1.0 - self.top_margin).clamp(0.0, 1.0) * 0.25)
            .clamp(0.0, 1.0);
        CognitiveSignals {
            complexity: context_pressure,
            uncertainty,
            prior_failure,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedText {
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SamplingError {
    EmptyDistribution,
    NonFinite,
    NegativeProbability,
    ZeroMass,
    InvalidTemperature,
    InvalidTopK,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GenerationError {
    EmptyPrompt,
    EmptyCandidate,
    InvalidContextLimit,
    InvalidVocabulary,
    MissingDistributionOutput(usize),
    InvalidDistributionShape,
    TokenIdOutOfRange(u32),
    Tensor(TensorError),
    Execution(ExecutionError),
    Sampling(SamplingError),
    Tokenizer(TokenizerError),
}

pub struct GraphGenerator<P>
where
    P: ExecutionProvider,
{
    graph: Graph,
    executor: GraphExecutor<P>,
    static_inputs: BTreeMap<ValueId, Tensor>,
    token_input: ValueId,
    distribution_output: usize,
    vocab_size: usize,
}

impl<P> GraphGenerator<P>
where
    P: ExecutionProvider,
{
    pub fn new(
        graph: Graph,
        provider: P,
        static_inputs: BTreeMap<ValueId, Tensor>,
        token_input: ValueId,
        distribution_output: usize,
        vocab_size: usize,
    ) -> Result<Self, GenerationError> {
        if vocab_size == 0 {
            return Err(GenerationError::InvalidVocabulary);
        }

        Ok(Self {
            graph,
            executor: GraphExecutor::new(provider),
            static_inputs,
            token_input,
            distribution_output,
            vocab_size,
        })
    }

    pub fn next_distribution_with_resolver<R: TensorResolver>(
        &self,
        resolver: &R,
        prompt_tokens: &[u32],
        context_limit: usize,
    ) -> Result<Vec<f32>, GenerationError> {
        validate_prompt_tokens(prompt_tokens, self.vocab_size, context_limit)?;
        let start = prompt_tokens.len().saturating_sub(context_limit);
        let window = &prompt_tokens[start..];
        let token_tensor = Tensor::new(
            vec![window.len()],
            window.iter().map(|token| *token as f32).collect(),
        )
        .map_err(GenerationError::Tensor)?;

        let mut inputs = self.static_inputs.clone();
        inputs.insert(self.token_input, token_tensor);

        let outputs = self
            .executor
            .execute_with_resolver(&self.graph, inputs, resolver)
            .map_err(GenerationError::Execution)?;
        let output = outputs.get(self.distribution_output).ok_or(
            GenerationError::MissingDistributionOutput(self.distribution_output),
        )?;
        Ok(last_distribution(output, self.vocab_size)?.to_vec())
    }

    pub fn score_continuation_with_resolver<R: TensorResolver>(
        &self,
        resolver: &R,
        prompt_tokens: &[u32],
        candidate_tokens: &[u32],
        context_limit: usize,
    ) -> Result<f32, GenerationError> {
        validate_prompt_tokens(prompt_tokens, self.vocab_size, context_limit)?;
        if candidate_tokens.is_empty() {
            return Err(GenerationError::EmptyCandidate);
        }
        for token in candidate_tokens {
            let id =
                usize::try_from(*token).map_err(|_| GenerationError::TokenIdOutOfRange(*token))?;
            if id >= self.vocab_size {
                return Err(GenerationError::TokenIdOutOfRange(*token));
            }
        }

        let mut context = prompt_tokens.to_vec();
        let mut total_log_probability = 0.0f64;
        for token in candidate_tokens {
            let logits = self.next_distribution_with_resolver(resolver, &context, context_limit)?;
            total_log_probability += token_log_probability(
                &logits,
                usize::try_from(*token).map_err(|_| GenerationError::TokenIdOutOfRange(*token))?,
            )
            .map_err(GenerationError::Sampling)? as f64;
            context.push(*token);
        }

        Ok((total_log_probability / candidate_tokens.len() as f64) as f32)
    }

    pub fn generate_tokens(
        &self,
        prompt_tokens: &[u32],
        config: GenerationConfig,
    ) -> Result<GenerationResult, GenerationError> {
        if prompt_tokens.is_empty() {
            return Err(GenerationError::EmptyPrompt);
        }
        if config.context_limit == 0 {
            return Err(GenerationError::InvalidContextLimit);
        }

        for token in prompt_tokens {
            let id =
                usize::try_from(*token).map_err(|_| GenerationError::TokenIdOutOfRange(*token))?;
            if id >= self.vocab_size {
                return Err(GenerationError::TokenIdOutOfRange(*token));
            }
        }

        let mut all_tokens = prompt_tokens.to_vec();
        let mut generated_tokens = Vec::with_capacity(config.max_new_tokens);

        for step in 0..config.max_new_tokens {
            let start = all_tokens.len().saturating_sub(config.context_limit);
            let window = &all_tokens[start..];
            let token_tensor = Tensor::new(
                vec![window.len()],
                window.iter().map(|token| *token as f32).collect(),
            )
            .map_err(GenerationError::Tensor)?;

            let mut inputs = self.static_inputs.clone();
            inputs.insert(self.token_input, token_tensor);

            let outputs = self
                .executor
                .execute(&self.graph, inputs)
                .map_err(GenerationError::Execution)?;
            let output = outputs.get(self.distribution_output).ok_or(
                GenerationError::MissingDistributionOutput(self.distribution_output),
            )?;
            let distribution = last_distribution(output, self.vocab_size)?;
            let next = sample_token(distribution, config.distribution, config.sampling, step)
                .map_err(GenerationError::Sampling)?;

            let next_u32 = u32::try_from(next).map_err(|_| GenerationError::InvalidVocabulary)?;
            generated_tokens.push(next_u32);
            all_tokens.push(next_u32);

            if config.eos_token == Some(next_u32) {
                break;
            }
        }

        Ok(GenerationResult {
            prompt_tokens: prompt_tokens.to_vec(),
            generated_tokens,
            all_tokens,
        })
    }

    pub fn generate_tokens_with_resolver<R: TensorResolver>(
        &self,
        resolver: &R,
        prompt_tokens: &[u32],
        config: GenerationConfig,
    ) -> Result<GenerationResult, GenerationError> {
        if prompt_tokens.is_empty() {
            return Err(GenerationError::EmptyPrompt);
        }
        if config.context_limit == 0 {
            return Err(GenerationError::InvalidContextLimit);
        }

        for token in prompt_tokens {
            let id =
                usize::try_from(*token).map_err(|_| GenerationError::TokenIdOutOfRange(*token))?;
            if id >= self.vocab_size {
                return Err(GenerationError::TokenIdOutOfRange(*token));
            }
        }

        let mut all_tokens = prompt_tokens.to_vec();
        let mut generated_tokens = Vec::with_capacity(config.max_new_tokens);

        for step in 0..config.max_new_tokens {
            let start = all_tokens.len().saturating_sub(config.context_limit);
            let window = &all_tokens[start..];
            let token_tensor = Tensor::new(
                vec![window.len()],
                window.iter().map(|token| *token as f32).collect(),
            )
            .map_err(GenerationError::Tensor)?;

            let mut inputs = self.static_inputs.clone();
            inputs.insert(self.token_input, token_tensor);

            let outputs = self
                .executor
                .execute_with_resolver(&self.graph, inputs, resolver)
                .map_err(GenerationError::Execution)?;
            let output = outputs.get(self.distribution_output).ok_or(
                GenerationError::MissingDistributionOutput(self.distribution_output),
            )?;
            let distribution = last_distribution(output, self.vocab_size)?;
            let next = sample_token(distribution, config.distribution, config.sampling, step)
                .map_err(GenerationError::Sampling)?;

            let next_u32 = u32::try_from(next).map_err(|_| GenerationError::InvalidVocabulary)?;
            generated_tokens.push(next_u32);
            all_tokens.push(next_u32);

            if config.eos_token == Some(next_u32) {
                break;
            }
        }

        Ok(GenerationResult {
            prompt_tokens: prompt_tokens.to_vec(),
            generated_tokens,
            all_tokens,
        })
    }

    pub fn generate_tokens_with_resolver_streaming<R, F>(
        &self,
        resolver: &R,
        prompt_tokens: &[u32],
        config: GenerationConfig,
        mut on_token: F,
    ) -> Result<StreamingGenerationResult, GenerationError>
    where
        R: TensorResolver,
        F: FnMut(u32, &[u32]) -> GenerationControl,
    {
        if prompt_tokens.is_empty() {
            return Err(GenerationError::EmptyPrompt);
        }
        if config.context_limit == 0 {
            return Err(GenerationError::InvalidContextLimit);
        }

        for token in prompt_tokens {
            let id =
                usize::try_from(*token).map_err(|_| GenerationError::TokenIdOutOfRange(*token))?;
            if id >= self.vocab_size {
                return Err(GenerationError::TokenIdOutOfRange(*token));
            }
        }

        let mut all_tokens = prompt_tokens.to_vec();
        let mut generated_tokens = Vec::with_capacity(config.max_new_tokens);
        let mut finish_reason = GenerationFinishReason::MaxTokens;

        for step in 0..config.max_new_tokens {
            let start = all_tokens.len().saturating_sub(config.context_limit);
            let window = &all_tokens[start..];
            let token_tensor = Tensor::new(
                vec![window.len()],
                window.iter().map(|token| *token as f32).collect(),
            )
            .map_err(GenerationError::Tensor)?;

            let mut inputs = self.static_inputs.clone();
            inputs.insert(self.token_input, token_tensor);

            let outputs = self
                .executor
                .execute_with_resolver(&self.graph, inputs, resolver)
                .map_err(GenerationError::Execution)?;
            let output = outputs.get(self.distribution_output).ok_or(
                GenerationError::MissingDistributionOutput(self.distribution_output),
            )?;
            let distribution = last_distribution(output, self.vocab_size)?;
            let next = sample_token(distribution, config.distribution, config.sampling, step)
                .map_err(GenerationError::Sampling)?;

            let next_u32 = u32::try_from(next).map_err(|_| GenerationError::InvalidVocabulary)?;
            generated_tokens.push(next_u32);
            all_tokens.push(next_u32);
            let control = on_token(next_u32, &generated_tokens);

            if config.eos_token == Some(next_u32) {
                finish_reason = GenerationFinishReason::Eos;
                break;
            }
            if control == GenerationControl::Cancel {
                finish_reason = GenerationFinishReason::Cancelled;
                break;
            }
        }

        Ok(StreamingGenerationResult {
            generation: GenerationResult {
                prompt_tokens: prompt_tokens.to_vec(),
                generated_tokens,
                all_tokens,
            },
            finish_reason,
        })
    }

    pub fn generate_text_with_resolver<T: TextTokenizer, R: TensorResolver>(
        &self,
        resolver: &R,
        tokenizer: &T,
        prompt: &str,
        add_special_tokens: bool,
        mut config: GenerationConfig,
    ) -> Result<GeneratedText, GenerationError> {
        if tokenizer.vocab_size() != self.vocab_size {
            return Err(GenerationError::InvalidVocabulary);
        }

        let prompt_tokens = tokenizer
            .encode_text(prompt, add_special_tokens)
            .map_err(GenerationError::Tokenizer)?;
        if config.eos_token.is_none() {
            config.eos_token = tokenizer.eos_token();
        }

        let generated = self.generate_tokens_with_resolver(resolver, &prompt_tokens, config)?;
        let text = tokenizer
            .decode_text(&generated.generated_tokens, true)
            .map_err(GenerationError::Tokenizer)?;

        Ok(GeneratedText {
            prompt_tokens: generated.prompt_tokens,
            generated_tokens: generated.generated_tokens,
            text,
        })
    }

    pub fn generate_text<T: TextTokenizer>(
        &self,
        tokenizer: &T,
        prompt: &str,
        add_special_tokens: bool,
        mut config: GenerationConfig,
    ) -> Result<GeneratedText, GenerationError> {
        if tokenizer.vocab_size() != self.vocab_size {
            return Err(GenerationError::InvalidVocabulary);
        }

        let prompt_tokens = tokenizer
            .encode_text(prompt, add_special_tokens)
            .map_err(GenerationError::Tokenizer)?;
        if config.eos_token.is_none() {
            config.eos_token = tokenizer.eos_token();
        }

        let generated = self.generate_tokens(&prompt_tokens, config)?;
        let text = tokenizer
            .decode_text(&generated.generated_tokens, true)
            .map_err(GenerationError::Tokenizer)?;

        Ok(GeneratedText {
            prompt_tokens: generated.prompt_tokens,
            generated_tokens: generated.generated_tokens,
            text,
        })
    }
}

fn token_log_probability(logits: &[f32], token: usize) -> Result<f32, SamplingError> {
    if logits.is_empty() || token >= logits.len() {
        return Err(SamplingError::EmptyDistribution);
    }
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(SamplingError::NonFinite);
    }

    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut mass = 0.0f64;
    for value in logits {
        mass += f64::from((*value - max).exp());
    }
    if !mass.is_finite() || mass <= 0.0 {
        return Err(SamplingError::ZeroMass);
    }

    Ok(logits[token] - max - (mass.ln() as f32))
}

pub fn model_inference_signals(logits: &[f32]) -> Result<ModelInferenceSignals, SamplingError> {
    if logits.is_empty() {
        return Err(SamplingError::EmptyDistribution);
    }
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(SamplingError::NonFinite);
    }

    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut weights = Vec::with_capacity(logits.len());
    let mut total = 0.0f32;
    for value in logits {
        let weight = (*value - max).exp();
        weights.push(weight);
        total += weight;
    }
    if !total.is_finite() || total <= 0.0 {
        return Err(SamplingError::ZeroMass);
    }

    let mut entropy = 0.0f32;
    let mut top = 0.0f32;
    let mut second = 0.0f32;
    for weight in weights {
        let probability = weight / total;
        if probability > 0.0 {
            entropy -= probability * probability.ln();
        }
        if probability > top {
            second = top;
            top = probability;
        } else if probability > second {
            second = probability;
        }
    }

    let normalized_entropy = if logits.len() <= 1 {
        0.0
    } else {
        (entropy / (logits.len() as f32).ln()).clamp(0.0, 1.0)
    };
    Ok(ModelInferenceSignals {
        normalized_entropy,
        top_probability: top.clamp(0.0, 1.0),
        top_margin: (top - second).clamp(0.0, 1.0),
    })
}

pub fn sample_token(
    values: &[f32],
    kind: DistributionKind,
    mode: SamplingMode,
    step: usize,
) -> Result<usize, SamplingError> {
    if values.is_empty() {
        return Err(SamplingError::EmptyDistribution);
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(SamplingError::NonFinite);
    }

    if matches!(mode, SamplingMode::Greedy) {
        return values
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                left.total_cmp(right)
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index)
            .ok_or(SamplingError::EmptyDistribution);
    }

    let SamplingMode::Stochastic {
        temperature,
        top_k,
        seed,
    } = mode
    else {
        return Err(SamplingError::EmptyDistribution);
    };

    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(SamplingError::InvalidTemperature);
    }
    if matches!(top_k, Some(0)) {
        return Err(SamplingError::InvalidTopK);
    }

    let mut weights = match kind {
        DistributionKind::Logits => logits_to_weights(values, temperature),
        DistributionKind::Probabilities => probabilities_to_weights(values, temperature)?,
    };

    if let Some(k) = top_k {
        let limit = k.min(weights.len());
        keep_top_k(&mut weights, limit)?;
    }

    let sum = weights.iter().sum::<f32>();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(SamplingError::ZeroMass);
    }

    let random = deterministic_unit(seed, step);
    let mut target = random * sum;
    let mut fallback = None;

    for (index, weight) in weights.iter().enumerate() {
        if *weight <= 0.0 {
            continue;
        }
        fallback = Some(index);
        if target <= *weight {
            return Ok(index);
        }
        target -= *weight;
    }

    fallback.ok_or(SamplingError::ZeroMass)
}

fn validate_prompt_tokens(
    prompt_tokens: &[u32],
    vocab_size: usize,
    context_limit: usize,
) -> Result<(), GenerationError> {
    if prompt_tokens.is_empty() {
        return Err(GenerationError::EmptyPrompt);
    }
    if context_limit == 0 {
        return Err(GenerationError::InvalidContextLimit);
    }
    for token in prompt_tokens {
        let id = usize::try_from(*token).map_err(|_| GenerationError::TokenIdOutOfRange(*token))?;
        if id >= vocab_size {
            return Err(GenerationError::TokenIdOutOfRange(*token));
        }
    }
    Ok(())
}

fn last_distribution(tensor: &Tensor, vocab_size: usize) -> Result<&[f32], GenerationError> {
    if tensor.data().len() < vocab_size || tensor.data().len() % vocab_size != 0 {
        return Err(GenerationError::InvalidDistributionShape);
    }
    let start = tensor.data().len() - vocab_size;
    Ok(&tensor.data()[start..])
}

fn logits_to_weights(values: &[f32], temperature: f32) -> Vec<f32> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    values
        .iter()
        .map(|value| ((*value - max) / temperature).exp())
        .collect()
}

fn probabilities_to_weights(values: &[f32], temperature: f32) -> Result<Vec<f32>, SamplingError> {
    if values.iter().any(|value| *value < 0.0) {
        return Err(SamplingError::NegativeProbability);
    }
    let exponent = 1.0 / temperature;
    Ok(values.iter().map(|value| value.powf(exponent)).collect())
}

fn keep_top_k(weights: &mut [f32], k: usize) -> Result<(), SamplingError> {
    if k == 0 {
        return Err(SamplingError::InvalidTopK);
    }

    let mut indices = (0..weights.len()).collect::<Vec<_>>();
    indices.sort_by(|left, right| {
        weights[*right]
            .total_cmp(&weights[*left])
            .then_with(|| left.cmp(right))
    });

    for index in indices.into_iter().skip(k) {
        weights[index] = 0.0;
    }
    Ok(())
}

fn deterministic_unit(seed: u64, step: usize) -> f32 {
    let mut value = seed
        .wrapping_add(u64::try_from(step).unwrap_or(u64::MAX))
        .wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;

    let mantissa = value >> 40;
    (mantissa as f32) / ((1u32 << 24) as f32)
}

#[derive(Debug, Clone, PartialEq)]
pub struct KvLayerCache {
    pub key: Tensor,
    pub value: Tensor,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct KvCache {
    layers: BTreeMap<u32, KvLayerCache>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvCacheError {
    InvalidRank,
    ShapeMismatch,
    Tensor,
}

impl KvCache {
    pub fn layer(&self, layer: u32) -> Option<&KvLayerCache> {
        self.layers.get(&layer)
    }

    pub fn append(&mut self, layer: u32, key: Tensor, value: Tensor) -> Result<(), KvCacheError> {
        validate_kv_pair(&key, &value)?;

        if let Some(existing) = self.layers.get_mut(&layer) {
            existing.key = concat_sequence(&existing.key, &key)?;
            existing.value = concat_sequence(&existing.value, &value)?;
        } else {
            self.layers.insert(layer, KvLayerCache { key, value });
        }
        Ok(())
    }

    pub fn truncate_left(&mut self, max_tokens: usize) -> Result<(), KvCacheError> {
        for cache in self.layers.values_mut() {
            cache.key = retain_last_sequence(&cache.key, max_tokens)?;
            cache.value = retain_last_sequence(&cache.value, max_tokens)?;
        }
        Ok(())
    }

    pub fn clear(&mut self) {
        self.layers.clear();
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct PrefixCache {
    pub tokens: Vec<u32>,
    pub kv: KvCache,
}

impl PrefixCache {
    pub fn reusable_prefix_len(&self, tokens: &[u32]) -> usize {
        self.tokens
            .iter()
            .zip(tokens.iter())
            .take_while(|(left, right)| left == right)
            .count()
    }
}

fn validate_kv_pair(key: &Tensor, value: &Tensor) -> Result<(), KvCacheError> {
    if key.shape().is_empty() || value.shape().is_empty() {
        return Err(KvCacheError::InvalidRank);
    }
    if key.shape()[0] != value.shape()[0] || key.shape()[1..] != value.shape()[1..] {
        return Err(KvCacheError::ShapeMismatch);
    }
    Ok(())
}

fn concat_sequence(left: &Tensor, right: &Tensor) -> Result<Tensor, KvCacheError> {
    if left.shape().is_empty()
        || right.shape().is_empty()
        || left.shape()[1..] != right.shape()[1..]
    {
        return Err(KvCacheError::ShapeMismatch);
    }

    let mut shape = left.shape().to_vec();
    shape[0] = shape[0]
        .checked_add(right.shape()[0])
        .ok_or(KvCacheError::Tensor)?;
    let mut data = Vec::with_capacity(
        left.data()
            .len()
            .checked_add(right.data().len())
            .ok_or(KvCacheError::Tensor)?,
    );
    data.extend_from_slice(left.data());
    data.extend_from_slice(right.data());
    Tensor::new(shape, data).map_err(|_| KvCacheError::Tensor)
}

fn retain_last_sequence(tensor: &Tensor, max_tokens: usize) -> Result<Tensor, KvCacheError> {
    if tensor.shape().is_empty() {
        return Err(KvCacheError::InvalidRank);
    }
    let sequence = tensor.shape()[0];
    if sequence <= max_tokens {
        return Ok(tensor.clone());
    }

    let row_width = tensor.data().len() / sequence;
    let keep = max_tokens;
    let start = sequence
        .checked_sub(keep)
        .and_then(|value| value.checked_mul(row_width))
        .ok_or(KvCacheError::Tensor)?;
    let mut shape = tensor.shape().to_vec();
    shape[0] = keep;
    Tensor::new(shape, tensor.data()[start..].to_vec()).map_err(|_| KvCacheError::Tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ntd_ir::{DType, IrVersion, Node, NodeId, OpKind, TensorOp, ValueDecl, ValueType};

    fn tensor_decl(id: u32, dtype: DType, rank: u8) -> ValueDecl {
        ValueDecl {
            id: ValueId(id),
            ty: ValueType::Tensor { dtype, rank },
        }
    }

    fn transition_graph() -> Graph {
        Graph {
            version: IrVersion::CURRENT,
            inputs: vec![tensor_decl(0, DType::I32, 1), tensor_decl(1, DType::F32, 2)],
            outputs: vec![ValueId(3)],
            nodes: vec![
                Node {
                    id: NodeId(0),
                    op: OpKind::Tensor(TensorOp::Gather),
                    inputs: vec![ValueId(1), ValueId(0)],
                    outputs: vec![tensor_decl(2, DType::F32, 2)],
                },
                Node {
                    id: NodeId(1),
                    op: OpKind::Tensor(TensorOp::Softmax),
                    inputs: vec![ValueId(2)],
                    outputs: vec![tensor_decl(3, DType::F32, 2)],
                },
            ],
        }
    }

    #[test]
    fn graph_generator_advances_autoregressively() {
        let transitions = Tensor::new(
            vec![4, 4],
            vec![
                0.0, 10.0, 0.0, 0.0, //
                0.0, 0.0, 10.0, 0.0, //
                0.0, 0.0, 0.0, 10.0, //
                0.0, 0.0, 0.0, 10.0,
            ],
        )
        .expect("transitions");

        let mut static_inputs = BTreeMap::new();
        static_inputs.insert(ValueId(1), transitions);

        let generator = GraphGenerator::new(
            transition_graph(),
            crate::CpuReferenceProvider,
            static_inputs,
            ValueId(0),
            0,
            4,
        )
        .expect("generator");

        let result = generator
            .generate_tokens(
                &[0],
                GenerationConfig {
                    max_new_tokens: 8,
                    context_limit: 4,
                    eos_token: Some(3),
                    distribution: DistributionKind::Probabilities,
                    sampling: SamplingMode::Greedy,
                },
            )
            .expect("generate");

        assert_eq!(result.generated_tokens, vec![1, 2, 3]);
    }

    #[test]
    fn continuation_scoring_prefers_higher_native_probability() {
        let transitions = Tensor::new(
            vec![4, 4],
            vec![
                0.0, 10.0, 1.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0,
            ],
        )
        .expect("transitions");
        let mut static_inputs = BTreeMap::new();
        static_inputs.insert(ValueId(1), transitions);
        let generator = GraphGenerator::new(
            transition_graph(),
            crate::CpuReferenceProvider,
            static_inputs,
            ValueId(0),
            0,
            4,
        )
        .expect("generator");

        let preferred = generator
            .score_continuation_with_resolver(&crate::EmptyTensorResolver, &[0], &[1], 4)
            .expect("preferred");
        let alternate = generator
            .score_continuation_with_resolver(&crate::EmptyTensorResolver, &[0], &[2], 4)
            .expect("alternate");

        assert!(preferred > alternate);
    }

    #[test]
    fn inference_signals_capture_confidence_and_map_to_cognition() {
        let confident = model_inference_signals(&[12.0, 1.0, 0.0]).expect("confident");
        let uncertain = model_inference_signals(&[1.0, 1.0, 1.0]).expect("uncertain");

        assert!(confident.top_probability > uncertain.top_probability);
        assert!(confident.top_margin > uncertain.top_margin);
        assert!(confident.normalized_entropy < uncertain.normalized_entropy);

        let signals = uncertain.cognitive_signals(90, 100, false);
        assert!(signals.complexity >= 0.9);
        assert!(signals.uncertainty >= 0.7);
    }

    #[test]
    fn next_distribution_probe_uses_native_graph_without_committing_a_token() {
        let transitions = Tensor::new(
            vec![4, 4],
            vec![
                0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0,
            ],
        )
        .expect("transitions");
        let mut static_inputs = BTreeMap::new();
        static_inputs.insert(ValueId(1), transitions);
        let generator = GraphGenerator::new(
            transition_graph(),
            crate::CpuReferenceProvider,
            static_inputs,
            ValueId(0),
            0,
            4,
        )
        .expect("generator");

        let distribution = generator
            .next_distribution_with_resolver(&crate::EmptyTensorResolver, &[0], 4)
            .expect("distribution");

        assert_eq!(distribution.len(), 4);
        assert!(distribution[1] > distribution[0]);
        assert!(distribution[1] > distribution[2]);
    }

    #[test]
    fn streaming_generation_can_cancel_between_tokens() {
        let transitions = Tensor::new(
            vec![4, 4],
            vec![
                0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0,
            ],
        )
        .expect("transitions");
        let mut static_inputs = BTreeMap::new();
        static_inputs.insert(ValueId(1), transitions);
        let generator = GraphGenerator::new(
            transition_graph(),
            crate::CpuReferenceProvider,
            static_inputs,
            ValueId(0),
            0,
            4,
        )
        .expect("generator");

        let mut streamed = Vec::new();
        let result = generator
            .generate_tokens_with_resolver_streaming(
                &crate::EmptyTensorResolver,
                &[0],
                GenerationConfig {
                    max_new_tokens: 8,
                    context_limit: 4,
                    eos_token: Some(3),
                    distribution: DistributionKind::Probabilities,
                    sampling: SamplingMode::Greedy,
                },
                |token, generated| {
                    streamed.push(token);
                    assert_eq!(generated, streamed.as_slice());
                    GenerationControl::Cancel
                },
            )
            .expect("stream");

        assert_eq!(streamed, vec![1]);
        assert_eq!(result.generation.generated_tokens, vec![1]);
        assert_eq!(result.generation.all_tokens, vec![0, 1]);
        assert_eq!(result.finish_reason, GenerationFinishReason::Cancelled);
    }

    #[test]
    fn stochastic_sampling_is_seeded() {
        let values = [0.1, 0.2, 0.7];
        let mode = SamplingMode::Stochastic {
            temperature: 1.0,
            top_k: Some(3),
            seed: 97,
        };
        assert_eq!(
            sample_token(&values, DistributionKind::Probabilities, mode, 4),
            sample_token(&values, DistributionKind::Probabilities, mode, 4)
        );
    }

    #[test]
    fn kv_cache_appends_and_truncates_sequence() {
        let mut cache = KvCache::default();
        cache
            .append(
                0,
                Tensor::new(vec![1, 1, 2], vec![1.0, 2.0]).expect("key 1"),
                Tensor::new(vec![1, 1, 2], vec![3.0, 4.0]).expect("value 1"),
            )
            .expect("append first");
        cache
            .append(
                0,
                Tensor::new(vec![1, 1, 2], vec![5.0, 6.0]).expect("key 2"),
                Tensor::new(vec![1, 1, 2], vec![7.0, 8.0]).expect("value 2"),
            )
            .expect("append second");

        assert_eq!(cache.layer(0).expect("layer").key.shape(), &[2, 1, 2]);
        cache.truncate_left(1).expect("truncate");
        assert_eq!(cache.layer(0).expect("layer").key.data(), &[5.0, 6.0]);
    }

    #[test]
    fn prefix_cache_reports_reusable_prefix() {
        let cache = PrefixCache {
            tokens: vec![1, 2, 3],
            kv: KvCache::default(),
        };
        assert_eq!(cache.reusable_prefix_len(&[1, 2, 9]), 2);
    }
}
