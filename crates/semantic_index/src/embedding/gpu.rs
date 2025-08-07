#[cfg(feature = "gpu-embeddings")]
use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::BertModel;
use futures::future::BoxFuture;
use futures::FutureExt;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;

use crate::embedding::{Embedding, EmbeddingProvider, TextToEmbed};

const MODEL_ID: &str = "Alibaba-NLP/gte-Qwen2-1.5B-instruct";
const EMBEDDING_DIM: usize = 1536;
const MAX_SEQUENCE_LENGTH: usize = 8192;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuEmbeddingSettings {
    pub model_path: Option<PathBuf>,
    pub device: GpuDevice,
    pub batch_size: usize,
    pub quantization: QuantizationType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GpuDevice {
    Auto,
    Cuda(usize),
    Metal,
    Cpu,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuantizationType {
    None,
    Int8,
}

impl Default for GpuEmbeddingSettings {
    fn default() -> Self {
        Self {
            model_path: None,
            device: GpuDevice::Auto,
            batch_size: 32,
            quantization: QuantizationType::Int8,
        }
    }
}

pub struct GpuEmbeddingProvider {
    model: Arc<Mutex<BertModel>>,
    tokenizer: Arc<Tokenizer>,
    device: Device,
    batch_size: usize,
}

impl GpuEmbeddingProvider {
    pub async fn new(settings: GpuEmbeddingSettings) -> Result<Self> {
        let device = match settings.device {
            GpuDevice::Auto => {
                if candle_core::utils::cuda_is_available() {
                    Device::new_cuda(0)?
                } else if candle_core::utils::metal_is_available() {
                    Device::new_metal(0)?
                } else {
                    Device::Cpu
                }
            }
            GpuDevice::Cuda(index) => Device::new_cuda(index)?,
            GpuDevice::Metal => Device::new_metal(0)?,
            GpuDevice::Cpu => Device::Cpu,
        };

        let model_path = if let Some(path) = settings.model_path {
            path
        } else {
            download_model(MODEL_ID).await?
        };

        let tokenizer = Tokenizer::from_file(&model_path.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let config = serde_json::from_reader(std::fs::File::open(&model_path.join("config.json"))?)
            .context("Failed to load model config")?;

        let weights_file = match settings.quantization {
            QuantizationType::Int8 => model_path.join("model.q8_0.gguf"),
            QuantizationType::None => model_path.join("model.safetensors"),
        };

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_file], candle_core::DType::F32, &device)?
        };

        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            tokenizer: Arc::new(tokenizer),
            device,
            batch_size: settings.batch_size,
        })
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>> {
        let encoded = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("Failed to tokenize texts: {}", e))?;

        let token_ids: Vec<Vec<u32>> = encoded
            .iter()
            .map(|encoding| encoding.get_ids().to_vec())
            .collect();

        let max_len = token_ids
            .iter()
            .map(|ids| ids.len())
            .max()
            .unwrap_or(0)
            .min(MAX_SEQUENCE_LENGTH);

        let padded_ids: Vec<u32> = token_ids
            .iter()
            .flat_map(|ids| {
                let mut padded = ids.clone();
                padded.resize(max_len, 0);
                padded
            })
            .collect();

        let input_ids = Tensor::from_vec(
            padded_ids,
            (texts.len(), max_len),
            &self.device,
        )?;

        let attention_mask = token_ids
            .iter()
            .flat_map(|ids| {
                let mut mask = vec![1f32; ids.len()];
                mask.resize(max_len, 0f32);
                mask
            })
            .collect::<Vec<_>>();

        let attention_mask = Tensor::from_vec(
            attention_mask,
            (texts.len(), max_len),
            &self.device,
        )?;

        // Create token type ids (all zeros for single sequence)
        let token_type_ids = Tensor::zeros_like(&input_ids)?;
        
        let model = self.model.lock();
        let outputs = model.forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

        let embeddings = outputs
            .mean_pool(&attention_mask)?
            .to_vec2::<f32>()?
            .into_iter()
            .map(Embedding::new)
            .collect();

        Ok(embeddings)
    }
}

impl EmbeddingProvider for GpuEmbeddingProvider {
    fn embed<'a>(&'a self, texts: &'a [TextToEmbed<'a>]) -> BoxFuture<'a, Result<Vec<Embedding>>> {
        async move {
            let mut all_embeddings = Vec::with_capacity(texts.len());

            for chunk in texts.chunks(self.batch_size) {
                let text_strs: Vec<&str> = chunk.iter().map(|t| t.text).collect();
                let embeddings = self.embed_batch(&text_strs).await?;
                all_embeddings.extend(embeddings);
            }

            Ok(all_embeddings)
        }
        .boxed()
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }
}

async fn download_model(model_id: &str) -> Result<PathBuf> {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join("zed")
        .join("models")
        .join(model_id.replace('/', "--"));

    if cache_dir.exists() {
        return Ok(cache_dir);
    }

    std::fs::create_dir_all(&cache_dir)?;

    // TODO: Implement actual model downloading from Hugging Face
    // For now, return error indicating manual download is needed
    anyhow::bail!(
        "Please download the model from https://huggingface.co/{} and place it in {:?}",
        model_id,
        cache_dir
    );
}

// Extension trait for tensor operations
trait TensorExt {
    fn mean_pool(&self, attention_mask: &Tensor) -> Result<Tensor>;
}

impl TensorExt for Tensor {
    fn mean_pool(&self, attention_mask: &Tensor) -> Result<Tensor> {
        let masked = self.broadcast_mul(attention_mask)?;
        let sum = masked.sum(1)?;
        let count = attention_mask.sum(1)?;
        Ok(sum.broadcast_div(&count)?)
    }
}