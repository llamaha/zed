use anyhow::Result;
use gpui::App;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use settings::{Settings, SettingsSources, VsCodeSettings};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct SemanticIndexSettings {
    pub enabled: bool,
    pub gpu_embeddings: Option<GpuEmbeddingsSettings>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GpuEmbeddingsSettings {
    pub enabled: bool,
    pub model_path: Option<String>,
    pub device: String,
    pub batch_size: usize,
    pub quantization: String,
    pub qdrant_url: String,
}

impl Default for SemanticIndexSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            gpu_embeddings: None,
        }
    }
}

impl Default for GpuEmbeddingsSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            model_path: None,
            device: "auto".to_string(),
            batch_size: 32,
            quantization: "int8".to_string(),
            qdrant_url: "http://localhost:6334".to_string(),
        }
    }
}

impl Settings for SemanticIndexSettings {
    const KEY: Option<&'static str> = Some("semantic_index");

    type FileContent = Self;

    fn load(sources: SettingsSources<Self::FileContent>, _: &mut App) -> Result<Self> {
        sources.json_merge()
    }

    fn import_from_vscode(_vscode: &VsCodeSettings, _current: &mut Self::FileContent) {
        // No VSCode settings to import for semantic index
    }
}