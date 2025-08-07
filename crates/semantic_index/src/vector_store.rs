use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

pub mod qdrant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorDocument {
    pub id: String,
    pub embedding: Vec<f32>,
    pub metadata: DocumentMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    pub file_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub content: String,
    pub language: String,
    pub element_type: String,
    pub project_id: String,
    pub worktree_id: String,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub file_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub content: String,
    pub language: String,
    pub element_type: String,
}

#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn create_collection(&self, name: &str, vector_size: usize) -> Result<()>;
    async fn insert_documents(&self, collection: &str, documents: Vec<VectorDocument>) -> Result<()>;
    async fn search(
        &self,
        collection: &str,
        query_vector: Vec<f32>,
        limit: usize,
        score_threshold: Option<f32>,
    ) -> Result<Vec<SearchResult>>;
    async fn delete_documents(&self, collection: &str, ids: Vec<String>) -> Result<()>;
    async fn collection_exists(&self, name: &str) -> Result<bool>;
}