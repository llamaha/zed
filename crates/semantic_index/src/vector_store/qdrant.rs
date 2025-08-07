use anyhow::{Context, Result};
use async_trait::async_trait;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    point_id::PointIdOptions, vectors_config::Config, CreateCollectionBuilder,
    Distance, PointId, PointStruct, SearchPointsBuilder, VectorParamsBuilder,
    VectorsConfig, Value, DeletePointsBuilder, PointsIdsList, UpsertPointsBuilder,
};
use serde_json::json;
use std::collections::HashMap;

use super::{SearchResult, VectorDocument, VectorStore};

pub struct QdrantVectorStore {
    client: Qdrant,
}

impl QdrantVectorStore {
    pub async fn new(url: &str) -> Result<Self> {
        let client = Qdrant::from_url(url)
            .build()
            .context("Failed to create Qdrant client")?;
        Ok(Self { client })
    }
}

#[async_trait]
impl VectorStore for QdrantVectorStore {
    async fn create_collection(&self, name: &str, vector_size: usize) -> Result<()> {
        if self.collection_exists(name).await? {
            return Ok(());
        }

        self.client
            .create_collection(
                CreateCollectionBuilder::new(name)
                    .vectors_config(VectorsConfig {
                        config: Some(Config::Params(
                            VectorParamsBuilder::new(vector_size as u64, Distance::Cosine)
                                .on_disk(false)
                                .build(),
                        )),
                    }),
            )
            .await
            .context("Failed to create collection")?;

        Ok(())
    }

    async fn insert_documents(&self, collection: &str, documents: Vec<VectorDocument>) -> Result<()> {
        let points: Vec<PointStruct> = documents
            .into_iter()
            .map(|doc| {
                let payload: HashMap<String, Value> = HashMap::from([
                    ("file_path".to_string(), json!(doc.metadata.file_path).into()),
                    ("start_line".to_string(), json!(doc.metadata.start_line).into()),
                    ("end_line".to_string(), json!(doc.metadata.end_line).into()),
                    ("content".to_string(), json!(doc.metadata.content).into()),
                    ("language".to_string(), json!(doc.metadata.language).into()),
                    ("element_type".to_string(), json!(doc.metadata.element_type).into()),
                    ("project_id".to_string(), json!(doc.metadata.project_id).into()),
                    ("worktree_id".to_string(), json!(doc.metadata.worktree_id).into()),
                ]);

                PointStruct::new(doc.id, doc.embedding, payload)
            })
            .collect();

        self.client
            .upsert_points(UpsertPointsBuilder::new(collection, points))
            .await
            .context("Failed to insert documents")?;

        Ok(())
    }

    async fn search(
        &self,
        collection: &str,
        query_vector: Vec<f32>,
        limit: usize,
        score_threshold: Option<f32>,
    ) -> Result<Vec<SearchResult>> {
        let search_result = self
            .client
            .search_points(
                SearchPointsBuilder::new(collection, query_vector, limit as u64)
                    .score_threshold(score_threshold.unwrap_or(0.0))
                    .with_payload(true),
            )
            .await
            .context("Failed to search points")?;

        let results = search_result
            .result
            .into_iter()
            .map(|point| {
                let payload = point.payload;
                SearchResult {
                    id: match point.id.as_ref().and_then(|id| id.point_id_options.as_ref()) {
                        Some(PointIdOptions::Uuid(uuid)) => uuid.clone(),
                        Some(PointIdOptions::Num(num)) => num.to_string(),
                        None => String::new(),
                    },
                    score: point.score,
                    file_path: payload
                        .get("file_path")
                        .and_then(|v| match &v.kind {
                            Some(qdrant_client::qdrant::value::Kind::StringValue(s)) => Some(s.clone()),
                            _ => None,
                        })
                        .unwrap_or_default(),
                    start_line: payload
                        .get("start_line")
                        .and_then(|v| match &v.kind {
                            Some(qdrant_client::qdrant::value::Kind::IntegerValue(n)) => Some(*n as usize),
                            _ => None,
                        })
                        .unwrap_or(0),
                    end_line: payload
                        .get("end_line")
                        .and_then(|v| match &v.kind {
                            Some(qdrant_client::qdrant::value::Kind::IntegerValue(n)) => Some(*n as usize),
                            _ => None,
                        })
                        .unwrap_or(0),
                    content: payload
                        .get("content")
                        .and_then(|v| match &v.kind {
                            Some(qdrant_client::qdrant::value::Kind::StringValue(s)) => Some(s.clone()),
                            _ => None,
                        })
                        .unwrap_or_default(),
                    language: payload
                        .get("language")
                        .and_then(|v| match &v.kind {
                            Some(qdrant_client::qdrant::value::Kind::StringValue(s)) => Some(s.clone()),
                            _ => None,
                        })
                        .unwrap_or_default(),
                    element_type: payload
                        .get("element_type")
                        .and_then(|v| match &v.kind {
                            Some(qdrant_client::qdrant::value::Kind::StringValue(s)) => Some(s.clone()),
                            _ => None,
                        })
                        .unwrap_or_default(),
                }
            })
            .collect();

        Ok(results)
    }

    async fn delete_documents(&self, collection: &str, ids: Vec<String>) -> Result<()> {
        let point_ids: Vec<PointId> = ids
            .into_iter()
            .map(|id| PointId {
                point_id_options: Some(PointIdOptions::Uuid(id)),
            })
            .collect();

        self.client
            .delete_points(
                DeletePointsBuilder::new(collection)
                    .points(PointsIdsList { ids: point_ids }),
            )
            .await
            .context("Failed to delete documents")?;

        Ok(())
    }

    async fn collection_exists(&self, name: &str) -> Result<bool> {
        match self.client.collection_info(name).await {
            Ok(_) => Ok(true),
            Err(e) => {
                if e.to_string().contains("doesn't exist") {
                    Ok(false)
                } else {
                    Err(e.into())
                }
            }
        }
    }
}