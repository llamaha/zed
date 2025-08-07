use anyhow::{anyhow, Result};
use assistant_tool::{ActionLog, Tool, ToolResult};
use gpui::{AnyWindowHandle, App, BorrowAppContext, Entity, Task};
use language_model::{LanguageModel, LanguageModelRequest, LanguageModelToolSchemaFormat};
use project::Project;
use schemars::JsonSchema;
use semantic_index::SemanticDb;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use ui::IconName;

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct SemanticSearchInput {
    /// The natural language query to search for
    pub query: String,
    /// Maximum number of results to return (default: 10)
    pub limit: Option<usize>,
    /// Minimum similarity score threshold (default: 0.5)
    pub threshold: Option<f32>,
}

pub struct SemanticSearchTool;

impl Tool for SemanticSearchTool {
    fn name(&self) -> String {
        "semantic_search".to_string()
    }

    fn description(&self) -> String {
        "Search for code semantically using natural language queries".to_string()
    }

    fn icon(&self) -> IconName {
        IconName::MagnifyingGlass
    }

    fn needs_confirmation(&self, _: &serde_json::Value, _: &Entity<Project>, _: &App) -> bool {
        false
    }

    fn may_perform_edits(&self) -> bool {
        false
    }

    fn input_schema(&self, _format: LanguageModelToolSchemaFormat) -> Result<serde_json::Value> {
        let schema = schemars::schema_for!(SemanticSearchInput);
        Ok(serde_json::to_value(schema)?)
    }

    fn ui_text(&self, input: &serde_json::Value) -> String {
        if let Ok(search_input) = serde_json::from_value::<SemanticSearchInput>(input.clone()) {
            format!("Search for: {}", search_input.query)
        } else {
            "Semantic search".to_string()
        }
    }

    fn run(
        self: Arc<Self>,
        input: serde_json::Value,
        _request: Arc<LanguageModelRequest>,
        project: Entity<Project>,
        _action_log: Entity<ActionLog>,
        _model: Arc<dyn LanguageModel>,
        _window: Option<AnyWindowHandle>,
        cx: &mut App,
    ) -> ToolResult {
        let search_input = match serde_json::from_value::<SemanticSearchInput>(input) {
            Ok(input) => input,
            Err(e) => {
                return ToolResult {
                    output: Task::ready(Err(anyhow!("Invalid input: {}", e))),
                    card: None,
                };
            }
        };

        let query = search_input.query.clone();
        let limit = search_input.limit.unwrap_or(10);
        let _threshold = search_input.threshold.unwrap_or(0.5);

        let output = cx.spawn(async move |cx| {
            let project_index = cx.update(|cx| {
                cx.update_global::<SemanticDb, _>(|db, cx| {
                    db.project_index(project.clone(), cx)
                        .or_else(|| Some(db.create_project_index(project.clone(), cx)))
                })
            })?;

            let project_index = match project_index {
                Some(index) => index,
                None => return Err(anyhow!("Failed to create project index")),
            };

            let results = cx.update(|cx| {
                project_index.update(cx, |index, cx| {
                    index.search(vec![query.clone()], limit, cx)
                })
            })?.await?;
            
            // Load the search results to get the actual content
            let fs = cx.update(|cx| project.read(cx).fs().clone())?;
            let loaded_results = SemanticDb::load_results(results, &fs, &cx).await?;

            let mut output = String::new();
            if loaded_results.is_empty() {
                output.push_str("No results found for the query.");
            } else {
                for result in loaded_results {
                    let start_line = result.row_range.start();
                    let end_line = result.row_range.end();
                    output.push_str(&format!(
                        "**{}:{}:{}**\n```\n{}\n```\n\n",
                        result.full_path.display(),
                        start_line,
                        end_line,
                        result.excerpt_content
                    ));
                }
            }

            Ok(output.into())
        });

        ToolResult { output, card: None }
    }
}