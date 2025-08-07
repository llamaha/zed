use anyhow::{Context, Result};
use std::collections::HashMap;
use streaming_iterator::StreamingIterator;
use tree_sitter::{Parser, Query, QueryCursor};

pub struct CodeChunk {
    pub start_line: usize,
    pub end_line: usize,
    pub content: String,
    pub element_type: String,
}

pub struct CodeParser {
    parsers: HashMap<String, (Parser, Query)>,
}

impl CodeParser {
    pub fn new() -> Result<Self> {
        let mut parsers = HashMap::new();

        // Rust
        let mut rust_parser = Parser::new();
        rust_parser.set_language(&tree_sitter_rust::LANGUAGE.into())?;
        let rust_query = Query::new(
            &tree_sitter_rust::LANGUAGE.into(),
            r#"
            (function_item) @function
            (impl_item) @impl
            (struct_item) @struct
            (enum_item) @enum
            (trait_item) @trait
            "#,
        )?;
        parsers.insert("rust".to_string(), (rust_parser, rust_query));

        // JavaScript/TypeScript
        let mut js_parser = Parser::new();
        js_parser.set_language(&tree_sitter_javascript::LANGUAGE.into())?;
        let js_query = Query::new(
            &tree_sitter_javascript::LANGUAGE.into(),
            r#"
            (function_declaration) @function
            (function_expression) @function
            (arrow_function) @function
            (class_declaration) @class
            (method_definition) @method
            "#,
        )?;
        // JavaScript parser
        parsers.insert("javascript".to_string(), (js_parser, js_query));
        
        // TypeScript uses same grammar as JavaScript
        let mut ts_parser = Parser::new();
        ts_parser.set_language(&tree_sitter_javascript::LANGUAGE.into())?;
        let ts_query = Query::new(
            &tree_sitter_javascript::LANGUAGE.into(),
            r#"
            (function_declaration) @function
            (function_expression) @function
            (arrow_function) @function
            (class_declaration) @class
            (method_definition) @method
            "#,
        )?;
        parsers.insert("typescript".to_string(), (ts_parser, ts_query));

        // Python
        let mut py_parser = Parser::new();
        py_parser.set_language(&tree_sitter_python::LANGUAGE.into())?;
        let py_query = Query::new(
            &tree_sitter_python::LANGUAGE.into(),
            r#"
            (function_definition) @function
            (class_definition) @class
            "#,
        )?;
        parsers.insert("python".to_string(), (py_parser, py_query));

        // Go
        let mut go_parser = Parser::new();
        go_parser.set_language(&tree_sitter_go::LANGUAGE.into())?;
        let go_query = Query::new(
            &tree_sitter_go::LANGUAGE.into(),
            r#"
            (function_declaration) @function
            (method_declaration) @method
            (type_declaration) @type
            "#,
        )?;
        parsers.insert("go".to_string(), (go_parser, go_query));

        Ok(Self { parsers })
    }

    pub fn get_chunks(&self, file_path: &str, content: &str) -> Result<Vec<CodeChunk>> {
        let language = Self::detect_language(file_path);
        
        if let Some((parser, query)) = self.parsers.get(language) {
            self.parse_with_query(content, parser, query)
        } else {
            // Fallback to line-based chunking
            Ok(self.chunk_by_lines(content, 50))
        }
    }

    pub fn get_chunks_from_content(&self, content: &str, language: &str) -> Result<Vec<CodeChunk>> {
        if let Some((parser, query)) = self.parsers.get(language) {
            self.parse_with_query(content, parser, query)
        } else {
            // Fallback to line-based chunking
            Ok(self.chunk_by_lines(content, 50))
        }
    }

    fn parse_with_query(&self, content: &str, parser: &Parser, query: &Query) -> Result<Vec<CodeChunk>> {
        // Create a new parser instance
        let mut new_parser = Parser::new();
        new_parser.set_language(&parser.language().unwrap())?;
        
        let tree = new_parser
            .parse(content, None)
            .context("Failed to parse code")?;

        let root_node = tree.root_node();
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(query, root_node, content.as_bytes());

        let mut chunks = Vec::new();

        while let Some(mat) = matches.next() {
            for capture in mat.captures {
                let node = capture.node;
                let start_line = node.start_position().row;
                let end_line = node.end_position().row;
                let element_type = Self::get_element_type(capture.index, query);

                chunks.push(CodeChunk {
                    start_line,
                    end_line,
                    content: content[node.byte_range()].to_string(),
                    element_type,
                });
            }
        }

        // If no chunks were found, fall back to line-based chunking
        if chunks.is_empty() {
            chunks = self.chunk_by_lines(content, 50);
        }

        Ok(chunks)
    }

    fn chunk_by_lines(&self, content: &str, chunk_size: usize) -> Vec<CodeChunk> {
        let lines: Vec<&str> = content.lines().collect();
        let mut chunks = Vec::new();

        for (i, chunk) in lines.chunks(chunk_size).enumerate() {
            let start_line = i * chunk_size;
            let end_line = start_line + chunk.len() - 1;

            chunks.push(CodeChunk {
                start_line,
                end_line,
                content: chunk.join("\n"),
                element_type: "code_block".to_string(),
            });
        }

        chunks
    }

    fn detect_language(file_path: &str) -> &str {
        if file_path.ends_with(".rs") {
            "rust"
        } else if file_path.ends_with(".js") || file_path.ends_with(".jsx") {
            "javascript"
        } else if file_path.ends_with(".ts") || file_path.ends_with(".tsx") {
            "typescript"
        } else if file_path.ends_with(".py") {
            "python"
        } else if file_path.ends_with(".go") {
            "go"
        } else {
            "unknown"
        }
    }

    fn get_element_type(capture_index: u32, query: &Query) -> String {
        query
            .capture_names()
            .get(capture_index as usize)
            .map(|s| s.as_ref())
            .unwrap_or("unknown")
            .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_parsing() {
        let parser = CodeParser::new().unwrap();
        let content = r#"
fn hello() {
    println!("Hello, world!");
}

struct MyStruct {
    field: i32,
}

impl MyStruct {
    fn new() -> Self {
        Self { field: 0 }
    }
}
"#;

        let chunks = parser.get_chunks("test.rs", content).unwrap();
        assert!(!chunks.is_empty());
        assert!(chunks.iter().any(|c| c.element_type == "function"));
        assert!(chunks.iter().any(|c| c.element_type == "struct"));
        assert!(chunks.iter().any(|c| c.element_type == "impl"));
    }
}