use anyhow::Result;
use code_parsers::CodeParser;
use language::Language;
use std::ops::Range;
use std::sync::Arc;

pub struct Chunk {
    pub range: Range<usize>,
    pub content: String,
    pub element_type: String,
}

pub fn chunk_text(text: &str, language: Option<&Arc<Language>>) -> Result<Vec<Chunk>> {
    if let Some(lang) = language {
        let parser = CodeParser::new()?;
        let language_name = lang.code_fence_block_name();
        
        let parser_chunks = parser.get_chunks_from_content(text, &language_name)?;
        
        Ok(parser_chunks
            .into_iter()
            .map(|pc| {
                let start_byte = text
                    .lines()
                    .take(pc.start_line)
                    .map(|l| l.len() + 1) // +1 for newline
                    .sum::<usize>();
                
                let end_byte = text
                    .lines()
                    .take(pc.end_line + 1)
                    .map(|l| l.len() + 1)
                    .sum::<usize>()
                    .saturating_sub(1);
                
                Chunk {
                    range: start_byte..end_byte,
                    content: pc.content,
                    element_type: pc.element_type,
                }
            })
            .collect())
    } else {
        // Fallback to simple chunking for unknown languages
        Ok(chunk_text_simple(text, 1000))
    }
}

fn chunk_text_simple(text: &str, max_chunk_size: usize) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut start = 0;
    
    while start < text.len() {
        let end = (start + max_chunk_size).min(text.len());
        
        // Try to break at a newline
        let chunk_end = if end < text.len() {
            text[start..end]
                .rfind('\n')
                .map(|i| start + i + 1)
                .unwrap_or(end)
        } else {
            end
        };
        
        chunks.push(Chunk {
            range: start..chunk_end,
            content: text[start..chunk_end].to_string(),
            element_type: "text_block".to_string(),
        });
        
        start = chunk_end;
    }
    
    chunks
}