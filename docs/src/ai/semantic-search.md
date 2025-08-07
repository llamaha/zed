# Semantic Search

Semantic search allows you to find code using natural language queries, powered by AI embeddings. This feature can understand the intent and context of your search query, making it easier to find relevant code even when you don't know the exact keywords or function names.

## Overview

Semantic search uses machine learning models to convert both your code and search queries into high-dimensional vectors (embeddings). When you search, it finds code segments with similar semantic meaning, not just text matches.

## Configuration

To enable semantic search with GPU acceleration:

```json
{
  "semantic_index": {
    "enabled": true,
    "gpu_embeddings": {
      "enabled": true,
      "device": "auto",
      "batch_size": 32,
      "quantization": "int8",
      "qdrant_url": "http://localhost:6334"
    }
  }
}
```

### Settings

- **enabled**: Enable or disable semantic indexing
- **gpu_embeddings**: GPU acceleration settings (optional)
  - **enabled**: Enable GPU-accelerated embeddings
  - **device**: Device to use (`"auto"`, `"cuda:0"`, `"metal"`, `"cpu"`)
  - **batch_size**: Number of documents to process in parallel
  - **quantization**: Model quantization (`"int8"` or `"none"`)
  - **qdrant_url**: URL of the Qdrant vector database

## Prerequisites

### Running Qdrant

Semantic search with GPU embeddings requires a running Qdrant instance:

```bash
# Using Docker
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

### GPU Support

- **NVIDIA GPUs**: Requires CUDA 11.8 or later
- **Apple Silicon**: Metal support is built-in
- **CPU fallback**: Available but slower

## Using Semantic Search

### In the Agent Panel

Use the semantic search tool in agent interactions:

```
Please find all authentication-related code in the project
```

The agent will use semantic search to find relevant code segments.

### Search Examples

- "Find the database connection logic"
- "Show me all error handling code"
- "Where is user authentication implemented?"
- "Find code that processes payments"

## How It Works

1. **Indexing**: Your code is automatically chunked into semantic units (functions, classes, etc.) and converted into embeddings
2. **Storage**: Embeddings are stored in the Qdrant vector database for fast retrieval
3. **Search**: Your natural language query is converted to an embedding and compared against stored embeddings
4. **Results**: The most semantically similar code segments are returned

## Performance Considerations

- Initial indexing may take time for large projects
- GPU acceleration significantly improves indexing and search speed
- The index is incrementally updated as you modify code
- Quantized models (INT8) use less memory with minimal quality loss

## Troubleshooting

### Qdrant Connection Issues

If you see connection errors:
1. Ensure Qdrant is running on the configured URL
2. Check firewall settings
3. Verify the URL in your settings

### GPU Not Detected

If GPU acceleration isn't working:
1. Check CUDA installation (for NVIDIA)
2. Verify GPU drivers are up to date
3. Set device to "cpu" as a fallback

### Model Download

The first run will need to download the embedding model. Ensure you have:
- Stable internet connection
- ~2GB free disk space
- Write permissions to the cache directory