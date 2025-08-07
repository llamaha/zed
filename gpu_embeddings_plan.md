# GPU Embeddings Implementation Plan

## Overview
This document outlines the plan to add GPU-accelerated embeddings to Zed's semantic code search using Qwen3-Embed-0.6 model with INT8 quantization and Qdrant vector database.

## Goals
1. Enable fast, local embeddings generation using GPU acceleration
2. Support semantic code search with natural language queries  
3. Integrate with existing Zed infrastructure (settings, agent tools, etc.)
4. Make it opt-in through configuration settings (not enabled by default)

## Architecture

### Components

1. **GPU Embedding Provider**
   - Location: `crates/semantic_index/src/embedding/gpu.rs`
   - Uses Candle for Rust-native ML inference
   - Supports CUDA (Linux/Windows) and Metal (macOS)
   - Implements Qwen3-Embed-0.6 model with INT8 quantization
   - Configurable batch size for throughput optimization

2. **Vector Store Integration**
   - Qdrant client integration for vector storage
   - Location: `crates/semantic_index/src/vector_store/`
   - Handles document insertion, search, and management
   - Supports collection management and metadata filtering

3. **Code Chunking System**
   - Location: `crates/code-parsers/`
   - Language-aware chunking using Tree-sitter
   - Chunks code at semantic boundaries (functions, classes, etc.)
   - Fallback to line-based chunking for unsupported languages

4. **Agent Tool Integration**
   - Location: `crates/assistant_tools/src/semantic_search_tool.rs`
   - Exposes semantic search to AI agent
   - Natural language query interface

### Settings Structure

```json
{
  "semantic_index": {
    "enabled": true,
    "gpu_embeddings": {
      "enabled": false,  // Opt-in, not default
      "model_path": null,  // Optional local model path
      "device": "auto",    // auto, cuda:0, metal, cpu
      "batch_size": 32,
      "quantization": "int8",  // int8 or none
      "qdrant_url": "http://localhost:6334"
    }
  }
}
```

## Implementation Steps

### Phase 1: Core Infrastructure
1. ✅ Add Candle dependencies with feature flags
2. ✅ Implement GPU embedding provider
3. ✅ Add Qdrant vector store implementation
4. ✅ Create settings structure with opt-in configuration

### Phase 2: Integration
1. ✅ Integrate with existing semantic_index crate
2. ✅ Add code-parsers for intelligent chunking
3. ✅ Create agent tool for semantic search
4. ✅ Add feature flags to control compilation

### Phase 3: User Experience
1. ✅ Add configuration UI in settings
2. ✅ Create documentation for setup and usage
3. ✅ Add progress indicators for indexing
4. ⏳ Add search UI improvements

## Technical Details

### Model: Qwen3-Embed-0.6
- Embedding dimension: 1536
- Context length: 8192 tokens
- Quantization: INT8 (reduces memory by 75%)
- Performance: ~1000 embeddings/second on RTX 3080

### Qdrant Configuration
- Default URL: http://localhost:6334
- Collections: One per project/workspace
- Metadata: file_path, line numbers, language, element type
- Distance metric: Cosine similarity

### Dependencies
```toml
[dependencies]
candle-core = { version = "0.9", features = ["cuda"] }  # or ["metal"] for macOS
candle-nn = "0.9"
candle-transformers = "0.9"
tokenizers = "0.20"
qdrant-client = "1.10"
```

## Benefits
1. **Performance**: 10-100x faster than API-based embeddings
2. **Privacy**: All processing happens locally
3. **Cost**: No API fees after initial model download
4. **Latency**: Near-instant search results
5. **Offline**: Works without internet connection

## Considerations
1. **Memory**: Requires ~2GB GPU memory with INT8 quantization
2. **Storage**: Model weights ~1.5GB, Qdrant storage varies by project size
3. **Setup**: Requires Qdrant to be running locally
4. **Compatibility**: CUDA 11.8+ for NVIDIA, any Apple Silicon Mac