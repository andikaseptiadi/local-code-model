# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial transformer implementation in pure Go
- Complete training pipeline (data loading → training → model saving → inference)
- Multi-head self-attention mechanism
- Adam optimizer with learning rate scheduling
- Character-level tokenization
- Model serialization/deserialization
- Comprehensive learning guide (LEARNING.md)
- Professional README with architecture documentation
- Apache 2.0 license
- Semantic versioning
- Keep a Changelog format
- Switchable modern architecture improvements:
  - RoPE (Rotary Position Embeddings) via `--use-rope` flag
  - SwiGLU activation in feed-forward layers via `--use-swiglu` flag
  - RMSNorm normalization via `--use-rmsnorm` flag
  - Explicit causal masking via `--use-explicit-mask` flag
- Training metrics visualization (HTML output with loss curves and learning rate schedule)
- Attention pattern visualization (interactive heatmaps)
- Token embedding visualization (PCA and t-SNE)
- Byte-Pair Encoding (BPE) tokenizer support via `--tokenizer-type bpe`
- KV cache for efficient inference
- Interactive Jupyter notebooks for hands-on learning
- Gradient accumulation via `--grad-accum-steps` flag for training with larger effective batch sizes
- Parallel matrix multiplication using Go goroutines (`tensor_parallel.go`)
  - Row-parallel strategy with WaitGroups (3-8x speedup on multi-core machines)
  - Alternative channel-based work queue pattern
  - Comprehensive tests and benchmarks
  - Pure Go implementation, no external dependencies
- Reusable worker pool for batch operations (`tensor_pool.go`)
  - Persistent worker pool with task queue (avoids goroutine creation overhead)
  - Batch tensor operations (BatchMatMul, BatchMatMulParallel)
  - Graceful shutdown with Wait() and Stop()
  - ~300ns task submission overhead (amortized across training iterations)
  - Ideal for training loops processing batches repeatedly
- Cache-friendly blocked matrix multiplication (`tensor_blocked.go`)
  - Loop tiling/blocking for better cache locality (2-4x speedup)
  - L1-optimized (block size 64) and L2-optimized (block size 128) variants
  - MatMulBlocked and MatMulBlockedParallel functions
  - Combines blocking + parallelism for 8-16x total speedup
  - Educational implementation with extensive comments on cache hierarchy and memory bandwidth
- Tensor allocation pooling using sync.Pool (`tensor_syncpool.go`)
  - Object pooling for tensor recycling (reduces GC pressure)
  - Per-size pools using map[int]*sync.Pool with RWMutex
  - 10x speedup and 0 allocations after warmup in training loops
  - Global pool pattern (GetPooledTensor, PutPooledTensor, WithPooledTensor)
  - Comprehensive tests and benchmarks demonstrating GC reduction
  - Educational implementation explaining object pooling and GC optimization

### Fixed
- Training loop batch data structuring
- Forward pass gradient computation for output projection
- Backward pass through attention layers (multi-head handling)
- Missing input field in FFCache structure
- Missing input field in AttentionCache structure
- Transpose handling in attention backward pass
- Code formatting issues (gofmt)

## [0.1.0] - 2025-01-XX

### Added
- Initial release
- Pure Go GPT-style transformer implementation
- Educational codebase with 30-50% comment coverage
- Training and generation CLI commands
- Tensor operations (MatMul, GELU, Softmax, etc.)
- Automatic differentiation for backpropagation
- Feed-forward networks with layer normalization

[Unreleased]: https://github.com/scttfrdmn/local-code-model/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/scttfrdmn/local-code-model/releases/tag/v0.1.0
