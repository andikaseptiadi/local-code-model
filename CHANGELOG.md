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
