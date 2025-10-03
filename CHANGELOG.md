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
