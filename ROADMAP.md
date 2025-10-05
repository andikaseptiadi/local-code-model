# Local Code Model - Roadmap

This document outlines planned improvements and features for the project.

## Immediate (This Week)

- [x] **Get Go Report Card Badge**
  - Submit to https://goreportcard.com/report/github.com/scttfrdmn/local-code-model
  - Add A+ badge to README
  - Validates all quality work

- [x] **Add GitHub Actions CI/CD**
  - Run `make quality` on every PR/push
  - Automatically run tests
  - Ensure quality stays high as project evolves

- [x] **Add Example Outputs to README**
  - Show what the model actually generates after training
  - Include loss curves
  - Make it more tangible for learners

## Short Term (This Month)

- [x] **Add KV Cache for Inference**
  - Currently generating is slow (recomputes everything each token)
  - KV caching dramatically speeds up generation
  - Good teaching opportunity about optimization

- [x] **Create Blog Posts/Tutorials** (3/3 complete)
  - ✅ Attention mechanism deep dive ([docs/attention-mechanism.md](docs/attention-mechanism.md))
  - ✅ Backpropagation through transformers ([docs/backpropagation.md](docs/backpropagation.md))
  - ✅ Training dynamics and loss curves ([docs/training-dynamics.md](docs/training-dynamics.md))

- [x] **Add Visualization Tools**
  - [x] Attention weight visualization
  - [x] Training loss/perplexity plots
  - [x] Token embedding visualizations (PCA/t-SNE)

## Medium Term (Next 3 Months)

- [x] **BPE Tokenizer Implementation**
  - Byte-level BPE with merge rules
  - Better vocabulary efficiency
  - More realistic training
  - Supports both BPE and character-level tokenization

- [x] **Interactive Notebooks/Tutorials**
  - Created Jupyter notebooks (using gophernotes)
  - Step-by-step tutorials for tensors, attention, and training
  - Available in `notebooks/` directory

- [x] **Implement Modern Architectural Improvements**
  - ✅ RoPE (Rotary Position Embeddings) - `rope.go`
  - ✅ SwiGLU activation - `swiglu.go`
  - ✅ RMSNorm instead of LayerNorm - `rmsnorm.go`
  - ✅ Switchable architecture via CLI flags: `--use-rope`, `--use-swiglu`, `--use-rmsnorm`
  - Shows evolution of transformer architecture from GPT-2 to modern LLMs like LLaMA

- [ ] **Build Community**
  - Discord/community forum
  - Place for learners to ask questions
  - Share what they've built

## Educational Enhancements

- [ ] **"Build Your Own GPT" Tutorial Series**
  - Step-by-step guide building from scratch
  - Each step is a git commit
  - Learners can follow along commit by commit

- [ ] **Add Troubleshooting Guide**
  - Common issues (NaN loss, poor convergence)
  - How to debug training problems
  - Performance tuning tips

- [ ] **Video Tutorials**
  - Walkthrough of key concepts
  - Live coding sessions
  - Architecture explanations

## Performance/Features

- [x] **Implement Proper Causal Masking**
  - ✅ Implemented as switchable option (`--use-explicit-mask`)
  - Dynamic masking (default): faster, implicit, computed on-the-fly
  - Explicit masking (optional): architecturally correct, uses pre-computed tensor
  - Properly handles KV cache scenarios with fallback logic

- [x] **Gradient Accumulation**
  - ✅ Implemented as switchable option (`--grad-accum-steps`)
  - Allows training with larger effective batch sizes (effective batch = batch × accum steps)
  - Important for memory-constrained environments
  - Default: 1 (no accumulation)

- [x] **Parallel Matrix Multiplication with Goroutines**
  - ✅ Implemented in `tensor_parallel.go`
  - Row-parallel strategy using goroutines and WaitGroups
  - Alternative channel-based work queue pattern
  - 3-8x speedup on multi-core machines (tested: 8.75x on M4 Max)
  - Pure Go, no external dependencies
  - Demonstrates Go concurrency patterns for ML workloads

- [x] **Reusable Worker Pool for Batch Operations**
  - ✅ Implemented in `tensor_pool.go`
  - Persistent worker pool with task queue (avoids goroutine creation overhead)
  - Batch tensor operations (BatchMatMul, BatchMatMulParallel)
  - Graceful shutdown with Wait() and Stop()
  - ~300ns task submission overhead (amortized across training iterations)
  - Ideal for training loops processing batches repeatedly

- [x] **Cache-Friendly Blocked Matrix Multiplication**
  - ✅ Implemented in `tensor_blocked.go`
  - Loop tiling/blocking for better cache locality (2-4x speedup)
  - L1-optimized (64) and L2-optimized (128) block sizes
  - MatMulBlocked, MatMulBlockedParallel for maximum performance
  - Combines blocking + parallelism for 8-16x total speedup
  - Educational implementation explaining cache hierarchy and memory bandwidth

- [ ] **Add More Architecture Variants**
  - BERT-style (bidirectional)
  - Encoder-decoder (T5-style)
  - Shows different transformer architectures

- [ ] **Implement Common Optimizations**
  - Mixed precision training
  - Gradient checkpointing
  - Flash Attention (educational implementation)

## Advanced Features

- [ ] **Add Evaluation Metrics**
  - Perplexity tracking
  - BLEU/ROUGE for generation quality
  - Benchmark against reference implementations

- [ ] **Implement Training from Scratch on Real Dataset**
  - Use WikiText or similar
  - Show full training pipeline
  - Demonstrate convergence to reasonable perplexity

- [ ] **Add Model Analysis Tools**
  - Attention pattern analysis
  - Layer-wise learning rate analysis
  - Gradient flow visualization

- [ ] **Benchmarking Suite**
  - Track training speed improvements
  - Memory usage profiling
  - Compare different implementation choices

## Contributing

See issues tagged with roadmap items. Pull requests welcome!

## Priority Levels

- **Immediate**: High impact, enables other work
- **Short Term**: Important for usability and learning
- **Medium Term**: Significant enhancements
- **Long Term**: Nice to have, research-oriented
