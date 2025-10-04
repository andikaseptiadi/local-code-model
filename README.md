# Local Code Model

A **pure Go implementation** of a GPT-style transformer from scratch, designed for learning and understanding how large language models work at a fundamental level.

[![Go Report Card](https://goreportcard.com/badge/github.com/scttfrdmn/local-code-model)](https://goreportcard.com/report/github.com/scttfrdmn/local-code-model)
[![Go Reference](https://pkg.go.dev/badge/github.com/scttfrdmn/local-code-model.svg)](https://pkg.go.dev/github.com/scttfrdmn/local-code-model)

## What Is This?

This project implements a complete transformer-based language model (similar to GPT) entirely in Go, with **no external ML libraries**. Every component—from tensor operations to backpropagation to the transformer architecture itself—is implemented from first principles.

**Key Features:**
- ✅ Pure Go implementation (no Python, no TensorFlow, no PyTorch)
- ✅ Complete training pipeline (data loading → training → model saving → inference)
- ✅ Extensively commented for learning (~30-50% comments throughout)
- ✅ Working backpropagation with automatic differentiation
- ✅ Multi-head self-attention mechanism
- ✅ Adam optimizer with learning rate scheduling
- ✅ Character-level tokenization
- ✅ Model serialization/deserialization

**Educational Philosophy:**
This codebase prioritizes **clarity over performance**. The code is deliberately naive to make core concepts visible. Each file includes detailed "WHAT'S GOING ON HERE" sections explaining the theory, mathematics, and design decisions.

## Quick Start

### Installation

```bash
git clone https://github.com/scttfrdmn/local-code-model.git
cd local-code-model
go build
```

### Train a Model

Train a tiny GPT model on your local Go source files:

```bash
./local-code-model train -epochs 2 -batch 4 -seq 64
```

This trains a 2-layer, 64-dimensional model in ~1 minute on a laptop. The model learns basic patterns in Go code.

### Generate Text

Generate text from your trained model:

```bash
./local-code-model generate -model tiny_model.bin -tokenizer tiny_tokenizer.bin -prompt "func main"
```

### Example Output

Here's what training looks like when you run the tiny model on local Go source code:

```
===========================================================================
TRAINING A TINY GPT MODEL ON GO CODE
===========================================================================

Model: 2 layers, 64 embed dim, 2 heads, 64 seq len
Training: 2 epochs, batch size 4, lr 0.0010

Step 1: Loading training data from .
  Loaded 146614 characters from .go files

Step 2: Building character-level tokenizer
  Vocabulary size: 259 characters

Step 3: Creating training dataset
  Created 573 batches

Step 4: Initializing model
  Total parameters: 136832

Step 5: Creating Adam optimizer

Step 6: Training...
-------------------------------------------------------------------
Epoch 1/2, Batch 1/573, Loss: 5.4727, LR: 0.000100
Epoch 1/2, Batch 11/573, Loss: 5.5216, LR: 0.001000
Epoch 1/2, Batch 21/573, Loss: 5.6214, LR: 0.001000
...
Epoch 1/2, Batch 231/573, Loss: 5.6322, LR: 0.000919
Epoch 1/2, Batch 241/573, Loss: 5.5906, LR: 0.000911
Epoch 1/2, Batch 251/573, Loss: 5.5779, LR: 0.000904
```

**What you're seeing:**
- **Loss curves**: Starting around 5.47, briefly rising, then slowly decreasing as the model learns
- **Learning rate schedule**: Starts at 0.0001 (warmup), ramps to 0.001, then gradually decays with cosine annealing
- **~136K parameters**: Small enough to train quickly, large enough to learn basic patterns
- **Training time**: ~1-2 minutes on a laptop (M1/M2) or ~3-5 minutes on older CPUs

The model won't produce perfect Go code, but it learns basic syntax patterns, indentation, and common keywords. This demonstrates the full training pipeline working end-to-end.

### Visualizing Training Progress

Track your training progress with interactive HTML visualizations:

```bash
./local-code-model train -epochs 2 -batch 4 -seq 64 -metrics training_metrics.html
```

This creates a self-contained HTML file with:
- **Loss curve**: See how loss decreases during training
- **Learning rate schedule**: Visualize warmup → constant → decay
- **Summary statistics**: Final loss, min loss, average loss, total steps

The visualization opens in any browser with no external dependencies. Perfect for monitoring training runs, comparing hyperparameters, and understanding training dynamics.

### Visualizing Attention Patterns

Understand what your model "pays attention to" with interactive attention heatmaps:

```bash
./local-code-model visualize -model model.bin -tokenizer tok.bin -prompt "func main"
```

This generates an HTML visualization showing:
- **Attention weights**: Interactive heatmap showing which tokens attend to which
- **Layer and head selection**: Dropdown menus to explore different layers and attention heads
- **Token labels**: See exactly which tokens are attending to each other

The visualization helps you:
- Debug model behavior (why did it generate this token?)
- Understand learned patterns (does it attend to function names? keywords?)
- Explore multi-head attention (different heads learn different patterns)
- Educational tool for understanding transformer internals

Example output: attention.html (self-contained, opens in any browser)

### Token Embedding Visualization

Explore how your model represents tokens in high-dimensional embedding space:

```bash
./local-code-model embeddings -model model.bin -tokenizer tok.bin -method pca
```

This generates an interactive 2D visualization of token embeddings:
- **PCA** (Principal Component Analysis): Fast, preserves global structure
- **t-SNE** (t-Distributed Stochastic Neighbor Embedding): Slower, preserves local neighborhoods

The visualization shows:
- **Token clusters**: Semantically similar tokens appear close together
- **Interactive exploration**: Hover over points to see which token they represent
- **Embedding relationships**: Understand what the model learned about token similarity

This helps you:
- See which tokens the model considers similar (keywords, operators, identifiers)
- Validate embedding quality (are related tokens clustered?)
- Understand the model's internal representation
- Educational tool for understanding embeddings and dimensionality reduction

Example output: embeddings.html (self-contained, opens in any browser)

### Tokenizer Options

This project supports two tokenizer types:

**Character-level (`-tokenizer-type char`)** (default):
- Simple: Each unique character becomes a token
- Small vocabulary (~100-200 tokens for typical code)
- Good for learning and experimentation
- Fast training on small datasets

**Byte-Pair Encoding (`-tokenizer-type bpe`)**:
- More efficient: Learns subword units from data
- Larger vocabulary (configurable, default 512 tokens)
- Better for real-world training
- Handles rare words through composition

Example:
```bash
# Character-level (default, simple)
./local-code-model train -tokenizer-type char

# BPE (more efficient)
./local-code-model train -tokenizer-type bpe
```

### Options

```bash
# Training
./local-code-model train \
  -layers 2 \           # Number of transformer layers
  -embed 64 \           # Embedding dimension
  -heads 2 \            # Number of attention heads
  -seq 64 \             # Sequence length (context window)
  -epochs 2 \           # Training epochs
  -batch 4 \            # Batch size
  -lr 0.001 \           # Learning rate
  -data . \             # Directory with .go files
  -model model.bin \    # Output model path
  -tokenizer tok.bin \  # Output tokenizer path
  -tokenizer-type char \  # Tokenizer type: 'char' (default) or 'bpe'
  -metrics metrics.html # Optional: Save training metrics visualization

# Generation
./local-code-model generate \
  -model model.bin \
  -tokenizer tok.bin \
  -prompt "func " \
  -length 200 \         # Number of tokens to generate
  -temp 0.8 \           # Sampling temperature
  -topk 40              # Top-k sampling
```

## Learning Path

This codebase is designed to be read like a textbook. See **[LEARNING.md](LEARNING.md)** for a comprehensive guided tour through the code.

### Interactive Notebooks

For hands-on learning, try the **[Jupyter notebooks](notebooks/)** - interactive tutorials using gophernotes (Go kernel):

1. **[Tensor Basics](notebooks/01-tensor-basics.ipynb)** - Learn tensor operations, matrix multiplication, and activation functions
2. **[Attention Mechanism](notebooks/02-attention-mechanism.ipynb)** - Build self-attention from scratch with Q, K, V projections
3. **[Training a Transformer](notebooks/03-training-transformer.ipynb)** - Train a complete model end-to-end

See [notebooks/README.md](notebooks/README.md) for setup instructions (requires Jupyter + gophernotes).

### Code Reading Order

**Recommended reading order:**
1. `tensor.go` - Multi-dimensional arrays (like NumPy)
2. `autograd.go` - Automatic differentiation for backpropagation
3. `transformer.go` - The transformer architecture (forward pass)
4. `transformer_backward.go` - Backpropagation through the transformer
5. `train.go` - Training loop implementation
6. `cmd_train.go` - End-to-end training example

**Deep dive tutorials:**
- **[Understanding the Attention Mechanism](docs/attention-mechanism.md)** - How attention works, with intuition, mathematics, and practical examples
- **[Backpropagation Through Transformers](docs/backpropagation.md)** - How gradients flow backward and learning happens, with step-by-step walkthrough
- **[Training Dynamics and Loss Curves](docs/training-dynamics.md)** - How models learn, loss curves, learning rate schedules, and debugging training problems

**Time to understand:**
- Quick overview: 2-3 hours
- Deep understanding: 1-2 days
- Master the details: 1 week

## Architecture

```
Input Tokens
    ↓
Token Embedding + Positional Embedding
    ↓
┌─────────────────────────────────┐
│  N × Transformer Layers:        │
│    - Multi-Head Self-Attention  │
│    - Feed-Forward Network       │
│    - Layer Normalization        │
└─────────────────────────────────┘
    ↓
Final Layer Norm
    ↓
Output Projection (to vocabulary)
    ↓
Logits (probability distribution over next token)
```

**Default tiny model:**
- 2 layers, 64 embedding dimensions, 2 attention heads
- ~50K parameters (vs 125M for GPT-2 small)
- Trains in seconds, not hours
- Demonstrates the pipeline, not SOTA performance

## Project Structure

```
.
├── README.md                  # This file
├── LEARNING.md                # Comprehensive learning guide
├── go.mod                     # Go module definition
│
├── main.go                    # CLI entry point
├── cmd_train.go               # Training command implementation
├── cmd_generate.go            # Text generation command
│
├── tensor.go                  # Tensor operations (like NumPy)
├── autograd.go                # Automatic differentiation
├── transformer.go             # Transformer forward pass
├── transformer_backward.go    # Transformer backward pass
│
├── train.go                   # Training loop
├── loss.go                    # Loss functions
├── optimizer.go               # SGD & Adam optimizers
│
├── tokenizer.go               # Character-level tokenization
├── serialization.go           # Model save/load
├── visualization.go           # Training metrics visualization
└── model.go                   # Model utilities
```

## Performance Expectations

This is **educational code**, not production ML infrastructure. Performance characteristics:

| Operation | This Implementation | Production (GPU) |
|-----------|---------------------|------------------|
| MatMul (512×512) | ~10 ms | ~0.01 ms |
| Training tiny model | ~1 minute | ~1 second |
| Inference (1 token) | ~50 ms | ~0.1 ms |

**Why so slow?**
- Naive O(n³) matrix multiplication in pure Go
- No GPU, SIMD, or cache optimization
- Single-threaded execution
- Focused on clarity, not speed

**How to make it faster?**
The comments explain optimization paths: parallel Go routines → cache blocking → SIMD → GPU → Neural Engine (500-1000x faster).

## Key Concepts Demonstrated

After working through this codebase, you'll understand:

1. **Why transformers work:** Self-attention learns relationships between any positions
2. **How backpropagation works:** Chain rule applied systematically through computational graphs
3. **Why MatMul dominates:** O(n³) matrix multiplication happens everywhere
4. **Training dynamics:** Loss curves, gradient flow, optimizer behavior
5. **Transformer architecture:** Multi-head attention, feed-forward networks, layer normalization
6. **Why they're expensive:** O(n²) attention over sequence length

## What This Is NOT

❌ A recipe for training production models
❌ An optimized training harness
❌ A demonstration of ML best practices
❌ A replacement for PyTorch/TensorFlow

## What This IS

✅ Proof that the training infrastructure works
✅ A starting point for your own experiments
✅ An educational example of the full pipeline
✅ A reference implementation for learning
✅ A demonstration that transformers aren't magic

## Requirements

- Go 1.19 or later
- ~100MB disk space for training data + models
- 2GB+ RAM recommended

No external ML libraries required. Pure Go.

## Testing

```bash
# Run all tests
go test ./...

# Run with verbose output
go test -v ./...

# Run benchmarks
go test -bench=. -benchtime=3s
```

## Contributing

This is an educational project. Contributions that improve clarity, fix bugs, or add educational value are welcome. Please maintain the extensive commenting style.

**Before submitting a PR:**
```bash
go fmt ./...
go vet ./...
go test ./...
```

## Related Projects

- [llm.c](https://github.com/karpathy/llm.c) - LLM training in pure C
- [minGPT](https://github.com/karpathy/minGPT) - Minimal PyTorch GPT implementation
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Simplest, fastest GPT training

## References

**Papers:**
- "Attention Is All You Need" (Vaswani et al., 2017) - Original transformer paper
- "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019) - GPT-2

**Books:**
- "Deep Learning" (Goodfellow, Bengio, Courville, 2016) - Chapters 2, 6, 8
- "Numerical Linear Algebra" (Trefethen & Bau, 1997) - Matrix operations

**Blogs:**
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual explanations
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) - Detailed walkthrough

## Version

This project follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

Current version: **0.1.0** (Initial release)

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

## License

Apache License 2.0 - See LICENSE file for details.

## Acknowledgments

Inspired by Andrej Karpathy's educational ML projects that demystify deep learning through clear, readable code.

---

**Questions?** Open an issue or check [LEARNING.md](LEARNING.md) for detailed explanations.

**Want to learn more?** Read the code! It's designed to be understood.
