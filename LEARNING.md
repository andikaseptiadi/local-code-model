# Learning Path: Building a Transformer from Scratch

This codebase implements a GPT-style transformer entirely in Go, with extensive educational comments. It's designed to be read and understood, not just used.

## Philosophy

This project prioritizes **clarity over performance**. The code is deliberately naive to make the core concepts visible. Each file contains detailed "WHAT'S GOING ON HERE" sections that explain the theory, mathematics, and design decisions.

## Recommended Learning Path

### Level 0: Foundation (Start Here)

Read these files in order to understand the building blocks:

#### 1. **tensor.go** (429 lines, 31% comments)
**Start here.** This implements multi-dimensional arrays - the fundamental data structure for neural networks.

**Key concepts:**
- Tensor abstraction (like NumPy arrays)
- Row-major memory layout
- Basic operations: Add, Mul, MatMul, Transpose
- Activation functions: ReLU, GELU, Softmax
- Performance characteristics (why MatMul dominates everything)

**Read:** Lines 1-72 (header), then skim the operations starting at line 245.

**Important functions:**
- `NewTensor()` - tensor creation (line 100)
- `MatMul()` - matrix multiplication, the heart of neural networks (line 292)
- `Softmax()` - converts logits to probabilities (line 380)
- `GELU()` - activation function used in transformers (line 358)

---

#### 2. **autograd.go** (358 lines, 50% comments)
**Next.** This implements automatic differentiation for backpropagation.

**Key concepts:**
- The chain rule: how gradients flow backward through operations
- Forward vs. backward pass
- Why backward pass costs 2x forward pass
- Gradient computation for each operation

**Read:** Lines 1-41 (header explaining chain rule), then focus on:
- `MatMulBackward()` - line 65 (most important!)
- `ReLUBackward()` - line 129 (simpler example)
- `CrossEntropyBackward()` - line 318 (loss function gradient)

**Mathematics recap:**
```
Forward:  C = A @ B
Backward: ∂L/∂A = gradC @ B^T
          ∂L/∂B = A^T @ gradC
```

---

### Level 1: Building Blocks

#### 3. **tokenizer.go**
Character-level tokenization (simplest approach).

**Key concepts:**
- Converting text → integers → text
- Vocabulary building
- Why BPE/WordPiece are better (but more complex)

**Skim this - it's straightforward mapping.**

---

#### 4. **optimizer.go**
SGD and Adam optimizers.

**Key concepts:**
- How learning rates work
- Why Adam uses momentum and adaptive learning rates
- Learning rate scheduling

**Focus on:**
- `AdamOptimizer.Step()` - the core update rule
- Why we need separate learning rates per parameter

---

### Level 2: The Transformer Architecture

Now you're ready for the main event. Read these together:

#### 5. **transformer.go** (1023 lines, 26% comments) - FORWARD PASS
This implements the forward pass: input tokens → output logits.

**Architecture overview (read lines 1-90 first):**
```
Input: Token IDs
  ↓
Token Embedding + Position Embedding
  ↓
N × Transformer Layers:
    - Multi-Head Attention  (learns relationships between tokens)
    - Feed-Forward Network  (processes each position independently)
    - Layer Normalization   (stabilizes training)
  ↓
Output: Logits (scores for each vocabulary token)
```

**Read in this order:**

1. **Config struct** (line 92) - hyperparameters explained
2. **GPT struct** (line 123) - model architecture
3. **Attention.Forward()** (line 264) - THE CORE MECHANISM
   - This is where transformers learn relationships between words
   - Query, Key, Value projections
   - Attention scores = how much each token "attends" to others
   - Multi-head: running multiple attention operations in parallel
4. **FeedForward.Forward()** (line 432) - simple MLP applied to each position
5. **TransformerLayer.Forward()** (line 478) - combines attention + FF
6. **GPT.Forward()** (line 546) - ties everything together

**Most important:** Spend time on the Attention mechanism (lines 234-420). This is what made transformers revolutionary.

---

#### 6. **transformer_backward.go** (523 lines, 29% comments) - BACKWARD PASS
This implements backpropagation through the entire transformer.

**Why this is complex:**
- Must reverse every operation from forward pass
- Multi-head attention requires careful reshaping
- Gradients must accumulate correctly

**Read after understanding transformer.go forward pass.**

**Key sections:**
1. **Cache structures** (lines 77-114) - what we save from forward pass
2. **GPT.Backward()** (line 284) - top-level backward
3. **Attention.Backward()** (line 417) - most complex part
4. **FeedForward.Backward()** (line 388) - simpler example

**Pro tip:** Compare each backward function with its forward counterpart in transformer.go.

---

### Level 3: Training & Inference

#### 7. **train.go** (586 lines, 33% comments)
Implements the training loop.

**Key concepts:**
- Forward → Loss → Backward → Optimizer step
- Gradient accumulation
- Why we zero gradients each step
- Cross-entropy loss for language modeling

**Read:**
- `TrainStep()` - line 505 - single training iteration
- `CrossEntropyLoss()` - line 375 - how we measure prediction quality

---

#### 8. **cmd_train.go** (395 lines, 31% comments)
End-to-end training example with extensive documentation.

**Read the header (lines 1-73) for:**
- Why the model is tiny (for fast experimentation)
- What to expect (loss curves, generated text quality)
- Design decisions explained

**Then trace through `RunTrainCommand()`:**
1. Load data (Go source files)
2. Build tokenizer
3. Create batches
4. Train!
5. Save model
6. Generate samples

**This is the capstone** - see all pieces working together.

---

#### 9. **cmd_generate.go**
Inference: generate text from a trained model.

**Key concepts:**
- Autoregressive generation (one token at a time)
- Temperature sampling
- Top-k/top-p filtering
- Why generation is slow (no KV cache)

---

### Level 4: Advanced Topics (Optional)

#### **loss.go** - Loss functions with detailed math

#### **serialization.go** - Model saving/loading

#### **model.go** - Additional utilities

---

## How to Learn Effectively

### 1. **Read the Headers First**
Every file starts with a detailed explanation. Read these before diving into code.

### 2. **Follow the Data Flow**
Track a single example through the system:
- Text: "func main"
- Tokens: [102, 117, 110, 99, ...]
- Embeddings: (seqLen, embedDim) tensor
- Through transformer layers
- Logits: (seqLen, vocabSize) tensor
- Sample next token
- Repeat

### 3. **Understand Tensor Shapes**
The most common bugs are shape mismatches. Always ask:
- What shape is this tensor?
- What shape does the next operation expect?

**Common shapes:**
- Embeddings: `(seqLen, embedDim)`
- Attention Q,K,V: `(seqLen, embedDim)`
- After multi-head split: `(seqLen, numHeads, headDim)`
- Logits: `(seqLen, vocabSize)`

### 4. **Run the Code**
Don't just read - experiment!

```bash
# Train a tiny model (runs in ~1 minute)
go run . train -epochs 2 -batch 4 -seq 32

# Generate text
go run . generate -model tiny_model.bin -tokenizer tiny_tokenizer.bin -prompt "func main"
```

### 5. **Read Forward and Backward Together**
For each operation (Attention, FeedForward), read:
1. Forward pass in transformer.go
2. Backward pass in transformer_backward.go
3. Compare them - backward should mirror forward

### 6. **Consult the Math**
The code includes gradient derivations. When confused:
- Look at the comments explaining ∂L/∂x
- Verify the shapes match the mathematics
- Work through a small example on paper

---

## Key Insights You'll Gain

After working through this codebase, you'll understand:

1. **Why transformers work:** Self-attention learns relationships between any positions
2. **Why they're fast:** All positions processed in parallel (unlike RNNs)
3. **Why they're expensive:** O(n²) attention over sequence length
4. **How backprop works:** Chain rule applied systematically
5. **Why MatMul dominates:** It's O(n³) and happens everywhere
6. **Training dynamics:** Loss curves, learning rates, gradient flow

---

## Common Questions

### "Why is this so slow?"
This is intentionally naive Go code for learning. Production transformers use:
- GPU acceleration (50-200x faster)
- Optimized BLAS libraries
- Mixed precision training
- KV caching for inference

### "Why character-level tokenization?"
Simplest approach for a demo. Real models use BPE or WordPiece (better, but more complex).

### "Why such a tiny model?"
To train in seconds, not hours. The architecture scales to billions of parameters.

### "Where's the attention mask?"
Causal masking (preventing attention to future tokens) is implicit in our generation code. A full implementation would include explicit masks.

---

## Further Reading

The code includes references to key papers and textbooks. After understanding this implementation:

1. **"Attention Is All You Need"** (Vaswani et al., 2017) - Original transformer paper
2. **"Deep Learning"** (Goodfellow, Bengio, Courville) - Chapters 2, 6, 8
3. **"The Illustrated Transformer"** (Jay Alammar) - Visual explanations
4. **GPT-2 paper** (Radford et al., 2019) - Architecture this code follows

---

## Getting Help

If you get stuck:
1. Re-read the header comments in each file
2. Print tensor shapes to debug dimension mismatches
3. Trace through a tiny example (seq length = 2, embed dim = 4)
4. Compare forward and backward passes side-by-side

---

## Summary: Reading Order

**Minimum path (understand the basics):**
1. tensor.go - header + MatMul
2. autograd.go - header + MatMulBackward
3. transformer.go - header + Attention.Forward
4. cmd_train.go - header + RunTrainCommand

**Complete path (understand everything):**
1. tensor.go (all)
2. autograd.go (all)
3. tokenizer.go (skim)
4. optimizer.go (focus on Adam)
5. transformer.go (all - take your time!)
6. transformer_backward.go (all)
7. train.go (TrainStep + loss functions)
8. cmd_train.go (complete example)
9. cmd_generate.go (inference)

**Time investment:**
- Quick overview: 2-3 hours
- Deep understanding: 1-2 days
- Master the details: 1 week

The code is designed to be read like a textbook. Take your time, work through the examples, and you'll emerge with a deep understanding of how transformers really work.

Happy learning!
