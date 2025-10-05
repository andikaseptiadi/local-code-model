# Transformer Mathematics: A Visual Guide

This guide explains the mathematical foundations of transformers with clear examples and visual representations.

## Table of Contents

1. [Attention Mechanism Mathematics](#attention-mechanism-mathematics)
2. [Multi-Head Attention](#multi-head-attention)
3. [Position Encodings](#position-encodings)
4. [Layer Normalization](#layer-normalization)
5. [Feed-Forward Networks](#feed-forward-networks)
6. [Backpropagation Through Transformers](#backpropagation-through-transformers)

---

## Attention Mechanism Mathematics

### The Core Formula

**Attention(Q, K, V) = softmax(QK^T / √d_k) · V**

This simple formula is the heart of transformers. Let's break it down step by step.

### Step 1: Query-Key Similarity

```
Given:
  Q (Queries):  [seq_len, d_k]  - "What am I looking for?"
  K (Keys):     [seq_len, d_k]  - "What do I contain?"
  V (Values):   [seq_len, d_v]  - "What information do I have?"

Compute similarity scores:
  Scores = Q · K^T
  Shape: [seq_len, seq_len]
```

**Example with numbers:**

```
Sentence: "The cat sat"
Tokens:    [0]  [1]  [2]
          "The" "cat" "sat"

Q (simplified, d_k=3):
  The: [1.0, 0.2, 0.1]
  cat: [0.3, 1.5, 0.4]
  sat: [0.2, 0.5, 1.8]

K (same as Q for self-attention):
  The: [1.0, 0.2, 0.1]
  cat: [0.3, 1.5, 0.4]
  sat: [0.2, 0.5, 1.8]

Compute Q·K^T:
           The   cat   sat
  The:   [1.05  0.74  0.49]  ← "The" attention to all tokens
  cat:   [0.74  2.66  1.51]  ← "cat" attention to all tokens
  sat:   [0.49  1.51  3.53]  ← "sat" attention to all tokens
```

**Interpretation**: Higher scores mean stronger relationships.
- "cat" has strong self-attention (2.66)
- "sat" attends strongly to itself (3.53) and moderately to "cat" (1.51)

### Step 2: Scaling

```
Scaled scores = Scores / √d_k
```

**Why scale?** For large d_k, dot products grow large, pushing softmax into regions with tiny gradients.

```
With d_k = 3, √d_k ≈ 1.73

Scaled scores:
           The   cat   sat
  The:   [0.61  0.43  0.28]
  cat:   [0.43  1.54  0.87]
  sat:   [0.28  0.87  2.04]
```

### Step 3: Causal Masking (for GPT)

```
For autoregressive models, mask future tokens:

Mask (apply -∞ to future positions):
           The   cat   sat
  The:   [0.61  -∞    -∞  ]  ← "The" can only see itself
  cat:   [0.43  1.54  -∞  ]  ← "cat" can see "The" and itself
  sat:   [0.28  0.87  2.04]  ← "sat" can see all previous
```

**For BERT**: No masking (bidirectional attention)

### Step 4: Softmax

```
Attention weights = softmax(Scaled scores)
```

Convert scores to probabilities (sum to 1.0 per row):

```
GPT (with causal mask):
           The    cat    sat
  The:   [1.00   0.00   0.00]  ← 100% attention to itself
  cat:   [0.23   0.77   0.00]  ← 23% to "The", 77% to "cat"
  sat:   [0.11   0.29   0.60]  ← 11% to "The", 29% to "cat", 60% to "sat"

BERT (no mask):
           The    cat    sat
  The:   [0.40   0.35   0.25]  ← Distributed attention
  cat:   [0.13   0.69   0.18]  ← Focuses on "cat"
  sat:   [0.10   0.24   0.66]  ← Focuses on "sat"
```

### Step 5: Weighted Sum of Values

```
Output = Attention_weights · V
Shape: [seq_len, d_v]
```

**Example:**

```
V (values, d_v=4):
  The: [0.5, 0.2, 0.1, 0.3]
  cat: [0.8, 0.9, 0.4, 0.2]
  sat: [0.3, 0.5, 1.2, 0.7]

For token "sat" in GPT:
  Output_sat = 0.11 × V_The + 0.29 × V_cat + 0.60 × V_sat
             = 0.11 × [0.5, 0.2, 0.1, 0.3] +
               0.29 × [0.8, 0.9, 0.4, 0.2] +
               0.60 × [0.3, 0.5, 1.2, 0.7]
             = [0.46, 0.61, 0.86, 0.50]
```

**Interpretation**: "sat" is a mixture of information from all previous tokens, weighted by attention.

---

## Multi-Head Attention

### The Concept

Instead of one attention mechanism, use **h parallel heads**, each learning different relationships.

```
MultiHead(Q, K, V) = Concat(head₁, ..., head_h) · W_O

where head_i = Attention(Q·W_Q^i, K·W_K^i, V·W_V^i)
```

### Example with 2 Heads

```
Given: d_model = 512, h = 8
Each head: d_k = d_v = d_model / h = 64

Head 1 might learn: syntactic relationships (subject-verb)
Head 2 might learn: semantic relationships (related concepts)
...
Head 8 might learn: positional relationships (nearby words)
```

**Numerical example (simplified):**

```
Input: "The cat sat"
d_model = 4, h = 2, d_k = d_v = 2 per head

Head 1 (syntactic):
  Attention weights for "sat":
    The: 0.2  (weak - not syntactically related)
    cat: 0.7  (strong - subject of "sat")
    sat: 0.1  (weak - doesn't attend to itself much)

Head 2 (positional):
  Attention weights for "sat":
    The: 0.1  (far away)
    cat: 0.3  (nearby)
    sat: 0.6  (current position)

Concatenate heads:
  head1_output: [0.5, 0.8]
  head2_output: [0.3, 0.6]
  concat: [0.5, 0.8, 0.3, 0.6]  ← Length 4 (2+2)

Final projection (W_O):
  Output = concat · W_O
  Shape: [4] → [4] (d_model)
```

### Why Multiple Heads?

1. **Diversity**: Each head specializes in different patterns
2. **Expressiveness**: Can attend to multiple positions simultaneously
3. **Robustness**: Multiple views of the same data

---

## Position Encodings

Transformers have no inherent notion of order. Position encodings add this information.

### 1. Sinusoidal Position Encoding (Original Transformer)

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

where:
  pos = position in sequence (0, 1, 2, ...)
  i = dimension index (0, 1, 2, ..., d_model/2)
```

**Example:**

```
Position 0 ("The"):
  dim 0: sin(0 / 10000^(0/512)) = sin(0) = 0.0
  dim 1: cos(0 / 10000^(0/512)) = cos(0) = 1.0
  dim 2: sin(0 / 10000^(2/512)) = sin(0) = 0.0
  dim 3: cos(0 / 10000^(2/512)) = cos(0) = 1.0
  ...
  PE_0 = [0.0, 1.0, 0.0, 1.0, ...]

Position 1 ("cat"):
  dim 0: sin(1 / 10000^(0/512)) = sin(1) = 0.841
  dim 1: cos(1 / 10000^(0/512)) = cos(1) = 0.540
  dim 2: sin(1 / 10000^(2/512)) ≈ 0.001
  dim 3: cos(1 / 10000^(2/512)) ≈ 1.0
  ...
  PE_1 = [0.841, 0.540, 0.001, 1.0, ...]
```

**Properties:**
- **Unique**: Each position gets a unique encoding
- **Relative**: PE(pos+k) can be expressed as a linear function of PE(pos)
- **Unbounded**: Works for any sequence length

### 2. Learned Position Embeddings (BERT)

```
Position embeddings = lookup table[position]
Shape: [max_seq_len, d_model]
```

**Example:**

```
max_seq_len = 512
d_model = 768

Position embedding lookup:
  pos 0: randomly initialized [0.12, -0.45, 0.78, ..., 0.33]  (768 dims)
  pos 1: randomly initialized [-0.23, 0.67, -0.11, ..., 0.88]
  ...
  pos 511: randomly initialized [0.45, 0.22, -0.67, ..., -0.12]

These are learned during training.
```

### 3. Rotary Position Embedding (RoPE)

Used in modern LLMs (LLaMA, GPT-NeoX). Rotates Q and K based on position.

```
RoPE(x, pos) = [
  x_0 · cos(pos·θ_0) - x_1 · sin(pos·θ_0),
  x_0 · sin(pos·θ_0) + x_1 · cos(pos·θ_0),
  x_2 · cos(pos·θ_1) - x_3 · sin(pos·θ_1),
  x_2 · sin(pos·θ_1) + x_3 · cos(pos·θ_1),
  ...
]

where θ_i = 10000^(-2i/d)
```

**Advantages:**
- Maintains relative position information in dot products
- Better extrapolation to longer sequences
- No explicit position embedding added to input

---

## Layer Normalization

Normalizes activations across features (not batch).

```
LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β

where:
  μ = mean(x) across features
  σ² = variance(x) across features
  γ, β = learnable scale and shift parameters
  ε = small constant (1e-5) for numerical stability
```

### Example

```
Input x (one token, d_model=4):
  x = [1.0, 3.0, 2.0, 4.0]

Step 1: Compute statistics
  μ = (1.0 + 3.0 + 2.0 + 4.0) / 4 = 2.5
  σ² = ((1.0-2.5)² + (3.0-2.5)² + (2.0-2.5)² + (4.0-2.5)²) / 4
     = (2.25 + 0.25 + 0.25 + 2.25) / 4
     = 1.25
  σ = √1.25 = 1.118

Step 2: Normalize
  x_norm = (x - μ) / σ
         = (x - 2.5) / 1.118
         = [-1.34, 0.45, -0.45, 1.34]

Step 3: Scale and shift (assume γ=1, β=0 initially)
  output = γ · x_norm + β
         = 1.0 · [-1.34, 0.45, -0.45, 1.34] + 0.0
         = [-1.34, 0.45, -0.45, 1.34]
```

**Properties:**
- Mean = 0, Variance = 1 (after normalization)
- Stabilizes training
- Allows higher learning rates

### RMSNorm (Modern Alternative)

Simpler, faster variant used in LLaMA:

```
RMSNorm(x) = γ · x / RMS(x)

where RMS(x) = √(mean(x²) + ε)
```

**Example:**

```
x = [1.0, 3.0, 2.0, 4.0]

RMS(x) = √((1² + 3² + 2² + 4²) / 4)
       = √((1 + 9 + 4 + 16) / 4)
       = √(7.5)
       = 2.74

x_norm = x / 2.74
       = [0.36, 1.09, 0.73, 1.46]
```

**Advantage**: No mean subtraction → ~15% faster

---

## Feed-Forward Networks

After attention, each position passes through a feed-forward network.

```
FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

where:
  W₁: [d_model, d_ff] (typically d_ff = 4 × d_model)
  W₂: [d_ff, d_model]
```

### Example

```
d_model = 4, d_ff = 16 (4×)

Input x = [0.5, 0.3, -0.2, 0.8]

Step 1: First linear layer
  W₁ (4×16 matrix, simplified to 4×4):
    [[0.2, 0.1, 0.3, 0.4],
     [0.5, 0.2, 0.1, 0.3],
     [0.1, 0.4, 0.2, 0.5],
     [0.3, 0.1, 0.5, 0.2]]

  h = x · W₁ + b₁
    = [0.5, 0.3, -0.2, 0.8] · W₁ + b₁
    ≈ [0.45, 0.23, 0.67, 0.58] (after bias)

Step 2: ReLU activation
  h_relu = max(0, h)
         = [0.45, 0.23, 0.67, 0.58]  (all positive, unchanged)

Step 3: Second linear layer
  W₂ (16×4, simplified):
    [[0.1, 0.2, 0.3, 0.1],
     [0.4, 0.1, 0.2, 0.3],
     [0.2, 0.3, 0.1, 0.4],
     [0.3, 0.4, 0.2, 0.1]]

  output = h_relu · W₂ + b₂
         ≈ [0.62, 0.51, 0.43, 0.54]
```

### Modern Alternatives

#### SwiGLU (Used in LLaMA)

```
SwiGLU(x) = Swish(x·W₁) ⊙ (x·V)

where:
  Swish(x) = x · σ(βx) = x / (1 + e^(-βx))
  ⊙ = element-wise multiplication
  W₁, V: separate weight matrices
```

**Example:**

```
x = [0.5, 0.3, -0.2, 0.8]

Gate: Swish(x·W₁) = [0.42, 0.19, 0.53, 0.45]
Value: x·V = [0.68, 0.34, 0.22, 0.71]

output = [0.42, 0.19, 0.53, 0.45] ⊙ [0.68, 0.34, 0.22, 0.71]
       = [0.29, 0.06, 0.12, 0.32]
```

**Advantages:**
- Better gradient flow
- ~5% improvement in model quality

---

## Backpropagation Through Transformers

### Chain Rule Through Attention

```
Loss → Output → Attention → Softmax → Scores → Q, K, V
```

#### Step 1: Gradient of Attention Output

```
Given:
  Attention = softmax(QK^T / √d_k) · V
  Let A = softmax(QK^T / √d_k)  (attention weights)

Forward: output = A · V

Backward:
  ∂Loss/∂V = A^T · ∂Loss/∂output
  ∂Loss/∂A = ∂Loss/∂output · V^T
```

**Numerical example:**

```
∂Loss/∂output = [0.1, 0.2, 0.3]  (gradient from next layer)
V = [[0.5, 0.2, 0.1],
     [0.8, 0.9, 0.4],
     [0.3, 0.5, 1.2]]
A = [[1.0, 0.0, 0.0],
     [0.2, 0.8, 0.0],
     [0.1, 0.3, 0.6]]

∂Loss/∂V = A^T · ∂Loss/∂output
         = [[1.0, 0.2, 0.1],    [[0.1],
            [0.0, 0.8, 0.3],  ·  [0.2],
            [0.0, 0.0, 0.6]]     [0.3]]
         = [[0.17],
            [0.25],
            [0.18]]
```

#### Step 2: Gradient of Softmax

```
∂Loss/∂scores = ∂Loss/∂A · ∂A/∂scores

For softmax, the Jacobian is:
  ∂A_i/∂scores_j = A_i · (δ_ij - A_j)

In practice:
  ∂Loss/∂scores = A ⊙ (∂Loss/∂A - sum(∂Loss/∂A ⊙ A))
```

#### Step 3: Gradient of Q and K

```
scores = QK^T / √d_k

∂Loss/∂Q = (∂Loss/∂scores · K) / √d_k
∂Loss/∂K = (∂Loss/∂scores^T · Q) / √d_k
```

### Memory Requirement

For sequence length N:
- **Forward pass**: Store attention weights A [N, N] and intermediate activations
- **Backward pass**: Need to recompute or store all intermediate values

**Memory usage**:
```
Forward: O(N² × d_model)
Backward: O(N² × d_model)
Total: O(2 × N² × d_model)
```

### Optimization: Gradient Checkpointing

Trade compute for memory:
```
Forward: Don't store all activations
Backward: Recompute activations when needed

Memory: O(√N × N × d_model) ← Much better!
Compute: +33% (need to recompute)
```

---

## Putting It All Together

### Full Forward Pass

```
Input: token embeddings + position embeddings

For each layer (repeated N times):
  1. Multi-head attention
     x = LayerNorm(x + MultiHeadAttention(x, x, x))

  2. Feed-forward
     x = LayerNorm(x + FFN(x))

Output: x
```

### Numerical Example (Tiny Transformer)

```
Input: "The cat"
Vocab: {The: 0, cat: 1}
d_model = 4, num_heads = 2, num_layers = 2

Step 1: Token embeddings
  The (id=0): embedding_table[0] = [0.2, 0.5, 0.3, 0.1]
  cat (id=1): embedding_table[1] = [0.8, 0.2, 0.6, 0.4]

Step 2: Position embeddings (learned)
  pos 0: [0.1, 0.0, 0.1, 0.0]
  pos 1: [0.0, 0.1, 0.0, 0.1]

Step 3: Add embeddings
  The: [0.2, 0.5, 0.3, 0.1] + [0.1, 0.0, 0.1, 0.0] = [0.3, 0.5, 0.4, 0.1]
  cat: [0.8, 0.2, 0.6, 0.4] + [0.0, 0.1, 0.0, 0.1] = [0.8, 0.3, 0.6, 0.5]

Layer 1:
  Step 4: Multi-head attention
    (Q, K, V projections, attention, output projection)
    → output: [[0.4, 0.6, 0.3, 0.2],
               [0.7, 0.4, 0.5, 0.6]]

  Step 5: Residual + LayerNorm
    x = LayerNorm([0.3, 0.5, 0.4, 0.1] + [0.4, 0.6, 0.3, 0.2])
      = LayerNorm([0.7, 1.1, 0.7, 0.3])
      ≈ [-0.5, 1.2, -0.5, -1.2]  (normalized)

  Step 6: Feed-forward
    → output: [[0.5, 0.8, 0.4, 0.3],
               [0.9, 0.5, 0.7, 0.6]]

  Step 7: Residual + LayerNorm
    (similar to step 5)

Layer 2: (repeat steps 4-7)

Final output:
  The: [0.62, 0.81, 0.43, 0.35]
  cat: [0.87, 0.52, 0.73, 0.68]
```

---

## Key Takeaways

### Attention is a Weighted Average
```
output = Σ attention_weight_i × value_i
```
Attention weights decide which inputs to focus on.

### Scaling is Critical
```
scores / √d_k
```
Prevents softmax saturation for large dimensions.

### Multi-Head = Multiple Perspectives
Each head learns different patterns independently.

### Position Encodings Add Order
Transformers are order-agnostic without position information.

### Layer Norm Stabilizes Training
Keeps activations in reasonable ranges.

### FFN Adds Non-Linearity
Attention is linear; FFN adds expressiveness.

---

## Further Reading

- Original Paper: "Attention is All You Need" (Vaswani et al., 2017)
- Our implementations: `transformer.go`, `attention.go`, `tensor.go`
- Visual attention: See `docs/attention-mechanism.md`
- Training guide: See `docs/training-workflows.md`
