# Backpropagation Through Transformers

**Understanding How Gradients Flow and Learning Happens**

## Table of Contents

1. [Introduction](#introduction)
2. [The Big Picture](#the-big-picture)
3. [Chain Rule Fundamentals](#chain-rule-fundamentals)
4. [Backward Pass Through Attention](#backward-pass-through-attention)
5. [Layer Normalization Gradients](#layer-normalization-gradients)
6. [Complete Transformer Backward Pass](#complete-transformer-backward-pass)
7. [Common Gradient Problems](#common-gradient-problems)
8. [Implementation Walkthrough](#implementation-walkthrough)
9. [Practical Exercises](#practical-exercises)

## Introduction

Backpropagation is how neural networks learn. It's the algorithm that computes gradients—telling us how to adjust each weight to reduce the loss.

For transformers, backpropagation is particularly interesting because:
1. **Deep networks**: Gradients must flow through many layers
2. **Complex operations**: Attention, layer norm, and residual connections
3. **Gradient flow challenges**: Vanishing/exploding gradients are real concerns

This tutorial walks through the backward pass step-by-step, showing exactly how gradients propagate from the loss back to every parameter.

## The Big Picture

### Forward Pass (Inference)

```
Input → Embedding → Layer 1 → Layer 2 → ... → Layer N → Output → Loss
```

Data flows forward, transforming the input into predictions.

### Backward Pass (Learning)

```
Loss → ∂L/∂Output → ... → ∂L/∂Layer2 → ∂L/∂Layer1 → ∂L/∂Embedding → ∂L/∂Weights
```

Gradients flow backward, computing how the loss changes with respect to each parameter.

### Key Insight

The forward pass computes **values**. The backward pass computes **derivatives** of the loss with respect to those values.

## Chain Rule Fundamentals

Backpropagation is just the **chain rule** applied systematically.

### Simple Example

Given: `y = f(g(x))`

By the chain rule:
```
dy/dx = (dy/dg) × (dg/dx)
```

### Neural Network Example

Given: `Loss = L(softmax(matmul(x, W)))`

Chain rule expansion:
```
∂L/∂W = (∂L/∂softmax) × (∂softmax/∂matmul) × (∂matmul/∂W)
```

Each "×" is a matrix multiplication or element-wise product, depending on the shapes.

### The Pattern

For any operation `y = f(x)`:
1. **Forward pass**: Compute `y = f(x)` and save `x`
2. **Backward pass**: Given `∂L/∂y`, compute `∂L/∂x = ∂L/∂y × ∂y/∂x`

## Backward Pass Through Attention

Attention is the most complex part of the transformer. Let's break down its backward pass.

### Forward Pass Recap

```go
// 1. Compute Q, K, V
Q = x.MatMul(W_Q)
K = x.MatMul(W_K)
V = x.MatMul(W_V)

// 2. Compute attention scores
scores = Q.MatMul(K.Transpose()) / sqrt(d_k)

// 3. Apply softmax
attn_weights = softmax(scores)

// 4. Compute output
output = attn_weights.MatMul(V)
```

### Backward Pass

Now we work backward. Suppose we have `∂L/∂output` from the next layer.

#### Step 1: Gradient w.r.t. attn_weights and V

```
output = attn_weights × V

∂L/∂attn_weights = ∂L/∂output × V^T
∂L/∂V = attn_weights^T × ∂L/∂output
```

**Why?** This is the gradient of matrix multiplication. If `C = A × B`, then:
- `∂L/∂A = ∂L/∂C × B^T`
- `∂L/∂B = A^T × ∂L/∂C`

#### Step 2: Gradient w.r.t. scores (softmax backward)

Softmax is tricky. The gradient is:
```
∂L/∂scores[i,j] = attn_weights[i,j] × (∂L/∂attn_weights[i,j] - Σ_k attn_weights[i,k] × ∂L/∂attn_weights[i,k])
```

In vectorized form:
```go
// Compute the sum term for each row
sum_term := ∂L/∂attn_weights * attn_weights  // element-wise
sum_term = sum_term.Sum(axis=1)  // sum across columns

// Apply softmax gradient formula
∂L/∂scores = attn_weights * (∂L/∂attn_weights - sum_term)
```

**Intuition**: Softmax couples all outputs. Changing one logit affects all probabilities.

#### Step 3: Gradient w.r.t. Q and K

```
scores = Q × K^T / sqrt(d_k)

∂L/∂Q = (∂L/∂scores × K) / sqrt(d_k)
∂L/∂K = (∂L/∂scores^T × Q) / sqrt(d_k)
```

#### Step 4: Gradient w.r.t. W_Q, W_K, W_V

```
Q = x × W_Q

∂L/∂W_Q = x^T × ∂L/∂Q
∂L/∂x = ∂L/∂Q × W_Q^T
```

Similarly for W_K and W_V.

#### Step 5: Accumulate input gradients

Since Q, K, V all come from the same input `x`:
```
∂L/∂x_total = ∂L/∂x_from_Q + ∂L/∂x_from_K + ∂L/∂x_from_V
```

### Complete Attention Backward

Putting it all together:

```go
func AttentionBackward(
    ∂L_∂output *Tensor,  // gradient from next layer
    x, Q, K, V *Tensor,   // saved from forward pass
    attn_weights *Tensor, // saved from forward pass
    W_Q, W_K, W_V *Tensor, // parameters
) (∂L_∂x, ∂L_∂W_Q, ∂L_∂W_K, ∂L_∂W_V *Tensor) {

    // Step 1: Output → attn_weights and V
    ∂L_∂attn_weights := ∂L_∂output.MatMul(V.Transpose())
    ∂L_∂V := attn_weights.Transpose().MatMul(∂L_∂output)

    // Step 2: Softmax backward
    ∂L_∂scores := SoftmaxBackward(∂L_∂attn_weights, attn_weights)

    // Step 3: Scores → Q and K
    ∂L_∂Q := ∂L_∂scores.MatMul(K).DivScalar(math.Sqrt(float64(d_k)))
    ∂L_∂K := ∂L_∂scores.Transpose().MatMul(Q).DivScalar(math.Sqrt(float64(d_k)))

    // Step 4: Q, K, V → weights
    ∂L_∂W_Q := x.Transpose().MatMul(∂L_∂Q)
    ∂L_∂W_K := x.Transpose().MatMul(∂L_∂K)
    ∂L_∂W_V := x.Transpose().MatMul(∂L_∂V)

    // Step 5: Q, K, V → input
    ∂L_∂x_Q := ∂L_∂Q.MatMul(W_Q.Transpose())
    ∂L_∂x_K := ∂L_∂K.MatMul(W_K.Transpose())
    ∂L_∂x_V := ∂L_∂V.MatMul(W_V.Transpose())

    ∂L_∂x := ∂L_∂x_Q.Add(∂L_∂x_K).Add(∂L_∂x_V)

    return ∂L_∂x, ∂L_∂W_Q, ∂L_∂W_K, ∂L_∂W_V
}
```

## Layer Normalization Gradients

Layer normalization normalizes across the feature dimension:

```
y = (x - mean) / sqrt(variance + epsilon) * gamma + beta
```

### Forward Pass

```go
mean := x.Mean(axis=1)  // mean across features
variance := x.Variance(axis=1)  // variance across features
x_normalized := (x - mean) / sqrt(variance + epsilon)
output := x_normalized * gamma + beta
```

### Backward Pass

The gradient computation is complex because mean and variance depend on all elements.

```go
func LayerNormBackward(
    ∂L_∂output *Tensor,  // gradient from next layer
    x *Tensor,           // saved from forward pass
    x_normalized *Tensor, // saved from forward pass
    gamma *Tensor,       // learned scale parameter
    variance *Tensor,    // saved from forward pass
) (∂L_∂x, ∂L_∂gamma, ∂L_∂beta *Tensor) {

    // Gradient w.r.t. gamma and beta (easy part)
    ∂L_∂gamma := (∂L_∂output * x_normalized).Sum(axis=0)
    ∂L_∂beta := ∂L_∂output.Sum(axis=0)

    // Gradient w.r.t. normalized x
    ∂L_∂x_normalized := ∂L_∂output * gamma

    // Gradient w.r.t. x (hard part - mean and variance couple everything)
    N := float64(x.shape[1])  // number of features
    std := sqrt(variance + epsilon)

    // This formula accounts for how x affects both the numerator and denominator
    ∂L_∂variance := (∂L_∂x_normalized * (x - mean)).Sum(axis=1) * (-0.5) * pow(std, -3)
    ∂L_∂mean := (∂L_∂x_normalized / -std).Sum(axis=1) + ∂L_∂variance * (-2.0 * (x - mean)).Sum(axis=1) / N

    ∂L_∂x := (∂L_∂x_normalized / std) +
             (∂L_∂variance * 2.0 * (x - mean) / N) +
             (∂L_∂mean / N)

    return ∂L_∂x, ∂L_∂gamma, ∂L_∂beta
}
```

**Key Insight**: Every element of `x` affects the mean and variance, so every element's gradient depends on all other elements.

## Complete Transformer Backward Pass

A transformer layer looks like:

```
x → LayerNorm → Attention → Add(x) → LayerNorm → FFN → Add(x) → output
     ↑_____________residual↑_____________residual↑
```

### Backward Through Residual Connections

Residual connections are gradient highways:

```
output = f(x) + x
```

Backward:
```
∂L/∂x = ∂L/∂output + ∂L/∂f(x)
```

The gradient **splits** and flows through both paths. This is why residual connections help:
- Direct path prevents vanishing gradients
- Gradient has two routes to flow backward

### Full Layer Backward

```go
func TransformerLayerBackward(
    ∂L_∂output *Tensor,
    saved_values map[string]*Tensor,  // everything from forward pass
    params map[string]*Tensor,        // all parameters
) (∂L_∂input *Tensor, gradients map[string]*Tensor) {

    gradients := make(map[string]*Tensor)

    // Backward through second residual connection
    ∂L_∂ffn_output := ∂L_∂output  // copy gradient
    ∂L_∂attn_output := ∂L_∂output // gradient also flows through residual

    // Backward through FFN + LayerNorm
    ∂L_∂ffn_ln_output := ∂L_∂ffn_output
    ∂L_∂ffn_ln_input, gradients["ffn_ln"] := LayerNormBackward(...)

    ∂L_∂ffn_input := ∂L_∂ffn_ln_output
    ∂L_∂ffn_hidden, gradients["ffn_W1"], gradients["ffn_b1"] := LinearBackward(...)
    ∂L_∂ffn_gelu_input := GELUBackward(∂L_∂ffn_hidden, ...)
    ∂L_∂ffn_input2, gradients["ffn_W2"], gradients["ffn_b2"] := LinearBackward(...)

    // Add FFN gradient to residual gradient
    ∂L_∂attn_output = ∂L_∂attn_output.Add(∂L_∂ffn_input2)

    // Backward through first residual connection
    ∂L_∂attn_ln_output := ∂L_∂attn_output
    ∂L_∂input_from_residual := ∂L_∂attn_output  // second copy

    // Backward through Attention + LayerNorm
    ∂L_∂attn_ln_input, gradients["attn_ln"] := LayerNormBackward(...)
    ∂L_∂input_from_attn, gradients["attn"] := AttentionBackward(...)

    // Combine gradients from both paths
    ∂L_∂input := ∂L_∂input_from_attn.Add(∂L_∂input_from_residual)

    return ∂L_∂input, gradients
}
```

## Common Gradient Problems

### 1. Vanishing Gradients

**Problem**: Gradients become extremely small as they flow backward through many layers.

**Why it happens**:
- Multiplying many small derivatives (< 1) makes the product approach 0
- Example: `sigmoid` has max gradient of 0.25, so after 10 layers: 0.25^10 ≈ 0.000001

**Solutions in transformers**:
- ✅ **Residual connections**: Provide direct gradient path
- ✅ **Layer normalization**: Keeps gradients in a reasonable range
- ✅ **Attention**: Direct connections between any positions (no sequential bottleneck like RNNs)

### 2. Exploding Gradients

**Problem**: Gradients become extremely large, causing NaN or training instability.

**Why it happens**:
- Multiplying many large derivatives (> 1) makes the product explode
- Poor initialization can cause this

**Solutions**:
- ✅ **Gradient clipping**: Cap maximum gradient norm
- ✅ **Proper initialization**: Xavier/He initialization keeps gradients stable
- ✅ **Layer normalization**: Prevents activation magnitudes from growing

### 3. Dead ReLU Problem

**Problem**: ReLU neurons get stuck outputting 0, stopping all gradient flow.

**Why it happens**: If `x < 0`, ReLU outputs 0, and gradient is 0.

**Solutions**:
- ✅ **GELU instead of ReLU**: Smooth function with non-zero gradients everywhere
- ✅ **Leaky ReLU**: Small gradient for negative inputs
- ✅ **Proper learning rate**: Prevents large updates that push neurons too negative

## Implementation Walkthrough

Let's trace through the actual backward pass in `transformer_backward.go`:

### Step 1: Loss Gradient

```go
// Cross-entropy loss backward
func CrossEntropyBackward(logits *Tensor, targets []int) *Tensor {
    ∂L_∂logits := logits.Softmax()  // probabilities

    // Subtract 1 from the true class probability
    for i, target := range targets {
        ∂L_∂logits.data[i*logits.shape[1] + target] -= 1.0
    }

    // Average over batch
    ∂L_∂logits = ∂L_∂logits.DivScalar(float64(len(targets)))

    return ∂L_∂logits
}
```

**Intuition**: The gradient pushes the predicted probability for the true class toward 1, and other classes toward 0.

### Step 2: Output Projection Backward

```go
// Linear layer: output = input × W + b
∂L_∂W_out := input_to_output.Transpose().MatMul(∂L_∂logits)
∂L_∂b_out := ∂L_∂logits.Sum(axis=0)
∂L_∂input := ∂L_∂logits.MatMul(W_out.Transpose())
```

### Step 3: Through All Transformer Layers

```go
∂L_∂layer_output := ∂L_∂input
for i := numLayers - 1; i >= 0; i-- {
    ∂L_∂layer_input, layer_grads := TransformerLayerBackward(
        ∂L_∂layer_output,
        saved_from_forward[i],
        params[i],
    )

    // Store gradients for this layer
    all_gradients[i] = layer_grads

    // Pass gradient to previous layer
    ∂L_∂layer_output = ∂L_∂layer_input
}
```

### Step 4: Embedding Backward

```go
// For each token, add gradient to the corresponding embedding row
∂L_∂embeddings := NewTensor(vocab_size, embed_dim)
for batch_idx, token_ids := range input_tokens {
    for pos_idx, token_id := range token_ids {
        grad := ∂L_∂layer_output.Get(batch_idx, pos_idx)
        ∂L_∂embeddings.AddRow(token_id, grad)
    }
}
```

## Practical Exercises

### Exercise 1: Matrix Multiplication Backward

Given:
```
C = A × B
∂L/∂C = [[1, 2],
         [3, 4]]
A = [[1, 2],
     [3, 4]]
B = [[5, 6],
     [7, 8]]
```

Compute `∂L/∂A` and `∂L/∂B`.

**Solution**:
```
∂L/∂A = ∂L/∂C × B^T
      = [[1, 2],    × [[5, 7],
         [3, 4]]      [6, 8]]
      = [[17, 23],
         [39, 53]]

∂L/∂B = A^T × ∂L/∂C
      = [[1, 3],    × [[1, 2],
         [2, 4]]      [3, 4]]
      = [[10, 14],
         [14, 20]]
```

### Exercise 2: Softmax Backward

Given:
```
softmax([1, 2, 3]) = [0.09, 0.24, 0.67]
∂L/∂softmax = [0.1, -0.5, 0.4]
```

Compute `∂L/∂logits`.

**Solution**:
```
For each element i:
∂L/∂logit[i] = softmax[i] × (∂L/∂softmax[i] - Σ_j softmax[j] × ∂L/∂softmax[j])

Σ term = 0.09×0.1 + 0.24×(-0.5) + 0.67×0.4 = 0.157

∂L/∂logit[0] = 0.09 × (0.1 - 0.157) = -0.005
∂L/∂logit[1] = 0.24 × (-0.5 - 0.157) = -0.158
∂L/∂logit[2] = 0.67 × (0.4 - 0.157) = 0.163
```

### Exercise 3: Residual Connection Gradient

Given a residual block:
```
output = f(x) + x
∂L/∂output = 2.0
∂L/∂f(x) = 1.5
```

Compute `∂L/∂x`.

**Solution**:
```
∂L/∂x = ∂L/∂output + ∂L/∂f(x)
      = 2.0 + 1.5
      = 3.5
```

The gradient is the **sum** of both paths, not a split.

## Key Takeaways

1. **Backpropagation is systematic chain rule application**: Each operation has a local gradient rule.

2. **Residual connections are gradient highways**: They provide direct paths for gradients to flow backward.

3. **Layer normalization couples all features**: Every element's gradient depends on all others through mean/variance.

4. **Softmax couples all classes**: Changing one logit affects all probabilities.

5. **Matrix multiplication gradients swap positions**: `∂L/∂A = ∂L/∂C × B^T`, `∂L/∂B = A^T × ∂L/∂C`

6. **Transformers avoid vanishing gradients**: Through residual connections, layer norm, and direct attention connections.

7. **Gradients accumulate at split points**: When one tensor feeds multiple operations, gradients add.

## Further Reading

- **Original Transformer Paper**: ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- **Backpropagation Tutorial**: [Yes you should understand backprop](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b) (Andrej Karpathy)
- **Gradient Flow**: ["Understanding the difficulty of training deep feedforward neural networks"](http://proceedings.mlr.press/v9/glorot10a.html) (Glorot & Bengio, 2010)

## Next Steps

Now that you understand backpropagation, explore:
1. [Training Dynamics](training-dynamics.md) - How loss evolves during training
2. [Attention Mechanism](attention-mechanism.md) - Deep dive into attention
3. [Implementation Details](transformer_backward.go) - See the actual code

---

*This tutorial is part of the [Local Code Model](https://github.com/scttfrdmn/local-code-model) educational project.*
