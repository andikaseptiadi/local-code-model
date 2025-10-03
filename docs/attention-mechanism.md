# Understanding the Attention Mechanism

**A Deep Dive into the Core of Transformers**

## Table of Contents

1. [Introduction](#introduction)
2. [The Intuition](#the-intuition)
3. [The Mathematics](#the-mathematics)
4. [Implementation Walkthrough](#implementation-walkthrough)
5. [Why It Works](#why-it-works)
6. [Multi-Head Attention](#multi-head-attention)
7. [Computational Complexity](#computational-complexity)
8. [Practical Exercises](#practical-exercises)

## Introduction

The attention mechanism is the heart of the transformer architecture. It's what allows models like GPT to understand relationships between words, no matter how far apart they are in a sentence.

Before attention, recurrent neural networks (RNNs) processed text sequentially, struggling with long-range dependencies. Attention changed everything by allowing the model to look at **all** positions simultaneously.

## The Intuition

### The Cocktail Party Problem

Imagine you're at a crowded party. Your friend is talking to you, but there are dozens of other conversations happening around you. Your brain performs "attention":

- **Query**: What your friend is saying (what you're focusing on)
- **Keys**: All the other sounds in the room (potential things to focus on)
- **Values**: The actual content of each sound

Your brain computes how relevant each sound is to what you're trying to understand, giving more "weight" to your friend's voice and less to background noise.

**This is exactly what attention does in transformers.**

### A Concrete Example

Consider the sentence: "The animal didn't cross the street because **it** was too tired."

When processing the word "it", the model needs to figure out what "it" refers to:
- Does "it" refer to "animal"? ✓ (makes sense)
- Does "it" refer to "street"? ✗ (streets don't get tired)

Attention allows the model to look back at "animal" and give it high weight when processing "it", while giving "street" low weight.

## The Mathematics

### Step 1: Creating Q, K, V

For each word (token) in our input, we create three vectors:

```
Query (Q)  = What am I looking for?
Key (K)    = What do I contain?
Value (V)  = What information do I hold?
```

These are created by multiplying the input embedding by learned weight matrices:

```
Q = X × W_Q
K = X × W_K
V = X × W_V
```

Where:
- `X` is our input (shape: `[seq_len, embed_dim]`)
- `W_Q`, `W_K`, `W_V` are learned weight matrices (shape: `[embed_dim, head_dim]`)

### Step 2: Computing Attention Scores

For each query, we compute how much it should attend to each key:

```
scores = Q × K^T / sqrt(head_dim)
```

The division by `sqrt(head_dim)` is called **scaled dot-product attention**. It prevents the dot products from becoming too large, which would make gradients vanish after softmax.

**Why dot product?** If two vectors point in similar directions, their dot product is large. This measures semantic similarity.

### Step 3: Applying Softmax

We convert scores into probabilities using softmax:

```
attention_weights = softmax(scores)
```

This ensures:
1. All weights sum to 1.0
2. Larger scores get exponentially more weight
3. We have a probability distribution over all tokens

### Step 4: Computing Output

Finally, we use the attention weights to compute a weighted average of the values:

```
output = attention_weights × V
```

This gives us a context-aware representation for each token.

## Implementation Walkthrough

Let's trace through the actual code in `transformer.go`. Here's the simplified attention computation:

```go
// 1. Create Q, K, V projections
Q := x.MatMul(layer.WQ)  // [seq_len, head_dim]
K := x.MatMul(layer.WK)  // [seq_len, head_dim]
V := x.MatMul(layer.WV)  // [seq_len, head_dim]

// 2. Compute attention scores
scores := Q.MatMul(K.Transpose())  // [seq_len, seq_len]
scores = scores.DivScalar(math.Sqrt(float64(headDim)))

// 3. Apply causal masking (for GPT-style models)
// This prevents position i from attending to positions > i
for i := 0; i < seqLen; i++ {
    for j := i + 1; j < seqLen; j++ {
        scores.Set(-1e9, i, j)  // -infinity effectively
    }
}

// 4. Apply softmax to get attention weights
attnWeights := scores.Softmax(1)  // Softmax over dim 1

// 5. Compute weighted sum of values
output := attnWeights.MatMul(V)  // [seq_len, head_dim]
```

### Visualizing Attention Weights

The attention weights matrix tells us which tokens attend to which:

```
        token1  token2  token3  token4
token1  [ 1.0    0.0    0.0    0.0  ]
token2  [ 0.3    0.7    0.0    0.0  ]
token3  [ 0.1    0.2    0.7    0.0  ]
token4  [ 0.05   0.15   0.3    0.5  ]
```

Notice the upper triangle is all zeros (causal masking) - each token can only attend to previous tokens.

## Why It Works

### 1. Parallel Processing

Unlike RNNs that process sequentially:
```
RNN:  token1 → token2 → token3 → token4  (sequential)
```

Attention processes all tokens simultaneously:
```
Attention:  token1 ┐
            token2 ├→ [attention] → all outputs at once
            token3 │
            token4 ┘
```

This is **much faster** and allows for better gradient flow.

### 2. Direct Connections

In an RNN, information from token1 to token4 must pass through tokens 2 and 3:
```
token1 → token2 → token3 → token4
```

With attention, token4 can directly attend to token1:
```
token1 ────────────────────→ token4
```

This solves the **vanishing gradient problem** for long sequences.

### 3. Learned Relationships

The weight matrices `W_Q`, `W_K`, `W_V` are learned during training. The model learns to encode semantic relationships:

- "cat" and "kitten" might have similar keys (related concepts)
- Question words might have query patterns that match answer patterns
- Pronouns learn to attend to their antecedents

## Multi-Head Attention

Instead of having one attention mechanism, we use multiple "heads" in parallel:

```go
// We have numHeads = 8 in a typical model
for head := 0; head < numHeads; head++ {
    Q_h := x.MatMul(W_Q[head])
    K_h := x.MatMul(W_K[head])
    V_h := x.MatMul(W_V[head])

    output[head] = Attention(Q_h, K_h, V_h)
}

// Concatenate all heads
finalOutput = Concat(output[0], output[1], ..., output[7])
```

### Why Multiple Heads?

Each head can learn different types of relationships:

- **Head 1**: Syntax (subject-verb agreement)
- **Head 2**: Semantic similarity (synonyms, related concepts)
- **Head 3**: Coreference (pronouns to entities)
- **Head 4**: Sequential patterns (word order)
- etc.

Think of it like having multiple "perspectives" on the same text.

## Computational Complexity

### Time Complexity

The attention mechanism is **O(n² × d)** where:
- `n` = sequence length
- `d` = embedding dimension

Breaking it down:

1. **Q × K^T**: `O(n² × d)` - This is the bottleneck!
2. **Softmax**: `O(n²)` - Relatively cheap
3. **Weights × V**: `O(n² × d)` - Also expensive

For a sequence of 2048 tokens and 768 dimensions:
- Operations: 2048² × 768 ≈ **3.2 billion operations per attention layer**
- With 12 layers: **38 billion operations**

This is why:
- Long contexts are expensive
- Optimizations like Flash Attention are crucial
- KV caching makes generation 50-500x faster

### Memory Complexity

We must store the attention weights matrix: **O(n²)**

For 2048 tokens:
- Matrix size: 2048 × 2048 = **4.2 million floats** = 16.8 MB per head
- With 8 heads and 12 layers: **1.6 GB just for attention weights**

This is why context length is limited in practice!

## Practical Exercises

### Exercise 1: Manual Attention Calculation

Given:
```
Query:  [1.0, 0.0]
Keys:   [[1.0, 0.0],   # Token A
         [0.0, 1.0],   # Token B
         [0.7, 0.7]]   # Token C
Values: [[1.0, 2.0],   # Token A
         [3.0, 4.0],   # Token B
         [5.0, 6.0]]   # Token C
```

**Step 1**: Compute attention scores (Q · K^T):
```
Token A: [1.0, 0.0] · [1.0, 0.0] = 1.0
Token B: [1.0, 0.0] · [0.0, 1.0] = 0.0
Token C: [1.0, 0.0] · [0.7, 0.7] = 0.7
```

**Step 2**: Apply softmax (simplified, ignoring temperature):
```
exp(1.0) = 2.718
exp(0.0) = 1.0
exp(0.7) = 2.014

Sum = 5.732

Weights:
Token A: 2.718 / 5.732 = 0.474
Token B: 1.0 / 5.732 = 0.174
Token C: 2.014 / 5.732 = 0.351
```

**Step 3**: Weighted sum of values:
```
Output = 0.474 * [1.0, 2.0] + 0.174 * [3.0, 4.0] + 0.351 * [5.0, 6.0]
       = [0.474, 0.948] + [0.522, 0.696] + [1.755, 2.106]
       = [2.751, 3.750]
```

**Interpretation**: The output is most influenced by Token A (47.4%), followed by Token C (35.1%), with less from Token B (17.4%).

### Exercise 2: Implementing Attention from Scratch

Try implementing attention yourself in Go:

```go
func SimpleAttention(Q, K, V *Tensor) *Tensor {
    // 1. Compute scores
    scores := Q.MatMul(K.Transpose())

    // 2. Scale by sqrt(d_k)
    headDim := K.Shape()[1]
    scores = scores.DivScalar(math.Sqrt(float64(headDim)))

    // 3. Apply softmax
    attnWeights := scores.Softmax(1)

    // 4. Weighted sum of values
    output := attnWeights.MatMul(V)

    return output
}
```

Test this implementation with small examples and verify the output matches your hand calculations.

### Exercise 3: Visualizing Attention Patterns

Modify the generation code to output attention weights, then visualize them:

1. Generate text with your trained model
2. Extract attention weights for each layer
3. Create a heatmap showing which tokens attend to which
4. Look for patterns:
   - Do pronouns attend to their referents?
   - Do verbs attend to their subjects?
   - What about punctuation?

## Key Takeaways

1. **Attention is weighted averaging**: Each token computes a weighted average of all other tokens' values.

2. **Queries, Keys, Values**: The Q/K/V paradigm allows the model to learn what to look for (Q), what each token represents (K), and what information to extract (V).

3. **Softmax creates focus**: The softmax operation creates sharp attention patterns - the model learns to focus strongly on relevant tokens.

4. **Multi-head is crucial**: Multiple heads allow the model to attend to different aspects simultaneously.

5. **Complexity is O(n²)**: This quadratic complexity is both the strength (all-to-all connections) and weakness (expensive for long sequences) of attention.

6. **Causal masking enables autoregression**: By preventing tokens from attending to the future, we can train the model to predict the next token.

## Further Reading

- **Original Paper**: ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- **Illustrated Guide**: [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- **Flash Attention**: [Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)

## Next Steps

Now that you understand attention, explore:
1. [Backpropagation Through Attention](backpropagation.md) - How gradients flow through attention
2. [Training Dynamics](training-dynamics.md) - How the model learns these patterns
3. [Implementation Details](implementation-details.md) - Optimizations and tricks

---

*This tutorial is part of the [Local Code Model](https://github.com/scttfrdmn/local-code-model) educational project.*
