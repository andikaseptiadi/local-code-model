package main

import (
	"fmt"
	"math"
)

// RECOMMENDED READING:
//
// Transformer Architecture:
// - "Attention Is All You Need" by Vaswani et al. (2017)
//   https://arxiv.org/abs/1706.03762
//
// - "The Illustrated Transformer" by Jay Alammar
//   https://jalammar.github.io/illustrated-transformer/
//
// GPT Architecture:
// - "Language Models are Unsupervised Multitask Learners" (GPT-2)
//   https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
//
// - "Improving Language Understanding by Generative Pre-Training" (GPT-1)
//   https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

// Config holds hyperparameters for the transformer model.
type Config struct {
	VocabSize int     // Size of vocabulary
	SeqLen    int     // Maximum sequence length (context window)
	EmbedDim  int     // Embedding dimension (d_model)
	NumHeads  int     // Number of attention heads
	NumLayers int     // Number of transformer blocks
	FFHidden  int     // Feed-forward hidden dimension (typically 4 * EmbedDim)
	Dropout   float64 // Dropout probability
}

// DefaultConfig returns a small transformer configuration for testing.
func DefaultConfig() Config {
	return Config{
		VocabSize: 1000,
		SeqLen:    128,
		EmbedDim:  256,
		NumHeads:  4,
		NumLayers: 4,
		FFHidden:  1024,
		Dropout:   0.1,
	}
}

// Attention implements multi-head self-attention.
//
// INTUITION:
// Attention allows each position to "look at" other positions in the sequence
// to gather context. It's answering: "what other tokens are relevant when
// processing this token?"
//
// Mechanism:
//   1. Project input to Query, Key, Value matrices
//   2. Compute attention scores: softmax(Q·K^T / √d_k)
//   3. Weight values by attention scores: Attention(Q,K,V) = softmax(QK^T/√d_k)V
//
// Multi-head: Run h parallel attention operations with different projections,
// then concatenate. This allows attending to different representation subspaces.
type Attention struct {
	embedDim int
	numHeads int
	headDim  int

	// Linear projections
	wq, wk, wv, wo *Tensor

	// Causal mask for autoregressive generation
	mask *Tensor
}

// NewAttention creates a new attention layer.
func NewAttention(embedDim, numHeads, seqLen int) *Attention {
	if embedDim%numHeads != 0 {
		panic(fmt.Sprintf("transformer: embedDim (%d) must be divisible by numHeads (%d)", embedDim, numHeads))
	}

	headDim := embedDim / numHeads

	// Xavier/Glorot initialization scaled for transformers
	scale := math.Sqrt(2.0 / float64(embedDim))

	wq := NewTensorRand(embedDim, embedDim)
	wk := NewTensorRand(embedDim, embedDim)
	wv := NewTensorRand(embedDim, embedDim)
	wo := NewTensorRand(embedDim, embedDim)

	// Scale weights
	for i := range wq.data {
		wq.data[i] *= scale
		wk.data[i] *= scale
		wv.data[i] *= scale
		wo.data[i] *= scale
	}

	// Create causal mask: lower triangular matrix
	// Prevents attending to future tokens in autoregressive generation
	mask := NewTensor(seqLen, seqLen)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			if i >= j {
				mask.Set(1.0, i, j)
			}
		}
	}

	return &Attention{
		embedDim: embedDim,
		numHeads: numHeads,
		headDim:  headDim,
		wq:       wq,
		wk:       wk,
		wv:       wv,
		wo:       wo,
		mask:     mask,
	}
}

// Forward computes attention output for input x.
// x shape: (seqLen, embedDim)
// Returns: (seqLen, embedDim)
func (a *Attention) Forward(x *Tensor) *Tensor {
	if len(x.shape) != 2 {
		panic("transformer: attention input must be 2D (seqLen, embedDim)")
	}

	seqLen := x.shape[0]

	// Project to Q, K, V
	q := MatMul(x, a.wq) // (seqLen, embedDim)
	k := MatMul(x, a.wk)
	v := MatMul(x, a.wv)

	// Compute attention scores: Q @ K^T / sqrt(d_k)
	kt := Transpose(k)
	scores := MatMul(q, kt) // (seqLen, seqLen)

	// Scale for numerical stability
	scale := 1.0 / math.Sqrt(float64(a.embedDim))
	scores = Scale(scores, scale)

	// Apply causal mask (set future positions to -inf)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			if a.mask.At(i, j) == 0 {
				scores.Set(-1e9, i, j)
			}
		}
	}

	// Softmax to get attention weights
	weights := Softmax(scores) // (seqLen, seqLen)

	// Apply attention to values
	output := MatMul(weights, v) // (seqLen, embedDim)

	// Final projection
	output = MatMul(output, a.wo)

	return output
}

// LayerNorm implements layer normalization.
//
// PAPER: "Layer Normalization" by Ba, Kiros, Hinton (2016)
// https://arxiv.org/abs/1607.06450
//
// Normalizes activations across features for each example independently.
// Critical for training deep transformers.
//
// Formula: y = γ * (x - μ) / σ + β
// where μ, σ are computed per-layer, γ, β are learned parameters.
type LayerNorm struct {
	dim   int
	eps   float64
	gamma *Tensor // Scale parameter
	beta  *Tensor // Shift parameter
}

// NewLayerNorm creates a layer normalization layer.
func NewLayerNorm(dim int) *LayerNorm {
	gamma := NewTensor(dim)
	beta := NewTensor(dim)

	// Initialize: gamma=1, beta=0 (identity transform)
	for i := 0; i < dim; i++ {
		gamma.data[i] = 1.0
		beta.data[i] = 0.0
	}

	return &LayerNorm{
		dim:   dim,
		eps:   1e-5,
		gamma: gamma,
		beta:  beta,
	}
}

// Forward applies layer normalization.
// x shape: (seqLen, features)
func (ln *LayerNorm) Forward(x *Tensor) *Tensor {
	if len(x.shape) != 2 {
		panic("transformer: LayerNorm input must be 2D")
	}

	seqLen, features := x.shape[0], x.shape[1]
	out := NewTensor(seqLen, features)

	// Normalize each position independently
	for i := 0; i < seqLen; i++ {
		// Compute mean
		mean := 0.0
		for j := 0; j < features; j++ {
			mean += x.At(i, j)
		}
		mean /= float64(features)

		// Compute variance
		variance := 0.0
		for j := 0; j < features; j++ {
			diff := x.At(i, j) - mean
			variance += diff * diff
		}
		variance /= float64(features)

		// Normalize and scale
		std := math.Sqrt(variance + ln.eps)
		for j := 0; j < features; j++ {
			normalized := (x.At(i, j) - mean) / std
			scaled := normalized*ln.gamma.data[j] + ln.beta.data[j]
			out.Set(scaled, i, j)
		}
	}

	return out
}

// FeedForward implements the position-wise feed-forward network.
//
// This is a simple two-layer MLP applied independently to each position:
//   FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
//
// The hidden dimension is typically 4x the embedding dimension.
// This is where most of the model's parameters reside.
type FeedForward struct {
	w1, b1 *Tensor
	w2, b2 *Tensor
}

// NewFeedForward creates a feed-forward layer.
func NewFeedForward(embedDim, hiddenDim int) *FeedForward {
	return &FeedForward{
		w1: NewTensorRand(embedDim, hiddenDim),
		b1: NewTensor(hiddenDim),
		w2: NewTensorRand(hiddenDim, embedDim),
		b2: NewTensor(embedDim),
	}
}

// Forward applies the feed-forward network.
// x shape: (seqLen, embedDim)
func (ff *FeedForward) Forward(x *Tensor) *Tensor {
	// First layer with GELU activation
	hidden := MatMul(x, ff.w1)
	hidden = addBias(hidden, ff.b1)
	hidden = GELU(hidden)

	// Second layer
	output := MatMul(hidden, ff.w2)
	output = addBias(output, ff.b2)

	return output
}

// TransformerBlock combines attention, layer norm, and feed-forward layers.
//
// Architecture (GPT-style, pre-norm):
//   x = x + Attention(LayerNorm(x))
//   x = x + FeedForward(LayerNorm(x))
//
// The residual connections (x + ...) are crucial for training deep networks.
type TransformerBlock struct {
	attn *Attention
	ln1  *LayerNorm
	ff   *FeedForward
	ln2  *LayerNorm
}

// NewTransformerBlock creates a transformer block.
func NewTransformerBlock(config Config) *TransformerBlock {
	return &TransformerBlock{
		attn: NewAttention(config.EmbedDim, config.NumHeads, config.SeqLen),
		ln1:  NewLayerNorm(config.EmbedDim),
		ff:   NewFeedForward(config.EmbedDim, config.FFHidden),
		ln2:  NewLayerNorm(config.EmbedDim),
	}
}

// Forward applies the transformer block.
// x shape: (seqLen, embedDim)
func (tb *TransformerBlock) Forward(x *Tensor) *Tensor {
	// Self-attention with residual connection
	normed := tb.ln1.Forward(x)
	attended := tb.attn.Forward(normed)
	x = Add(x, attended)

	// Feed-forward with residual connection
	normed = tb.ln2.Forward(x)
	ff := tb.ff.Forward(normed)
	x = Add(x, ff)

	return x
}

// GPT implements a GPT-style transformer language model.
//
// Architecture:
//   1. Token + positional embeddings
//   2. Stack of transformer blocks
//   3. Final layer norm
//   4. Linear projection to vocabulary logits
type GPT struct {
	config Config

	// Embeddings
	tokenEmbed *Tensor // (vocabSize, embedDim)
	posEmbed   *Tensor // (seqLen, embedDim)

	// Transformer blocks
	blocks []*TransformerBlock

	// Output
	lnFinal *LayerNorm
	lmHead  *Tensor // (embedDim, vocabSize) - language model head
}

// NewGPT creates a new GPT model.
func NewGPT(config Config) *GPT {
	// Initialize embeddings
	tokenEmbed := NewTensorRand(config.VocabSize, config.EmbedDim)
	posEmbed := NewTensorRand(config.SeqLen, config.EmbedDim)

	// Scale embeddings
	scale := 0.02
	for i := range tokenEmbed.data {
		tokenEmbed.data[i] *= scale
		if i < len(posEmbed.data) {
			posEmbed.data[i] *= scale
		}
	}

	// Create transformer blocks
	blocks := make([]*TransformerBlock, config.NumLayers)
	for i := range blocks {
		blocks[i] = NewTransformerBlock(config)
	}

	// Output projection (tied weights with token embedding is common)
	lmHead := NewTensorRand(config.EmbedDim, config.VocabSize)

	return &GPT{
		config:     config,
		tokenEmbed: tokenEmbed,
		posEmbed:   posEmbed,
		blocks:     blocks,
		lnFinal:    NewLayerNorm(config.EmbedDim),
		lmHead:     lmHead,
	}
}

// Forward computes logits for input token IDs.
// inputIDs: []int of token indices
// Returns: (seqLen, vocabSize) logits
func (g *GPT) Forward(inputIDs []int) *Tensor {
	seqLen := len(inputIDs)
	if seqLen > g.config.SeqLen {
		panic(fmt.Sprintf("transformer: sequence length %d exceeds maximum %d", seqLen, g.config.SeqLen))
	}

	// Embedding lookup and summation
	x := NewTensor(seqLen, g.config.EmbedDim)
	for i, tokenID := range inputIDs {
		if tokenID < 0 || tokenID >= g.config.VocabSize {
			panic(fmt.Sprintf("transformer: token ID %d out of vocabulary range [0,%d)", tokenID, g.config.VocabSize))
		}

		// Add token embedding + position embedding
		for j := 0; j < g.config.EmbedDim; j++ {
			tokEmb := g.tokenEmbed.At(tokenID, j)
			posEmb := g.posEmbed.At(i, j)
			x.Set(tokEmb+posEmb, i, j)
		}
	}

	// Pass through transformer blocks
	for _, block := range g.blocks {
		x = block.Forward(x)
	}

	// Final layer norm
	x = g.lnFinal.Forward(x)

	// Project to vocabulary logits
	logits := MatMul(x, g.lmHead)

	return logits
}

// Generate produces tokens autoregressively.
// prompt: initial token IDs
// maxTokens: maximum number of new tokens to generate
// Returns: generated token IDs (including prompt)
func (g *GPT) Generate(prompt []int, maxTokens int) []int {
	tokens := make([]int, len(prompt))
	copy(tokens, prompt)

	for i := 0; i < maxTokens; i++ {
		// Forward pass
		logits := g.Forward(tokens)

		// Get logits for last position
		lastPos := len(tokens) - 1
		lastLogits := NewTensor(g.config.VocabSize)
		for j := 0; j < g.config.VocabSize; j++ {
			lastLogits.data[j] = logits.At(lastPos, j)
		}

		// Sample next token (greedy for now - take argmax)
		nextToken := argmax(lastLogits.data)

		// Append to sequence
		tokens = append(tokens, nextToken)

		// Stop if we've reached max length
		if len(tokens) >= g.config.SeqLen {
			break
		}
	}

	return tokens
}

// ===========================================================================
// HELPERS
// ===========================================================================

// addBias adds a bias vector to each row of a 2D tensor.
// x: (seqLen, features), bias: (features,)
func addBias(x, bias *Tensor) *Tensor {
	if len(x.shape) != 2 {
		panic("addBias: x must be 2D")
	}
	if len(bias.shape) != 1 {
		panic("addBias: bias must be 1D")
	}
	if x.shape[1] != bias.shape[0] {
		panic(fmt.Sprintf("addBias: dimension mismatch %d vs %d", x.shape[1], bias.shape[0]))
	}

	out := x.Clone()
	seqLen, features := x.shape[0], x.shape[1]

	for i := 0; i < seqLen; i++ {
		for j := 0; j < features; j++ {
			out.Set(out.At(i, j)+bias.data[j], i, j)
		}
	}

	return out
}

// argmax returns the index of the maximum value.
func argmax(data []float64) int {
	if len(data) == 0 {
		return -1
	}

	maxIdx := 0
	maxVal := data[0]

	for i := 1; i < len(data); i++ {
		if data[i] > maxVal {
			maxVal = data[i]
			maxIdx = i
		}
	}

	return maxIdx
}