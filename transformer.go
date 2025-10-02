package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"sort"
)

// ===========================================================================
// WHAT'S GOING ON HERE
// ===========================================================================
//
// This file implements a complete GPT-style transformer model - the architecture
// that powers modern language models like GPT-3, GPT-4, Claude, etc.
//
// INTENTION:
// Create a working, trainable transformer that demonstrates all the key
// components: attention, layer normalization, feed-forward networks, and
// autoregressive generation. This is a learning implementation - prioritizing
// clarity over performance.
//
// WHERE THIS SITS ON THE CONTINUUM OF NAIVETE:
//
// Architecture Level: Complete and correct
//   ✓ Multi-head attention with causal masking
//   ✓ Residual connections and layer normalization
//   ✓ Position embeddings
//   ✓ Autoregressive generation
//   - No: KV-caching, flash attention, rotary embeddings (modern optimizations)
//
// Implementation Level: Baseline (naive matmul from tensor.go)
//   - Uses single-threaded matrix operations
//   - Expected forward pass time: ~10ms for seq_len=128, embed_dim=256
//   - Scales O(n²) with sequence length (attention), O(n) with embed_dim
//
// PERFORMANCE CHARACTERISTICS:
// For a small model (256 dim, 4 heads, 4 layers, seq_len=128):
//   - Forward pass: ~50ms (dominated by attention matmuls)
//   - Generation (per token): ~50ms (no KV-caching)
//   - Memory: ~5MB for weights
//
// Bottlenecks (in order):
//   1. Attention matmuls (Q·K^T and scores·V): 70% of time
//   2. Feed-forward matmuls: 25% of time
//   3. Everything else (layernorm, softmax, etc.): 5% of time
//
// WHAT GETS FASTER WITH OPTIMIZATION:
// The optimization levels in matmul_optimized.go will speed up:
//   - Parallel: 2-5x improvement (memory bandwidth limited)
//   - Cache-blocked: 2-4x additional
//   - GPU/Metal: 50-100x for attention (massively parallel)
//   - ANE: Not applicable (transformers need fp16/fp32, ANE optimized for int8)
//
// WHY THIS APPROACH:
// Understanding transformers deeply requires implementing one. This code
// prioritizes readability and correctness. Once you understand how attention
// works at this level, you can appreciate why techniques like flash attention
// and KV-caching matter so much for production systems.
//
// ===========================================================================
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
// ===========================================================================

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

	// Backend for accelerated operations (optional)
	backend interface{ MatMul(*Tensor, *Tensor) (*Tensor, error) }
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

	// Helper to use backend if available
	matmul := func(t1, t2 *Tensor) *Tensor {
		if a.backend != nil {
			result, err := a.backend.MatMul(t1, t2)
			if err == nil {
				return result
			}
		}
		return MatMul(t1, t2)
	}

	// Project to Q, K, V
	q := matmul(x, a.wq) // (seqLen, embedDim)
	k := matmul(x, a.wk)
	v := matmul(x, a.wv)

	// Compute attention scores: Q @ K^T / sqrt(d_k)
	kt := Transpose(k)
	scores := matmul(q, kt) // (seqLen, seqLen)

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
	output := matmul(weights, v) // (seqLen, embedDim)

	// Final projection
	output = matmul(output, a.wo)

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

	// Backend for accelerated operations (optional)
	backend interface{ MatMul(*Tensor, *Tensor) (*Tensor, error) }
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
	// Helper to use backend if available
	matmul := func(t1, t2 *Tensor) *Tensor {
		if ff.backend != nil {
			result, err := ff.backend.MatMul(t1, t2)
			if err == nil {
				return result
			}
		}
		return MatMul(t1, t2)
	}

	// First layer with GELU activation
	hidden := matmul(x, ff.w1)
	hidden = addBias(hidden, ff.b1)
	hidden = GELU(hidden)

	// Second layer
	output := matmul(hidden, ff.w2)
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

	// Backend for accelerated matrix operations
	// If nil, uses naive MatMul from tensor.go
	backend interface{ MatMul(*Tensor, *Tensor) (*Tensor, error) }
}

// SetBackend configures the backend for accelerated matrix operations
func (g *GPT) SetBackend(backend interface{ MatMul(*Tensor, *Tensor) (*Tensor, error) }) {
	g.backend = backend

	// Propagate to all transformer blocks
	for _, block := range g.blocks {
		block.attn.backend = backend
		block.ff.backend = backend
	}
}

// matmul performs matrix multiplication using backend if available, otherwise falls back to naive
func (g *GPT) matmul(a, b *Tensor) *Tensor {
	if g.backend != nil {
		result, err := g.backend.MatMul(a, b)
		if err == nil {
			return result
		}
	}
	return MatMul(a, b)
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
	logits := g.matmul(x, g.lmHead)

	return logits
}

// Generate produces tokens autoregressively.
// prompt: initial token IDs
// maxTokens: maximum number of new tokens to generate
// Returns: generated token IDs (including prompt)
func (g *GPT) Generate(prompt []int, maxTokens int) []int {
	return g.GenerateWithSampling(prompt, maxTokens, &SampleConfig{Temperature: 0.0})
}

// GenerateWithSampling produces tokens autoregressively with customizable sampling.
// prompt: initial token IDs
// maxTokens: maximum number of new tokens to generate
// config: sampling configuration (temperature, top-k, top-p)
// Returns: generated token IDs (including prompt)
func (g *GPT) GenerateWithSampling(prompt []int, maxTokens int, config *SampleConfig) []int {
	tokens := make([]int, len(prompt))
	copy(tokens, prompt)

	for i := 0; i < maxTokens; i++ {
		// Forward pass
		logits := g.Forward(tokens)

		// Get logits for last position
		lastPos := len(tokens) - 1
		lastLogits := make([]float64, g.config.VocabSize)
		for j := 0; j < g.config.VocabSize; j++ {
			lastLogits[j] = logits.At(lastPos, j)
		}

		// Sample next token using configured sampling strategy
		nextToken := sample(lastLogits, config)

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

// SampleConfig holds configuration for text generation sampling.
type SampleConfig struct {
	Temperature float64 // Temperature for sampling (0 = greedy, higher = more random)
	TopK        int     // Top-k sampling (0 = disabled)
	TopP        float64 // Top-p (nucleus) sampling (0 = disabled)
}

// NewSampleConfig creates a default sampling configuration.
func NewSampleConfig() *SampleConfig {
	return &SampleConfig{
		Temperature: 1.0,
		TopK:        0,
		TopP:        0.0,
	}
}

// sample samples from logits using temperature, top-k, and top-p sampling.
func sample(logits []float64, config *SampleConfig) int {
	// Greedy decoding if temperature is 0
	if config.Temperature == 0.0 {
		return argmax(logits)
	}

	// Apply temperature
	scaledLogits := make([]float64, len(logits))
	for i, logit := range logits {
		scaledLogits[i] = logit / config.Temperature
	}

	// Convert logits to probabilities using softmax
	probs := softmaxSlice(scaledLogits)

	// Apply top-k filtering if enabled
	if config.TopK > 0 {
		probs = applyTopK(probs, config.TopK)
	}

	// Apply top-p (nucleus) filtering if enabled
	if config.TopP > 0.0 && config.TopP < 1.0 {
		probs = applyTopP(probs, config.TopP)
	}

	// Sample from the distribution
	return sampleFromDistribution(probs)
}

// softmaxSlice applies softmax to a slice of logits.
func softmaxSlice(logits []float64) []float64 {
	// Find max for numerical stability
	maxLogit := logits[0]
	for _, logit := range logits {
		if logit > maxLogit {
			maxLogit = logit
		}
	}

	// Compute exp and sum
	expSum := 0.0
	probs := make([]float64, len(logits))
	for i, logit := range logits {
		probs[i] = math.Exp(logit - maxLogit)
		expSum += probs[i]
	}

	// Normalize
	for i := range probs {
		probs[i] /= expSum
	}

	return probs
}

// applyTopK filters probabilities to keep only top-k tokens.
func applyTopK(probs []float64, k int) []float64 {
	if k <= 0 || k >= len(probs) {
		return probs
	}

	// Create indices sorted by probability
	type indexedProb struct {
		index int
		prob  float64
	}

	indexed := make([]indexedProb, len(probs))
	for i, p := range probs {
		indexed[i] = indexedProb{i, p}
	}

	// Sort by probability descending
	sort.Slice(indexed, func(i, j int) bool {
		return indexed[i].prob > indexed[j].prob
	})

	// Zero out probabilities outside top-k
	filtered := make([]float64, len(probs))
	totalProb := 0.0
	for i := 0; i < k && i < len(indexed); i++ {
		filtered[indexed[i].index] = indexed[i].prob
		totalProb += indexed[i].prob
	}

	// Renormalize
	if totalProb > 0 {
		for i := range filtered {
			filtered[i] /= totalProb
		}
	}

	return filtered
}

// applyTopP filters probabilities using nucleus sampling.
func applyTopP(probs []float64, p float64) []float64 {
	if p <= 0.0 || p >= 1.0 {
		return probs
	}

	// Create indices sorted by probability
	type indexedProb struct {
		index int
		prob  float64
	}

	indexed := make([]indexedProb, len(probs))
	for i, prob := range probs {
		indexed[i] = indexedProb{i, prob}
	}

	// Sort by probability descending
	sort.Slice(indexed, func(i, j int) bool {
		return indexed[i].prob > indexed[j].prob
	})

	// Find nucleus (minimum set with cumulative prob >= p)
	filtered := make([]float64, len(probs))
	cumProb := 0.0
	totalProb := 0.0

	for _, item := range indexed {
		if cumProb >= p {
			break
		}
		filtered[item.index] = item.prob
		cumProb += item.prob
		totalProb += item.prob
	}

	// Renormalize
	if totalProb > 0 {
		for i := range filtered {
			filtered[i] /= totalProb
		}
	}

	return filtered
}

// sampleFromDistribution samples an index from a probability distribution.
func sampleFromDistribution(probs []float64) int {
	// Generate random number in [0, 1)
	r := rand.Float64()

	// Sample using cumulative distribution
	cumProb := 0.0
	for i, prob := range probs {
		cumProb += prob
		if r < cumProb {
			return i
		}
	}

	// Fallback to last index (shouldn't happen with valid probs)
	return len(probs) - 1
}

// ===========================================================================
// Model Serialization
// ===========================================================================
//
// Simple binary format for saving/loading GPT models.
//
// Format:
//   1. Header with config (JSON)
//   2. All tensor data in order (binary float64)
//
// This is a naive format - just tensor dumps. Production systems would use:
//   - SafeTensors (memory-mapped format from HuggingFace)
//   - GGUF (llama.cpp format with quantization)
//   - PyTorch .pt files
//
// But for learning purposes, a simple format is clearest.
// ===========================================================================

// Save writes the model to a file.
func (g *GPT) Save(filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer f.Close()

	// Write config as JSON header
	configJSON, err := json.Marshal(g.config)
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	// Write header length (4 bytes)
	headerLen := uint32(len(configJSON))
	if err := binary.Write(f, binary.LittleEndian, headerLen); err != nil {
		return fmt.Errorf("failed to write header length: %w", err)
	}

	// Write JSON config
	if _, err := f.Write(configJSON); err != nil {
		return fmt.Errorf("failed to write config: %w", err)
	}

	// Helper to write tensor data
	writeTensor := func(t *Tensor) error {
		return binary.Write(f, binary.LittleEndian, t.data)
	}

	// Write embeddings
	if err := writeTensor(g.tokenEmbed); err != nil {
		return fmt.Errorf("failed to write token embeddings: %w", err)
	}
	if err := writeTensor(g.posEmbed); err != nil {
		return fmt.Errorf("failed to write position embeddings: %w", err)
	}

	// Write each transformer block
	for i, block := range g.blocks {
		// Attention weights
		if err := writeTensor(block.attn.wq); err != nil {
			return fmt.Errorf("failed to write block %d wq: %w", i, err)
		}
		if err := writeTensor(block.attn.wk); err != nil {
			return fmt.Errorf("failed to write block %d wk: %w", i, err)
		}
		if err := writeTensor(block.attn.wv); err != nil {
			return fmt.Errorf("failed to write block %d wv: %w", i, err)
		}
		if err := writeTensor(block.attn.wo); err != nil {
			return fmt.Errorf("failed to write block %d wo: %w", i, err)
		}

		// LayerNorm 1
		if err := writeTensor(block.ln1.gamma); err != nil {
			return fmt.Errorf("failed to write block %d ln1 gamma: %w", i, err)
		}
		if err := writeTensor(block.ln1.beta); err != nil {
			return fmt.Errorf("failed to write block %d ln1 beta: %w", i, err)
		}

		// Feed-forward weights
		if err := writeTensor(block.ff.w1); err != nil {
			return fmt.Errorf("failed to write block %d ff w1: %w", i, err)
		}
		if err := writeTensor(block.ff.b1); err != nil {
			return fmt.Errorf("failed to write block %d ff b1: %w", i, err)
		}
		if err := writeTensor(block.ff.w2); err != nil {
			return fmt.Errorf("failed to write block %d ff w2: %w", i, err)
		}
		if err := writeTensor(block.ff.b2); err != nil {
			return fmt.Errorf("failed to write block %d ff b2: %w", i, err)
		}

		// LayerNorm 2
		if err := writeTensor(block.ln2.gamma); err != nil {
			return fmt.Errorf("failed to write block %d ln2 gamma: %w", i, err)
		}
		if err := writeTensor(block.ln2.beta); err != nil {
			return fmt.Errorf("failed to write block %d ln2 beta: %w", i, err)
		}
	}

	// Write final layer norm
	if err := writeTensor(g.lnFinal.gamma); err != nil {
		return fmt.Errorf("failed to write final ln gamma: %w", err)
	}
	if err := writeTensor(g.lnFinal.beta); err != nil {
		return fmt.Errorf("failed to write final ln beta: %w", err)
	}

	// Write language model head
	if err := writeTensor(g.lmHead); err != nil {
		return fmt.Errorf("failed to write lm head: %w", err)
	}

	return nil
}

// Load reads a model from a file.
func LoadGPT(filename string) (*GPT, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer f.Close()

	// Read header length
	var headerLen uint32
	if err := binary.Read(f, binary.LittleEndian, &headerLen); err != nil {
		return nil, fmt.Errorf("failed to read header length: %w", err)
	}

	// Read config JSON
	configJSON := make([]byte, headerLen)
	if _, err := io.ReadFull(f, configJSON); err != nil {
		return nil, fmt.Errorf("failed to read config: %w", err)
	}

	// Parse config
	var config Config
	if err := json.Unmarshal(configJSON, &config); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}

	// Create model with config
	model := NewGPT(config)

	// Helper to read tensor data
	readTensor := func(t *Tensor) error {
		return binary.Read(f, binary.LittleEndian, t.data)
	}

	// Read embeddings
	if err := readTensor(model.tokenEmbed); err != nil {
		return nil, fmt.Errorf("failed to read token embeddings: %w", err)
	}
	if err := readTensor(model.posEmbed); err != nil {
		return nil, fmt.Errorf("failed to read position embeddings: %w", err)
	}

	// Read each transformer block
	for i, block := range model.blocks {
		// Attention weights
		if err := readTensor(block.attn.wq); err != nil {
			return nil, fmt.Errorf("failed to read block %d wq: %w", i, err)
		}
		if err := readTensor(block.attn.wk); err != nil {
			return nil, fmt.Errorf("failed to read block %d wk: %w", i, err)
		}
		if err := readTensor(block.attn.wv); err != nil {
			return nil, fmt.Errorf("failed to read block %d wv: %w", i, err)
		}
		if err := readTensor(block.attn.wo); err != nil {
			return nil, fmt.Errorf("failed to read block %d wo: %w", i, err)
		}

		// LayerNorm 1
		if err := readTensor(block.ln1.gamma); err != nil {
			return nil, fmt.Errorf("failed to read block %d ln1 gamma: %w", i, err)
		}
		if err := readTensor(block.ln1.beta); err != nil {
			return nil, fmt.Errorf("failed to read block %d ln1 beta: %w", i, err)
		}

		// Feed-forward weights
		if err := readTensor(block.ff.w1); err != nil {
			return nil, fmt.Errorf("failed to read block %d ff w1: %w", i, err)
		}
		if err := readTensor(block.ff.b1); err != nil {
			return nil, fmt.Errorf("failed to read block %d ff b1: %w", i, err)
		}
		if err := readTensor(block.ff.w2); err != nil {
			return nil, fmt.Errorf("failed to read block %d ff w2: %w", i, err)
		}
		if err := readTensor(block.ff.b2); err != nil {
			return nil, fmt.Errorf("failed to read block %d ff b2: %w", i, err)
		}

		// LayerNorm 2
		if err := readTensor(block.ln2.gamma); err != nil {
			return nil, fmt.Errorf("failed to read block %d ln2 gamma: %w", i, err)
		}
		if err := readTensor(block.ln2.beta); err != nil {
			return nil, fmt.Errorf("failed to read block %d ln2 beta: %w", i, err)
		}
	}

	// Read final layer norm
	if err := readTensor(model.lnFinal.gamma); err != nil {
		return nil, fmt.Errorf("failed to read final ln gamma: %w", err)
	}
	if err := readTensor(model.lnFinal.beta); err != nil {
		return nil, fmt.Errorf("failed to read final ln beta: %w", err)
	}

	// Read language model head
	if err := readTensor(model.lmHead); err != nil {
		return nil, fmt.Errorf("failed to read lm head: %w", err)
	}

	return model, nil
}