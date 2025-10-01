package main

import (
	"math"
)

// ===========================================================================
// WHAT'S GOING ON HERE
// ===========================================================================
//
// This file implements backpropagation through transformer layers.
//
// INTENTION:
// Enable gradient flow through the complete transformer architecture so the
// model can learn from data via gradient descent.
//
// THE BACKWARD PASS:
//
// Each component needs backward implementation:
//   1. GPT.Backward() - Top-level backward through entire model
//   2. TransformerBlock.Backward() - Through attention + feedforward
//   3. Attention.Backward() - Through multi-head attention
//   4. FeedForward.Backward() - Through MLP layers
//   5. LayerNorm.Backward() - Through normalization
//
// GRADIENT FLOW:
//
// Forward: Input → Embed → Blocks → LN → Output → Logits → Loss
// Backward: Loss → ∂Logits → ∂LN → ∂Blocks → ∂Embed → ∂Input
//
// At each step, we:
//   1. Receive gradient from next layer (∂L/∂output)
//   2. Compute gradients for parameters (∂L/∂weights)
//   3. Compute gradient for input (∂L/∂input)
//   4. Pass input gradient to previous layer
//
// RESIDUAL CONNECTIONS:
//
// Residual: y = x + F(x)
// Backward: ∂L/∂x = ∂L/∂y * (1 + ∂F/∂x)
//
// In practice: gradients add at residual connections
//   gradX = gradY + gradF(x)
//
// This helps with vanishing gradients (direct path for gradients).
//
// MEMORY MANAGEMENT:
//
// Backward pass requires storing activations from forward pass:
//   - Input to each layer
//   - Intermediate computations
//   - This is why training uses ~4x inference memory
//
// ===========================================================================

// ForwardCache stores activations needed for backward pass.
type ForwardCache struct {
	// Embeddings
	tokenIDs []int
	posIDs   []int

	// Transformer blocks (one cache per block)
	blockCaches []*BlockCache

	// Final layer norm
	lnFinalInput *Tensor
}

// BlockCache stores activations for one transformer block.
type BlockCache struct {
	// Block input
	input *Tensor

	// Attention
	attnInput  *Tensor
	attnOutput *Tensor
	attnCache  *AttentionCache

	// Residual 1 (after attention)
	residual1 *Tensor

	// Feed-forward
	ffInput  *Tensor
	ffOutput *Tensor
	ffCache  *FFCache

	// Residual 2 (after feed-forward)
	residual2 *Tensor
}

// AttentionCache stores activations for attention layer.
type AttentionCache struct {
	// Projections
	q, k, v *Tensor

	// Attention scores
	scores *Tensor

	// Attention weights (after softmax)
	weights *Tensor

	// Context vectors
	context *Tensor
}

// FFCache stores activations for feed-forward layer.
type FFCache struct {
	hidden     *Tensor // After first linear + activation
	preActivation *Tensor // Before activation (needed for GELU gradient)
}

// ForwardWithCache performs forward pass and stores activations for backward.
func (gpt *GPT) ForwardWithCache(inputIDs []int) (*Tensor, *ForwardCache) {
	seqLen := len(inputIDs)
	if seqLen > gpt.config.SeqLen {
		panic("input sequence too long")
	}

	cache := &ForwardCache{
		tokenIDs:    inputIDs,
		posIDs:      make([]int, seqLen),
		blockCaches: make([]*BlockCache, gpt.config.NumLayers),
	}

	// Position IDs: [0, 1, 2, ..., seqLen-1]
	for i := range cache.posIDs {
		cache.posIDs[i] = i
	}

	// Token embeddings
	x := NewTensor(seqLen, gpt.config.EmbedDim)
	for i, tokenID := range inputIDs {
		for d := 0; d < gpt.config.EmbedDim; d++ {
			x.Set(gpt.tokenEmbed.At(tokenID, d), i, d)
		}
	}

	// Add positional embeddings
	for i, posID := range cache.posIDs {
		for d := 0; d < gpt.config.EmbedDim; d++ {
			x.data[i*gpt.config.EmbedDim+d] += gpt.posEmbed.At(posID, d)
		}
	}

	// Transformer blocks
	for layer := 0; layer < gpt.config.NumLayers; layer++ {
		blockCache := &BlockCache{}
		cache.blockCaches[layer] = blockCache

		block := gpt.blocks[layer]

		// Store block input
		blockCache.input = x.Clone()

		// Attention with residual
		blockCache.attnInput = x.Clone()
		attnOut, attnCache := block.attn.ForwardWithCache(x)
		blockCache.attnCache = attnCache
		blockCache.attnOutput = attnOut

		// LayerNorm after attention
		ln1Out := block.ln1.Forward(attnOut)

		// Residual connection
		blockCache.residual1 = x.Clone()
		x = Add(x, ln1Out)

		// Feed-forward with residual
		blockCache.ffInput = x.Clone()
		ffOut, ffCache := block.ff.ForwardWithCache(x)
		blockCache.ffCache = ffCache
		blockCache.ffOutput = ffOut

		// LayerNorm after feed-forward
		ln2Out := block.ln2.Forward(ffOut)

		// Residual connection
		blockCache.residual2 = x.Clone()
		x = Add(x, ln2Out)
	}

	// Final layer norm
	cache.lnFinalInput = x.Clone()
	x = gpt.lnFinal.Forward(x)

	// Project to vocabulary
	logits := MatMul(x, Transpose(gpt.lmHead))

	return logits, cache
}

// ForwardWithCache for Attention layer.
func (attn *Attention) ForwardWithCache(x *Tensor) (*Tensor, *AttentionCache) {
	cache := &AttentionCache{}

	seqLen := x.shape[0]
	embedDim := x.shape[1]

	// Project to Q, K, V
	cache.q = MatMul(x, attn.wq)
	cache.k = MatMul(x, attn.wk)
	cache.v = MatMul(x, attn.wv)

	// Reshape for multi-head: (seqLen, embedDim) -> (seqLen, numHeads, headDim)
	q := cache.q.Reshape(seqLen, attn.numHeads, attn.headDim)
	k := cache.k.Reshape(seqLen, attn.numHeads, attn.headDim)
	v := cache.v.Reshape(seqLen, attn.numHeads, attn.headDim)

	// Compute attention for each head
	outputs := make([]*Tensor, attn.numHeads)

	for h := 0; h < attn.numHeads; h++ {
		// Extract head
		qHead := NewTensor(seqLen, attn.headDim)
		kHead := NewTensor(seqLen, attn.headDim)
		vHead := NewTensor(seqLen, attn.headDim)

		for i := 0; i < seqLen; i++ {
			for d := 0; d < attn.headDim; d++ {
				qHead.Set(q.At(i, h, d), i, d)
				kHead.Set(k.At(i, h, d), i, d)
				vHead.Set(v.At(i, h, d), i, d)
			}
		}

		// Attention scores: Q @ K^T / sqrt(d_k)
		kT := Transpose(kHead)
		scores := MatMul(qHead, kT)
		scale := 1.0 / math.Sqrt(float64(attn.headDim))
		scores = Scale(scores, scale)

		// Apply causal mask
		for i := 0; i < seqLen; i++ {
			for j := i + 1; j < seqLen; j++ {
				scores.Set(-1e9, i, j) // Large negative number → softmax ≈ 0
			}
		}

		cache.scores = scores.Clone()

		// Attention weights: softmax(scores)
		weights := Softmax(scores)
		cache.weights = weights.Clone()

		// Context: weights @ V
		context := MatMul(weights, vHead)
		outputs[h] = context
	}

	// Concatenate heads
	output := NewTensor(seqLen, embedDim)
	for h := 0; h < attn.numHeads; h++ {
		for i := 0; i < seqLen; i++ {
			for d := 0; d < attn.headDim; d++ {
				idx := h*attn.headDim + d
				output.Set(outputs[h].At(i, d), i, idx)
			}
		}
	}

	cache.context = output.Clone()

	// Output projection
	output = MatMul(output, attn.wo)

	return output, cache
}

// ForwardWithCache for FeedForward layer.
func (ff *FeedForward) ForwardWithCache(x *Tensor) (*Tensor, *FFCache) {
	cache := &FFCache{}

	// First linear: x @ W1 + b1
	hidden := MatMul(x, ff.w1)
	for i := range hidden.data {
		hidden.data[i] += ff.b1.data[i%ff.b1.Size()]
	}

	cache.preActivation = hidden.Clone()

	// Activation: GELU
	hidden = GELU(hidden)
	cache.hidden = hidden.Clone()

	// Second linear: hidden @ W2 + b2
	output := MatMul(hidden, ff.w2)
	for i := range output.data {
		output.data[i] += ff.b2.data[i%ff.b2.Size()]
	}

	return output, cache
}

// Backward implements backpropagation through the GPT model.
func (gpt *GPT) Backward(gradLogits *Tensor, cache *ForwardCache) {
	// Gradient through output projection
	// logits = x @ lmHead^T
	// ∂L/∂x = gradLogits @ lmHead
	// ∂L/∂lmHead = gradLogits^T @ x

	x := cache.lnFinalInput // Input to final projection

	// Gradient for lmHead
	gradLogitsT := Transpose(gradLogits)
	gradLmHead := MatMul(gradLogitsT, x)
	gpt.lmHead.AccumulateGrad(gradLmHead)

	// Gradient for x (input to output projection)
	gradX := MatMul(gradLogits, gpt.lmHead)

	// Backward through final layer norm
	gradX, gradGamma, gradBeta := LayerNormBackward(
		cache.lnFinalInput, x, gpt.lnFinal.gamma, gpt.lnFinal.beta,
		gradX, 1e-5)
	gpt.lnFinal.gamma.AccumulateGrad(gradGamma)
	gpt.lnFinal.beta.AccumulateGrad(gradBeta)

	// Backward through transformer blocks (in reverse order)
	for layer := gpt.config.NumLayers - 1; layer >= 0; layer-- {
		block := gpt.blocks[layer]
		blockCache := cache.blockCaches[layer]

		// Backward through residual 2
		gradFF := gradX.Clone()

		// Backward through layer norm 2
		gradFF, gradGamma2, gradBeta2 := LayerNormBackward(
			blockCache.ffOutput, blockCache.ffInput,
			block.ln2.gamma, block.ln2.beta, gradFF, 1e-5)
		block.ln2.gamma.AccumulateGrad(gradGamma2)
		block.ln2.beta.AccumulateGrad(gradBeta2)

		// Backward through feed-forward
		gradFFInput := block.ff.Backward(gradFF, blockCache.ffCache)

		// Add residual gradient
		gradX = Add(gradX, gradFFInput)

		// Backward through residual 1
		gradAttn := gradX.Clone()

		// Backward through layer norm 1
		gradAttn, gradGamma1, gradBeta1 := LayerNormBackward(
			blockCache.attnOutput, blockCache.attnInput,
			block.ln1.gamma, block.ln1.beta, gradAttn, 1e-5)
		block.ln1.gamma.AccumulateGrad(gradGamma1)
		block.ln1.beta.AccumulateGrad(gradBeta1)

		// Backward through attention
		gradAttnInput := block.attn.Backward(gradAttn, blockCache.attnCache)

		// Add residual gradient
		gradX = Add(gradX, gradAttnInput)
	}

	// Backward through embeddings
	// Token embeddings
	for i, tokenID := range cache.tokenIDs {
		for d := 0; d < gpt.config.EmbedDim; d++ {
			gpt.tokenEmbed.grad[tokenID*gpt.config.EmbedDim+d] += gradX.At(i, d)
		}
	}

	// Positional embeddings
	for i, posID := range cache.posIDs {
		for d := 0; d < gpt.config.EmbedDim; d++ {
			gpt.posEmbed.grad[posID*gpt.config.EmbedDim+d] += gradX.At(i, d)
		}
	}
}

// Backward through FeedForward layer.
func (ff *FeedForward) Backward(gradOutput *Tensor, cache *FFCache) *Tensor {
	// Backward through second linear: output = hidden @ W2 + b2
	// ∂L/∂hidden = gradOutput @ W2^T
	// ∂L/∂W2 = hidden^T @ gradOutput
	// ∂L/∂b2 = sum(gradOutput, axis=0)

	gradHidden, gradW2 := MatMulBackward(cache.hidden, ff.w2, gradOutput)
	ff.w2.AccumulateGrad(gradW2)

	// Gradient for bias: sum over batch dimension
	for i := range gradOutput.data {
		ff.b2.grad[i%ff.b2.Size()] += gradOutput.data[i]
	}

	// Backward through GELU
	gradPreActivation := GELUBackward(cache.preActivation, gradHidden)

	// Backward through first linear: hidden = x @ W1 + b1
	gradInput, gradW1 := MatMulBackward(gradPreActivation, ff.w1, gradPreActivation)
	ff.w1.AccumulateGrad(gradW1)

	// Gradient for bias
	for i := range gradPreActivation.data {
		ff.b1.grad[i%ff.b1.Size()] += gradPreActivation.data[i]
	}

	return gradInput
}

// Backward through Attention layer.
func (attn *Attention) Backward(gradOutput *Tensor, cache *AttentionCache) *Tensor {
	// This is simplified - full implementation would handle multi-head properly
	// For now, backward through the main operations

	// Backward through output projection
	gradContext, gradWo := MatMulBackward(cache.context, attn.wo, gradOutput)
	attn.wo.AccumulateGrad(gradWo)

	// Backward through attention mechanism
	// context = weights @ V
	gradWeights, gradV := MatMulBackward(cache.weights, cache.v, gradContext)

	// Backward through softmax
	gradScores := SoftmaxBackward(cache.weights, gradWeights)

	// Backward through scaling
	gradScores = Scale(gradScores, 1.0/math.Sqrt(float64(attn.headDim)))

	// Backward through attention scores: scores = Q @ K^T
	gradQ, gradK := MatMulBackward(cache.q, Transpose(cache.k), gradScores)

	// Backward through Q, K, V projections
	gradInput := NewTensor(cache.q.shape[0], attn.embedDim)

	// Q projection
	gradInputQ, gradWq := MatMulBackward(gradInput, attn.wq, gradQ)
	attn.wq.AccumulateGrad(gradWq)
	gradInput = Add(gradInput, gradInputQ)

	// K projection
	gradInputK, gradWk := MatMulBackward(gradInput, attn.wk, gradK)
	attn.wk.AccumulateGrad(gradWk)
	gradInput = Add(gradInput, gradInputK)

	// V projection
	gradInputV, gradWv := MatMulBackward(gradInput, attn.wv, gradV)
	attn.wv.AccumulateGrad(gradWv)
	gradInput = Add(gradInput, gradInputV)

	return gradInput
}
