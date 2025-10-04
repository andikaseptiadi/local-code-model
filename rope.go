package main

import "math"

// ===========================================================================
// WHAT'S GOING ON HERE
// ===========================================================================
//
// RoPE (Rotary Position Embeddings) is a modern alternative to learned
// positional embeddings. Instead of adding position embeddings to token
// embeddings, RoPE rotates the query and key vectors by an angle proportional
// to their position in the sequence.
//
// WHY ROPE IS BETTER:
// - Relative position encoding (naturally handles variable lengths)
// - Better extrapolation to longer sequences than seen during training
// - No learned parameters (saves memory and training time)
// - Used in modern LLMs: LLaMA, PaLM, GPT-NeoX, etc.
//
// PAPER: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
//        https://arxiv.org/abs/2104.09864
//
// INTUITION:
// Think of each pair of dimensions as a 2D plane. We rotate vectors in these
// planes by an angle that increases with position. This creates a natural
// "distance" metric: vectors at nearby positions have similar rotations.
//
// MATHEMATICS:
// For position m and dimension pair (2i, 2i+1):
//   θ_i = 10000^(-2i/d)
//   R(m) = rotation matrix by angle m·θ_i
//
// Apply rotation to Q and K (not V):
//   Q' = R(m) @ Q
//   K' = R(n) @ K
//
// The attention score Q'·K' naturally includes relative position information!
//
// ===========================================================================

// RoPE applies rotary position embeddings to a tensor.
//
// x: input tensor of shape (seqLen, embedDim)
// posOffset: starting position in sequence (for KV caching)
//
// Returns: tensor with RoPE applied
func RoPE(x *Tensor, posOffset int) *Tensor {
	if len(x.shape) != 2 {
		panic("rope: input must be 2D (seqLen, embedDim)")
	}

	seqLen := x.shape[0]
	embedDim := x.shape[1]

	// RoPE works on pairs of dimensions
	if embedDim%2 != 0 {
		panic("rope: embedDim must be even")
	}

	out := NewTensor(seqLen, embedDim)

	// Apply rotation to each position
	for pos := 0; pos < seqLen; pos++ {
		actualPos := posOffset + pos

		// Process dimension pairs
		for i := 0; i < embedDim/2; i++ {
			// Compute rotation frequency for this dimension pair
			// θ_i = 10000^(-2i/d)
			theta := 1.0 / math.Pow(10000.0, float64(2*i)/float64(embedDim))

			// Rotation angle for this position
			angle := float64(actualPos) * theta

			cos := math.Cos(angle)
			sin := math.Sin(angle)

			// Get the pair of values
			x0 := x.At(pos, 2*i)
			x1 := x.At(pos, 2*i+1)

			// Apply 2D rotation matrix:
			// [cos -sin] [x0]
			// [sin  cos] [x1]
			out.Set(x0*cos-x1*sin, pos, 2*i)
			out.Set(x0*sin+x1*cos, pos, 2*i+1)
		}
	}

	return out
}

// RoPEWithCache applies RoPE and is compatible with KV caching.
// This is a convenience wrapper that determines the position offset automatically.
func RoPEWithCache(x *Tensor, cache *KVCache, layerIdx int) *Tensor {
	posOffset := 0
	if cache != nil {
		posOffset = cache.CachedLen()
	}
	return RoPE(x, posOffset)
}

// ===========================================================================
// COMPARISON: RoPE vs Learned Positional Embeddings
// ===========================================================================
//
// Learned Positional Embeddings (Original Transformer):
//   - Fixed maximum sequence length
//   - Learned embeddings for each position: posEmbed[pos]
//   - Added to token embeddings: x = tokEmbed + posEmbed
//   - Parameters: seqLen × embedDim
//
// RoPE (Modern LLMs):
//   - No maximum sequence length (within reason)
//   - Computed on-the-fly using sin/cos
//   - Applied as rotation to Q and K
//   - Parameters: 0
//
// ADVANTAGES OF ROPE:
//   1. Zero parameters (saves memory)
//   2. Extrapolates to longer sequences
//   3. Relative position encoding (learns "X tokens apart")
//   4. Better performance in practice
//
// TRADE-OFFS:
//   - Slightly more computation (sin/cos per forward pass)
//   - More complex to implement
//   - Harder to understand initially
//
// ===========================================================================
