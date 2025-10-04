package main

import "math"

// ===========================================================================
// WHAT'S GOING ON HERE
// ===========================================================================
//
// RMSNorm (Root Mean Square Layer Normalization) is a simpler and faster
// alternative to LayerNorm that achieves comparable performance.
//
// WHY RMSNORM IS BETTER:
// - Simpler: No mean subtraction, only RMS scaling
// - Faster: ~10-30% less computation than LayerNorm
// - Comparable performance in practice
// - Used in modern LLMs: LLaMA, GPT-NeoX, T5
//
// PAPER: "Root Mean Square Layer Normalization"
//        https://arxiv.org/abs/1910.07467
//
// INTUITION:
// LayerNorm normalizes by removing mean and scaling by standard deviation.
// RMSNorm observes that the mean removal doesn't contribute much to
// performance. We can just scale by the RMS (root mean square) instead.
//
// Think of it as: "Make all activations have similar magnitudes" without
// worrying about their center point.
//
// MATHEMATICS:
//
// LayerNorm:
//   y = γ * (x - μ) / σ + β
//   where μ = mean(x), σ = std(x)
//
// RMSNorm:
//   y = γ * x / RMS(x)
//   where RMS(x) = sqrt(mean(x²))
//
// Key difference: No mean subtraction (μ) and no bias term (β)
//
// ===========================================================================

// RMSNorm implements root mean square layer normalization.
type RMSNorm struct {
	dim   int
	eps   float64
	gamma *Tensor // Scale parameter (no beta unlike LayerNorm)
}

// NewRMSNorm creates an RMSNorm layer.
func NewRMSNorm(dim int) *RMSNorm {
	gamma := NewTensor(dim)

	// Initialize gamma to 1 (identity transform)
	for i := 0; i < dim; i++ {
		gamma.data[i] = 1.0
	}

	return &RMSNorm{
		dim:   dim,
		eps:   1e-5, // Small constant for numerical stability
		gamma: gamma,
	}
}

// Forward applies RMSNorm.
// x shape: (seqLen, features)
// Returns: (seqLen, features)
func (rms *RMSNorm) Forward(x *Tensor) *Tensor {
	if len(x.shape) != 2 {
		panic("rmsnorm: input must be 2D (seqLen, features)")
	}

	seqLen, features := x.shape[0], x.shape[1]
	out := NewTensor(seqLen, features)

	// Normalize each position independently
	for i := 0; i < seqLen; i++ {
		// Compute RMS: sqrt(mean(x²))
		sumSquares := 0.0
		for j := 0; j < features; j++ {
			val := x.At(i, j)
			sumSquares += val * val
		}
		rmsValue := math.Sqrt(sumSquares/float64(features) + rms.eps)

		// Scale by 1/RMS and apply learned gamma
		for j := 0; j < features; j++ {
			normalized := x.At(i, j) / rmsValue
			scaled := normalized * rms.gamma.data[j]
			out.Set(scaled, i, j)
		}
	}

	return out
}

// ===========================================================================
// COMPARISON: RMSNorm vs LayerNorm
// ===========================================================================
//
// LayerNorm (Original Transformer):
//   1. Compute mean: μ = Σx / n
//   2. Compute variance: σ² = Σ(x - μ)² / n
//   3. Normalize: x_norm = (x - μ) / √(σ² + ε)
//   4. Scale and shift: y = γ * x_norm + β
//   Operations: 2n additions, 2n multiplications, 1 sqrt, 1 division per feature
//   Parameters: 2d (gamma and beta)
//
// RMSNorm (Modern LLMs):
//   1. Compute RMS: rms = √(Σx² / n + ε)
//   2. Normalize and scale: y = γ * x / rms
//   Operations: n additions (for squares), n multiplications, 1 sqrt, 1 division per feature
//   Parameters: d (gamma only)
//
// COMPUTATIONAL SAVINGS:
//   - No mean computation (saves 1 pass over data)
//   - No mean subtraction (saves n subtractions)
//   - No beta parameter (saves n additions and d parameters)
//   - Overall: ~10-30% faster than LayerNorm
//
// WHEN TO USE RMSNORM:
//   ✓ Training large models (speed matters at scale)
//   ✓ Following modern architectures (LLaMA, GPT-NeoX)
//   ✓ Want simpler implementation
//   ✓ Inference speed is critical
//
// WHEN TO USE LAYERNORM:
//   ✓ Following original Transformer paper
//   ✓ Working with pre-trained models that use LayerNorm
//   ✓ Compatibility with existing codebases
//   ✓ Extra 10-30% compute time is acceptable
//
// PERFORMANCE IN PRACTICE:
//   - RMSNorm and LayerNorm achieve similar model quality
//   - RMSNorm is faster but difference shrinks with larger models
//   - Modern research shows no significant quality gap
//   - Most new large models use RMSNorm for efficiency
//
// ===========================================================================
// EDUCATIONAL NOTE: Why does removing mean subtraction work?
//
// The key insight: What matters for deep learning is the *shape* of the
// activation distribution (its scale), not its exact location (mean).
//
// LayerNorm's mean subtraction:
//   - Ensures zero mean (centers distribution)
//   - Helps with training stability in shallow networks
//   - Less critical in deep networks with residual connections
//
// RMSNorm's observation:
//   - Residual connections already provide mean-centering effect
//   - Scaling is more important than centering
//   - Simpler normalization is sufficient
//
// This is validated empirically: RMSNorm matches LayerNorm performance
// in transformers while being simpler and faster.
// ===========================================================================
