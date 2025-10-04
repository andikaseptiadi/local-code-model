package main

// ===========================================================================
// WHAT'S GOING ON HERE
// ===========================================================================
//
// SwiGLU is a modern activation function used in feed-forward networks.
// It's a variant of GLU (Gated Linear Unit) that uses the Swish activation
// instead of sigmoid.
//
// WHY SWIGLU IS BETTER:
// - Better performance than GELU and ReLU in practice
// - Used in modern LLMs: PaLM, LLaMA 2, etc.
// - Gating mechanism provides more expressiveness
// - Smooth, non-monotonic activation
//
// PAPER: "GLU Variants Improve Transformer"
//        https://arxiv.org/abs/2002.05202
//
// INTUITION:
// Instead of a single activation:
//   FFN(x) = activation(x @ W1) @ W2
//
// SwiGLU splits the hidden layer into two paths:
//   FFN(x) = (Swish(x @ W_gate) ⊙ (x @ W)) @ W2
//
// One path (gate) controls which information from the other path flows through.
// Think of it as a learned "relevance filter" for each neuron.
//
// MATHEMATICS:
// Swish(x) = x * sigmoid(x)
// SwiGLU(x, W, V, W2) = (Swish(xW) ⊙ xV) @ W2
//
// Where:
//   W, V: Two separate weight matrices for gate and value
//   ⊙: Element-wise multiplication (Hadamard product)
//   W2: Output projection
//
// ===========================================================================

// Swish activation: x * sigmoid(x)
// Also known as SiLU (Sigmoid Linear Unit) in some papers
func Swish(x *Tensor) *Tensor {
	out := NewTensor(x.shape...)
	for i := range x.data {
		// Swish(x) = x * sigmoid(x) = x / (1 + e^(-x))
		sigmoid := 1.0 / (1.0 + exp(-x.data[i]))
		out.data[i] = x.data[i] * sigmoid
	}
	return out
}

// exp is a helper for computing e^x
func exp(x float64) float64 {
	// For numerical stability, clamp extreme values
	if x > 20.0 {
		return 485165195.4  // e^20
	}
	if x < -20.0 {
		return 2.061153622e-09  // e^-20
	}

	// Use math.Exp for actual computation
	// This is a placeholder - production would use math.Exp
	// (avoiding import here to keep file self-contained)
	result := 1.0
	term := 1.0
	for n := 1; n < 20; n++ {
		term *= x / float64(n)
		result += term
		if abs(term) < 1e-10 {
			break
		}
	}
	return result
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// SwiGLUFeedForward is a feed-forward layer using SwiGLU activation.
// Unlike standard FFN with one hidden layer, SwiGLU uses TWO parallel
// projections that interact through gating.
type SwiGLUFeedForward struct {
	wGate *Tensor // Gate projection: (embedDim, hiddenDim)
	wValue *Tensor // Value projection: (embedDim, hiddenDim)
	w2     *Tensor // Output projection: (hiddenDim, embedDim)
	bGate  *Tensor // Bias for gate (optional, often omitted)
	bValue *Tensor // Bias for value (optional, often omitted)
	b2     *Tensor // Bias for output (optional, often omitted)

	// Backend for accelerated operations (optional)
	backend interface {
		MatMul(*Tensor, *Tensor) (*Tensor, error)
	}
}

// NewSwiGLUFeedForward creates a SwiGLU feed-forward layer.
// Note: hiddenDim for SwiGLU is typically 2/3 the size of standard FFN
// to keep parameter count similar (since we have two parallel projections).
func NewSwiGLUFeedForward(embedDim, hiddenDim int) *SwiGLUFeedForward {
	return &SwiGLUFeedForward{
		wGate:  NewTensorRand(embedDim, hiddenDim),
		wValue: NewTensorRand(embedDim, hiddenDim),
		w2:     NewTensorRand(hiddenDim, embedDim),
		bGate:  NewTensor(hiddenDim),
		bValue: NewTensor(hiddenDim),
		b2:     NewTensor(embedDim),
	}
}

// Forward applies the SwiGLU feed-forward transformation.
// x shape: (seqLen, embedDim)
// Returns: (seqLen, embedDim)
func (ff *SwiGLUFeedForward) Forward(x *Tensor) *Tensor {
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

	// Gate path: Swish(x @ W_gate + b_gate)
	gate := matmul(x, ff.wGate)
	gate = addBias(gate, ff.bGate)
	gate = Swish(gate)

	// Value path: x @ W_value + b_value
	value := matmul(x, ff.wValue)
	value = addBias(value, ff.bValue)

	// Element-wise multiplication (gating)
	hidden := Mul(gate, value)

	// Output projection
	output := matmul(hidden, ff.w2)
	output = addBias(output, ff.b2)

	return output
}

// Mul is defined in tensor.go - element-wise multiplication (Hadamard product)

// ===========================================================================
// COMPARISON: SwiGLU vs GELU Feed-Forward
// ===========================================================================
//
// Standard FFN with GELU (Original Transformer):
//   hidden = GELU(x @ W1 + b1)
//   output = hidden @ W2 + b2
//   Parameters: (d × 4d) + (4d × d) = 8d² + 5d
//
// SwiGLU FFN (Modern LLMs):
//   gate = Swish(x @ W_gate + b_gate)
//   value = x @ W_value + b_value
//   hidden = gate ⊙ value
//   output = hidden @ W2 + b2
//   Parameters: (d × h) + (d × h) + (h × d) = 3dh + 3d
//
// For comparable parameters, set h = 8d²/3d = (8/3)d ≈ 2.67d
// In practice, often use h = (8/3)d or h = 3d
//
// ADVANTAGES OF SWIGLU:
//   1. Better performance in practice (empirically proven)
//   2. Gating provides more flexible information flow
//   3. Swish is smooth and non-monotonic
//   4. Used in state-of-the-art models (LLaMA, PaLM)
//
// TRADE-OFFS:
//   - Two matrix multiplications instead of one in first layer
//   - Slightly more complex to implement
//   - Element-wise multiplication adds computation (but it's cheap)
//
// WHEN TO USE:
//   - Use SwiGLU for state-of-the-art performance
//   - Use GELU for simplicity and following original Transformer
//   - The difference matters more at scale (large models, long training)
//
// ===========================================================================
