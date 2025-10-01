package main

// ===========================================================================
// WHAT'S GOING ON HERE
// ===========================================================================
//
// This file implements automatic differentiation (autograd) for backpropagation.
//
// INTENTION:
// Enable gradient computation through computational graphs. Each operation
// (MatMul, Add, ReLU, etc.) needs a forward pass (compute output) and a
// backward pass (compute gradients).
//
// THE CHAIN RULE:
//
// Given: y = f(x) and z = g(y)
// Want: ∂z/∂x (how z changes with x)
//
// Chain rule: ∂z/∂x = ∂z/∂y · ∂y/∂x
//
// In backpropagation:
//   - Forward: Compute y = f(x), z = g(y)
//   - Backward: Given ∂L/∂z, compute ∂L/∂x = ∂L/∂z · ∂z/∂y · ∂y/∂x
//
// EXAMPLE: Matrix Multiplication
//
// Forward: C = A @ B
// Backward:
//   - ∂L/∂A = ∂L/∂C @ B^T
//   - ∂L/∂B = A^T @ ∂L/∂C
//
// WHERE WE ARE:
// This is Level -1 (before Level 0): The foundational math that makes
// learning possible. Without this, we can only do inference.
//
// PERFORMANCE:
// Backward pass is typically 2x the cost of forward pass:
//   - Forward: One matmul
//   - Backward: Two matmuls (one for each input gradient)
//
// ===========================================================================

import (
	"math"
)

// Backward operations for automatic differentiation.
// Each operation needs both forward and backward implementations.

// MatMulBackward computes gradients for matrix multiplication.
//
// Given:
//   - C = A @ B
//   - gradC = ∂L/∂C (gradient flowing back from loss)
//
// Compute:
//   - gradA = ∂L/∂A = gradC @ B^T
//   - gradB = ∂L/∂B = A^T @ gradC
//
// Derivation:
//   C[i,j] = Σ_k A[i,k] * B[k,j]
//   ∂C[i,j]/∂A[i,k] = B[k,j]
//   ∂L/∂A[i,k] = Σ_j ∂L/∂C[i,j] * B[k,j] = (gradC @ B^T)[i,k]
func MatMulBackward(a, b, gradC *Tensor) (gradA, gradB *Tensor) {
	// ∂L/∂A = gradC @ B^T
	bT := Transpose(b)
	gradA = MatMul(gradC, bT)

	// ∂L/∂B = A^T @ gradC
	aT := Transpose(a)
	gradB = MatMul(aT, gradC)

	return gradA, gradB
}

// AddBackward computes gradients for element-wise addition.
//
// Given:
//   - C = A + B
//   - gradC = ∂L/∂C
//
// Compute:
//   - gradA = ∂L/∂A = gradC (gradient passes through unchanged)
//   - gradB = ∂L/∂B = gradC
//
// Derivation:
//   C[i] = A[i] + B[i]
//   ∂C[i]/∂A[i] = 1
//   ∂L/∂A[i] = ∂L/∂C[i] * 1 = gradC[i]
func AddBackward(gradC *Tensor) (gradA, gradB *Tensor) {
	// Addition distributes gradients equally to both inputs
	return gradC.Clone(), gradC.Clone()
}

// ScaleBackward computes gradient for scalar multiplication.
//
// Given:
//   - Y = scalar * X
//   - gradY = ∂L/∂Y
//
// Compute:
//   - gradX = ∂L/∂X = scalar * gradY
//
// Derivation:
//   Y[i] = scalar * X[i]
//   ∂Y[i]/∂X[i] = scalar
//   ∂L/∂X[i] = ∂L/∂Y[i] * scalar
func ScaleBackward(scalar float64, gradY *Tensor) *Tensor {
	return Scale(gradY, scalar)
}

// ReLUBackward computes gradient for ReLU activation.
//
// Given:
//   - Y = ReLU(X) = max(0, X)
//   - gradY = ∂L/∂Y
//
// Compute:
//   - gradX = ∂L/∂X = gradY * (X > 0)
//
// Derivation:
//   Y[i] = max(0, X[i])
//   ∂Y[i]/∂X[i] = 1 if X[i] > 0, else 0
//   ∂L/∂X[i] = ∂L/∂Y[i] * indicator(X[i] > 0)
func ReLUBackward(x, gradY *Tensor) *Tensor {
	gradX := NewTensor(x.shape...)

	for i := range x.data {
		if x.data[i] > 0 {
			gradX.data[i] = gradY.data[i]
		}
		// else: gradient is 0 (ReLU derivative is 0 for negative inputs)
	}

	return gradX
}

// GELUBackward computes gradient for GELU activation.
//
// GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
//
// Derivative is complex but can be computed analytically.
// For simplicity, we use numerical approximation or the exact formula.
func GELUBackward(x, gradY *Tensor) *Tensor {
	gradX := NewTensor(x.shape...)

	const (
		sqrt2OverPi = 0.7978845608028654
		coeff       = 0.044715
	)

	for i := range x.data {
		v := x.data[i]

		// Compute GELU and its derivative
		inner := sqrt2OverPi * (v + coeff*v*v*v)
		tanhInner := math.Tanh(inner)

		// d/dx GELU(x) using chain rule
		// This is the derivative of GELU w.r.t. x
		tanhDeriv := 1.0 - tanhInner*tanhInner // sech²(inner)
		innerDeriv := sqrt2OverPi * (1.0 + 3.0*coeff*v*v)
		geluDeriv := 0.5*(1.0+tanhInner) + 0.5*v*tanhDeriv*innerDeriv

		// Apply chain rule
		gradX.data[i] = gradY.data[i] * geluDeriv
	}

	return gradX
}

// SoftmaxBackward computes gradient for softmax.
//
// Given:
//   - Y = softmax(X)
//   - gradY = ∂L/∂Y
//
// Compute:
//   - gradX = ∂L/∂X
//
// Derivation:
//   Y[i] = exp(X[i]) / Σ_j exp(X[j])
//   ∂Y[i]/∂X[j] = Y[i] * (δ[i,j] - Y[j])
//   where δ[i,j] = 1 if i==j, else 0
//
// Simplifies to:
//   gradX[i] = Y[i] * (gradY[i] - Σ_j gradY[j] * Y[j])
func SoftmaxBackward(y, gradY *Tensor) *Tensor {
	if len(y.shape) != 2 {
		panic("SoftmaxBackward: requires 2D tensor")
	}

	batch := y.shape[0]
	features := y.shape[1]

	gradX := NewTensor(y.shape...)

	for b := 0; b < batch; b++ {
		// Compute dot product: Σ_j gradY[j] * Y[j]
		dot := 0.0
		for f := 0; f < features; f++ {
			dot += gradY.At(b, f) * y.At(b, f)
		}

		// Compute gradient: Y[i] * (gradY[i] - dot)
		for f := 0; f < features; f++ {
			gradX.Set(y.At(b, f)*(gradY.At(b, f)-dot), b, f)
		}
	}

	return gradX
}

// LayerNormBackward computes gradients for layer normalization.
//
// LayerNorm: y = gamma * (x - mean) / std + beta
//
// where:
//   - mean = Σ x[i] / n
//   - variance = Σ (x[i] - mean)² / n
//   - std = sqrt(variance + epsilon)
//
// Gradients:
//   - ∂L/∂x involves the chain rule through mean and variance
//   - ∂L/∂gamma = Σ ∂L/∂y * (x - mean) / std
//   - ∂L/∂beta = Σ ∂L/∂y
func LayerNormBackward(x, y, gamma, beta, gradY *Tensor, epsilon float64) (gradX, gradGamma, gradBeta *Tensor) {
	if len(x.shape) != 2 {
		panic("LayerNormBackward: requires 2D tensor")
	}

	batch := x.shape[0]
	features := x.shape[1]

	gradX = NewTensor(x.shape...)
	gradGamma = NewTensor(gamma.shape...)
	gradBeta = NewTensor(beta.shape...)

	n := float64(features)

	for b := 0; b < batch; b++ {
		// Recompute statistics (needed for backward pass)
		mean := 0.0
		for f := 0; f < features; f++ {
			mean += x.At(b, f)
		}
		mean /= n

		variance := 0.0
		for f := 0; f < features; f++ {
			diff := x.At(b, f) - mean
			variance += diff * diff
		}
		variance /= n

		std := math.Sqrt(variance + epsilon)

		// Compute gradients for gamma and beta
		for f := 0; f < features; f++ {
			xNorm := (x.At(b, f) - mean) / std

			// ∂L/∂gamma[f] += ∂L/∂y[b,f] * xNorm
			gradGamma.data[f] += gradY.At(b, f) * xNorm

			// ∂L/∂beta[f] += ∂L/∂y[b,f]
			gradBeta.data[f] += gradY.At(b, f)
		}

		// Compute gradient for x (complex due to mean/variance dependencies)
		// Using the standard formula for batch norm backward pass

		// Sum of gradients
		sumGradY := 0.0
		sumGradYXNorm := 0.0
		for f := 0; f < features; f++ {
			xNorm := (x.At(b, f) - mean) / std
			sumGradY += gradY.At(b, f) * gamma.data[f]
			sumGradYXNorm += gradY.At(b, f) * gamma.data[f] * xNorm
		}

		for f := 0; f < features; f++ {
			xNorm := (x.At(b, f) - mean) / std

			// Gradient through normalization
			gradXNorm := gradY.At(b, f) * gamma.data[f]

			// Gradient through variance and mean
			// This is the standard batch norm gradient formula
			gradX.Set((n*gradXNorm - sumGradY - xNorm*sumGradYXNorm) / (n * std), b, f)
		}
	}

	return gradX, gradGamma, gradBeta
}

// CrossEntropyBackward computes gradient for cross-entropy loss.
//
// Given:
//   - logits: (batch, vocab_size)
//   - targets: (batch) - target indices
//   - loss = -log(softmax(logits)[target])
//
// Compute:
//   - gradLogits = ∂L/∂logits
//
// Derivation:
//   For the target class: ∂L/∂logit[target] = softmax[target] - 1
//   For other classes: ∂L/∂logit[i] = softmax[i]
//
// Simplified: gradLogits = softmax(logits) - one_hot(targets)
func CrossEntropyBackward(logits *Tensor, targets []int) *Tensor {
	if len(logits.shape) != 2 {
		panic("CrossEntropyBackward: requires 2D logits")
	}

	batchSize := logits.shape[0]
	vocabSize := logits.shape[1]

	// Compute softmax
	probs := Softmax(logits)

	// Gradient: probs - one_hot(targets)
	// Averaged over batch (matches loss computation)
	gradLogits := NewTensor(batchSize, vocabSize)

	for b := 0; b < batchSize; b++ {
		for v := 0; v < vocabSize; v++ {
			if v == targets[b] {
				// Target class: gradient is (prob - 1) / batch_size
				gradLogits.Set((probs.At(b, v)-1.0)/float64(batchSize), b, v)
			} else {
				// Non-target class: gradient is prob / batch_size
				gradLogits.Set(probs.At(b, v)/float64(batchSize), b, v)
			}
		}
	}

	return gradLogits
}

// AccumulateGrad adds gradient to a tensor's gradient buffer.
// Used when a tensor is used multiple times in the forward pass.
func (t *Tensor) AccumulateGrad(grad *Tensor) {
	if !shapeEqual(t.shape, grad.shape) {
		panic("AccumulateGrad: shape mismatch")
	}

	for i := range t.grad {
		t.grad[i] += grad.data[i]
	}
}
