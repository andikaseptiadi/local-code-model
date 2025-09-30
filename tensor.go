package main

import (
	"fmt"
	"math"
	"math/rand"
)

// Tensor represents a multi-dimensional array of floats.
// This is our basic building block - everything in neural networks
// operates on tensors (scalars, vectors, matrices, etc.)
//
// For learning purposes, we're starting with a simple implementation.
// Later you can optimize with SIMD, GPU kernels, etc.
type Tensor struct {
	Data  []float64 // Flat array storing all elements
	Shape []int     // Dimensions [batch, seq_len, features, etc.]
	Grad  []float64 // Gradient for backpropagation (same shape as Data)
}

// NewTensor creates a new tensor with given shape, initialized to zeros.
func NewTensor(shape ...int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}

	return &Tensor{
		Data:  make([]float64, size),
		Shape: shape,
		Grad:  make([]float64, size),
	}
}

// NewTensorRand creates a tensor with random values from normal distribution.
// This is used for weight initialization - random weights break symmetry
// and allow the network to learn different features.
//
// Xavier/He initialization (scaled by sqrt(fanIn)) comes later.
func NewTensorRand(shape ...int) *Tensor {
	t := NewTensor(shape...)
	for i := range t.Data {
		// Box-Muller transform for normal distribution
		u1 := rand.Float64()
		u2 := rand.Float64()
		t.Data[i] = math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2) * 0.02
	}
	return t
}

// Size returns the total number of elements in the tensor.
func (t *Tensor) Size() int {
	size := 1
	for _, dim := range t.Shape {
		size *= dim
	}
	return size
}

// ZeroGrad clears the gradient. Call this before each backward pass.
func (t *Tensor) ZeroGrad() {
	for i := range t.Grad {
		t.Grad[i] = 0
	}
}

// At returns the value at the given indices.
// For a 2D matrix: At(i, j)
// For a 3D tensor: At(batch, seq, feature)
func (t *Tensor) At(indices ...int) float64 {
	if len(indices) != len(t.Shape) {
		panic(fmt.Sprintf("expected %d indices, got %d", len(t.Shape), len(indices)))
	}

	// Convert multi-dimensional indices to flat index
	idx := 0
	stride := 1
	for i := len(indices) - 1; i >= 0; i-- {
		idx += indices[i] * stride
		stride *= t.Shape[i]
	}

	return t.Data[idx]
}

// Set sets the value at the given indices.
func (t *Tensor) Set(value float64, indices ...int) {
	if len(indices) != len(t.Shape) {
		panic(fmt.Sprintf("expected %d indices, got %d", len(t.Shape), len(indices)))
	}

	idx := 0
	stride := 1
	for i := len(indices) - 1; i >= 0; i-- {
		idx += indices[i] * stride
		stride *= t.Shape[i]
	}

	t.Data[idx] = value
}

// Reshape returns a new view of the tensor with different shape.
// Total number of elements must remain the same.
func (t *Tensor) Reshape(newShape ...int) *Tensor {
	newSize := 1
	for _, dim := range newShape {
		newSize *= dim
	}

	if newSize != t.Size() {
		panic(fmt.Sprintf("cannot reshape tensor of size %d to %v", t.Size(), newShape))
	}

	// Shallow copy - shares underlying data
	return &Tensor{
		Data:  t.Data,
		Shape: newShape,
		Grad:  t.Grad,
	}
}

// Copy creates a deep copy of the tensor.
func (t *Tensor) Copy() *Tensor {
	data := make([]float64, len(t.Data))
	grad := make([]float64, len(t.Grad))
	shape := make([]int, len(t.Shape))

	copy(data, t.Data)
	copy(grad, t.Grad)
	copy(shape, t.Shape)

	return &Tensor{
		Data:  data,
		Shape: shape,
		Grad:  grad,
	}
}

// String returns a string representation of the tensor.
func (t *Tensor) String() string {
	return fmt.Sprintf("Tensor(shape=%v, size=%d)", t.Shape, t.Size())
}

// ===========================================================================
// BASIC OPERATIONS
// ===========================================================================

// Add performs element-wise addition: out = a + b
// Both tensors must have the same shape (broadcasting comes later).
func Add(a, b *Tensor) *Tensor {
	if !shapeEqual(a.Shape, b.Shape) {
		panic(fmt.Sprintf("shape mismatch: %v vs %v", a.Shape, b.Shape))
	}

	out := NewTensor(a.Shape...)
	for i := range out.Data {
		out.Data[i] = a.Data[i] + b.Data[i]
	}

	return out
}

// Multiply performs element-wise multiplication: out = a * b
func Multiply(a, b *Tensor) *Tensor {
	if !shapeEqual(a.Shape, b.Shape) {
		panic(fmt.Sprintf("shape mismatch: %v vs %v", a.Shape, b.Shape))
	}

	out := NewTensor(a.Shape...)
	for i := range out.Data {
		out.Data[i] = a.Data[i] * b.Data[i]
	}

	return out
}

// Scale multiplies tensor by a scalar: out = a * scalar
func Scale(a *Tensor, scalar float64) *Tensor {
	out := NewTensor(a.Shape...)
	for i := range out.Data {
		out.Data[i] = a.Data[i] * scalar
	}
	return out
}

// MatMul performs matrix multiplication: C = A @ B
// A: (M, K), B: (K, N) -> C: (M, N)
//
// This is the O(n^3) operation at the heart of neural networks.
// Later optimizations: BLAS, GPU, blocked algorithms, etc.
func MatMul(a, b *Tensor) *Tensor {
	// Only handle 2D matrices for now (batched matmul comes later)
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		panic("MatMul currently only supports 2D tensors")
	}

	M, K1 := a.Shape[0], a.Shape[1]
	K2, N := b.Shape[0], b.Shape[1]

	if K1 != K2 {
		panic(fmt.Sprintf("incompatible dimensions for matmul: (%d,%d) @ (%d,%d)", M, K1, K2, N))
	}

	K := K1
	out := NewTensor(M, N)

	// Naive triple loop - O(M*N*K)
	// Your systems background will immediately see the cache misses here.
	// Optimizations: loop tiling, SIMD, Strassen's algorithm, GPU kernels
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			sum := 0.0
			for k := 0; k < K; k++ {
				sum += a.At(i, k) * b.At(k, j)
			}
			out.Set(sum, i, j)
		}
	}

	return out
}

// Transpose returns the transpose of a 2D matrix.
// A: (M, N) -> A^T: (N, M)
func Transpose(a *Tensor) *Tensor {
	if len(a.Shape) != 2 {
		panic("Transpose currently only supports 2D tensors")
	}

	M, N := a.Shape[0], a.Shape[1]
	out := NewTensor(N, M)

	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			out.Set(a.At(i, j), j, i)
		}
	}

	return out
}

// ===========================================================================
// ACTIVATION FUNCTIONS
// ===========================================================================

// ReLU applies Rectified Linear Unit: f(x) = max(0, x)
// Simple but effective - this is what made deep learning work.
func ReLU(x *Tensor) *Tensor {
	out := NewTensor(x.Shape...)
	for i := range x.Data {
		if x.Data[i] > 0 {
			out.Data[i] = x.Data[i]
		} else {
			out.Data[i] = 0
		}
	}
	return out
}

// GELU applies Gaussian Error Linear Unit.
// Used in GPT-2 and modern transformers - smoother than ReLU.
// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
func GELU(x *Tensor) *Tensor {
	out := NewTensor(x.Shape...)
	for i := range x.Data {
		v := x.Data[i]
		// Approximation used in GPT-2
		out.Data[i] = 0.5 * v * (1.0 + math.Tanh(math.Sqrt(2.0/math.Pi)*(v+0.044715*v*v*v)))
	}
	return out
}

// Softmax applies softmax function along the last dimension.
// Converts logits to probabilities: p_i = exp(x_i) / sum(exp(x_j))
//
// Numerically stable version: subtract max before exp to avoid overflow.
func Softmax(x *Tensor) *Tensor {
	out := NewTensor(x.Shape...)

	// For simplicity, only handle 2D tensors (batch, features)
	if len(x.Shape) != 2 {
		panic("Softmax currently only supports 2D tensors")
	}

	batch, features := x.Shape[0], x.Shape[1]

	for b := 0; b < batch; b++ {
		// Find max for numerical stability
		maxVal := x.At(b, 0)
		for f := 1; f < features; f++ {
			if v := x.At(b, f); v > maxVal {
				maxVal = v
			}
		}

		// Compute exp and sum
		sum := 0.0
		for f := 0; f < features; f++ {
			expVal := math.Exp(x.At(b, f) - maxVal)
			out.Set(expVal, b, f)
			sum += expVal
		}

		// Normalize
		for f := 0; f < features; f++ {
			out.Set(out.At(b, f)/sum, b, f)
		}
	}

	return out
}

// ===========================================================================
// HELPER FUNCTIONS
// ===========================================================================

// shapeEqual checks if two shapes are equal.
func shapeEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// PrintTensor prints the tensor data (useful for debugging).
// Only prints first few elements for large tensors.
func PrintTensor(t *Tensor) {
	fmt.Printf("%s\n", t)
	maxPrint := 10
	if len(t.Data) <= maxPrint {
		fmt.Printf("Data: %v\n", t.Data)
	} else {
		fmt.Printf("Data: %v ... %v\n", t.Data[:maxPrint/2], t.Data[len(t.Data)-maxPrint/2:])
	}
}