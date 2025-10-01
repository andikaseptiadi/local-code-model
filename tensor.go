package main

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
)

// RECOMMENDED READING:
//
// Deep Learning Foundations:
// - "Deep Learning" by Goodfellow, Bengio, Courville (2016)
//   Chapter 2: Linear Algebra - tensor operations
//   Chapter 6: Deep Feedforward Networks - backpropagation
//
// Numerical Computing:
// - "Numerical Linear Algebra" by Trefethen & Bau (1997)
//   Explains stability, conditioning of matrix operations
//
// Implementation:
// - "Programming Massively Parallel Processors" by Hwu, Kirk, Hajj (2022)
//   GPU optimization techniques for tensor operations

var (
	// ErrShapeMismatch indicates incompatible tensor shapes for an operation.
	ErrShapeMismatch = errors.New("tensor: shape mismatch")

	// ErrInvalidShape indicates an invalid tensor shape.
	ErrInvalidShape = errors.New("tensor: invalid shape")

	// ErrInvalidIndex indicates an out-of-bounds index access.
	ErrInvalidIndex = errors.New("tensor: invalid index")
)

// Tensor represents a multi-dimensional array of float64 values.
// It stores data in row-major (C-contiguous) order.
//
// Tensor is not safe for concurrent use. Synchronization must be
// handled by the caller if needed.
type Tensor struct {
	data  []float64 // Flat array storing all elements
	shape []int     // Dimensions [batch, seq_len, features, etc.]
	grad  []float64 // Gradient for backpropagation
}

// NewTensor creates a tensor with the given shape, initialized to zero.
// Panics if shape is invalid (empty or contains non-positive dimensions).
//
// This is idiomatic Go for ML code - shape errors are programmer bugs,
// not runtime conditions that should be handled gracefully.
func NewTensor(shape ...int) *Tensor {
	if len(shape) == 0 {
		panic("tensor: shape cannot be empty")
	}

	size := 1
	for i, dim := range shape {
		if dim <= 0 {
			panic(fmt.Sprintf("tensor: shape[%d] must be positive, got %d", i, dim))
		}
		size *= dim
	}

	// Copy shape slice to prevent external mutation
	shapeCopy := make([]int, len(shape))
	copy(shapeCopy, shape)

	return &Tensor{
		data:  make([]float64, size),
		shape: shapeCopy,
		grad:  make([]float64, size),
	}
}

// NewTensorRand creates a tensor with values from a normal distribution.
// Uses Box-Muller transform for sampling. Scale parameter controls
// the standard deviation (default 0.02 for small random initialization).
func NewTensorRand(shape ...int) *Tensor {
	t := NewTensor(shape...)

	// Box-Muller transform for normal distribution
	// Generate pairs of independent standard normal variables
	for i := 0; i < len(t.data); i += 2 {
		u1, u2 := rand.Float64(), rand.Float64()
		mag := 0.02 * math.Sqrt(-2*math.Log(u1))
		z0 := mag * math.Cos(2*math.Pi*u2)

		t.data[i] = z0
		if i+1 < len(t.data) {
			z1 := mag * math.Sin(2*math.Pi*u2)
			t.data[i+1] = z1
		}
	}

	return t
}

// Shape returns a copy of the tensor's shape.
// The returned slice can be safely modified without affecting the tensor.
func (t *Tensor) Shape() []int {
	shape := make([]int, len(t.shape))
	copy(shape, t.shape)
	return shape
}

// Dims returns the number of dimensions (rank) of the tensor.
func (t *Tensor) Dims() int {
	return len(t.shape)
}

// Size returns the total number of elements in the tensor.
func (t *Tensor) Size() int {
	return len(t.data)
}

// At returns the element at the given indices.
// Panics if indices are invalid - this is a programmer error.
func (t *Tensor) At(indices ...int) float64 {
	idx := t.flatIndex(indices)
	return t.data[idx]
}

// Set sets the element at the given indices.
// Panics if indices are invalid.
func (t *Tensor) Set(value float64, indices ...int) {
	idx := t.flatIndex(indices)
	t.data[idx] = value
}

// flatIndex converts multi-dimensional indices to a flat index.
// Panics on invalid indices.
func (t *Tensor) flatIndex(indices []int) int {
	if len(indices) != len(t.shape) {
		panic(fmt.Sprintf("tensor: expected %d indices, got %d", len(t.shape), len(indices)))
	}

	idx := 0
	stride := 1

	// Compute flat index in row-major order
	for i := len(indices) - 1; i >= 0; i-- {
		if indices[i] < 0 || indices[i] >= t.shape[i] {
			panic(fmt.Sprintf("tensor: index[%d]=%d out of bounds [0,%d)", i, indices[i], t.shape[i]))
		}
		idx += indices[i] * stride
		stride *= t.shape[i]
	}

	return idx
}

// ZeroGrad clears the gradient tensor. Call before backward pass.
func (t *Tensor) ZeroGrad() {
	for i := range t.grad {
		t.grad[i] = 0
	}
}

// Clone creates a deep copy of the tensor.
func (t *Tensor) Clone() *Tensor {
	clone := NewTensor(t.shape...)
	copy(clone.data, t.data)
	copy(clone.grad, t.grad)
	return clone
}

// Reshape returns a new view of the tensor with a different shape.
// The total number of elements must remain the same.
// The returned tensor shares the underlying data.
func (t *Tensor) Reshape(newShape ...int) *Tensor {
	newSize := 1
	for _, dim := range newShape {
		newSize *= dim
	}

	if newSize != len(t.data) {
		panic(fmt.Sprintf("tensor: cannot reshape size %d to %v (size %d)", len(t.data), newShape, newSize))
	}

	shapeCopy := make([]int, len(newShape))
	copy(shapeCopy, newShape)

	return &Tensor{
		data:  t.data, // Share underlying data
		shape: shapeCopy,
		grad:  t.grad, // Share gradient too
	}
}

// String returns a string representation of the tensor for debugging.
func (t *Tensor) String() string {
	return fmt.Sprintf("Tensor(shape=%v, size=%d)", t.shape, len(t.data))
}

// ===========================================================================
// OPERATIONS
// ===========================================================================

// Add performs element-wise addition: out = a + b.
// Panics if shapes don't match.
func Add(a, b *Tensor) *Tensor {
	if !shapeEqual(a.shape, b.shape) {
		panic(fmt.Sprintf("tensor: cannot add shapes %v and %v", a.shape, b.shape))
	}

	out := NewTensor(a.shape...)
	for i := range out.data {
		out.data[i] = a.data[i] + b.data[i]
	}

	return out
}

// Mul performs element-wise multiplication: out = a * b (Hadamard product).
// Panics if shapes don't match.
func Mul(a, b *Tensor) *Tensor {
	if !shapeEqual(a.shape, b.shape) {
		panic(fmt.Sprintf("tensor: cannot multiply shapes %v and %v", a.shape, b.shape))
	}

	out := NewTensor(a.shape...)
	for i := range out.data {
		out.data[i] = a.data[i] * b.data[i]
	}

	return out
}

// Scale multiplies all elements by a scalar: out = a * scalar.
func Scale(a *Tensor, scalar float64) *Tensor {
	out := NewTensor(a.shape...)
	for i := range out.data {
		out.data[i] = a.data[i] * scalar
	}
	return out
}

// MatMul performs matrix multiplication: C = A @ B.
// A must be (M, K), B must be (K, N), result is (M, N).
//
// This is the O(M*N*K) operation at the heart of neural networks.
// Uses global compute configuration to determine parallel execution.
func MatMul(a, b *Tensor) *Tensor {
	return MatMulWithConfig(a, b, globalComputeConfig)
}

// Transpose returns the transpose of a 2D matrix: A^T.
// A: (M, N) -> A^T: (N, M).
func Transpose(a *Tensor) *Tensor {
	if len(a.shape) != 2 {
		panic("tensor: Transpose requires 2D tensor")
	}

	m, n := a.shape[0], a.shape[1]
	out := NewTensor(n, m)

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			out.Set(a.At(i, j), j, i)
		}
	}

	return out
}

// ===========================================================================
// ACTIVATION FUNCTIONS
// ===========================================================================

// ReLU applies Rectified Linear Unit: f(x) = max(0, x).
// The most widely used activation - simple and effective.
func ReLU(x *Tensor) *Tensor {
	out := NewTensor(x.shape...)
	for i := range x.data {
		out.data[i] = math.Max(0, x.data[i])
	}
	return out
}

// GELU applies Gaussian Error Linear Unit.
// Used in transformers (GPT, BERT). Smoother than ReLU.
//
// GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
func GELU(x *Tensor) *Tensor {
	out := NewTensor(x.shape...)

	const (
		sqrt2OverPi = 0.7978845608028654  // sqrt(2/π)
		coeff       = 0.044715
	)

	for i := range x.data {
		v := x.data[i]
		inner := sqrt2OverPi * (v + coeff*v*v*v)
		out.data[i] = 0.5 * v * (1.0 + math.Tanh(inner))
	}

	return out
}

// Softmax applies softmax function: p_i = exp(x_i) / Σ exp(x_j).
// Converts logits to probabilities (sum to 1).
//
// Numerically stable version: subtract max before exp to prevent overflow.
// Currently only supports 2D tensors (batch, features).
func Softmax(x *Tensor) *Tensor {
	if len(x.shape) != 2 {
		panic("tensor: Softmax currently requires 2D tensor")
	}

	batch, features := x.shape[0], x.shape[1]
	out := NewTensor(batch, features)

	// Process each row independently
	for b := 0; b < batch; b++ {
		// Find max for numerical stability
		maxVal := x.At(b, 0)
		for f := 1; f < features; f++ {
			if v := x.At(b, f); v > maxVal {
				maxVal = v
			}
		}

		// Compute exp(x - max) and sum
		sum := 0.0
		for f := 0; f < features; f++ {
			expVal := math.Exp(x.At(b, f) - maxVal)
			out.Set(expVal, b, f)
			sum += expVal
		}

		// Normalize to get probabilities
		for f := 0; f < features; f++ {
			out.Set(out.At(b, f)/sum, b, f)
		}
	}

	return out
}

// ===========================================================================
// HELPERS
// ===========================================================================

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