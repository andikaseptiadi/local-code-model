package main

import (
	"math"
	"testing"
)

// TestTensorBasics tests basic tensor creation and access.
func TestTensorBasics(t *testing.T) {
	// Create a 2x3 matrix
	tensor := NewTensor(2, 3)

	// Verify shape
	if len(tensor.Shape) != 2 || tensor.Shape[0] != 2 || tensor.Shape[1] != 3 {
		t.Errorf("expected shape [2 3], got %v", tensor.Shape)
	}

	// Verify size
	if tensor.Size() != 6 {
		t.Errorf("expected size 6, got %d", tensor.Size())
	}

	// Test setting and getting values
	tensor.Set(1.5, 0, 0)
	tensor.Set(2.5, 1, 2)

	if v := tensor.At(0, 0); v != 1.5 {
		t.Errorf("expected 1.5, got %f", v)
	}

	if v := tensor.At(1, 2); v != 2.5 {
		t.Errorf("expected 2.5, got %f", v)
	}
}

// TestMatMul tests matrix multiplication.
func TestMatMul(t *testing.T) {
	// Create two matrices: A (2x3) and B (3x2)
	a := NewTensor(2, 3)
	a.Set(1, 0, 0)
	a.Set(2, 0, 1)
	a.Set(3, 0, 2)
	a.Set(4, 1, 0)
	a.Set(5, 1, 1)
	a.Set(6, 1, 2)

	b := NewTensor(3, 2)
	b.Set(1, 0, 0)
	b.Set(2, 0, 1)
	b.Set(3, 1, 0)
	b.Set(4, 1, 1)
	b.Set(5, 2, 0)
	b.Set(6, 2, 1)

	// C = A @ B should be (2x2)
	c := MatMul(a, b)

	// Verify shape
	if len(c.Shape) != 2 || c.Shape[0] != 2 || c.Shape[1] != 2 {
		t.Errorf("expected shape [2 2], got %v", c.Shape)
	}

	// Verify values
	// C[0,0] = 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
	// C[0,1] = 1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28
	// C[1,0] = 4*1 + 5*3 + 6*5 = 4 + 15 + 30 = 49
	// C[1,1] = 4*2 + 5*4 + 6*6 = 8 + 20 + 36 = 64

	expected := [][]float64{
		{22, 28},
		{49, 64},
	}

	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if v := c.At(i, j); v != expected[i][j] {
				t.Errorf("C[%d,%d]: expected %f, got %f", i, j, expected[i][j], v)
			}
		}
	}
}

// TestTranspose tests matrix transpose.
func TestTranspose(t *testing.T) {
	a := NewTensor(2, 3)
	a.Set(1, 0, 0)
	a.Set(2, 0, 1)
	a.Set(3, 0, 2)
	a.Set(4, 1, 0)
	a.Set(5, 1, 1)
	a.Set(6, 1, 2)

	aT := Transpose(a)

	// Verify shape
	if len(aT.Shape) != 2 || aT.Shape[0] != 3 || aT.Shape[1] != 2 {
		t.Errorf("expected shape [3 2], got %v", aT.Shape)
	}

	// Verify values
	if v := aT.At(0, 0); v != 1 {
		t.Errorf("expected 1, got %f", v)
	}
	if v := aT.At(1, 0); v != 2 {
		t.Errorf("expected 2, got %f", v)
	}
	if v := aT.At(2, 1); v != 6 {
		t.Errorf("expected 6, got %f", v)
	}
}

// TestSoftmax tests the softmax function.
func TestSoftmax(t *testing.T) {
	// Create a simple 1x3 tensor
	x := NewTensor(1, 3)
	x.Set(1.0, 0, 0)
	x.Set(2.0, 0, 1)
	x.Set(3.0, 0, 2)

	out := Softmax(x)

	// Verify probabilities sum to 1
	sum := 0.0
	for i := 0; i < 3; i++ {
		sum += out.At(0, i)
	}

	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("softmax should sum to 1, got %f", sum)
	}

	// Verify largest input has largest probability
	if out.At(0, 2) <= out.At(0, 1) || out.At(0, 2) <= out.At(0, 0) {
		t.Errorf("softmax should give highest probability to largest input")
	}
}

// TestReLU tests the ReLU activation.
func TestReLU(t *testing.T) {
	x := NewTensor(1, 4)
	x.Set(-2.0, 0, 0)
	x.Set(-1.0, 0, 1)
	x.Set(1.0, 0, 2)
	x.Set(2.0, 0, 3)

	out := ReLU(x)

	// Verify negative values become 0
	if v := out.At(0, 0); v != 0 {
		t.Errorf("ReLU(-2) should be 0, got %f", v)
	}
	if v := out.At(0, 1); v != 0 {
		t.Errorf("ReLU(-1) should be 0, got %f", v)
	}

	// Verify positive values unchanged
	if v := out.At(0, 2); v != 1.0 {
		t.Errorf("ReLU(1) should be 1, got %f", v)
	}
	if v := out.At(0, 3); v != 2.0 {
		t.Errorf("ReLU(2) should be 2, got %f", v)
	}
}