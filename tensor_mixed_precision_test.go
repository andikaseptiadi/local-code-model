package main

import (
	"fmt"
	"math"
	"testing"
)

// ===========================================================================
// MIXED PRECISION TRAINING TESTS
// ===========================================================================
//
// This test file verifies the correctness of mixed precision training
// components: float16 conversion, loss scaling, gradient unscaling, and
// the complete training loop integration.
//
// Test coverage:
// 1. Float32 ↔ Float16 conversion accuracy
// 2. Special value handling (infinity, NaN, zero, denormals)
// 3. Loss scaling and gradient unscaling
// 4. Parameter updates with master weights
// 5. Overflow detection
// 6. End-to-end training with mixed precision
//
// ===========================================================================

// TestFloat16Conversion tests basic float32 to float16 conversion.
func TestFloat16Conversion(t *testing.T) {
	testCases := []struct {
		name     string
		input    float32
		expected float32 // Expected after round-trip conversion
		tolerance float32
	}{
		{"Zero", 0.0, 0.0, 0.0},
		{"One", 1.0, 1.0, 0.0},
		{"MinusOne", -1.0, -1.0, 0.0},
		{"Small", 0.0001, 0.0001, 0.00001},
		{"Large", 1000.0, 1000.0, 0.1},
		{"MaxFloat16", 65504.0, 65504.0, 1.0},
		{"Pi", 3.14159, 3.14159, 0.001},
		{"E", 2.71828, 2.71828, 0.002}, // Float16 has limited precision (~3-4 digits)
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Convert to float16 and back
			f16 := Float32ToFloat16(tc.input)
			result := Float16ToFloat32(f16)

			// Check accuracy
			diff := math.Abs(float64(result - tc.expected))
			if diff > float64(tc.tolerance) {
				t.Errorf("%s: expected %v, got %v (diff: %v, tolerance: %v)",
					tc.name, tc.expected, result, diff, tc.tolerance)
			}
		})
	}
}

// TestFloat16SpecialValues tests handling of special float values.
func TestFloat16SpecialValues(t *testing.T) {
	testCases := []struct {
		name  string
		input float32
		check func(float32) bool
	}{
		{
			"PositiveInfinity",
			float32(math.Inf(1)),
			func(f float32) bool { return math.IsInf(float64(f), 1) },
		},
		{
			"NegativeInfinity",
			float32(math.Inf(-1)),
			func(f float32) bool { return math.IsInf(float64(f), -1) },
		},
		{
			"NaN",
			float32(math.NaN()),
			func(f float32) bool { return math.IsNaN(float64(f)) },
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			f16 := Float32ToFloat16(tc.input)
			result := Float16ToFloat32(f16)

			if !tc.check(result) {
				t.Errorf("%s: check failed for input %v -> %v", tc.name, tc.input, result)
			}
		})
	}
}

// TestFloat16Overflow tests overflow handling (clamping to max value).
func TestFloat16Overflow(t *testing.T) {
	// Values larger than max float16 should clamp to infinity
	testCases := []float32{70000.0, 100000.0, 1e10}

	for _, input := range testCases {
		f16 := Float32ToFloat16(input)
		result := Float16ToFloat32(f16)

		if !math.IsInf(float64(result), 1) {
			t.Errorf("Overflow test failed for %v: expected infinity, got %v", input, result)
		}
	}
}

// TestFloat16Underflow tests underflow handling (flush to zero).
func TestFloat16Underflow(t *testing.T) {
	// Values smaller than min normal float16 should flush to zero
	testCases := []float32{1e-6, 1e-8, 1e-10}

	for _, input := range testCases {
		f16 := Float32ToFloat16(input)
		result := Float16ToFloat32(f16)

		if result != 0.0 {
			t.Errorf("Underflow test failed for %v: expected 0, got %v", input, result)
		}
	}
}

// TestFloat16Precision tests that float16 maintains expected precision.
func TestFloat16Precision(t *testing.T) {
	// Float16 has ~3-4 decimal digits of precision
	// Test that values are accurate to within 0.1%
	testValues := []float32{
		1.0, 2.0, 3.14159, 10.0, 100.0, 1000.0, 10000.0,
	}

	for _, input := range testValues {
		f16 := Float32ToFloat16(input)
		result := Float16ToFloat32(f16)

		relativeError := math.Abs(float64(result-input)) / float64(input)
		if relativeError > 0.001 { // 0.1% tolerance
			t.Errorf("Precision test failed for %v: got %v (relative error: %.4f%%)",
				input, result, relativeError*100)
		}
	}
}

// TestTensorFloat16Conversion tests conversion between Tensor and TensorFloat16.
func TestTensorFloat16Conversion(t *testing.T) {
	// Create a test tensor
	shape := []int{2, 3}
	original := NewTensor(shape...)
	original.data = []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}

	// Convert to float16 and back
	f16 := NewTensorFloat16(shape...)
	f16.FromTensor(original)
	result := f16.ToTensor()

	// Check shape
	if len(result.shape) != len(original.shape) {
		t.Errorf("Shape mismatch: expected %v, got %v", original.shape, result.shape)
	}
	for i := range result.shape {
		if result.shape[i] != original.shape[i] {
			t.Errorf("Shape mismatch at dim %d: expected %d, got %d",
				i, original.shape[i], result.shape[i])
		}
	}

	// Check values (with tolerance for float16 precision)
	for i := range original.data {
		diff := math.Abs(result.data[i] - original.data[i])
		if diff > 0.01 {
			t.Errorf("Value mismatch at index %d: expected %v, got %v (diff: %v)",
				i, original.data[i], result.data[i], diff)
		}
	}
}

// TestMixedPrecisionConfig tests basic config operations.
func TestMixedPrecisionConfig(t *testing.T) {
	cfg := NewMixedPrecisionConfig()

	// Check defaults
	if !cfg.Enabled {
		t.Error("Expected Enabled to be true by default")
	}
	if cfg.LossScale != 1024.0 {
		t.Errorf("Expected LossScale to be 1024.0, got %v", cfg.LossScale)
	}
	if cfg.MasterWeights == nil {
		t.Error("Expected MasterWeights map to be initialized")
	}
	if cfg.Float16Weights == nil {
		t.Error("Expected Float16Weights map to be initialized")
	}
}

// TestLossScaling tests loss scaling and gradient unscaling.
func TestLossScaling(t *testing.T) {
	cfg := NewMixedPrecisionConfig()
	cfg.LossScale = 1024.0

	// Test loss scaling
	originalLoss := 0.5
	scaledLoss := cfg.ScaleLoss(originalLoss)
	expectedScaled := originalLoss * 1024.0

	if scaledLoss != expectedScaled {
		t.Errorf("Loss scaling failed: expected %v, got %v", expectedScaled, scaledLoss)
	}

	// Test gradient unscaling
	grad := NewTensor(2, 2)
	grad.data = []float64{1024.0, 2048.0, 3072.0, 4096.0}
	gradients := map[string]*Tensor{"grad": grad}

	cfg.UnscaleGradients(gradients)

	expectedUnscaled := []float64{1.0, 2.0, 3.0, 4.0}
	for i, expected := range expectedUnscaled {
		if math.Abs(grad.data[i]-expected) > 1e-9 {
			t.Errorf("Gradient unscaling failed at index %d: expected %v, got %v",
				i, expected, grad.data[i])
		}
	}
}

// TestRegisterParameter tests parameter registration.
func TestRegisterParameter(t *testing.T) {
	cfg := NewMixedPrecisionConfig()

	// Create a test parameter
	param := NewTensor(2, 2)
	param.data = []float64{1.0, 2.0, 3.0, 4.0}

	// Register parameter
	cfg.RegisterParameter("test_param", param)

	// Check that master weights were created
	master, ok := cfg.MasterWeights["test_param"]
	if !ok {
		t.Fatal("Master weights not created")
	}

	// Check that master weights match original
	for i := range param.data {
		if master.data[i] != param.data[i] {
			t.Errorf("Master weights mismatch at index %d: expected %v, got %v",
				i, param.data[i], master.data[i])
		}
	}

	// Check that float16 weights were created
	_, ok = cfg.Float16Weights["test_param"]
	if !ok {
		t.Fatal("Float16 weights not created")
	}
}

// TestGetFloat16Weight tests retrieving float16 weights.
func TestGetFloat16Weight(t *testing.T) {
	cfg := NewMixedPrecisionConfig()

	// Create and register a parameter
	param := NewTensor(2, 2)
	param.data = []float64{1.0, 2.0, 3.0, 4.0}
	cfg.RegisterParameter("test_param", param)

	// Retrieve float16 version
	f16Tensor := cfg.GetFloat16Weight("test_param")

	// Check that values are approximately correct (within float16 precision)
	for i := range param.data {
		diff := math.Abs(f16Tensor.data[i] - param.data[i])
		if diff > 0.01 {
			t.Errorf("Float16 weight mismatch at index %d: expected %v, got %v (diff: %v)",
				i, param.data[i], f16Tensor.data[i], diff)
		}
	}
}

// TestUpdateParameters tests parameter updates with mixed precision.
func TestUpdateParameters(t *testing.T) {
	cfg := NewMixedPrecisionConfig()
	cfg.LossScale = 1.0 // Disable loss scaling for this test

	// Create and register a parameter
	param := NewTensor(2, 2)
	param.data = []float64{1.0, 2.0, 3.0, 4.0}
	cfg.RegisterParameter("test_param", param)

	// Create gradients
	grad := NewTensor(2, 2)
	grad.data = []float64{0.1, 0.1, 0.1, 0.1}
	gradients := map[string]*Tensor{"test_param": grad}

	// Update parameters with learning rate 1.0
	cfg.UpdateParameters(gradients, 1.0)

	// Check that master weights were updated
	master := cfg.MasterWeights["test_param"]
	expectedValues := []float64{0.9, 1.9, 2.9, 3.9}

	for i, expected := range expectedValues {
		if math.Abs(master.data[i]-expected) > 1e-9 {
			t.Errorf("Parameter update failed at index %d: expected %v, got %v",
				i, expected, master.data[i])
		}
	}

	// Check that float16 weights were updated
	f16Tensor := cfg.GetFloat16Weight("test_param")
	for i, expected := range expectedValues {
		diff := math.Abs(f16Tensor.data[i] - expected)
		if diff > 0.01 {
			t.Errorf("Float16 weight update failed at index %d: expected %v, got %v (diff: %v)",
				i, expected, f16Tensor.data[i], diff)
		}
	}
}

// TestCheckOverflow tests overflow detection in gradients.
func TestCheckOverflow(t *testing.T) {
	cfg := NewMixedPrecisionConfig()

	testCases := []struct {
		name     string
		values   []float64
		overflow bool
	}{
		{
			"NoOverflow",
			[]float64{1.0, 2.0, 3.0, 4.0},
			false,
		},
		{
			"InfinityOverflow",
			[]float64{1.0, math.Inf(1), 3.0, 4.0},
			true,
		},
		{
			"NegativeInfinityOverflow",
			[]float64{1.0, 2.0, math.Inf(-1), 4.0},
			true,
		},
		{
			"NaNOverflow",
			[]float64{1.0, 2.0, 3.0, math.NaN()},
			true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			grad := NewTensor(2, 2)
			grad.data = tc.values
			gradients := map[string]*Tensor{"grad": grad}

			overflow := cfg.CheckOverflow(gradients)
			if overflow != tc.overflow {
				t.Errorf("%s: expected overflow=%v, got overflow=%v", tc.name, tc.overflow, overflow)
			}
		})
	}
}

// TestMixedPrecisionDisabled tests that operations work correctly when disabled.
func TestMixedPrecisionDisabled(t *testing.T) {
	cfg := NewMixedPrecisionConfig()
	cfg.Enabled = false

	// Loss scaling should be a no-op
	loss := 0.5
	scaledLoss := cfg.ScaleLoss(loss)
	if scaledLoss != loss {
		t.Errorf("Loss scaling should be no-op when disabled: expected %v, got %v", loss, scaledLoss)
	}

	// Gradient unscaling should be a no-op
	grad := NewTensor(2, 2)
	originalData := []float64{1.0, 2.0, 3.0, 4.0}
	grad.data = append([]float64(nil), originalData...)
	gradients := map[string]*Tensor{"grad": grad}

	cfg.UnscaleGradients(gradients)

	for i := range originalData {
		if grad.data[i] != originalData[i] {
			t.Errorf("Gradient unscaling should be no-op when disabled at index %d: expected %v, got %v",
				i, originalData[i], grad.data[i])
		}
	}

	// Register parameter should be a no-op (no float16 weights created)
	param := NewTensor(2, 2)
	param.data = []float64{1.0, 2.0, 3.0, 4.0}
	cfg.RegisterParameter("test_param", param)

	if len(cfg.Float16Weights) != 0 {
		t.Error("Float16 weights should not be created when disabled")
	}
}

// TestFloat16MemoryFootprint tests that float16 uses half the memory.
func TestFloat16MemoryFootprint(t *testing.T) {
	shape := []int{1000, 1000}

	// Create float32 tensor
	f32 := NewTensor(shape...)
	f32Size := len(f32.data) * 8 // 8 bytes per float64

	// Create float16 tensor
	f16 := NewTensorFloat16(shape...)
	f16Size := len(f16.data) * 2 // 2 bytes per Float16

	// Check that float16 uses approximately half the memory
	expectedRatio := 4.0 // float64 is 8 bytes, Float16 is 2 bytes (stored as uint16)
	actualRatio := float64(f32Size) / float64(f16Size)

	if math.Abs(actualRatio-expectedRatio) > 0.1 {
		t.Errorf("Memory footprint ratio incorrect: expected %.1f, got %.1f",
			expectedRatio, actualRatio)
	}

	t.Logf("Float32 tensor: %d bytes", f32Size)
	t.Logf("Float16 tensor: %d bytes", f16Size)
	t.Logf("Memory reduction: %.1f%%", (1.0-float64(f16Size)/float64(f32Size))*100)
}

// TestEndToEndMixedPrecision tests a complete training iteration with mixed precision.
func TestEndToEndMixedPrecision(t *testing.T) {
	cfg := NewMixedPrecisionConfig()
	cfg.LossScale = 1024.0

	// Create and register model parameters
	weights := NewTensor(2, 2)
	weights.data = []float64{1.0, 2.0, 3.0, 4.0}
	cfg.RegisterParameter("weights", weights)

	// Simulate forward pass with float16 weights
	f16Weights := cfg.GetFloat16Weight("weights")
	_ = f16Weights // In real code, use this for forward pass

	// Simulate loss computation and scaling
	loss := 0.5
	scaledLoss := cfg.ScaleLoss(loss)

	// Verify loss is scaled
	if scaledLoss != loss*cfg.LossScale {
		t.Errorf("Loss scaling failed: expected %v, got %v", loss*cfg.LossScale, scaledLoss)
	}

	// Simulate backward pass (gradients in float32)
	grad := NewTensor(2, 2)
	grad.data = []float64{102.4, 204.8, 307.2, 409.6} // Scaled gradients
	gradients := map[string]*Tensor{"weights": grad}

	// Unscale gradients
	cfg.UnscaleGradients(gradients)

	// Verify gradients are unscaled
	expectedGrads := []float64{0.1, 0.2, 0.3, 0.4}
	for i, expected := range expectedGrads {
		if math.Abs(grad.data[i]-expected) > 1e-9 {
			t.Errorf("Gradient unscaling failed at index %d: expected %v, got %v",
				i, expected, grad.data[i])
		}
	}

	// Check for overflow
	if cfg.CheckOverflow(gradients) {
		t.Error("Unexpected overflow detected")
	}

	// Update parameters
	learningRate := 1.0
	cfg.UpdateParameters(gradients, learningRate)

	// Verify master weights are updated correctly
	master := cfg.MasterWeights["weights"]
	expectedWeights := []float64{0.9, 1.8, 2.7, 3.6}
	for i, expected := range expectedWeights {
		if math.Abs(master.data[i]-expected) > 1e-9 {
			t.Errorf("Weight update failed at index %d: expected %v, got %v",
				i, expected, master.data[i])
		}
	}

	// Verify float16 weights are synced
	f16WeightsUpdated := cfg.GetFloat16Weight("weights")
	for i, expected := range expectedWeights {
		diff := math.Abs(f16WeightsUpdated.data[i] - expected)
		if diff > 0.01 { // Float16 precision tolerance
			t.Errorf("Float16 weight sync failed at index %d: expected %v, got %v (diff: %v)",
				i, expected, f16WeightsUpdated.data[i], diff)
		}
	}
}

// BenchmarkFloat32ToFloat16 benchmarks float16 conversion.
func BenchmarkFloat32ToFloat16(b *testing.B) {
	input := float32(3.14159)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = Float32ToFloat16(input)
	}
}

// BenchmarkFloat16ToFloat32 benchmarks float16 conversion.
func BenchmarkFloat16ToFloat32(b *testing.B) {
	f16 := Float32ToFloat16(3.14159)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = Float16ToFloat32(f16)
	}
}

// BenchmarkTensorConversionToFloat16 benchmarks tensor conversion to float16.
func BenchmarkTensorConversionToFloat16(b *testing.B) {
	sizes := []int{64, 256, 1024}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size%d", size), func(b *testing.B) {
			src := NewTensor(size, size)
			for i := range src.data {
				src.data[i] = float64(i)
			}
			dst := NewTensorFloat16(size, size)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				dst.FromTensor(src)
			}
		})
	}
}

// BenchmarkTensorConversionToFloat32 benchmarks tensor conversion from float16.
func BenchmarkTensorConversionToFloat32(b *testing.B) {
	sizes := []int{64, 256, 1024}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size%d", size), func(b *testing.B) {
			src := NewTensorFloat16(size, size)
			for i := range src.data {
				src.data[i] = Float32ToFloat16(float32(i))
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = src.ToTensor()
			}
		})
	}
}
