package main

import (
	"fmt"
	"testing"
)

// ===========================================================================
// GRADIENT CHECKPOINTING TESTS
// ===========================================================================
//
// This test file verifies gradient checkpointing functionality, memory savings,
// and correctness of recomputation during backward passes.
//
// Test coverage:
// 1. Basic checkpoint segment creation and execution
// 2. Forward pass with and without checkpointing
// 3. Recomputation during backward pass
// 4. Memory savings calculations
// 5. Checkpoint configuration and policy
// 6. End-to-end training with checkpointing
//
// ===========================================================================

// TestCheckpointSegmentCreation tests basic checkpoint segment creation.
func TestCheckpointSegmentCreation(t *testing.T) {
	// Create a simple forward function
	forward := func(inputs ...*Tensor) []*Tensor {
		// Simple addition: out = in + 1
		out := NewTensor(inputs[0].shape...)
		for i := range out.data {
			out.data[i] = inputs[0].data[i] + 1.0
		}
		return []*Tensor{out}
	}

	segment := NewCheckpointSegment(forward)

	if segment == nil {
		t.Fatal("Failed to create checkpoint segment")
	}
	if segment.Forward == nil {
		t.Error("Forward function not set")
	}
	if segment.Recomputed {
		t.Error("Recomputed should be false initially")
	}
}

// TestCheckpointRunForward tests forward pass execution with checkpointing.
func TestCheckpointRunForward(t *testing.T) {
	// Create a forward function that doubles the input
	forward := func(inputs ...*Tensor) []*Tensor {
		out := NewTensor(inputs[0].shape...)
		for i := range out.data {
			out.data[i] = inputs[0].data[i] * 2.0
		}
		return []*Tensor{out}
	}

	segment := NewCheckpointSegment(forward)

	// Create input
	input := NewTensor(2, 2)
	input.data = []float64{1.0, 2.0, 3.0, 4.0}

	// Run forward pass
	outputs := segment.RunForward(input)

	// Check that inputs were saved
	if len(segment.Inputs) != 1 {
		t.Errorf("Expected 1 input saved, got %d", len(segment.Inputs))
	}

	// Check that outputs are correct
	expected := []float64{2.0, 4.0, 6.0, 8.0}
	for i, exp := range expected {
		if outputs[0].data[i] != exp {
			t.Errorf("Output mismatch at index %d: expected %v, got %v",
				i, exp, outputs[0].data[i])
		}
	}

	// Check that outputs were NOT saved (memory optimization!)
	if segment.Outputs != nil {
		t.Error("Outputs should not be saved in checkpoint segment")
	}
}

// TestCheckpointRecompute tests recomputation during backward pass.
func TestCheckpointRecompute(t *testing.T) {
	callCount := 0

	// Create a forward function that tracks how many times it's called
	forward := func(inputs ...*Tensor) []*Tensor {
		callCount++
		out := NewTensor(inputs[0].shape...)
		for i := range out.data {
			out.data[i] = inputs[0].data[i] * 3.0
		}
		return []*Tensor{out}
	}

	segment := NewCheckpointSegment(forward)

	// Create input
	input := NewTensor(2, 2)
	input.data = []float64{1.0, 2.0, 3.0, 4.0}

	// Run forward pass (first call)
	_ = segment.RunForward(input)
	if callCount != 1 {
		t.Errorf("Expected 1 forward call, got %d", callCount)
	}

	// Recompute forward (second call)
	outputs := segment.RecomputeForward()
	if callCount != 2 {
		t.Errorf("Expected 2 forward calls after recomputation, got %d", callCount)
	}

	// Check recomputed outputs
	expected := []float64{3.0, 6.0, 9.0, 12.0}
	for i, exp := range expected {
		if outputs[0].data[i] != exp {
			t.Errorf("Recomputed output mismatch at index %d: expected %v, got %v",
				i, exp, outputs[0].data[i])
		}
	}

	// Verify that recomputed flag is set
	if !segment.Recomputed {
		t.Error("Recomputed flag should be true after recomputation")
	}

	// Call recompute again - should use cached result
	_ = segment.RecomputeForward()
	if callCount != 2 {
		t.Errorf("Expected still 2 forward calls (cached), got %d", callCount)
	}
}

// TestCheckpointClearOutputs tests clearing recomputed outputs to free memory.
func TestCheckpointClearOutputs(t *testing.T) {
	forward := func(inputs ...*Tensor) []*Tensor {
		out := NewTensor(inputs[0].shape...)
		for i := range out.data {
			out.data[i] = inputs[0].data[i] + 10.0
		}
		return []*Tensor{out}
	}

	segment := NewCheckpointSegment(forward)

	// Run forward and recompute
	input := NewTensor(2, 2)
	input.data = []float64{1.0, 2.0, 3.0, 4.0}
	_ = segment.RunForward(input)
	_ = segment.RecomputeForward()

	// Check that outputs exist
	if segment.Outputs == nil {
		t.Fatal("Outputs should exist after recomputation")
	}

	// Clear outputs
	segment.ClearOutputs()

	// Verify outputs are cleared
	if segment.Outputs != nil {
		t.Error("Outputs should be nil after clearing")
	}
	if segment.Recomputed {
		t.Error("Recomputed flag should be false after clearing")
	}
}

// TestCheckpointConfigCreation tests checkpoint configuration creation.
func TestCheckpointConfigCreation(t *testing.T) {
	cfg := NewCheckpointConfig()

	if cfg == nil {
		t.Fatal("Failed to create checkpoint config")
	}

	// Check defaults
	if cfg.Enabled {
		t.Error("Checkpointing should be disabled by default")
	}
	if cfg.CheckpointEveryN != 2 {
		t.Errorf("Expected CheckpointEveryN=2, got %d", cfg.CheckpointEveryN)
	}
	if cfg.CurrentLayer != 0 {
		t.Errorf("Expected CurrentLayer=0, got %d", cfg.CurrentLayer)
	}
	if cfg.Segments == nil {
		t.Error("Segments slice should be initialized")
	}
}

// TestCheckpointShouldCheckpoint tests checkpoint policy.
func TestCheckpointShouldCheckpoint(t *testing.T) {
	cfg := NewCheckpointConfig()
	cfg.Enabled = true
	cfg.CheckpointEveryN = 2

	testCases := []struct {
		layer    int
		expected bool
	}{
		{0, true},  // Layer 0: checkpoint (0 % 2 == 0)
		{1, false}, // Layer 1: no checkpoint
		{2, true},  // Layer 2: checkpoint (2 % 2 == 0)
		{3, false}, // Layer 3: no checkpoint
		{4, true},  // Layer 4: checkpoint (4 % 2 == 0)
	}

	for _, tc := range testCases {
		cfg.CurrentLayer = tc.layer
		result := cfg.ShouldCheckpoint()
		if result != tc.expected {
			t.Errorf("Layer %d: expected checkpoint=%v, got %v",
				tc.layer, tc.expected, result)
		}
	}
}

// TestCheckpointDisabledPolicy tests that checkpointing is disabled when Enabled=false.
func TestCheckpointDisabledPolicy(t *testing.T) {
	cfg := NewCheckpointConfig()
	cfg.Enabled = false
	cfg.CheckpointEveryN = 1 // Would checkpoint every layer if enabled

	for layer := 0; layer < 10; layer++ {
		cfg.CurrentLayer = layer
		if cfg.ShouldCheckpoint() {
			t.Errorf("Checkpointing should be disabled but got checkpoint at layer %d", layer)
		}
	}
}

// TestCheckpointAddSegment tests adding segments and tracking layers.
func TestCheckpointAddSegment(t *testing.T) {
	cfg := NewCheckpointConfig()

	forward := func(inputs ...*Tensor) []*Tensor {
		return []*Tensor{inputs[0]}
	}

	// Add 3 segments
	for i := 0; i < 3; i++ {
		segment := NewCheckpointSegment(forward)
		cfg.AddSegment(segment)
	}

	if len(cfg.Segments) != 3 {
		t.Errorf("Expected 3 segments, got %d", len(cfg.Segments))
	}
	if cfg.CurrentLayer != 3 {
		t.Errorf("Expected CurrentLayer=3, got %d", cfg.CurrentLayer)
	}
}

// TestCheckpointReset tests resetting checkpoint state.
func TestCheckpointReset(t *testing.T) {
	cfg := NewCheckpointConfig()

	// Add some segments
	forward := func(inputs ...*Tensor) []*Tensor {
		return []*Tensor{inputs[0]}
	}
	for i := 0; i < 5; i++ {
		segment := NewCheckpointSegment(forward)
		cfg.AddSegment(segment)
	}

	// Reset
	cfg.Reset()

	if len(cfg.Segments) != 0 {
		t.Errorf("Expected 0 segments after reset, got %d", len(cfg.Segments))
	}
	if cfg.CurrentLayer != 0 {
		t.Errorf("Expected CurrentLayer=0 after reset, got %d", cfg.CurrentLayer)
	}
}

// TestMemorySavingsEstimate tests memory savings calculation.
func TestMemorySavingsEstimate(t *testing.T) {
	cfg := NewCheckpointConfig()
	cfg.Enabled = true
	cfg.CheckpointEveryN = 2

	// Test case: 24 layers, batch 8, seq 1024, hidden 768
	memWithout, memWith, ratio := cfg.EstimateMemorySavings(24, 8, 1024, 768)

	// Memory without: 24 layers × 8 × 1024 × 768 × 4 bytes / 1024² ≈ 576 MB
	expectedWithout := 24.0 * 8.0 * 1024.0 * 768.0 * 4.0 / (1024.0 * 1024.0)
	if abs(memWithout-expectedWithout) > 1.0 {
		t.Errorf("Memory without checkpointing: expected %.1f MB, got %.1f MB",
			expectedWithout, memWithout)
	}

	// Memory with: (24/2 + 1) layers = 13 layers worth
	expectedCheckpoints := (24 + 2 - 1) / 2 // Ceiling division
	expectedWith := float64(expectedCheckpoints+1) * expectedWithout / 24.0
	if abs(memWith-expectedWith) > 1.0 {
		t.Errorf("Memory with checkpointing: expected %.1f MB, got %.1f MB",
			expectedWith, memWith)
	}

	// Savings ratio should be ~2x for CheckpointEveryN=2
	expectedRatio := expectedWithout / expectedWith
	if abs(ratio-expectedRatio) > 0.1 {
		t.Errorf("Savings ratio: expected %.2fx, got %.2fx", expectedRatio, ratio)
	}

	t.Logf("Memory savings: %.1f MB → %.1f MB (%.2fx reduction)",
		memWithout, memWith, ratio)
}

// TestMemorySavingsDifferentConfigs tests memory savings with different checkpoint frequencies.
func TestMemorySavingsDifferentConfigs(t *testing.T) {
	testCases := []struct {
		checkpointEveryN int
		expectedRatio    float64 // Approximate expected ratio
	}{
		{1, 1.0}, // Checkpoint every layer: no savings
		{2, 1.8}, // Checkpoint every 2 layers: ~2x savings
		{4, 3.2}, // Checkpoint every 4 layers: ~3-4x savings
	}

	numLayers := 24
	batchSize := 8
	seqLen := 512
	hiddenDim := 768

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("Every%dLayers", tc.checkpointEveryN), func(t *testing.T) {
			cfg := NewCheckpointConfig()
			cfg.Enabled = true
			cfg.CheckpointEveryN = tc.checkpointEveryN

			_, _, ratio := cfg.EstimateMemorySavings(numLayers, batchSize, seqLen, hiddenDim)

			// Allow 20% tolerance due to integer rounding
			tolerance := tc.expectedRatio * 0.2
			if abs(ratio-tc.expectedRatio) > tolerance {
				t.Errorf("CheckpointEveryN=%d: expected ratio ~%.2fx, got %.2fx",
					tc.checkpointEveryN, tc.expectedRatio, ratio)
			}

			t.Logf("CheckpointEveryN=%d: %.2fx memory savings", tc.checkpointEveryN, ratio)
		})
	}
}

// TestMemorySavingsDisabled tests that savings are 1x when checkpointing is disabled.
func TestMemorySavingsDisabled(t *testing.T) {
	cfg := NewCheckpointConfig()
	cfg.Enabled = false

	memWithout, memWith, ratio := cfg.EstimateMemorySavings(12, 4, 512, 512)

	// All values should be 0 or 1x when disabled
	if memWithout != 0 || memWith != 0 {
		t.Errorf("Memory values should be 0 when disabled, got without=%v, with=%v",
			memWithout, memWith)
	}
	if ratio != 1.0 {
		t.Errorf("Ratio should be 1.0 when disabled, got %v", ratio)
	}
}

// TestEndToEndCheckpointing tests a complete training iteration with checkpointing.
func TestEndToEndCheckpointing(t *testing.T) {
	cfg := NewCheckpointConfig()
	cfg.Enabled = true
	cfg.CheckpointEveryN = 2

	// Simulate a simple 4-layer network
	numLayers := 4
	layerFuncs := make([]CheckpointFunction, numLayers)

	// Each layer adds its layer number to the input
	for i := 0; i < numLayers; i++ {
		layerNum := i
		layerFuncs[i] = func(inputs ...*Tensor) []*Tensor {
			out := NewTensor(inputs[0].shape...)
			for j := range out.data {
				out.data[j] = inputs[0].data[j] + float64(layerNum)
			}
			return []*Tensor{out}
		}
	}

	// Forward pass
	cfg.Reset()
	input := NewTensor(2, 2)
	input.data = []float64{1.0, 1.0, 1.0, 1.0}

	x := input
	for layer := 0; layer < numLayers; layer++ {
		cfg.CurrentLayer = layer
		if cfg.ShouldCheckpoint() {
			// Checkpointed layer
			segment := NewCheckpointSegment(layerFuncs[layer])
			x = segment.RunForward(x)[0]
			cfg.AddSegment(segment)
		} else {
			// Non-checkpointed layer (direct execution)
			x = layerFuncs[layer](x)[0]
		}
	}

	// After 4 layers (0+1+2+3), output should be input + 6
	// Layers 0 and 2 are checkpointed (every 2)
	// Expected: 1 + 0 + 1 + 2 + 3 = 7
	expected := 7.0
	for i := range x.data {
		if abs(x.data[i]-expected) > 1e-9 {
			t.Errorf("Forward pass mismatch at index %d: expected %v, got %v",
				i, expected, x.data[i])
		}
	}

	// Check that we created checkpoints for layers 0 and 2
	if len(cfg.Segments) != 2 {
		t.Errorf("Expected 2 checkpoint segments, got %d", len(cfg.Segments))
	}

	// Simulate backward pass: recompute checkpointed segments
	for i := len(cfg.Segments) - 1; i >= 0; i-- {
		segment := cfg.Segments[i]

		// Recompute activations
		outputs := segment.RecomputeForward()
		if outputs == nil {
			t.Errorf("Failed to recompute segment %d", i)
		}

		// In real training, we'd compute gradients here
		// For this test, just verify recomputation worked
		if !segment.Recomputed {
			t.Errorf("Segment %d should be marked as recomputed", i)
		}

		// Clear outputs to free memory
		segment.ClearOutputs()
	}

	t.Log("End-to-end checkpointing test passed")
}

// BenchmarkCheckpointOverhead benchmarks the overhead of checkpointing.
func BenchmarkCheckpointOverhead(b *testing.B) {
	// Create a computation that mimics a small layer
	forward := func(inputs ...*Tensor) []*Tensor {
		input := inputs[0]
		out := NewTensor(input.shape...)

		// Simulate some computation
		for i := range out.data {
			out.data[i] = input.data[i]*2.0 + 1.0
		}
		return []*Tensor{out}
	}

	input := NewTensor(128, 128)
	for i := range input.data {
		input.data[i] = float64(i)
	}

	segment := NewCheckpointSegment(forward)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// Forward pass
		_ = segment.RunForward(input)

		// Recompute (simulating backward pass)
		_ = segment.RecomputeForward()

		// Clear
		segment.ClearOutputs()
	}
}

// BenchmarkCheckpointMemoryAccess benchmarks different tensor sizes.
func BenchmarkCheckpointMemoryAccess(b *testing.B) {
	sizes := []int{64, 256, 1024}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size%d", size), func(b *testing.B) {
			forward := func(inputs ...*Tensor) []*Tensor {
				input := inputs[0]
				out := NewTensor(input.shape...)
				for i := range out.data {
					out.data[i] = input.data[i] * 1.5
				}
				return []*Tensor{out}
			}

			input := NewTensor(size, size)
			for i := range input.data {
				input.data[i] = float64(i)
			}

			segment := NewCheckpointSegment(forward)

			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				_ = segment.RunForward(input)
				_ = segment.RecomputeForward()
				segment.ClearOutputs()
			}
		})
	}
}
