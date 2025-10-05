package main

import (
	"math"
	"testing"
)

// TestFlashAttentionConfig tests Flash Attention configuration creation and defaults.
func TestFlashAttentionConfig(t *testing.T) {
	headDim := 64
	config := NewFlashAttentionConfig(headDim)

	if !config.Enabled {
		t.Error("Expected config.Enabled to be true")
	}

	if config.BlockSize != 64 {
		t.Errorf("Expected BlockSize=64, got %d", config.BlockSize)
	}

	if config.CausalMask {
		t.Error("Expected CausalMask to be false by default")
	}

	expectedScale := 1.0 / math.Sqrt(float64(headDim))
	if math.Abs(config.SoftmaxScale-expectedScale) > 1e-10 {
		t.Errorf("Expected SoftmaxScale=%.10f, got %.10f", expectedScale, config.SoftmaxScale)
	}
}

// TestStandardAttention tests the reference implementation of attention.
func TestStandardAttention(t *testing.T) {
	// Small test case: batch=1, heads=1, seq_len=4, head_dim=8
	batch, numHeads, seqLen, headDim := 1, 1, 4, 8

	Q := NewTensor(batch, numHeads, seqLen, headDim)
	K := NewTensor(batch, numHeads, seqLen, headDim)
	V := NewTensor(batch, numHeads, seqLen, headDim)

	// Fill with simple test values
	for i := 0; i < seqLen; i++ {
		for d := 0; d < headDim; d++ {
			idx := i*headDim + d
			Q.data[idx] = float64(i + 1)
			K.data[idx] = float64(i + 1)
			V.data[idx] = float64(d + 1)
		}
	}

	scale := 1.0 / math.Sqrt(float64(headDim))
	output := StandardAttention(Q, K, V, scale, false)

	// Verify output shape
	if len(output.shape) != 4 {
		t.Fatalf("Expected 4D output, got %dD", len(output.shape))
	}
	if output.shape[0] != batch || output.shape[1] != numHeads ||
		output.shape[2] != seqLen || output.shape[3] != headDim {
		t.Errorf("Output shape mismatch: got %v", output.shape)
	}

	// Verify output is not all zeros
	hasNonZero := false
	for _, v := range output.data {
		if v != 0 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Error("Output is all zeros")
	}

	t.Logf("Standard attention computed successfully for %d×%d", seqLen, headDim)
}

// TestFlashAttentionVsStandard compares Flash Attention with Standard Attention.
// They should produce very similar results (within numerical precision).
func TestFlashAttentionVsStandard(t *testing.T) {
	batch, numHeads, seqLen, headDim := 1, 2, 32, 64

	Q := NewTensor(batch, numHeads, seqLen, headDim)
	K := NewTensor(batch, numHeads, seqLen, headDim)
	V := NewTensor(batch, numHeads, seqLen, headDim)

	// Fill with pseudo-random values
	for i := range Q.data {
		Q.data[i] = math.Sin(float64(i) * 0.1)
		K.data[i] = math.Cos(float64(i) * 0.1)
		V.data[i] = math.Sin(float64(i) * 0.2)
	}

	// Compute standard attention
	config := NewFlashAttentionConfig(headDim)
	config.Enabled = false
	outputStandard := FlashAttentionForward(Q, K, V, config)

	// Compute Flash Attention
	config.Enabled = true
	config.BlockSize = 16 // Use small blocks for this test
	outputFlash := FlashAttentionForward(Q, K, V, config)

	// Compare results
	maxDiff := 0.0
	avgDiff := 0.0
	for i := range outputStandard.data {
		diff := math.Abs(outputStandard.data[i] - outputFlash.data[i])
		if diff > maxDiff {
			maxDiff = diff
		}
		avgDiff += diff
	}
	avgDiff /= float64(len(outputStandard.data))

	// Allow small numerical differences due to different computation order
	tolerance := 1e-6
	if maxDiff > tolerance {
		t.Errorf("Max difference %.10f exceeds tolerance %.10f", maxDiff, tolerance)
	}

	t.Logf("Flash vs Standard attention: max_diff=%.10f, avg_diff=%.10f", maxDiff, avgDiff)
}

// TestFlashAttentionCausalMask tests causal masking in Flash Attention.
func TestFlashAttentionCausalMask(t *testing.T) {
	batch, numHeads, seqLen, headDim := 1, 1, 8, 16

	Q := NewTensor(batch, numHeads, seqLen, headDim)
	K := NewTensor(batch, numHeads, seqLen, headDim)
	V := NewTensor(batch, numHeads, seqLen, headDim)

	// Fill tensors
	for i := range Q.data {
		Q.data[i] = float64(i % 10)
		K.data[i] = float64(i % 10)
		V.data[i] = float64(i % 10)
	}

	// Test with causal mask
	config := NewFlashAttentionConfig(headDim)
	config.CausalMask = true
	config.BlockSize = 4
	outputCausal := FlashAttentionForward(Q, K, V, config)

	// Test without causal mask
	config.CausalMask = false
	outputNonCausal := FlashAttentionForward(Q, K, V, config)

	// Outputs should be different (causal mask affects attention)
	same := true
	for i := range outputCausal.data {
		if math.Abs(outputCausal.data[i]-outputNonCausal.data[i]) > 1e-10 {
			same = false
			break
		}
	}

	if same {
		t.Error("Causal and non-causal outputs are identical (mask not applied)")
	}

	t.Log("Causal masking produces different output as expected")
}

// TestFlashAttentionBlockSizes tests different block sizes.
func TestFlashAttentionBlockSizes(t *testing.T) {
	batch, numHeads, seqLen, headDim := 1, 1, 64, 32

	Q := NewTensor(batch, numHeads, seqLen, headDim)
	K := NewTensor(batch, numHeads, seqLen, headDim)
	V := NewTensor(batch, numHeads, seqLen, headDim)

	for i := range Q.data {
		Q.data[i] = math.Sin(float64(i) * 0.01)
		K.data[i] = math.Cos(float64(i) * 0.01)
		V.data[i] = math.Sin(float64(i) * 0.02)
	}

	// Test various block sizes
	blockSizes := []int{8, 16, 32, 64}
	var outputs []*Tensor

	for _, blockSize := range blockSizes {
		config := NewFlashAttentionConfig(headDim)
		config.BlockSize = blockSize
		output := FlashAttentionForward(Q, K, V, config)
		outputs = append(outputs, output)
	}

	// All block sizes should produce similar results
	for i := 1; i < len(outputs); i++ {
		maxDiff := 0.0
		for j := range outputs[0].data {
			diff := math.Abs(outputs[0].data[j] - outputs[i].data[j])
			if diff > maxDiff {
				maxDiff = diff
			}
		}

		tolerance := 1e-6
		if maxDiff > tolerance {
			t.Errorf("BlockSize %d vs %d: max_diff=%.10f exceeds tolerance %.10f",
				blockSizes[0], blockSizes[i], maxDiff, tolerance)
		}

		t.Logf("BlockSize %d vs %d: max_diff=%.10f", blockSizes[0], blockSizes[i], maxDiff)
	}
}

// TestFlashAttentionLongSequence tests Flash Attention with longer sequences.
func TestFlashAttentionLongSequence(t *testing.T) {
	batch, numHeads, seqLen, headDim := 1, 4, 256, 64

	Q := NewTensor(batch, numHeads, seqLen, headDim)
	K := NewTensor(batch, numHeads, seqLen, headDim)
	V := NewTensor(batch, numHeads, seqLen, headDim)

	// Fill with values
	for i := range Q.data {
		Q.data[i] = math.Sin(float64(i) * 0.001)
		K.data[i] = math.Cos(float64(i) * 0.001)
		V.data[i] = math.Sin(float64(i) * 0.002)
	}

	config := NewFlashAttentionConfig(headDim)
	config.BlockSize = 32
	output := FlashAttentionForward(Q, K, V, config)

	// Verify output shape
	if output.shape[2] != seqLen {
		t.Errorf("Expected output seq_len=%d, got %d", seqLen, output.shape[2])
	}

	// Verify output is reasonable (not all zeros, not NaN/Inf)
	hasNonZero := false
	for _, v := range output.data {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("Output contains NaN or Inf: %f", v)
		}
		if v != 0 {
			hasNonZero = true
		}
	}

	if !hasNonZero {
		t.Error("Output is all zeros")
	}

	t.Logf("Long sequence (%d tokens) processed successfully", seqLen)
}

// TestEstimateFlashAttentionSpeedup tests the speedup estimation function.
func TestEstimateFlashAttentionSpeedup(t *testing.T) {
	testCases := []struct {
		seqLen    int
		headDim   int
		blockSize int
		minSpeedup float64
	}{
		{128, 64, 32, 1.5},
		{256, 64, 32, 2.0},
		{512, 64, 32, 3.0},
		{1024, 64, 64, 4.0},
	}

	for _, tc := range testCases {
		stdTime, flashTime, speedup := EstimateFlashAttentionSpeedup(
			tc.seqLen, tc.headDim, tc.blockSize)

		if speedup < tc.minSpeedup {
			t.Errorf("SeqLen=%d: Expected speedup >= %.1fx, got %.2fx",
				tc.seqLen, tc.minSpeedup, speedup)
		}

		if flashTime >= stdTime {
			t.Errorf("SeqLen=%d: Flash time %.2f >= Standard time %.2f",
				tc.seqLen, flashTime, stdTime)
		}

		t.Logf("SeqLen=%d: %.2fx speedup (std=%.0f, flash=%.0f)",
			tc.seqLen, speedup, stdTime, flashTime)
	}
}

// TestFlashAttentionMultiHead tests multi-head attention.
func TestFlashAttentionMultiHead(t *testing.T) {
	batch, numHeads, seqLen, headDim := 2, 8, 64, 32

	Q := NewTensor(batch, numHeads, seqLen, headDim)
	K := NewTensor(batch, numHeads, seqLen, headDim)
	V := NewTensor(batch, numHeads, seqLen, headDim)

	// Fill tensors with different patterns per head
	for b := 0; b < batch; b++ {
		for h := 0; h < numHeads; h++ {
			for i := 0; i < seqLen; i++ {
				for d := 0; d < headDim; d++ {
					idx := ((b*numHeads+h)*seqLen+i)*headDim + d
					Q.data[idx] = math.Sin(float64(h*100+i) * 0.01)
					K.data[idx] = math.Cos(float64(h*100+i) * 0.01)
					V.data[idx] = math.Sin(float64(h*100+d) * 0.01)
				}
			}
		}
	}

	config := NewFlashAttentionConfig(headDim)
	config.BlockSize = 16
	output := FlashAttentionForward(Q, K, V, config)

	// Verify different heads produce different outputs
	head0 := make([]float64, seqLen*headDim)
	head1 := make([]float64, seqLen*headDim)

	for i := 0; i < seqLen; i++ {
		for d := 0; d < headDim; d++ {
			idx0 := ((0*numHeads+0)*seqLen+i)*headDim + d
			idx1 := ((0*numHeads+1)*seqLen+i)*headDim + d
			head0[i*headDim+d] = output.data[idx0]
			head1[i*headDim+d] = output.data[idx1]
		}
	}

	same := true
	for i := range head0 {
		if math.Abs(head0[i]-head1[i]) > 1e-10 {
			same = false
			break
		}
	}

	if same {
		t.Error("Different heads produce identical output")
	}

	t.Logf("Multi-head attention: %d heads produce different outputs", numHeads)
}

// TestFlashAttentionNumericalStability tests numerical stability with extreme values.
func TestFlashAttentionNumericalStability(t *testing.T) {
	batch, numHeads, seqLen, headDim := 1, 1, 16, 8

	Q := NewTensor(batch, numHeads, seqLen, headDim)
	K := NewTensor(batch, numHeads, seqLen, headDim)
	V := NewTensor(batch, numHeads, seqLen, headDim)

	// Fill with large values
	for i := range Q.data {
		Q.data[i] = 100.0 * math.Sin(float64(i))
		K.data[i] = 100.0 * math.Cos(float64(i))
		V.data[i] = 10.0 * math.Sin(float64(i))
	}

	config := NewFlashAttentionConfig(headDim)
	output := FlashAttentionForward(Q, K, V, config)

	// Check for NaN or Inf
	for i, v := range output.data {
		if math.IsNaN(v) {
			t.Errorf("Output[%d] is NaN", i)
		}
		if math.IsInf(v, 0) {
			t.Errorf("Output[%d] is Inf", i)
		}
	}

	t.Log("Numerical stability test passed (no NaN/Inf)")
}

// BenchmarkFlashAttentionSmall benchmarks Flash Attention on small sequences.
func BenchmarkFlashAttentionSmall(b *testing.B) {
	batch, numHeads, seqLen, headDim := 1, 8, 128, 64

	Q := NewTensor(batch, numHeads, seqLen, headDim)
	K := NewTensor(batch, numHeads, seqLen, headDim)
	V := NewTensor(batch, numHeads, seqLen, headDim)

	for i := range Q.data {
		Q.data[i] = float64(i) * 0.001
		K.data[i] = float64(i) * 0.001
		V.data[i] = float64(i) * 0.001
	}

	config := NewFlashAttentionConfig(headDim)
	config.BlockSize = 32

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = FlashAttentionForward(Q, K, V, config)
	}
}

// BenchmarkFlashAttentionLarge benchmarks Flash Attention on large sequences.
func BenchmarkFlashAttentionLarge(b *testing.B) {
	batch, numHeads, seqLen, headDim := 1, 8, 512, 64

	Q := NewTensor(batch, numHeads, seqLen, headDim)
	K := NewTensor(batch, numHeads, seqLen, headDim)
	V := NewTensor(batch, numHeads, seqLen, headDim)

	for i := range Q.data {
		Q.data[i] = float64(i) * 0.001
		K.data[i] = float64(i) * 0.001
		V.data[i] = float64(i) * 0.001
	}

	config := NewFlashAttentionConfig(headDim)
	config.BlockSize = 64

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = FlashAttentionForward(Q, K, V, config)
	}
}

// BenchmarkStandardAttentionSmall benchmarks standard attention for comparison.
func BenchmarkStandardAttentionSmall(b *testing.B) {
	batch, numHeads, seqLen, headDim := 1, 8, 128, 64

	Q := NewTensor(batch, numHeads, seqLen, headDim)
	K := NewTensor(batch, numHeads, seqLen, headDim)
	V := NewTensor(batch, numHeads, seqLen, headDim)

	for i := range Q.data {
		Q.data[i] = float64(i) * 0.001
		K.data[i] = float64(i) * 0.001
		V.data[i] = float64(i) * 0.001
	}

	scale := 1.0 / math.Sqrt(float64(headDim))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = StandardAttention(Q, K, V, scale, false)
	}
}
