package main

import (
	"testing"
)

func TestTransformerBlock(t *testing.T) {
	config := Config{
		VocabSize: 100,
		SeqLen:    16,
		EmbedDim:  64,
		NumHeads:  4,
		NumLayers: 2,
		FFHidden:  256,
		Dropout:   0.0,
	}

	block := NewTransformerBlock(config)

	// Create input
	x := NewTensorRand(8, config.EmbedDim) // seqLen=8

	// Forward pass
	output := block.Forward(x)

	// Verify output shape
	if len(output.Shape()) != 2 {
		t.Errorf("expected 2D output, got %dD", len(output.Shape()))
	}

	shape := output.Shape()
	if shape[0] != 8 || shape[1] != config.EmbedDim {
		t.Errorf("expected shape [8 %d], got %v", config.EmbedDim, shape)
	}
}

func TestGPTForward(t *testing.T) {
	config := DefaultConfig()
	config.VocabSize = 100
	config.SeqLen = 16
	config.EmbedDim = 64
	config.NumHeads = 4
	config.NumLayers = 2
	config.FFHidden = 256

	model := NewGPT(config)

	// Create input token IDs
	inputIDs := []int{1, 2, 3, 4, 5}

	// Forward pass
	logits := model.Forward(inputIDs)

	// Verify output shape: (seqLen, vocabSize)
	shape := logits.Shape()
	if len(shape) != 2 {
		t.Fatalf("expected 2D logits, got %dD", len(shape))
	}

	if shape[0] != len(inputIDs) {
		t.Errorf("expected seqLen %d, got %d", len(inputIDs), shape[0])
	}

	if shape[1] != config.VocabSize {
		t.Errorf("expected vocabSize %d, got %d", config.VocabSize, shape[1])
	}
}

func TestGPTGenerate(t *testing.T) {
	config := DefaultConfig()
	config.VocabSize = 50
	config.SeqLen = 16
	config.EmbedDim = 32
	config.NumHeads = 2
	config.NumLayers = 1
	config.FFHidden = 128

	model := NewGPT(config)

	// Start with a prompt
	prompt := []int{1, 2, 3}

	// Generate tokens
	generated := model.Generate(prompt, 5)

	// Should have prompt + 5 new tokens
	expectedLen := len(prompt) + 5
	if len(generated) != expectedLen {
		t.Errorf("expected %d tokens, got %d", expectedLen, len(generated))
	}

	// Verify prompt is preserved
	for i := range prompt {
		if generated[i] != prompt[i] {
			t.Errorf("prompt token %d: expected %d, got %d", i, prompt[i], generated[i])
		}
	}

	// Verify all tokens are in valid range
	for i, tok := range generated {
		if tok < 0 || tok >= config.VocabSize {
			t.Errorf("token %d: value %d out of range [0,%d)", i, tok, config.VocabSize)
		}
	}
}

func TestAttentionShape(t *testing.T) {
	embedDim := 64
	numHeads := 4
	seqLen := 8

	attn := NewAttention(embedDim, numHeads, seqLen)

	x := NewTensorRand(seqLen, embedDim)
	output := attn.Forward(x)

	shape := output.Shape()
	if len(shape) != 2 || shape[0] != seqLen || shape[1] != embedDim {
		t.Errorf("expected shape [%d %d], got %v", seqLen, embedDim, shape)
	}
}

func TestLayerNormalization(t *testing.T) {
	ln := NewLayerNorm(64)

	x := NewTensorRand(8, 64)
	output := ln.Forward(x)

	// Verify shape preserved
	if !shapeEqual(output.Shape(), x.Shape()) {
		t.Errorf("shape mismatch: input %v, output %v", x.Shape(), output.Shape())
	}

	// Verify mean ≈ 0, std ≈ 1 for each position
	seqLen, features := output.Shape()[0], output.Shape()[1]
	for i := 0; i < seqLen; i++ {
		mean := 0.0
		for j := 0; j < features; j++ {
			mean += output.At(i, j)
		}
		mean /= float64(features)

		if mean > 1e-5 || mean < -1e-5 {
			t.Errorf("position %d: mean should be ~0, got %f", i, mean)
		}
	}
}

func TestFeedForward(t *testing.T) {
	embedDim := 64
	hiddenDim := 256

	ff := NewFeedForward(embedDim, hiddenDim)

	x := NewTensorRand(8, embedDim)
	output := ff.Forward(x)

	// Verify shape preserved
	if !shapeEqual(output.Shape(), x.Shape()) {
		t.Errorf("shape mismatch: input %v, output %v", x.Shape(), output.Shape())
	}
}

func BenchmarkAttention(b *testing.B) {
	embedDim := 256
	numHeads := 8
	seqLen := 128

	attn := NewAttention(embedDim, numHeads, seqLen)
	x := NewTensorRand(seqLen, embedDim)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = attn.Forward(x)
	}
}

func BenchmarkTransformerBlock(b *testing.B) {
	config := Config{
		VocabSize: 1000,
		SeqLen:    128,
		EmbedDim:  256,
		NumHeads:  8,
		NumLayers: 1,
		FFHidden:  1024,
		Dropout:   0.0,
	}

	block := NewTransformerBlock(config)
	x := NewTensorRand(128, 256)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = block.Forward(x)
	}
}

func BenchmarkGPTForward(b *testing.B) {
	config := DefaultConfig()
	model := NewGPT(config)

	inputIDs := make([]int, 64)
	for i := range inputIDs {
		inputIDs[i] = i % config.VocabSize
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = model.Forward(inputIDs)
	}
}