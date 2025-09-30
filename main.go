package main

import (
	"fmt"
	"math/rand"
	"time"
)

func main() {
	// Seed random number generator
	rand.Seed(time.Now().UnixNano())

	fmt.Println("=== GPT-style Transformer in Pure Go ===")
	fmt.Println()

	// Create a small model for demonstration
	config := Config{
		VocabSize: 100,   // Small vocabulary
		SeqLen:    32,    // Short context window
		EmbedDim:  128,   // Small embedding dimension
		NumHeads:  4,     // 4 attention heads
		NumLayers: 2,     // 2 transformer blocks
		FFHidden:  512,   // 4x embedding dimension
		Dropout:   0.0,   // No dropout for now
	}

	fmt.Printf("Model Configuration:\n")
	fmt.Printf("  Vocabulary Size: %d\n", config.VocabSize)
	fmt.Printf("  Sequence Length: %d\n", config.SeqLen)
	fmt.Printf("  Embedding Dim:   %d\n", config.EmbedDim)
	fmt.Printf("  Attention Heads: %d\n", config.NumHeads)
	fmt.Printf("  Num Layers:      %d\n", config.NumLayers)
	fmt.Printf("  FF Hidden Dim:   %d\n", config.FFHidden)
	fmt.Println()

	// Calculate approximate parameter count
	params := calculateParameters(config)
	fmt.Printf("Approximate Parameters: %d (%.2fM)\n", params, float64(params)/1e6)
	fmt.Println()

	// Initialize model
	fmt.Println("Initializing model...")
	model := NewGPT(config)
	fmt.Println("✓ Model initialized")
	fmt.Println()

	// Demonstrate forward pass
	fmt.Println("=== Forward Pass Demo ===")
	inputIDs := []int{1, 2, 3, 4, 5}
	fmt.Printf("Input tokens: %v\n", inputIDs)

	start := time.Now()
	logits := model.Forward(inputIDs)
	elapsed := time.Since(start)

	fmt.Printf("Output shape: %v\n", logits.Shape())
	fmt.Printf("Forward pass time: %v\n", elapsed)
	fmt.Println()

	// Show sample logits for first position
	fmt.Println("Sample logits for position 0 (first 10 vocab items):")
	for i := 0; i < 10; i++ {
		fmt.Printf("  token %d: %.4f\n", i, logits.At(0, i))
	}
	fmt.Println()

	// Demonstrate generation
	fmt.Println("=== Autoregressive Generation Demo ===")
	prompt := []int{10, 20, 30}
	fmt.Printf("Prompt tokens: %v\n", prompt)

	start = time.Now()
	generated := model.Generate(prompt, 10)
	elapsed = time.Since(start)

	fmt.Printf("Generated tokens: %v\n", generated)
	fmt.Printf("Generation time: %v\n", elapsed)
	fmt.Println()

	// Component benchmarks
	fmt.Println("=== Component Performance ===")
	benchmarkComponents(config)
}

// calculateParameters estimates the number of parameters in the model.
func calculateParameters(c Config) int {
	params := 0

	// Token embeddings: vocab_size * embed_dim
	params += c.VocabSize * c.EmbedDim

	// Position embeddings: seq_len * embed_dim
	params += c.SeqLen * c.EmbedDim

	// Per transformer block:
	for i := 0; i < c.NumLayers; i++ {
		// Attention: 4 * (embed_dim * embed_dim) for Q, K, V, O projections
		params += 4 * c.EmbedDim * c.EmbedDim

		// LayerNorm (x2): 2 * embed_dim for gamma and beta each
		params += 2 * (2 * c.EmbedDim)

		// FeedForward:
		// W1: embed_dim * ff_hidden
		// b1: ff_hidden
		// W2: ff_hidden * embed_dim
		// b2: embed_dim
		params += c.EmbedDim*c.FFHidden + c.FFHidden + c.FFHidden*c.EmbedDim + c.EmbedDim
	}

	// Final LayerNorm: 2 * embed_dim
	params += 2 * c.EmbedDim

	// Output projection: embed_dim * vocab_size
	params += c.EmbedDim * c.VocabSize

	return params
}

// benchmarkComponents measures performance of individual components.
func benchmarkComponents(config Config) {
	const iterations = 10

	// Benchmark attention
	{
		attn := NewAttention(config.EmbedDim, config.NumHeads, config.SeqLen)
		x := NewTensorRand(16, config.EmbedDim)

		start := time.Now()
		for i := 0; i < iterations; i++ {
			_ = attn.Forward(x)
		}
		elapsed := time.Since(start) / iterations
		fmt.Printf("Attention (seq_len=16):         %v\n", elapsed)
	}

	// Benchmark layer norm
	{
		ln := NewLayerNorm(config.EmbedDim)
		x := NewTensorRand(16, config.EmbedDim)

		start := time.Now()
		for i := 0; i < iterations; i++ {
			_ = ln.Forward(x)
		}
		elapsed := time.Since(start) / iterations
		fmt.Printf("LayerNorm (seq_len=16):         %v\n", elapsed)
	}

	// Benchmark feed-forward
	{
		ff := NewFeedForward(config.EmbedDim, config.FFHidden)
		x := NewTensorRand(16, config.EmbedDim)

		start := time.Now()
		for i := 0; i < iterations; i++ {
			_ = ff.Forward(x)
		}
		elapsed := time.Since(start) / iterations
		fmt.Printf("FeedForward (seq_len=16):       %v\n", elapsed)
	}

	// Benchmark full transformer block
	{
		block := NewTransformerBlock(config)
		x := NewTensorRand(16, config.EmbedDim)

		start := time.Now()
		for i := 0; i < iterations; i++ {
			_ = block.Forward(x)
		}
		elapsed := time.Since(start) / iterations
		fmt.Printf("TransformerBlock (seq_len=16):  %v\n", elapsed)
	}

	// Benchmark matrix multiplication (the bottleneck)
	{
		a := NewTensorRand(128, 128)
		b := NewTensorRand(128, 128)

		start := time.Now()
		for i := 0; i < iterations; i++ {
			_ = MatMul(a, b)
		}
		elapsed := time.Since(start) / iterations
		fmt.Printf("MatMul (128x128 @ 128x128):     %v\n", elapsed)
	}

	fmt.Println()
	fmt.Println("Performance notes:")
	fmt.Println("- MatMul dominates compute time (O(n³) operation)")
	fmt.Println("- Attention is O(n²) in sequence length")
	fmt.Println("- Optimizations: BLAS, GPU, cache-aware algorithms")
}