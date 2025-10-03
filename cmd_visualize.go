package main

import (
	"flag"
	"fmt"
)

// ===========================================================================
// VISUALIZATION CLI - Generating Attention Heatmaps
// ===========================================================================
//
// This file implements a CLI command for visualizing attention patterns
// in trained models. It loads a model, runs inference on a prompt while
// capturing attention weights, and generates an interactive HTML heatmap.
//
// INTENTION:
// - Provide insight into what the model "pays attention to"
// - Help understand how different layers learn different patterns
// - Educational tool for exploring transformer internals
//
// WHY THIS MATTERS:
// - Attention visualization reveals model behavior
// - Helps debug training issues (dead heads, collapsed attention)
// - Educational: makes abstract attention mechanism concrete
//
// USAGE:
//   go run . visualize -model=model.bin -tokenizer=tok.bin \
//                      -prompt="func main" -output=attention.html
//
// ===========================================================================

// RunVisualizeCommand implements the visualization CLI.
//
// This command:
// 1. Loads a trained model and tokenizer
// 2. Encodes the user's prompt
// 3. Runs forward pass with attention weight capture
// 4. Generates interactive HTML visualization
func RunVisualizeCommand(args []string) error {
	fs := flag.NewFlagSet("visualize", flag.ExitOnError)

	// Required flags
	modelPath := fs.String("model", "", "Path to trained model file (required)")
	tokenizerPath := fs.String("tokenizer", "", "Path to tokenizer file (required)")
	prompt := fs.String("prompt", "func main", "Text prompt to visualize attention for")

	// Output
	outputPath := fs.String("output", "attention.html", "Output HTML file path")

	if err := fs.Parse(args); err != nil {
		return err
	}

	// Validate required flags
	if *modelPath == "" {
		return fmt.Errorf("--model flag is required")
	}
	if *tokenizerPath == "" {
		return fmt.Errorf("--tokenizer flag is required")
	}

	fmt.Println("===========================================================================")
	fmt.Println("ATTENTION WEIGHT VISUALIZATION")
	fmt.Println("===========================================================================")
	fmt.Println()

	// Step 1: Load model
	fmt.Println("Step 1: Loading model from", *modelPath)
	model, err := LoadGPT(*modelPath)
	if err != nil {
		return fmt.Errorf("failed to load model: %v", err)
	}
	fmt.Printf("  Model loaded: %d layers, %d embed dim, %d heads\n",
		model.config.NumLayers, model.config.EmbedDim, model.config.NumHeads)
	fmt.Println()

	// Step 2: Load tokenizer
	fmt.Println("Step 2: Loading tokenizer from", *tokenizerPath)
	var tokenizer TokenizerInterface

	// Try BPE tokenizer first
	bpeTokenizer := NewTokenizer()
	if err := bpeTokenizer.Load(*tokenizerPath); err == nil {
		tokenizer = bpeTokenizer
	} else {
		// Fall back to simple tokenizer
		simpleTokenizer := NewSimpleTokenizer()
		if err := simpleTokenizer.Load(*tokenizerPath); err != nil {
			return fmt.Errorf("failed to load tokenizer: %v", err)
		}
		tokenizer = simpleTokenizer
	}
	fmt.Printf("  Tokenizer loaded: vocab size %d\n", tokenizer.VocabSize())
	fmt.Println()

	// Step 3: Encode prompt
	fmt.Printf("Step 3: Encoding prompt: %q\n", *prompt)
	tokens := tokenizer.Encode(*prompt)
	if len(tokens) == 0 {
		return fmt.Errorf("prompt encoded to zero tokens")
	}
	if len(tokens) > model.config.SeqLen {
		tokens = tokens[:model.config.SeqLen]
		fmt.Printf("  Warning: Truncated to %d tokens (model's max sequence length)\n", model.config.SeqLen)
	}
	fmt.Printf("  Encoded to %d tokens\n", len(tokens))
	fmt.Println()

	// Step 4: Run forward pass with attention capture
	fmt.Println("Step 4: Running forward pass and capturing attention weights")
	collector := &AttentionCollector{
		Weights: make([][][]float64, 0),
	}

	// Perform embeddings manually (same as GPT.Forward in transformer.go)
	x := NewTensor(len(tokens), model.config.EmbedDim)
	for i, tokenID := range tokens {
		// Add token embedding + position embedding
		for j := 0; j < model.config.EmbedDim; j++ {
			tokEmb := model.tokenEmbed.At(tokenID, j)
			posEmb := model.posEmbed.At(i, j)
			x.Set(tokEmb+posEmb, i, j)
		}
	}

	// Process through transformer blocks with collector
	for layerIdx, block := range model.blocks {
		x = block.ForwardWithCollector(x, nil, layerIdx, collector)
	}

	// Final layer norm
	x = model.lnFinal.Forward(x)

	fmt.Printf("  Captured attention weights from %d layers\n", len(collector.Weights))
	fmt.Println()

	// Step 5: Convert tokens back to strings for labels
	fmt.Println("Step 5: Preparing token labels")
	tokenStrings := make([]string, len(tokens))
	for i, tokenID := range tokens {
		// Decode single token to get string representation
		tokenStrings[i] = tokenizer.Decode([]int{tokenID})
		// If empty or whitespace, use token ID
		if tokenStrings[i] == "" || tokenStrings[i] == " " {
			tokenStrings[i] = fmt.Sprintf("[%d]", tokenID)
		}
	}
	fmt.Println()

	// Step 6: Generate HTML visualization
	fmt.Println("Step 6: Generating HTML visualization")
	if err := SaveAttentionHTML(*outputPath, collector, tokenStrings, len(tokens)); err != nil {
		return fmt.Errorf("failed to save visualization: %v", err)
	}
	fmt.Printf("  Visualization saved to: %s\n", *outputPath)
	fmt.Println()

	fmt.Println("===========================================================================")
	fmt.Println("Visualization complete! Open the HTML file in your browser:")
	fmt.Printf("  open %s\n", *outputPath)
	fmt.Println("===========================================================================")
	fmt.Println()

	return nil
}
