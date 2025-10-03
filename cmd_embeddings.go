package main

import (
	"flag"
	"fmt"
)

// ===========================================================================
// EMBEDDINGS CLI - Visualizing Token Embeddings in 2D
// ===========================================================================
//
// This file implements a CLI command for visualizing token embeddings from
// a trained model. It reduces high-dimensional embeddings to 2D using either
// PCA or t-SNE, then generates an interactive HTML scatter plot.
//
// INTENTION:
// - Explore which tokens are "close" to each other in embedding space
// - Understand how the model represents different tokens
// - Educational tool for understanding embeddings
//
// WHY THIS MATTERS:
// - Embeddings encode semantic meaning in high-dimensional space
// - Visualization makes this abstract concept concrete
// - Helps debug training (are similar tokens close together?)
//
// USAGE:
//   go run . embeddings -model=model.bin -tokenizer=tok.bin \
//                        -method=pca -output=embeddings.html
//
// ===========================================================================

// RunEmbeddingsCommand implements the embedding visualization CLI.
//
// This command:
// 1. Loads a trained model and tokenizer
// 2. Extracts token embeddings from the model
// 3. Reduces to 2D using PCA or t-SNE
// 4. Generates interactive HTML visualization
func RunEmbeddingsCommand(args []string) error {
	fs := flag.NewFlagSet("embeddings", flag.ExitOnError)

	// Required flags
	modelPath := fs.String("model", "", "Path to trained model file (required)")
	tokenizerPath := fs.String("tokenizer", "", "Path to tokenizer file (required)")

	// Method selection
	method := fs.String("method", "pca", "Dimensionality reduction method: 'pca' (fast) or 'tsne' (slower, better)")

	// t-SNE parameters
	perplexity := fs.Float64("perplexity", 30.0, "t-SNE perplexity (5-50, controls neighborhood size)")
	iterations := fs.Int("iterations", 1000, "t-SNE iterations (500-2000)")
	learningRate := fs.Float64("lr", 200.0, "t-SNE learning rate (100-500)")

	// Output
	outputPath := fs.String("output", "embeddings.html", "Output HTML file path")

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
	if *method != "pca" && *method != "tsne" {
		return fmt.Errorf("--method must be 'pca' or 'tsne', got: %s", *method)
	}

	fmt.Println("===========================================================================")
	fmt.Println("TOKEN EMBEDDING VISUALIZATION")
	fmt.Println("===========================================================================")
	fmt.Println()

	// Step 1: Load model
	fmt.Println("Step 1: Loading model from", *modelPath)
	model, err := LoadGPT(*modelPath)
	if err != nil {
		return fmt.Errorf("failed to load model: %v", err)
	}
	fmt.Printf("  Model loaded: %d layers, %d embed dim\n", model.config.NumLayers, model.config.EmbedDim)
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

	// Step 3: Extract embeddings
	fmt.Println("Step 3: Extracting token embeddings from model")
	// model.tokenEmbed is a (vocabSize, embedDim) tensor
	embeddings := model.tokenEmbed
	fmt.Printf("  Extracted %d token embeddings of dimension %d\n",
		embeddings.shape[0], embeddings.shape[1])
	fmt.Println()

	// Step 4: Reduce to 2D
	fmt.Printf("Step 4: Reducing to 2D using %s\n", *method)
	var coords2D *Tensor

	if *method == "pca" {
		fmt.Println("  Running PCA (fast, preserves global structure)...")
		coords2D, err = PCA(embeddings)
		if err != nil {
			return fmt.Errorf("PCA failed: %v", err)
		}
	} else {
		fmt.Printf("  Running t-SNE (slower, perplexity=%.1f, %d iterations)...\n",
			*perplexity, *iterations)
		fmt.Println("  This may take a minute or two...")
		coords2D, err = TSNE(embeddings, *perplexity, *iterations, *learningRate)
		if err != nil {
			return fmt.Errorf("t-SNE failed: %v", err)
		}
	}
	fmt.Printf("  Reduced to 2D coordinates (%d points)\n", coords2D.shape[0])
	fmt.Println()

	// Step 5: Generate token labels
	fmt.Println("Step 5: Generating token labels")
	labels := make([]string, tokenizer.VocabSize())
	for i := 0; i < tokenizer.VocabSize(); i++ {
		// Decode single token
		label := tokenizer.Decode([]int{i})
		if label == "" || label == " " || label == "\n" || label == "\t" {
			// Use visual representation for whitespace
			switch label {
			case "":
				label = fmt.Sprintf("[TOKEN_%d]", i)
			case " ":
				label = "[SPACE]"
			case "\n":
				label = "[NEWLINE]"
			case "\t":
				label = "[TAB]"
			}
		}
		labels[i] = label
	}
	fmt.Println()

	// Step 6: Generate HTML visualization
	fmt.Println("Step 6: Generating HTML visualization")
	methodName := "PCA"
	if *method == "tsne" {
		methodName = "t-SNE"
	}
	if err := SaveEmbeddingHTML(*outputPath, coords2D, labels, methodName); err != nil {
		return fmt.Errorf("failed to save visualization: %v", err)
	}
	fmt.Printf("  Visualization saved to: %s\n", *outputPath)
	fmt.Println()

	fmt.Println("===========================================================================")
	fmt.Println("Visualization complete! Open the HTML file in your browser:")
	fmt.Printf("  open %s\n", *outputPath)
	fmt.Println()
	fmt.Println("INTERPRETING THE VISUALIZATION:")
	fmt.Println("- Points close together = tokens with similar embeddings")
	fmt.Println("- You might see clusters of related tokens (e.g., keywords, operators)")
	fmt.Println("- Hover over points to see which token they represent")
	if *method == "pca" {
		fmt.Println("- PCA preserves global structure (overall relationships)")
	} else {
		fmt.Println("- t-SNE preserves local structure (neighborhoods)")
	}
	fmt.Println("===========================================================================")
	fmt.Println()

	return nil
}
