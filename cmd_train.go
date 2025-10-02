package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"path/filepath"
	"strings"
)

// ===========================================================================
// TRAINING CLI - Demonstrating the Complete Training Loop
// ===========================================================================
//
// This file implements a minimal training CLI that demonstrates the full
// pipeline: data preparation → training → model saving → inference testing.
//
// INTENTION:
// Provide a working end-to-end example of training a tiny GPT model on Go code.
// This is meant to be:
//   - Simple enough to run in seconds on a laptop
//   - Complete enough to demonstrate all components working together
//   - Educational: show how the pieces fit together
//
// WHY THIS MATTERS:
// - Validates that training infrastructure actually works
// - Demonstrates the full ML loop: train → save → load → infer
// - Provides a template for more serious training experiments
// - Shows that even tiny models can learn patterns from small datasets
//
// KEY DESIGN DECISIONS:
//
// 1. DATASET:
//    - Use local .go files as training data (meta-learning: model learns Go!)
//    - Character-level tokenization (simplest possible, works with tiny vocab)
//    - Context window of 32-64 tokens (tiny for fast training)
//    - Why? We want training to complete in seconds, not hours
//
// 2. MODEL SIZE:
//    - Default: 2 layers, 64 embed dim, 2 heads
//    - Total parameters: ~50K (vs 125M for GPT-2 small)
//    - Why? We're demonstrating the pipeline, not achieving SOTA performance
//    - A tiny model can still learn basic patterns (newlines, indentation, etc.)
//
// 3. TRAINING:
//    - Small batch size (4-8 examples)
//    - Few epochs (1-3)
//    - Simple optimizer (SGD or Adam with defaults)
//    - Why? Fast iteration for experimentation
//
// 4. EVALUATION:
//    - Generate a few samples after training
//    - Visual inspection: does it look vaguely Go-like?
//    - We're not optimizing for quality, just validating the pipeline
//
// WHAT YOU'LL SEE:
// - Initial loss: ~4.5 (random guessing across ~100 character vocab)
// - Final loss: ~2.0-3.0 (model has learned some patterns)
// - Generated text: probably incoherent but with Go-like tokens
//
// THIS IS NOT:
// - A recipe for training production models
// - An optimized training harness
// - A demonstration of good ML practices (no train/val split, etc.)
//
// THIS IS:
// - Proof that the training infrastructure works
// - A starting point for your own experiments
// - An educational example of the full pipeline
//
// ===========================================================================

// RunTrainCommand implements the training CLI.
//
// This is the entry point for training. It:
// 1. Parses command-line flags for hyperparameters
// 2. Loads training data from local .go files
// 3. Creates and initializes a tiny GPT model
// 4. Trains the model for a few epochs
// 5. Saves the trained model to disk
// 6. Optionally generates samples to validate the model works
//
// Design note: Keep everything simple and fast. The goal is to demonstrate
// the pipeline, not to train a good model. Training should complete in
// under a minute on a laptop.
func RunTrainCommand(args []string) error {
	fs := flag.NewFlagSet("train", flag.ExitOnError)

	// Model hyperparameters
	numLayers := fs.Int("layers", 2, "Number of transformer layers")
	embedDim := fs.Int("embed", 64, "Embedding dimension")
	numHeads := fs.Int("heads", 2, "Number of attention heads")
	seqLen := fs.Int("seq", 64, "Sequence length (context window)")

	// Training hyperparameters
	epochs := fs.Int("epochs", 2, "Number of training epochs")
	batchSize := fs.Int("batch", 4, "Batch size")
	lr := fs.Float64("lr", 0.001, "Learning rate")

	// I/O
	dataDir := fs.String("data", ".", "Directory containing .go files for training")
	modelPath := fs.String("model", "tiny_model.bin", "Output model path")
	tokenizerPath := fs.String("tokenizer", "tiny_tokenizer.bin", "Output tokenizer path")

	// Validation
	genSamples := fs.Int("gen-samples", 3, "Number of samples to generate after training")
	genLen := fs.Int("gen-len", 100, "Length of generated samples")

	if err := fs.Parse(args); err != nil {
		return err
	}

	fmt.Println("===========================================================================")
	fmt.Println("TRAINING A TINY GPT MODEL ON GO CODE")
	fmt.Println("===========================================================================")
	fmt.Println()
	fmt.Printf("Model: %d layers, %d embed dim, %d heads, %d seq len\n",
		*numLayers, *embedDim, *numHeads, *seqLen)
	fmt.Printf("Training: %d epochs, batch size %d, lr %.4f\n", *epochs, *batchSize, *lr)
	fmt.Println()

	// Step 1: Load training data
	fmt.Println("Step 1: Loading training data from", *dataDir)
	text, err := loadGoFiles(*dataDir)
	if err != nil {
		return fmt.Errorf("failed to load training data: %v", err)
	}
	fmt.Printf("  Loaded %d characters from .go files\n", len(text))
	fmt.Println()

	// Step 2: Build character-level tokenizer
	fmt.Println("Step 2: Building character-level tokenizer")
	tokenizer := buildCharTokenizer(text)
	fmt.Printf("  Vocabulary size: %d characters\n", tokenizer.VocabSize())
	fmt.Println()

	// Step 3: Create training dataset
	fmt.Println("Step 3: Creating training dataset")
	trainData := createDataset(text, tokenizer, *seqLen, *batchSize)
	fmt.Printf("  Created %d batches\n", len(trainData))
	fmt.Println()

	// Step 4: Initialize model
	fmt.Println("Step 4: Initializing model")
	config := Config{
		VocabSize: tokenizer.VocabSize(),
		SeqLen:    *seqLen,
		EmbedDim:  *embedDim,
		NumLayers: *numLayers,
		NumHeads:  *numHeads,
		FFHidden:  *embedDim * 4, // Standard GPT ratio (4x embed dim)
		Dropout:   0.1,           // Light dropout for regularization
	}
	model := NewGPT(config)
	params := model.Parameters()
	fmt.Printf("  Total parameters: %d\n", countParameters(params))
	fmt.Println()

	// Step 5: Create optimizer
	fmt.Println("Step 5: Creating Adam optimizer")
	optimizer := NewAdamOptimizer(params, 0.9, 0.999, 1e-8, 0.0)
	scheduler := NewLRScheduler(*lr, *lr*0.1, 10, len(trainData)**epochs)
	fmt.Println()

	// Step 6: Train!
	fmt.Println("Step 6: Training...")
	fmt.Println("-------------------------------------------------------------------")
	for epoch := 0; epoch < *epochs; epoch++ {
		epochLoss := 0.0
		for batchIdx, batch := range trainData {
			// Get current learning rate
			currentLR := scheduler.GetLR()

			// Training step
			inputs := batch[:len(batch)-1]
			targets := make([][]int, len(inputs))
			for i := range inputs {
				// Target is input shifted by 1 (next token prediction)
				targets[i] = []int{batch[i+1][0]}
			}

			loss := TrainStep(model, [][]int{inputs[0]}, targets, optimizer, currentLR)
			epochLoss += loss

			// Print progress every 10 batches
			if batchIdx%10 == 0 {
				fmt.Printf("Epoch %d/%d, Batch %d/%d, Loss: %.4f, LR: %.6f\n",
					epoch+1, *epochs, batchIdx+1, len(trainData), loss, currentLR)
			}
		}
		avgLoss := epochLoss / float64(len(trainData))
		fmt.Printf("Epoch %d complete. Average loss: %.4f\n", epoch+1, avgLoss)
		fmt.Println("-------------------------------------------------------------------")
	}
	fmt.Println()

	// Step 7: Save model
	fmt.Println("Step 7: Saving model and tokenizer")
	if err := model.Save(*modelPath); err != nil {
		return fmt.Errorf("failed to save model: %v", err)
	}
	if err := tokenizer.Save(*tokenizerPath); err != nil {
		return fmt.Errorf("failed to save tokenizer: %v", err)
	}
	fmt.Printf("  Model saved to: %s\n", *modelPath)
	fmt.Printf("  Tokenizer saved to: %s\n", *tokenizerPath)
	fmt.Println()

	// Step 8: Generate samples to validate
	if *genSamples > 0 {
		fmt.Println("Step 8: Generating sample outputs")
		fmt.Println("-------------------------------------------------------------------")
		prompts := []string{"func ", "package ", "type "}
		for i := 0; i < *genSamples && i < len(prompts); i++ {
			fmt.Printf("Prompt: %q\n", prompts[i])
			generated := generateSample(model, tokenizer, prompts[i], *genLen)
			fmt.Printf("Generated: %q\n", generated)
			fmt.Println()
		}
		fmt.Println("-------------------------------------------------------------------")
	}

	fmt.Println()
	fmt.Println("Training complete! Try:")
	fmt.Printf("  go run . generate -model=%s -tokenizer=%s -prompt=\"func main\"\n",
		*modelPath, *tokenizerPath)
	fmt.Println()

	return nil
}

// loadGoFiles loads all .go files from a directory and concatenates them.
//
// Design note: We don't shuffle or do train/test splits because this is
// a minimal example. For real training you'd want proper data handling.
func loadGoFiles(dir string) (string, error) {
	var texts []string

	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && strings.HasSuffix(path, ".go") {
			content, err := ioutil.ReadFile(path)
			if err != nil {
				return err
			}
			texts = append(texts, string(content))
		}
		return nil
	})

	if err != nil {
		return "", err
	}

	if len(texts) == 0 {
		return "", fmt.Errorf("no .go files found in %s", dir)
	}

	return strings.Join(texts, "\n\n"), nil
}

// buildCharTokenizer creates a character-level tokenizer from text.
//
// Why character-level? Simplest possible tokenization. BPE would be better
// for real use, but char-level is enough for this demo.
func buildCharTokenizer(text string) *Tokenizer {
	// Collect unique characters
	charSet := make(map[rune]bool)
	for _, ch := range text {
		charSet[ch] = true
	}

	// Build vocab
	vocab := make([]string, 0, len(charSet))
	for ch := range charSet {
		vocab = append(vocab, string(ch))
	}

	tokenizer := NewTokenizer()
	if err := tokenizer.Train([]string{text}, len(vocab)); err != nil {
		panic(err)
	}

	return tokenizer
}

// createDataset splits text into training examples.
//
// Each example is a sequence of seqLen tokens. We create batches of
// batchSize examples each.
//
// Design note: No shuffling, no validation set, no data augmentation.
// This is intentionally minimal.
func createDataset(text string, tokenizer *Tokenizer, seqLen int, batchSize int) [][][]int {
	// Tokenize entire text
	tokens := tokenizer.Encode(text)

	// Split into sequences
	var batches [][][]int
	batch := make([][]int, 0, batchSize)

	for i := 0; i+seqLen < len(tokens); i += seqLen {
		sequence := tokens[i : i+seqLen]
		batch = append(batch, sequence)

		if len(batch) == batchSize {
			batches = append(batches, batch)
			batch = make([][]int, 0, batchSize)
		}
	}

	// Add remaining batch if non-empty
	if len(batch) > 0 {
		batches = append(batches, batch)
	}

	return batches
}

// countParameters counts total parameters in model.
func countParameters(params []*Tensor) int {
	total := 0
	for _, p := range params {
		count := 1
		for _, dim := range p.shape {
			count *= dim
		}
		total += count
	}
	return total
}

// generateSample generates text from a trained model.
//
// This is a simple greedy generation for validation purposes.
// For real inference, use cmd_generate.go with proper sampling.
func generateSample(model *GPT, tokenizer *Tokenizer, prompt string, maxLen int) string {
	tokens := tokenizer.Encode(prompt)

	for i := 0; i < maxLen; i++ {
		// Forward pass
		logits := model.Forward(tokens)

		// Get last token's logits
		vocabSize := logits.shape[1]
		lastLogits := make([]float64, vocabSize)
		lastTokenIdx := logits.shape[0] - 1
		for v := 0; v < vocabSize; v++ {
			lastLogits[v] = logits.At(lastTokenIdx, v)
		}

		// Softmax
		maxLogit := lastLogits[0]
		for _, l := range lastLogits {
			if l > maxLogit {
				maxLogit = l
			}
		}
		sumExp := 0.0
		for i := range lastLogits {
			lastLogits[i] = math.Exp(lastLogits[i] - maxLogit)
			sumExp += lastLogits[i]
		}
		for i := range lastLogits {
			lastLogits[i] /= sumExp
		}

		// Sample (greedy for simplicity)
		nextToken := 0
		maxProb := lastLogits[0]
		for i, prob := range lastLogits {
			if prob > maxProb {
				maxProb = prob
				nextToken = i
			}
		}

		tokens = append(tokens, nextToken)

		// Truncate context if too long
		if len(tokens) > model.config.SeqLen {
			tokens = tokens[len(tokens)-model.config.SeqLen:]
		}
	}

	return tokenizer.Decode(tokens)
}
