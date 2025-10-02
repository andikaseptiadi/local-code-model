package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"
)

// RunGenerateCommand implements the text generation CLI.
func RunGenerateCommand(args []string) error {
	fs := flag.NewFlagSet("generate", flag.ExitOnError)

	// Model and tokenizer paths
	modelPath := fs.String("model", "", "Path to saved model file (required)")
	tokenizerPath := fs.String("tokenizer", "", "Path to saved tokenizer file (required)")

	// Generation parameters
	prompt := fs.String("prompt", "", "Text prompt for generation")
	interactive := fs.Bool("interactive", false, "Interactive mode (REPL)")
	maxTokens := fs.Int("max-tokens", 100, "Maximum number of tokens to generate")

	// Sampling parameters
	temperature := fs.Float64("temperature", 0.8, "Temperature for sampling (0=greedy, higher=more random)")
	topK := fs.Int("top-k", 40, "Top-k sampling (0=disabled)")
	topP := fs.Float64("top-p", 0.9, "Top-p (nucleus) sampling (0=disabled)")

	// Backend selection
	backend := fs.String("backend", "auto", "Compute backend: auto, naive, accelerate, metal, cuda, sve, openblas")

	if err := fs.Parse(args); err != nil {
		return err
	}

	// Validate required arguments
	if *modelPath == "" {
		return fmt.Errorf("--model is required")
	}
	if *tokenizerPath == "" {
		return fmt.Errorf("--tokenizer is required")
	}

	// Load model
	fmt.Printf("Loading model from %s...\n", *modelPath)
	model, err := LoadGPT(*modelPath)
	if err != nil {
		return fmt.Errorf("failed to load model: %w", err)
	}
	fmt.Printf("✓ Model loaded (vocab=%d, dim=%d, layers=%d)\n",
		model.config.VocabSize, model.config.EmbedDim, model.config.NumLayers)

	// Set backend if specified
	if *backend != "auto" && *backend != "naive" {
		if err := setupBackend(model, *backend); err != nil {
			fmt.Fprintf(os.Stderr, "Warning: Failed to setup backend '%s': %v\n", *backend, err)
			fmt.Fprintf(os.Stderr, "Falling back to naive implementation\n")
		}
	}

	// Load tokenizer
	fmt.Printf("Loading tokenizer from %s...\n", *tokenizerPath)
	tokenizer := NewTokenizer()
	if err := tokenizer.Load(*tokenizerPath); err != nil {
		return fmt.Errorf("failed to load tokenizer: %w", err)
	}
	fmt.Printf("✓ Tokenizer loaded (vocab size=%d)\n", tokenizer.VocabSize())

	// Configure sampling
	samplingConfig := &SampleConfig{
		Temperature: *temperature,
		TopK:        *topK,
		TopP:        *topP,
	}

	fmt.Println()
	fmt.Printf("Generation settings:\n")
	fmt.Printf("  Max tokens:   %d\n", *maxTokens)
	fmt.Printf("  Temperature:  %.2f\n", samplingConfig.Temperature)
	fmt.Printf("  Top-k:        %d\n", samplingConfig.TopK)
	fmt.Printf("  Top-p:        %.2f\n", samplingConfig.TopP)
	fmt.Println()

	if *interactive {
		// Interactive mode (REPL)
		return runInteractive(model, tokenizer, *maxTokens, samplingConfig)
	}

	// Single prompt mode
	if *prompt == "" {
		return fmt.Errorf("either --prompt or --interactive is required")
	}

	return generateText(model, tokenizer, *prompt, *maxTokens, samplingConfig)
}

// generateText generates text from a single prompt.
func generateText(model *GPT, tokenizer *Tokenizer, promptText string, maxTokens int, config *SampleConfig) error {
	fmt.Printf("Prompt: %s\n", promptText)
	fmt.Println()

	// Encode prompt
	promptTokens := tokenizer.Encode(promptText)
	if len(promptTokens) == 0 {
		return fmt.Errorf("prompt encoding resulted in zero tokens")
	}

	fmt.Printf("Encoded to %d tokens: %v\n", len(promptTokens), promptTokens)
	fmt.Println()

	// Generate
	fmt.Println("Generating...")
	generatedTokens := model.GenerateWithSampling(promptTokens, maxTokens, config)

	// Decode
	generatedText := tokenizer.Decode(generatedTokens)

	fmt.Println()
	fmt.Println("=== Generated Text ===")
	fmt.Println(generatedText)
	fmt.Println()
	fmt.Printf("Generated %d total tokens\n", len(generatedTokens))

	return nil
}

// runInteractive runs an interactive text generation REPL.
func runInteractive(model *GPT, tokenizer *Tokenizer, maxTokens int, config *SampleConfig) error {
	fmt.Println("=== Interactive Mode ===")
	fmt.Println("Enter prompts to generate text. Type 'quit' or 'exit' to stop.")
	fmt.Println("Commands:")
	fmt.Println("  /temp <value>    Set temperature (e.g., /temp 0.8)")
	fmt.Println("  /topk <value>    Set top-k (e.g., /topk 40)")
	fmt.Println("  /topp <value>    Set top-p (e.g., /topp 0.9)")
	fmt.Println("  /tokens <value>  Set max tokens (e.g., /tokens 50)")
	fmt.Println("  /config          Show current settings")
	fmt.Println()

	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			break
		}

		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		// Check for exit commands
		if line == "quit" || line == "exit" {
			fmt.Println("Goodbye!")
			return nil
		}

		// Check for configuration commands
		if strings.HasPrefix(line, "/") {
			if err := handleCommand(line, config, &maxTokens); err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			}
			continue
		}

		// Generate text
		if err := generateText(model, tokenizer, line, maxTokens, config); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		}
		fmt.Println()
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading input: %w", err)
	}

	return nil
}

// handleCommand handles interactive mode commands.
func handleCommand(cmd string, config *SampleConfig, maxTokens *int) error {
	parts := strings.Fields(cmd)
	if len(parts) == 0 {
		return nil
	}

	switch parts[0] {
	case "/temp":
		if len(parts) < 2 {
			return fmt.Errorf("usage: /temp <value>")
		}
		var val float64
		if _, err := fmt.Sscanf(parts[1], "%f", &val); err != nil {
			return fmt.Errorf("invalid temperature value: %v", err)
		}
		config.Temperature = val
		fmt.Printf("Temperature set to %.2f\n", val)

	case "/topk":
		if len(parts) < 2 {
			return fmt.Errorf("usage: /topk <value>")
		}
		var val int
		if _, err := fmt.Sscanf(parts[1], "%d", &val); err != nil {
			return fmt.Errorf("invalid top-k value: %v", err)
		}
		config.TopK = val
		fmt.Printf("Top-k set to %d\n", val)

	case "/topp":
		if len(parts) < 2 {
			return fmt.Errorf("usage: /topp <value>")
		}
		var val float64
		if _, err := fmt.Sscanf(parts[1], "%f", &val); err != nil {
			return fmt.Errorf("invalid top-p value: %v", err)
		}
		config.TopP = val
		fmt.Printf("Top-p set to %.2f\n", val)

	case "/tokens":
		if len(parts) < 2 {
			return fmt.Errorf("usage: /tokens <value>")
		}
		var val int
		if _, err := fmt.Sscanf(parts[1], "%d", &val); err != nil {
			return fmt.Errorf("invalid max tokens value: %v", err)
		}
		*maxTokens = val
		fmt.Printf("Max tokens set to %d\n", val)

	case "/config":
		fmt.Printf("Current settings:\n")
		fmt.Printf("  Temperature:  %.2f\n", config.Temperature)
		fmt.Printf("  Top-k:        %d\n", config.TopK)
		fmt.Printf("  Top-p:        %.2f\n", config.TopP)
		fmt.Printf("  Max tokens:   %d\n", *maxTokens)

	default:
		return fmt.Errorf("unknown command: %s", parts[0])
	}

	return nil
}

// setupBackend configures the compute backend for the model.
func setupBackend(model *GPT, backendName string) error {
	switch backendName {
	case "accelerate":
		backend, err := NewAccelerateBackend()
		if err != nil {
			return err
		}
		model.SetBackend(backend)
		fmt.Printf("✓ Using Accelerate framework backend\n")

	case "metal":
		backend, err := NewMetalBackend()
		if err != nil {
			return err
		}
		model.SetBackend(backend)
		fmt.Printf("✓ Using Metal GPU backend\n")

	case "cuda":
		backend, err := NewCUDABackend()
		if err != nil {
			return err
		}
		model.SetBackend(backend)
		fmt.Printf("✓ Using CUDA GPU backend\n")

	case "sve":
		backend, err := NewSVEBackend()
		if err != nil {
			return err
		}
		model.SetBackend(backend)
		fmt.Printf("✓ Using ARM SVE backend\n")

	case "openblas":
		backend, err := NewOpenBLASBackend()
		if err != nil {
			return err
		}
		model.SetBackend(backend)
		fmt.Printf("✓ Using OpenBLAS backend\n")

	default:
		return fmt.Errorf("unknown backend: %s", backendName)
	}

	return nil
}
