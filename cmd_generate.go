// ===========================================================================
// TEXT GENERATION CLI - Interactive and Batch Inference
// ===========================================================================
//
// This file implements the command-line interface for text generation using
// trained GPT models. It demonstrates the full inference pipeline: loading
// models, tokenizing text, generating tokens, and decoding back to text.
//
// INTENTION:
// Provide two modes of interaction:
//   1. Single-shot generation: Quick text completion for scripts/automation
//   2. Interactive REPL: Experimentation with sampling parameters in real-time
//
// WHY TWO MODES?
// - Single-shot is ideal for production use, scripting, benchmarking
// - Interactive mode is crucial for development: you need to experiment with
//   temperature, top-k, top-p to find settings that work for your model/task
// - Interactive mode lets you iterate quickly without reloading the model
//   (model loading can take seconds, experimentation requires dozens of tries)
//
// KEY CONCEPTS:
//
// 1. SAMPLING PARAMETERS (Control output randomness/quality)
//    - Temperature: Controls randomness of predictions
//      * 0.0 = greedy (always pick most likely token) - deterministic but boring
//      * 0.7 = focused (good for factual/code generation)
//      * 1.0 = neutral (match training distribution)
//      * 1.5+ = creative (diverse but potentially incoherent)
//    - Top-k: Only sample from k most likely tokens
//      * Prevents sampling very unlikely tokens (reduces nonsense)
//      * 40 is a good default (empirically determined by GPT-2 paper)
//    - Top-p (nucleus): Sample from smallest set of tokens with cumulative probability p
//      * More adaptive than top-k (adjusts set size based on confidence)
//      * 0.9 is a good default (keeps quality high while allowing diversity)
//
// 2. BACKEND SELECTION (See backend.go for details)
//    Why configurable? Because performance varies drastically by hardware:
//      - M4 Max: Accelerate framework gives 10-20x speedup over naive
//      - Linux ARM: SVE or OpenBLAS are best options
//      - NVIDIA: CUDA backend required for GPU acceleration
//    The "auto" backend will eventually auto-detect, but manual selection
//    lets you benchmark different approaches on the same hardware.
//
// 3. MODEL/TOKENIZER LOADING
//    Why separate files?
//      - Models are large (MBs-GBs), tokenizers are small (KBs)
//      - Different tokenizers can be used with same model architecture
//      - Allows sharing tokenizers across models (save disk space)
//
// USAGE EXAMPLES:
//
// Single prompt:
//   go run . generate -model=model.bin -tokenizer=tok.bin -prompt="Hello"
//
// Interactive experimentation:
//   go run . generate -model=model.bin -tokenizer=tok.bin -interactive
//   > Hello world
//   > /temp 1.2      # Make it more creative
//   > /tokens 50     # Generate longer text
//   > Hello world    # Try again with new settings
//
// Production use with specific backend:
//   go run . generate -model=model.bin -tokenizer=tok.bin \
//     -prompt="func sum(a, b int)" -backend=accelerate -temperature=0.3
//
// ===========================================================================

package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"
)

// RunGenerateCommand implements the text generation CLI.
//
// This is the main entry point for text generation. It:
// 1. Parses command-line flags
// 2. Loads model and tokenizer (the slow part - can take seconds)
// 3. Configures compute backend and sampling parameters
// 4. Dispatches to either interactive or single-shot mode
//
// Design decision: Load model ONCE, then reuse for multiple generations.
// This is why interactive mode is so much faster than repeatedly running
// single-shot mode - you pay the loading cost once, not per generation.
func RunGenerateCommand(args []string) error {
	fs := flag.NewFlagSet("generate", flag.ExitOnError)

	// Model and tokenizer paths (required)
	// Why separate files? See header comments for rationale.
	modelPath := fs.String("model", "", "Path to saved model file (required)")
	tokenizerPath := fs.String("tokenizer", "", "Path to saved tokenizer file (required)")

	// Generation parameters
	// Why both prompt and interactive? Different use cases:
	//   - prompt: for scripting, automation, benchmarking
	//   - interactive: for experimentation, debugging, finding good parameters
	prompt := fs.String("prompt", "", "Text prompt for generation")
	interactive := fs.Bool("interactive", false, "Interactive mode (REPL)")
	maxTokens := fs.Int("max-tokens", 100, "Maximum number of tokens to generate")

	// Sampling parameters (control output randomness/quality)
	// These defaults (0.8, 40, 0.9) are empirically good for most tasks.
	// See header comments for detailed explanation of each parameter.
	temperature := fs.Float64("temperature", 0.8, "Temperature for sampling (0=greedy, higher=more random)")
	topK := fs.Int("top-k", 40, "Top-k sampling (0=disabled)")
	topP := fs.Float64("top-p", 0.9, "Top-p (nucleus) sampling (0=disabled)")

	// Backend selection (affects performance, not output)
	// "auto" will eventually auto-detect best backend, currently uses naive.
	// Manual selection lets you benchmark or force specific hardware usage.
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

	// Load model (the slow part - can take seconds for large models)
	// Why so slow? Need to read MBs-GBs from disk and allocate all weight matrices.
	// This is why we load once and reuse for multiple generations.
	fmt.Printf("Loading model from %s...\n", *modelPath)
	model, err := LoadGPT(*modelPath)
	if err != nil {
		return fmt.Errorf("failed to load model: %w", err)
	}
	fmt.Printf("✓ Model loaded (vocab=%d, dim=%d, layers=%d)\n",
		model.config.VocabSize, model.config.EmbedDim, model.config.NumLayers)

	// Set backend if specified
	// Backend selection determines which hardware accelerates the matrix multiplications.
	// On failure, we fall back to naive implementation - slower but always works.
	// Why not fail hard? Because we want text generation to work even if hardware
	// acceleration isn't available (e.g., wrong driver version, missing libraries).
	if *backend != "auto" && *backend != "naive" {
		if err := setupBackend(model, *backend); err != nil {
			fmt.Fprintf(os.Stderr, "Warning: Failed to setup backend '%s': %v\n", *backend, err)
			fmt.Fprintf(os.Stderr, "Falling back to naive implementation\n")
		}
	}

	// Load tokenizer
	// Much faster than model loading (KBs vs MBs-GBs).
	// Tokenizer converts text <-> token IDs, required for both input and output.
	fmt.Printf("Loading tokenizer from %s...\n", *tokenizerPath)
	tokenizer := NewTokenizer()
	if err := tokenizer.Load(*tokenizerPath); err != nil {
		return fmt.Errorf("failed to load tokenizer: %w", err)
	}
	fmt.Printf("✓ Tokenizer loaded (vocab size=%d)\n", tokenizer.VocabSize())

	// Configure sampling
	// These parameters stay constant for the initial run, but can be changed
	// interactively in REPL mode using /temp, /topk, /topp commands.
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
//
// This demonstrates the complete inference pipeline:
// 1. Encode: Convert text -> token IDs (using BPE tokenizer)
// 2. Generate: Autoregressively predict next tokens (using transformer model)
// 3. Decode: Convert token IDs -> text (using BPE tokenizer)
//
// Why show token IDs? For debugging and understanding. When generation goes wrong,
// seeing the token IDs helps you understand if the problem is in tokenization
// (wrong token IDs) or generation (right tokens, wrong predictions).
func generateText(model *GPT, tokenizer *Tokenizer, promptText string, maxTokens int, config *SampleConfig) error {
	fmt.Printf("Prompt: %s\n", promptText)
	fmt.Println()

	// Encode prompt: text -> token IDs
	// BPE tokenizer breaks text into subword units (e.g., "hello" might be one token,
	// "helloworld" might be ["hello", "world"], or ["hel", "low", "or", "ld"]).
	promptTokens := tokenizer.Encode(promptText)
	if len(promptTokens) == 0 {
		return fmt.Errorf("prompt encoding resulted in zero tokens")
	}

	fmt.Printf("Encoded to %d tokens: %v\n", len(promptTokens), promptTokens)
	fmt.Println()

	// Generate: Autoregressive next-token prediction
	// The model generates one token at a time, feeding each new token back as input.
	// This is why generation is slow - it's inherently sequential, can't be parallelized.
	// Temperature/top-k/top-p control the randomness of the sampling process.
	fmt.Println("Generating...")
	generatedTokens := model.GenerateWithSampling(promptTokens, maxTokens, config)

	// Decode: token IDs -> text
	// Convert the full sequence (prompt + generated tokens) back to readable text.
	generatedText := tokenizer.Decode(generatedTokens)

	fmt.Println()
	fmt.Println("=== Generated Text ===")
	fmt.Println(generatedText)
	fmt.Println()
	fmt.Printf("Generated %d total tokens\n", len(generatedTokens))

	return nil
}

// runInteractive runs an interactive text generation REPL.
//
// WHY REPL MODE?
// When developing with LLMs, you need to experiment with different prompts and
// sampling parameters to find what works. Reloading the model each time is too slow
// (can take seconds). REPL mode loads once and lets you iterate quickly.
//
// DESIGN: Slash commands for configuration
// We use "/command" syntax (like Discord, Slack) instead of special flags because:
//   - It's familiar to users of modern chat apps
//   - It's unambiguous (any line starting with "/" is a command, not a prompt)
//   - It allows natural language prompts without escaping
//
// The config pointer is shared with generateText, so changes apply immediately.
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
		// Slash commands let you adjust sampling parameters without restarting.
		// This is crucial for experimentation: you can see how temperature affects
		// output quality in real-time.
		if strings.HasPrefix(line, "/") {
			if err := handleCommand(line, config, &maxTokens); err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			}
			continue
		}

		// Generate text from user input
		// Each generation uses the current config values, so changes from slash
		// commands take effect immediately.
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
//
// This parses and executes slash commands that modify generation parameters.
// Why mutable pointers? So changes persist across multiple generations.
// The config pointer is shared with generateText(), so modifications here
// immediately affect subsequent generations.
//
// Design note: We could have returned modified values instead of mutating,
// but pointer mutation makes the REPL simpler - no need to thread updated
// config through the call stack.
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
//
// This is where we select which hardware accelerates inference.
// The backend doesn't affect the output (semantics), only performance.
//
// WHY MULTIPLE BACKENDS?
// Different hardware has different optimal implementations:
//   - Accelerate (macOS): Uses BLAS/LAPACK from Apple's vDSP framework
//   - Metal (macOS): Uses GPU via Metal API
//   - CUDA (Linux+NVIDIA): Uses GPU via CUDA
//   - SVE (ARM): Uses ARM Scalable Vector Extension
//   - OpenBLAS (Linux): Optimized BLAS library for various CPUs
//
// Performance can vary 10-100x between backends on the same hardware!
// For example, on M4 Max: naive ~5 GFLOPS, Accelerate ~50-100 GFLOPS.
//
// See backend.go for detailed explanation of backend selection strategy.
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
