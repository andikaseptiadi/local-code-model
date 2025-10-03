package main

import (
	"fmt"
	"math/rand"
	"os"
	"time"
)

func main() {
	// Seed random number generator (using modern approach for Go 1.20+)
	_ = rand.New(rand.NewSource(time.Now().UnixNano()))

	// Check for command-line mode
	if len(os.Args) > 1 {
		cmd := os.Args[1]
		switch cmd {
		case "generate":
			if err := RunGenerateCommand(os.Args[2:]); err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				os.Exit(1)
			}
			return
		case "train":
			if err := RunTrainCommand(os.Args[2:]); err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				os.Exit(1)
			}
			return
		case "visualize":
			if err := RunVisualizeCommand(os.Args[2:]); err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				os.Exit(1)
			}
			return
		case "embeddings":
			if err := RunEmbeddingsCommand(os.Args[2:]); err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				os.Exit(1)
			}
			return
		case "help", "-h", "--help":
			printUsage()
			return
		default:
			fmt.Fprintf(os.Stderr, "Unknown command: %s\n", cmd)
			printUsage()
			os.Exit(1)
		}
	}

	// Default: show help
	printUsage()
}

func printUsage() {
	fmt.Println("Usage:")
	fmt.Println("  go run . [command] [options]")
	fmt.Println()
	fmt.Println("Commands:")
	fmt.Println("  train       Train a new model on local .go files")
	fmt.Println("  generate    Generate text from a trained model")
	fmt.Println("  visualize   Visualize attention weights for a prompt")
	fmt.Println("  embeddings  Visualize token embeddings in 2D (PCA/t-SNE)")
	fmt.Println("  help        Show this help message")
	fmt.Println()
	fmt.Println("Examples:")
	fmt.Println("  go run . train -data=. -epochs=2 -model=tiny_model.bin")
	fmt.Println("  go run . generate -model=model.bin -tokenizer=tok.bin -prompt=\"Hello world\"")
	fmt.Println("  go run . generate -model=model.bin -tokenizer=tok.bin -interactive")
	fmt.Println("  go run . visualize -model=model.bin -tokenizer=tok.bin -prompt=\"func main\"")
	fmt.Println("  go run . embeddings -model=model.bin -tokenizer=tok.bin -method=pca")
	fmt.Println()
}
