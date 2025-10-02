package main

import (
	"fmt"
	"math/rand"
	"os"
	"time"
)

func main() {
	// Seed random number generator
	rand.Seed(time.Now().UnixNano())

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
	fmt.Println("  generate   Generate text from a trained model")
	fmt.Println("  help       Show this help message")
	fmt.Println()
	fmt.Println("Examples:")
	fmt.Println("  go run . generate -model=model.bin -tokenizer=tok.bin -prompt=\"Hello world\"")
	fmt.Println("  go run . generate -model=model.bin -tokenizer=tok.bin -interactive")
	fmt.Println()
}

