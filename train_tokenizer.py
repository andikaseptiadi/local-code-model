#!/usr/bin/env python3
"""
Train a custom tokenizer optimized for Go code + English instructions.

This creates a BPE (Byte-Pair Encoding) tokenizer that efficiently handles:
- Go syntax (keywords, operators, identifiers)
- Natural language instructions
- Comments and documentation

Much more efficient than GPT-2's general-purpose tokenizer.
"""

import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
from typing import List

def collect_training_corpus(go_files: List[str], instruction_file: str = None) -> List[str]:
    """
    Collect all text for tokenizer training.

    Args:
        go_files: Paths to Go source files
        instruction_file: Optional file with natural language instructions

    Returns:
        List of file paths for training
    """
    corpus_files = []

    # Add Go source files
    corpus_files.extend(go_files)

    # Add instruction examples if provided
    if instruction_file and os.path.exists(instruction_file):
        corpus_files.append(instruction_file)

    return corpus_files

def train_go_tokenizer(
    training_files: List[str],
    vocab_size: int = 12000,
    output_path: str = "go_tokenizer.json"
):
    """
    Train a custom tokenizer for Go code generation.

    This uses Byte-Pair Encoding (BPE), which:
    1. Starts with individual bytes as tokens
    2. Iteratively merges the most frequent pairs
    3. Builds vocabulary of common subwords

    BPE is great for code because:
    - Handles any input (no unknown tokens)
    - Learns domain-specific patterns (func, package, etc.)
    - Efficient representation of identifiers

    Args:
        training_files: Files to train tokenizer on
        vocab_size: Target vocabulary size (12K is good for code+instructions)
        output_path: Where to save the trained tokenizer
    """

    print(f"Training tokenizer on {len(training_files)} files...")

    # Initialize BPE tokenizer
    # BPE starts with bytes and builds up to subwords
    tokenizer = Tokenizer(models.BPE())

    # Pre-tokenization: split on whitespace and punctuation
    # ByteLevel handles Unicode and special characters properly
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Decoder: converts tokens back to text
    tokenizer.decoder = decoders.ByteLevel()

    # Special tokens for the model
    special_tokens = [
        "<|endoftext|>",  # End of sequence (like GPT-2)
        "<|pad|>",        # Padding token
        "<|unk|>",        # Unknown token (shouldn't happen with BPE, but safety)
    ]

    # Trainer configuration
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,  # Token must appear at least twice to be included
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # Train on the corpus
    print(f"Target vocabulary size: {vocab_size}")
    tokenizer.train(training_files, trainer)

    # Post-processing: add special tokens to start/end of sequences
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # Save the tokenizer
    tokenizer.save(output_path)
    print(f"✓ Tokenizer saved to {output_path}")

    return tokenizer

def analyze_tokenizer_efficiency(tokenizer: Tokenizer, test_samples: List[str]):
    """
    Compare tokenization efficiency with examples.
    Shows how many tokens are needed for various Go constructs.
    """
    print("\n" + "="*60)
    print("TOKENIZER EFFICIENCY ANALYSIS")
    print("="*60)

    for sample in test_samples:
        encoding = tokenizer.encode(sample)
        tokens = encoding.tokens

        print(f"\nInput: {sample}")
        print(f"Tokens ({len(tokens)}): {tokens}")
        print(f"Token IDs: {encoding.ids}")

def create_sample_instruction_corpus():
    """
    Create a sample corpus of natural language instructions for Go coding.
    In production, you'd have a larger set of instruction-code pairs.
    """
    instructions = """
Write a function to handle HTTP requests
Create a struct for user data
Implement error handling for file operations
Build a concurrent worker pool
Parse JSON configuration from a file
Create a CLI tool with flags
Implement a TCP server
Add logging to the application
Write unit tests for the handler
Create a Dockerfile for the service
Set up graceful shutdown
Implement rate limiting
Add middleware for authentication
Create a database connection pool
Write a function to upload files to S3
Implement exponential backoff retry logic
Build a queue consumer
Add metrics and monitoring
Create a gRPC service
Implement context cancellation
"""

    # Create instructions file
    os.makedirs("tokenizer_data", exist_ok=True)
    instructions_path = "tokenizer_data/instructions.txt"

    with open(instructions_path, 'w') as f:
        f.write(instructions)

    return instructions_path

def create_sample_go_corpus():
    """
    Create sample Go code for tokenizer training.
    This augments the main.go we already have.
    """
    samples = []

    # Sample 1: AWS SDK patterns
    aws_code = '''package main

import (
    "context"
    "fmt"
    "github.com/aws/aws-sdk-go/aws"
    "github.com/aws/aws-sdk-go/aws/session"
    "github.com/aws/aws-sdk-go/service/s3"
)

type S3Client struct {
    svc *s3.S3
}

func NewS3Client() (*S3Client, error) {
    sess, err := session.NewSession(&aws.Config{
        Region: aws.String("us-west-2"),
    })
    if err != nil {
        return nil, fmt.Errorf("failed to create session: %w", err)
    }

    return &S3Client{
        svc: s3.New(sess),
    }, nil
}

func (c *S3Client) UploadFile(ctx context.Context, bucket, key string, data []byte) error {
    _, err := c.svc.PutObjectWithContext(ctx, &s3.PutObjectInput{
        Bucket: aws.String(bucket),
        Key:    aws.String(key),
        Body:   bytes.NewReader(data),
    })
    return err
}
'''

    # Sample 2: Concurrency patterns
    concurrent_code = '''package main

import (
    "context"
    "sync"
    "time"
)

type WorkerPool struct {
    workers   int
    tasks     chan func()
    wg        sync.WaitGroup
    ctx       context.Context
    cancel    context.CancelFunc
}

func NewWorkerPool(workers int) *WorkerPool {
    ctx, cancel := context.WithCancel(context.Background())
    return &WorkerPool{
        workers: workers,
        tasks:   make(chan func(), workers*2),
        ctx:     ctx,
        cancel:  cancel,
    }
}

func (p *WorkerPool) Start() {
    for i := 0; i < p.workers; i++ {
        p.wg.Add(1)
        go p.worker()
    }
}

func (p *WorkerPool) worker() {
    defer p.wg.Done()

    for {
        select {
        case task := <-p.tasks:
            task()
        case <-p.ctx.Done():
            return
        }
    }
}

func (p *WorkerPool) Submit(task func()) {
    select {
    case p.tasks <- task:
    case <-p.ctx.Done():
    }
}

func (p *WorkerPool) Shutdown(timeout time.Duration) error {
    p.cancel()

    done := make(chan struct{})
    go func() {
        p.wg.Wait()
        close(done)
    }()

    select {
    case <-done:
        return nil
    case <-time.After(timeout):
        return fmt.Errorf("shutdown timeout")
    }
}
'''

    # Sample 3: CLI patterns
    cli_code = '''package main

import (
    "flag"
    "fmt"
    "os"
)

type Config struct {
    Host     string
    Port     int
    Verbose  bool
    LogLevel string
}

func main() {
    cfg := &Config{}

    flag.StringVar(&cfg.Host, "host", "localhost", "Server host")
    flag.IntVar(&cfg.Port, "port", 8080, "Server port")
    flag.BoolVar(&cfg.Verbose, "verbose", false, "Verbose output")
    flag.StringVar(&cfg.LogLevel, "log-level", "info", "Log level (debug|info|warn|error)")

    flag.Parse()

    if err := cfg.Validate(); err != nil {
        fmt.Fprintf(os.Stderr, "Error: %v\\n", err)
        os.Exit(1)
    }

    if err := run(cfg); err != nil {
        fmt.Fprintf(os.Stderr, "Error: %v\\n", err)
        os.Exit(1)
    }
}

func (c *Config) Validate() error {
    if c.Port < 1 || c.Port > 65535 {
        return fmt.Errorf("invalid port: %d", c.Port)
    }

    validLevels := map[string]bool{
        "debug": true, "info": true, "warn": true, "error": true,
    }

    if !validLevels[c.LogLevel] {
        return fmt.Errorf("invalid log level: %s", c.LogLevel)
    }

    return nil
}

func run(cfg *Config) error {
    fmt.Printf("Starting server on %s:%d\\n", cfg.Host, cfg.Port)
    // Server implementation here
    return nil
}
'''

    # Write samples to files
    os.makedirs("tokenizer_data", exist_ok=True)

    samples_data = [
        ("aws_s3.go", aws_code),
        ("worker_pool.go", concurrent_code),
        ("cli_app.go", cli_code),
    ]

    file_paths = []
    for filename, code in samples_data:
        path = f"tokenizer_data/{filename}"
        with open(path, 'w') as f:
            f.write(code)
        file_paths.append(path)
        print(f"Created sample: {path}")

    return file_paths

def main():
    """
    Train a Go-specific tokenizer and analyze its efficiency.
    """
    print("="*60)
    print("TRAINING GO CODE TOKENIZER")
    print("="*60)

    # Create sample data
    print("\nCreating sample training data...")
    go_files = create_sample_go_corpus()

    # Add the main.go from training script
    if os.path.exists("sample_go_code/main.go"):
        go_files.append("sample_go_code/main.go")

    instruction_file = create_sample_instruction_corpus()

    # Collect all training files
    training_files = collect_training_corpus(go_files, instruction_file)
    print(f"Training corpus: {len(training_files)} files")

    # Train tokenizer
    tokenizer = train_go_tokenizer(
        training_files=training_files,
        vocab_size=12000,
        output_path="go_tokenizer.json"
    )

    # Test tokenization efficiency
    test_samples = [
        "func main() {",
        "package main",
        "import \"fmt\"",
        'fmt.Println("Hello, World!")',
        "type Server struct {",
        "if err != nil {",
        "ctx context.Context",
        "Write a function to handle HTTP requests",
        "Create a concurrent worker pool",
        "<-chan string",
        ":=",
        "...interface{}",
    ]

    analyze_tokenizer_efficiency(tokenizer, test_samples)

    print("\n" + "="*60)
    print("TOKENIZER TRAINING COMPLETE")
    print("="*60)
    print(f"\nTokenizer saved to: go_tokenizer.json")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print("\nNext step: Update train_go_model.py to use this tokenizer")
    print("  - Replace: GPT2Tokenizer.from_pretrained('gpt2')")
    print("  - With: Tokenizer.from_file('go_tokenizer.json')")

if __name__ == "__main__":
    main()