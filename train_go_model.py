#!/usr/bin/env python3
"""
Minimal GPT-2 style transformer for training on Go code from scratch.
Over-commented for systems folks who think Python is gross but necessary.

This is a "hello world" for transformer training - gets you from zero to
a working model that can generate Go-ish code tokens.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import math
from typing import Optional, Tuple, List

# =============================================================================
# CONFIGURATION - All the knobs you'll want to tune
# =============================================================================

class Config:
    """
    All hyperparameters in one place. Start small, scale up.
    These are "toy model" sizes - think of this as your proof of concept.
    """
    # Model architecture
    vocab_size: int = 50257        # GPT-2 tokenizer vocab size
    n_positions: int = 1024        # Max sequence length (context window)
    n_embd: int = 384             # Embedding dimension (small for testing)
    n_layer: int = 6              # Number of transformer blocks (tiny model)
    n_head: int = 6               # Number of attention heads

    # Training parameters
    batch_size: int = 8           # Small batch for RTX 5090 testing
    learning_rate: float = 3e-4   # Standard transformer learning rate
    max_epochs: int = 5           # Just enough to see it learning
    warmup_steps: int = 100       # Learning rate warmup

    # Data processing
    block_size: int = 128         # Training sequence length (smaller for demo)

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# DATASET - Converts raw Go code into training tokens
# =============================================================================

class GoCodeDataset(Dataset):
    """
    Dead simple dataset that takes Go source files and turns them into
    sequences of tokens for training. This is where your data curation
    magic will happen later.
    """

    def __init__(self, go_files: List[str], tokenizer, block_size: int):
        """
        Args:
            go_files: List of paths to .go source files
            tokenizer: HuggingFace tokenizer for encoding text
            block_size: Length of each training sequence
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = []

        # Read all Go files and concatenate them
        print(f"Loading {len(go_files)} Go files...")
        all_text = ""
        for file_path in go_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Add file separator to help model learn file boundaries
                    all_text += f"\n// FILE: {os.path.basename(file_path)}\n"
                    all_text += content + "\n\n"
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

        # Tokenize the entire corpus
        print("Tokenizing corpus...")
        tokens = tokenizer.encode(all_text)
        print(f"Total tokens: {len(tokens)}")

        # Split into training blocks
        # Each example is a sequence of block_size tokens
        for i in range(0, len(tokens) - block_size, block_size):
            self.examples.append(tokens[i:i + block_size])

        print(f"Created {len(self.examples)} training examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """
        Returns a training example as input/target pair.
        For language modeling: predict next token given previous tokens.
        """
        tokens = torch.tensor(self.examples[idx], dtype=torch.long)
        # Input: all tokens except last, Target: all tokens except first
        # This is the "shifted" training pattern for autoregressive models
        return tokens[:-1], tokens[1:]

# =============================================================================
# MODEL ARCHITECTURE - Simplified GPT-2 style transformer
# =============================================================================

class MultiHeadAttention(nn.Module):
    """
    The attention mechanism - the "smart" part of transformers.
    This is where quadratic scaling lives (n^2 in sequence length).

    Your systems background will immediately see why this is inefficient:
    - O(n^2) memory and compute for sequence length n
    - Most attention weights are near zero (sparse but computed densely)
    - Lots of matrix multiplications for simple pattern matching

    But it works, and we can optimize later.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Linear projections for queries, keys, values
        # This is where most of the parameters live
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        # Causal mask - prevents model from "cheating" by looking at future tokens
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.n_positions, config.n_positions))
        )

    def forward(self, x):
        B, T, C = x.shape  # Batch size, sequence length, embedding dim

        # Compute queries, keys, values in one shot (efficiency hack)
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        # Think of this as running multiple "attention processors" in parallel
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Attention computation: Q * K^T / sqrt(d_k)
        # This is the O(n^2) operation that makes transformers slow
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # Apply causal mask - zero out "future" tokens
        att = att.masked_fill(self.causal_mask[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        # Apply attention to values
        out = att @ v  # (B, n_head, T, head_dim)

        # Concatenate heads and project back
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class TransformerBlock(nn.Module):
    """
    One "layer" of the transformer. GPT-2 stacks many of these.
    Pattern: attention -> add & norm -> feedforward -> add & norm

    The residual connections (add) are crucial - without them,
    gradients vanish and deep networks don't train.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        # Feed-forward network - just two linear layers with GELU activation
        # This is where the model "processes" the attended information
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),  # Expand
            nn.GELU(),                                      # Nonlinearity
            nn.Linear(4 * config.n_embd, config.n_embd),  # Contract
        )

    def forward(self, x):
        # Pre-norm variant: normalize before applying transformations
        x = x + self.attention(self.ln1(x))  # Residual connection
        x = x + self.mlp(self.ln2(x))        # Residual connection
        return x

class GoCodeGPT(nn.Module):
    """
    The complete model: embedding -> transformer blocks -> output projection
    This is your "neural network that generates Go code"
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Token embeddings: map token IDs to dense vectors
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)

        # Position embeddings: help model understand token order
        # Transformers have no inherent notion of sequence order
        self.position_embedding = nn.Embedding(config.n_positions, config.n_embd)

        # Stack of transformer blocks - this is where the "intelligence" lives
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Output projection: map hidden states back to vocabulary logits
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights (important for training stability)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Weight initialization following GPT-2 paper.
        Bad initialization = model never learns.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape

        # Create position indices [0, 1, 2, ..., T-1]
        pos = torch.arange(0, T, device=input_ids.device)

        # Embedding lookup: tokens -> vectors, positions -> vectors
        tok_emb = self.token_embedding(input_ids)  # (B, T, n_embd)
        pos_emb = self.position_embedding(pos)     # (T, n_embd)

        # Add embeddings (broadcasting handles batch dimension)
        x = tok_emb + pos_emb

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.ln_f(x)

        # Project to vocabulary logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute loss if targets provided (training mode)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # Flatten to (B*T, vocab_size)
                targets.view(-1),                   # Flatten to (B*T,)
                ignore_index=-1
            )

        return logits, loss

# =============================================================================
# TRAINING LOOP - Where the magic happens
# =============================================================================

def get_sample_go_files():
    """
    For demo purposes - create some sample Go files.
    In reality, you'd point this at your curated corpus of high-quality Go code.
    """
    sample_files = []

    # Create a simple Go file for testing
    sample_code = '''package main

import (
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "time"
)

func main() {
    http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
        fmt.Fprintf(w, "Hello, World!")
    })

    log.Println("Server starting on :8080")
    if err := http.ListenAndServe(":8080", nil); err != nil {
        log.Fatal(err)
    }
}

func processData(data []string) ([]string, error) {
    if data == nil {
        return nil, fmt.Errorf("data cannot be nil")
    }

    result := make([]string, 0, len(data))
    for _, item := range data {
        if item != "" {
            result = append(result, item)
        }
    }

    return result, nil
}

type Config struct {
    Port     int    `json:"port"`
    Host     string `json:"host"`
    Database string `json:"database"`
}

func NewConfig() *Config {
    return &Config{
        Port:     8080,
        Host:     "localhost",
        Database: "app.db",
    }
}

func (c *Config) Validate() error {
    if c.Port <= 0 {
        return fmt.Errorf("port must be positive")
    }
    if c.Host == "" {
        return fmt.Errorf("host cannot be empty")
    }
    return nil
}

func handleRequest(w http.ResponseWriter, r *http.Request) {
    switch r.Method {
    case http.MethodGet:
        handleGet(w, r)
    case http.MethodPost:
        handlePost(w, r)
    default:
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
    }
}

func handleGet(w http.ResponseWriter, r *http.Request) {
    data := map[string]interface{}{
        "status": "ok",
        "time":   time.Now(),
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(data)
}

func handlePost(w http.ResponseWriter, r *http.Request) {
    var req struct {
        Message string `json:"message"`
    }

    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }

    log.Printf("Received message: %s", req.Message)
    fmt.Fprintf(w, "Message received: %s", req.Message)
}'''

    # Write sample file
    os.makedirs("sample_go_code", exist_ok=True)
    sample_path = "sample_go_code/main.go"
    with open(sample_path, 'w') as f:
        f.write(sample_code)
    sample_files.append(sample_path)

    return sample_files

def train_model():
    """
    Main training function. This is where you'd spend most of your time
    tuning hyperparameters and watching loss curves.
    """
    config = Config()
    print(f"Training on device: {config.device}")

    # Initialize tokenizer (using GPT-2's tokenizer for simplicity)
    # In production, you might train a Go-specific tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token

    # Get training data
    go_files = get_sample_go_files()
    print(f"Found {len(go_files)} Go files")

    # Create dataset and dataloader
    dataset = GoCodeDataset(go_files, tokenizer, config.block_size)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,  # Parallel data loading (your concurrency expertise!)
        pin_memory=True if config.device == "cuda" else False
    )

    # Initialize model
    print("Initializing model...")
    model = GoCodeGPT(config).to(config.device)

    # Count parameters (because we're curious)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params:,} parameters")

    # Initialize optimizer
    # AdamW is the standard choice for transformers
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),  # GPT-2 hyperparams
        weight_decay=0.1
    )

    # Learning rate scheduler with warmup
    # Warmup is crucial for transformer training stability
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_epochs * len(dataloader)
    )

    # Training loop
    print("Starting training...")
    model.train()

    for epoch in range(config.max_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Move data to device (GPU if available)
            inputs = inputs.to(config.device)
            targets = targets.to(config.device)

            # Forward pass
            logits, loss = model(inputs, targets)

            # Backward pass
            optimizer.zero_grad()  # Clear old gradients
            loss.backward()        # Compute gradients

            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters
            optimizer.step()
            scheduler.step()

            # Accumulate loss for logging
            epoch_loss += loss.item()
            num_batches += 1

            # Log progress every 10 batches
            if batch_idx % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1}/{config.max_epochs}, "
                      f"Batch {batch_idx}/{len(dataloader)}, "
                      f"Loss: {loss.item():.4f}, "
                      f"LR: {current_lr:.6f}")

        # End of epoch logging
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'config': config.__dict__
        }
        torch.save(checkpoint, f'go_model_epoch_{epoch+1}.pt')
        print(f"Checkpoint saved: go_model_epoch_{epoch+1}.pt")

    print("Training completed!")
    return model, tokenizer

def generate_sample(model, tokenizer, prompt="package main", max_length=200):
    """
    Test the trained model by generating Go code from a prompt.
    This is where you see if your model actually learned anything useful.
    """
    model.eval()  # Switch to evaluation mode

    # Encode prompt
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    input_ids = input_ids.to(next(model.parameters()).device)

    # Generate tokens
    with torch.no_grad():  # No gradient computation needed
        for _ in range(max_length):
            # Forward pass
            logits, _ = model(input_ids)

            # Get logits for last token
            next_token_logits = logits[0, -1, :]

            # Sample next token (using temperature for randomness)
            temperature = 0.8
            next_token_logits = next_token_logits / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            # Stop if we hit end of sequence token
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode generated tokens
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    Run this script to train a tiny Go code generation model.

    Expected behavior:
    - Downloads GPT-2 tokenizer (first run only)
    - Creates sample Go training data
    - Trains a small transformer for a few epochs
    - Saves model checkpoints
    - Generates sample code to test the model

    This is your "hello world" - once this works, you can scale up
    with real data and bigger models.
    """

    # Train the model
    model, tokenizer = train_model()

    # Test generation
    print("\n" + "="*50)
    print("TESTING CODE GENERATION")
    print("="*50)

    prompts = [
        "package main\n\nfunc main() {",
        "func processError(err error)",
        "type Server struct {"
    ]

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 30)
        generated = generate_sample(model, tokenizer, prompt)
        print(generated)
        print("\n")