# Transformer Architecture Comparison: GPT vs BERT vs T5

This notebook demonstrates the practical differences between the three major transformer architectures on a concrete text processing task.

## Overview

We'll use a simple example to show how each architecture approaches the same problem differently:

**Task**: Process the sentence "The cat sat on the mat" and understand/generate related text.

## 1. Setup

```go
import (
    "fmt"
    "math/rand"
)

// Simple vocabulary for our examples
var vocab = map[string]int{
    "[PAD]":  0,
    "[CLS]":  1,
    "[SEP]":  2,
    "[MASK]": 3,
    "the":    4,
    "cat":    5,
    "sat":    6,
    "on":     7,
    "mat":    8,
    "dog":    9,
    "stood":  10,
    "floor":  11,
}

var reverseVocab = map[int]string{
    0: "[PAD]", 1: "[CLS]", 2: "[SEP]", 3: "[MASK]",
    4: "the", 5: "cat", 6: "sat", 7: "on", 8: "mat",
    9: "dog", 10: "stood", 11: "floor",
}

// Example sentence: "The cat sat on the mat"
var exampleTokens = []int{4, 5, 6, 7, 4, 8}
```

## 2. GPT: Decoder-Only Architecture

**Purpose**: Text generation via autoregressive prediction
**Attention**: Causal (can only see previous tokens)
**Training**: Next token prediction

### How GPT Works

```go
func demonstrateGPT() {
    fmt.Println("=== GPT: Decoder-Only (Causal Attention) ===\n")

    // GPT configuration
    config := &TransformerConfig{
        VocabSize:  len(vocab),
        HiddenDim:  256,
        NumLayers:  6,
        NumHeads:   8,
        MaxSeqLen:  128,
    }

    // Create GPT model
    model := NewTransformer(config)

    // Autoregressive generation
    fmt.Println("Task: Continue the sentence")
    fmt.Println("Input:  'The cat'")
    fmt.Println("Output: 'The cat sat on the mat'\n")

    // Start with initial tokens: [the, cat]
    generated := []int{4, 5}

    fmt.Println("Generation process:")
    for step := 0; step < 4; step++ {
        // Forward pass (GPT sees only previous tokens)
        logits := model.Forward(generated)

        // Get last token's prediction
        nextToken := argmax(logits[len(generated)-1])
        generated = append(generated, nextToken)

        fmt.Printf("  Step %d: %v → predict '%s'\n",
            step+1, tokensToWords(generated[:len(generated)-1]),
            reverseVocab[nextToken])
    }

    fmt.Println("\nKey characteristics:")
    fmt.Println("✓ Causal attention (can't see future)")
    fmt.Println("✓ Excellent for text generation")
    fmt.Println("✓ One directional (left-to-right)")
    fmt.Println("✗ Can't use future context for understanding\n")
}
```

### Attention Pattern Visualization

```
Token positions:  0    1    2    3    4
Input:           the  cat  sat  on  mat

Attention allowed (✓) or masked (✗):
                  0    1    2    3    4
Token 0 (the):    ✓    ✗    ✗    ✗    ✗   ← Can only see itself
Token 1 (cat):    ✓    ✓    ✗    ✗    ✗   ← Can see "the cat"
Token 2 (sat):    ✓    ✓    ✓    ✗    ✗   ← Can see "the cat sat"
Token 3 (on):     ✓    ✓    ✓    ✓    ✗   ← Can see "the cat sat on"
Token 4 (mat):    ✓    ✓    ✓    ✓    ✓   ← Can see all previous

This is CAUSAL (autoregressive) attention.
```

## 3. BERT: Encoder-Only Architecture

**Purpose**: Understanding and representation learning
**Attention**: Bidirectional (can see all tokens)
**Training**: Masked language modeling (MLM)

### How BERT Works

```go
func demonstrateBERT() {
    fmt.Println("=== BERT: Encoder-Only (Bidirectional Attention) ===\n")

    // BERT configuration
    config := NewBERTConfig()
    config.VocabSize = len(vocab)

    // Create BERT model
    model := NewBERTForMaskedLM(config)

    // Masked language modeling
    fmt.Println("Task: Fill in the blank")
    fmt.Println("Input:  'The cat [MASK] on the mat'")
    fmt.Println("Output: 'The cat sat on the mat'\n")

    // Input with mask: [CLS] the cat [MASK] on the mat [SEP]
    input := []int{1, 4, 5, 3, 7, 4, 8, 2}
    maskedPosition := 3

    // Forward pass (BERT sees all tokens bidirectionally)
    logits := model.Forward(input)

    // Predict masked token
    predicted := argmax(logits[maskedPosition])

    fmt.Println("Prediction process:")
    fmt.Printf("  Context before mask: '%s'\n", tokensToWords(input[1:maskedPosition]))
    fmt.Printf("  Context after mask:  '%s'\n", tokensToWords(input[maskedPosition+1:len(input)-1]))
    fmt.Printf("  Predicted token:     '%s'\n", reverseVocab[predicted])

    fmt.Println("\nKey characteristics:")
    fmt.Println("✓ Bidirectional attention (sees full context)")
    fmt.Println("✓ Excellent for understanding/classification")
    fmt.Println("✓ Can use both past and future context")
    fmt.Println("✗ Cannot generate text autoregressively\n")
}
```

### Attention Pattern Visualization

```
Token positions:    0      1    2      3    4    5      6
Input:           [CLS]  the  cat  [MASK]  on  mat  [SEP]

Attention allowed (✓) for all tokens:
                    0    1    2    3    4    5    6
Token 0 ([CLS]):    ✓    ✓    ✓    ✓    ✓    ✓    ✓   ← Sees everything
Token 1 (the):      ✓    ✓    ✓    ✓    ✓    ✓    ✓   ← Sees everything
Token 2 (cat):      ✓    ✓    ✓    ✓    ✓    ✓    ✓   ← Sees everything
Token 3 ([MASK]):   ✓    ✓    ✓    ✓    ✓    ✓    ✓   ← Sees everything (including future!)
Token 4 (on):       ✓    ✓    ✓    ✓    ✓    ✓    ✓   ← Sees everything
Token 5 (mat):      ✓    ✓    ✓    ✓    ✓    ✓    ✓   ← Sees everything
Token 6 ([SEP]):    ✓    ✓    ✓    ✓    ✓    ✓    ✓   ← Sees everything

This is BIDIRECTIONAL attention.
When predicting [MASK], BERT uses both "the cat" (past) and "on mat" (future).
```

## 4. T5: Encoder-Decoder Architecture

**Purpose**: Sequence-to-sequence tasks (translation, summarization)
**Attention**: Bidirectional (encoder) + Causal + Cross-attention (decoder)
**Training**: Span corruption

### How T5 Works

```go
func demonstrateT5() {
    fmt.Println("=== T5: Encoder-Decoder (Bidirectional + Cross-Attention) ===\n")

    // T5 configuration
    config := NewT5Config()
    config.VocabSize = len(vocab)
    config.SentinelStartID = len(vocab) // Sentinels start after regular vocab

    // Create T5 model
    model := NewT5ForConditionalGeneration(config)

    // Span corruption task
    fmt.Println("Task: Text-to-text (span corruption)")
    fmt.Println("Input:  'The cat <X> the mat'")
    fmt.Println("Output: '<X> sat on <EOS>'\n")

    // Encoder input: corrupted sentence
    // Original: the cat sat on the mat
    // Corrupted: the cat <X> the mat (span "sat on" replaced with sentinel <X>)
    encoderInput := []int{4, 5, 100, 4, 8} // 100 = sentinel token

    // Decoder input: sentinel + corrupted span
    // Target: <X> sat on <EOS>
    decoderInput := []int{0, 100} // 0 = decoder start, 100 = sentinel
    decoderTarget := []int{100, 6, 7, 1} // <X> sat on <EOS>

    fmt.Println("Encoder-Decoder process:")

    // Step 1: Encoder processes corrupted input (bidirectional)
    encoderOutputs := model.Encoder.Forward(encoderInput)
    fmt.Printf("  1. Encoder reads: '%s'\n", tokensToWords(encoderInput))
    fmt.Printf("     (bidirectional attention, sees full corrupted context)\n\n")

    // Step 2: Decoder generates target autoregressively (causal + cross-attention)
    fmt.Println("  2. Decoder generates (with cross-attention to encoder):")
    generated := []int{0} // Start token
    for step := 0; step < 4; step++ {
        // Decoder sees its own previous tokens (causal) + encoder outputs (cross-attention)
        logits := model.Decoder.Forward(generated, encoderOutputs)
        nextToken := argmax(logits[len(generated)-1])
        generated = append(generated, nextToken)

        fmt.Printf("     Step %d: %v → '%s'\n",
            step+1, tokensToWords(generated[:len(generated)-1]),
            reverseVocab[nextToken])
    }

    fmt.Println("\nKey characteristics:")
    fmt.Println("✓ Encoder: bidirectional (understands input fully)")
    fmt.Println("✓ Decoder: causal + cross-attention (generates while attending to input)")
    fmt.Println("✓ Excellent for seq2seq tasks")
    fmt.Println("✓ Most flexible architecture\n")
}
```

### Attention Pattern Visualization

```
ENCODER (Bidirectional):
Token positions:  0    1      2    3    4
Encoder input:   the  cat   <X>  the  mat

All encoder tokens see each other (bidirectional):
                  0    1    2    3    4
Token 0 (the):    ✓    ✓    ✓    ✓    ✓
Token 1 (cat):    ✓    ✓    ✓    ✓    ✓
Token 2 (<X>):    ✓    ✓    ✓    ✓    ✓   ← Sentinel sees full context
Token 3 (the):    ✓    ✓    ✓    ✓    ✓
Token 4 (mat):    ✓    ✓    ✓    ✓    ✓

DECODER (Causal Self-Attention + Cross-Attention):
Decoder input:   <S>  <X>  sat   on

Self-attention (causal):
                  0    1    2    3
Token 0 (<S>):    ✓    ✗    ✗    ✗   ← Sees only itself
Token 1 (<X>):    ✓    ✓    ✗    ✗   ← Sees <S> <X>
Token 2 (sat):    ✓    ✓    ✓    ✗   ← Sees <S> <X> sat
Token 3 (on):     ✓    ✓    ✓    ✓   ← Sees <S> <X> sat on

Cross-attention (to encoder):
Each decoder token attends to ALL encoder tokens:
                       Encoder tokens →
                  0    1    2    3    4
Decoder 0 (<S>):  ✓    ✓    ✓    ✓    ✓   ← Attends to all encoder
Decoder 1 (<X>):  ✓    ✓    ✓    ✓    ✓   ← Attends to all encoder
Decoder 2 (sat):  ✓    ✓    ✓    ✓    ✓   ← Attends to all encoder
Decoder 3 (on):   ✓    ✓    ✓    ✓    ✓   ← Attends to all encoder

This is ENCODER-DECODER with CROSS-ATTENTION.
```

## 5. Side-by-Side Comparison

### Same Task, Different Approaches

Let's see how each architecture would approach the task: "Understand the sentence 'The cat sat on the mat'"

```go
func compareArchitectures() {
    fmt.Println("=== Task: Process 'The cat sat on the mat' ===\n")

    tokens := []int{4, 5, 6, 7, 4, 8} // the cat sat on the mat

    // GPT approach
    fmt.Println("1. GPT (Decoder-Only):")
    fmt.Println("   Use case: Generate continuation")
    fmt.Println("   Input:  [the, cat]")
    fmt.Println("   Output: [sat, on, the, mat] (predicted autoregressively)")
    fmt.Println("   Limitation: Each token only sees past context\n")

    // BERT approach
    fmt.Println("2. BERT (Encoder-Only):")
    fmt.Println("   Use case: Understand and classify")
    fmt.Println("   Input:  [CLS] the cat sat on the mat [SEP]")
    fmt.Println("   Output: Contextual embeddings for each token")
    fmt.Println("           → Can classify sentiment, NER, etc.")
    fmt.Println("   Advantage: Each token sees full sentence context\n")

    // T5 approach
    fmt.Println("3. T5 (Encoder-Decoder):")
    fmt.Println("   Use case: Transform text")
    fmt.Println("   Input:  'translate to French: The cat sat on the mat'")
    fmt.Println("   Encoder: Processes full English sentence (bidirectional)")
    fmt.Println("   Decoder: Generates French translation (autoregressive)")
    fmt.Println("   Output: 'Le chat était assis sur le tapis'")
    fmt.Println("   Advantage: Best of both worlds\n")
}
```

## 6. When to Use Each Architecture

### Decision Tree

```
What is your task?
│
├─ Text Generation (stories, chat, code)?
│  → Use GPT (decoder-only)
│     Examples: story continuation, chatbots, code completion
│
├─ Text Understanding (classification, extraction)?
│  → Use BERT (encoder-only)
│     Examples: sentiment analysis, NER, QA (answer selection)
│
└─ Text Transformation (translation, summarization)?
   → Use T5 (encoder-decoder)
      Examples: translation, summarization, question answering (generation)
```

### Detailed Comparison

```go
type ArchitectureComparison struct {
    Name          string
    AttentionType string
    CanGenerate   bool
    UseCase       []string
    Strength      string
    Limitation    string
}

var comparisons = []ArchitectureComparison{
    {
        Name:          "GPT",
        AttentionType: "Causal (left-to-right)",
        CanGenerate:   true,
        UseCase:       []string{"Text generation", "Chat", "Code completion"},
        Strength:      "Excellent at generating fluent, coherent text",
        Limitation:    "Cannot use future context for understanding",
    },
    {
        Name:          "BERT",
        AttentionType: "Bidirectional (sees all)",
        CanGenerate:   false,
        UseCase:       []string{"Classification", "NER", "Sentence embedding"},
        Strength:      "Best understanding of text with full context",
        Limitation:    "Cannot generate text autoregressively",
    },
    {
        Name:          "T5",
        AttentionType: "Bidirectional + Causal + Cross",
        CanGenerate:   true,
        UseCase:       []string{"Translation", "Summarization", "QA generation"},
        Strength:      "Most flexible, handles any seq2seq task",
        Limitation:    "More complex, requires more compute",
    },
}
```

## 7. Training Objectives

### GPT: Next Token Prediction

```go
// Training example
input := []int{4, 5, 6, 7, 4}      // the cat sat on the
target := []int{5, 6, 7, 4, 8}     // cat sat on the mat

// Loss: predict each next token given previous tokens
// Position 0: predict "cat" given "the"
// Position 1: predict "sat" given "the cat"
// Position 2: predict "on" given "the cat sat"
// ...
```

### BERT: Masked Language Modeling

```go
// Training example
input := []int{4, 5, 3, 7, 4, 8}   // the cat [MASK] on the mat
target := 6                         // sat

// Loss: predict masked token given bidirectional context
// Uses both "the cat" (before) and "on the mat" (after)
```

### T5: Span Corruption

```go
// Training example
encoderInput := []int{4, 5, 100, 4, 8}  // the cat <X> the mat
decoderTarget := []int{100, 6, 7, 1}     // <X> sat on <EOS>

// Loss: generate corrupted span given corrupted input
// More efficient than masking individual tokens
// Better for phrase-level understanding
```

## 8. Memory and Compute Comparison

### Memory Usage (for sequence length N)

```
GPT:  O(N × HiddenDim × NumLayers)
      - Stores KV cache: O(N × HiddenDim × NumLayers)

BERT: O(N × HiddenDim × NumLayers)
      - No KV cache needed (processes all at once)

T5:   O(N × HiddenDim × NumLayers × 2)
      - Encoder + Decoder (roughly 2× GPT/BERT)
      - Stores encoder outputs + decoder KV cache
```

### Compute (FLOPs per token)

```
Generation (per token):
  GPT:  O(HiddenDim²) × NumLayers
  BERT: Cannot generate
  T5:   O(HiddenDim²) × NumLayers × 2 (encoder + decoder)

Understanding (full sequence):
  GPT:  O(N² × HiddenDim) × NumLayers  [but only uses causal context]
  BERT: O(N² × HiddenDim) × NumLayers  [best understanding]
  T5:   O(N² × HiddenDim) × NumLayers  [encoder only]
```

## 9. Hands-On Exercise

Try modifying the code to:

1. **GPT**: Generate a longer continuation of "The cat"
2. **BERT**: Predict multiple masked tokens in a sentence
3. **T5**: Implement a simple translation task

### Exercise 1: GPT Text Generation

```go
func exerciseGPT() {
    // TODO: Implement greedy generation
    // 1. Start with initial tokens
    // 2. Loop: predict next token, append, repeat
    // 3. Stop at max length or [EOS] token

    // Bonus: Implement temperature sampling for more diverse outputs
}
```

### Exercise 2: BERT Multiple Masks

```go
func exerciseBERT() {
    // TODO: Predict multiple masks
    // Input:  "The [MASK] sat on the [MASK]"
    // Target: "The cat sat on the mat"

    // Challenge: Masks are predicted independently
    // Real BERT would need iterative refinement for dependent masks
}
```

### Exercise 3: T5 Translation

```go
func exerciseT5() {
    // TODO: Implement simple translation
    // Encoder input: "translate to French: Hello"
    // Decoder output: "Bonjour"

    // Steps:
    // 1. Encode source sentence
    // 2. Decode target sentence (with cross-attention)
    // 3. Use beam search for better quality (optional)
}
```

## 10. Key Takeaways

### Architecture Choice Matters

```
Generation Task?
  → GPT (causal attention, autoregressive)

Understanding Task?
  → BERT (bidirectional attention, MLM)

Transformation Task?
  → T5 (encoder-decoder, cross-attention)
```

### Attention Pattern is Key

- **Causal**: Token i can only see tokens 0..i (GPT decoder)
- **Bidirectional**: Token i can see all tokens (BERT encoder, T5 encoder)
- **Cross-attention**: Decoder tokens attend to all encoder tokens (T5 decoder)

### Training Objective Shapes Behavior

- **Next token prediction**: Good at generation (GPT)
- **Masked language modeling**: Good at understanding (BERT)
- **Span corruption**: Good at transformation (T5)

## 11. Further Reading

- Original papers: "Attention is All You Need", "BERT", "T5"
- Hugging Face documentation on transformers
- Our implementation: `transformer.go`, `transformer_bert.go`, `transformer_t5.go`
- Architecture deep dive: `docs/transformer-architectures.md`

---

**Next Steps**: Run this notebook's functions to see the architectures in action!
