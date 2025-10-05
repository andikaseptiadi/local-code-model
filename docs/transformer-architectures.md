# Transformer Architecture Variants: A Complete Guide

This guide provides an in-depth comparison of the three major transformer architecture families implemented in this codebase: GPT (decoder-only), BERT (encoder-only), and T5 (encoder-decoder). Understanding these architectural differences is crucial for choosing the right model for your task.

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Architecture Deep Dive](#architecture-deep-dive)
3. [Attention Mechanisms](#attention-mechanisms)
4. [Training Objectives](#training-objectives)
5. [Use Cases](#use-cases)
6. [Implementation Details](#implementation-details)
7. [Performance Considerations](#performance-considerations)

## Quick Reference

### Architecture Comparison Matrix

| Feature | GPT | BERT | T5 |
|---------|-----|------|-----|
| **Type** | Decoder-only | Encoder-only | Encoder-Decoder |
| **Attention** | Causal (unidirectional) | Bidirectional | Both + Cross-attention |
| **Training Objective** | Next token prediction | Masked Language Modeling (MLM) | Span corruption |
| **Position Encoding** | Learned absolute or RoPE | Learned absolute | Relative position biases |
| **Primary Use Cases** | Text generation, completion | Classification, understanding | Translation, summarization, seq2seq |
| **Can Generate?** | ✅ Yes (autoregressive) | ❌ No | ✅ Yes (via decoder) |
| **Sees Future Context?** | ❌ No (causal mask) | ✅ Yes (bidirectional) | Encoder: ✅, Decoder: ❌ |
| **Implementation Files** | `transformer.go` | `transformer_bert.go` | `transformer_t5.go` |

## Architecture Deep Dive

### GPT: Decoder-Only Architecture

**Philosophy**: Predict the next token given all previous tokens.

```
Input:  "The cat sat on"
Mask:   Can see: "The", "The cat", "The cat sat", ...
Output: "the" (next token)
```

**Key Components**:
- **Causal Self-Attention**: Token at position `i` can only attend to positions `≤ i`
- **Autoregressive Generation**: Each token depends on all previous tokens
- **Position Embeddings**: Usually learned absolute positions or RoPE (Rotary Position Embeddings)

**Architecture Flow**:
```
Input Tokens → Token Embeddings → Position Embeddings →
  ↓
Transformer Blocks (N layers):
  ├─ Causal Multi-Head Self-Attention
  ├─ Add & LayerNorm
  ├─ Feed-Forward Network (FFN)
  └─ Add & LayerNorm
  ↓
Language Modeling Head → Next Token Logits
```

**Strengths**:
- Excellent at text generation
- Natural fit for autoregressive tasks
- Simple, unified architecture
- Scales well (GPT-3, GPT-4 use this architecture)

**Weaknesses**:
- Cannot see future context
- Not ideal for understanding tasks requiring bidirectional context
- Each position sees less context than BERT

**Best For**:
- Text generation
- Code completion
- Dialogue systems
- Few-shot learning (prompting)

### BERT: Encoder-Only Architecture

**Philosophy**: Understand text by looking at context from both directions simultaneously.

```
Input:  "The [MASK] sat on the mat"
Mask:   Can see: All tokens (bidirectional)
Output: "cat" (predict masked token)
```

**Key Components**:
- **Bidirectional Self-Attention**: Every token can attend to all other tokens
- **Masked Language Modeling (MLM)**: Randomly mask 15% of tokens, predict them
- **Special Tokens**: [CLS], [SEP], [MASK], [PAD] for different purposes
- **Token Type Embeddings**: Distinguish between sentence segments

**Architecture Flow**:
```
Input Tokens → Token Embeddings → Position Embeddings → Token Type Embeddings →
  ↓
Transformer Blocks (N layers):
  ├─ Bidirectional Multi-Head Self-Attention
  ├─ Add & LayerNorm
  ├─ Feed-Forward Network (FFN)
  └─ Add & LayerNorm
  ↓
[CLS] Token → Classification Head (for sentence-level tasks)
All Tokens → MLM Head (for token-level predictions)
```

**Masking Strategy (80/10/10 split)**:
- 80%: Replace with [MASK] token
- 10%: Replace with random token
- 10%: Keep original token

This prevents the model from relying solely on the [MASK] token presence.

**Strengths**:
- Deep bidirectional context understanding
- Excellent for classification and understanding tasks
- Strong sentence-level representations ([CLS] token)
- Effective for token-level tasks (NER, POS tagging)

**Weaknesses**:
- Cannot generate text autoregressively
- Mismatch between pretraining ([MASK]) and inference (no [MASK])
- Not suitable for sequence-to-sequence tasks

**Best For**:
- Text classification (sentiment, topic, etc.)
- Question answering (extractive)
- Named Entity Recognition (NER)
- Semantic similarity
- Sentence embeddings

### T5: Encoder-Decoder Architecture

**Philosophy**: Treat all NLP tasks as text-to-text transformations.

```
Input:  "translate English to French: Hello"
Encoder: Bidirectional understanding of input
Decoder: Autoregressive generation with encoder context
Output: "Bonjour"
```

**Key Components**:
- **Encoder**: Bidirectional self-attention (like BERT encoder)
- **Decoder**: Causal self-attention + **cross-attention** to encoder
- **Cross-Attention**: Decoder queries attend to encoder outputs
- **Span Corruption**: Mask contiguous spans (not individual tokens)
- **Relative Position Biases**: No absolute positions, only relative

**Architecture Flow**:
```
INPUT SIDE (Encoder):
Input Tokens → Token Embeddings → Relative Position Biases →
  ↓
Encoder Blocks (N layers):
  ├─ Bidirectional Multi-Head Self-Attention
  ├─ Add & LayerNorm
  ├─ Feed-Forward Network (FFN)
  └─ Add & LayerNorm
  ↓
Encoder Outputs (memory)

OUTPUT SIDE (Decoder):
Output Tokens → Token Embeddings (shared with encoder) → Relative Position Biases →
  ↓
Decoder Blocks (N layers):
  ├─ Causal Multi-Head Self-Attention (look at previous decoder tokens)
  ├─ Add & LayerNorm
  ├─ Cross-Attention (Q: decoder, K&V: encoder outputs)
  ├─ Add & LayerNorm
  ├─ Feed-Forward Network (FFN)
  └─ Add & LayerNorm
  ↓
Language Modeling Head → Next Token Logits
```

**Span Corruption Training**:
```
Original:  [token1, token2, token3, token4, token5, token6]
Corrupted: [token1, token2, <extra_id_0>, token5, <extra_id_1>]
Target:    [<extra_id_0>, token3, token4, <extra_id_1>, token6, <eos>]
```

Contiguous spans are more efficient than BERT's individual token masking.

**Cross-Attention Explained**:
```
Decoder Token: "Bonjour" (current generation)
  ↓ (Queries)
Cross-Attention
  ↑ (Keys & Values)
Encoder Tokens: "translate English to French: Hello"

Result: Decoder attends to relevant encoder context when generating
```

**Strengths**:
- Unified text-to-text framework (all tasks have same format)
- Both understanding (encoder) and generation (decoder) capabilities
- Cross-attention allows decoder to focus on relevant input parts
- Relative position biases generalize better to different sequence lengths
- More efficient training objective (span corruption vs token masking)

**Weaknesses**:
- Larger model (encoder + decoder)
- More complex than single-stack architectures
- Slower inference than GPT (two-pass: encode then decode)

**Best For**:
- Machine translation
- Text summarization
- Question answering (abstractive)
- Text-to-text transformations
- Any sequence-to-sequence task

## Attention Mechanisms

### 1. Causal Self-Attention (GPT)

**Attention Mask**:
```
     t1   t2   t3   t4
t1 [ ✓    ✗    ✗    ✗  ]
t2 [ ✓    ✓    ✗    ✗  ]
t3 [ ✓    ✓    ✓    ✗  ]
t4 [ ✓    ✓    ✓    ✓  ]
```

Token `i` can attend to tokens `1..i` (including itself, excluding future tokens).

**Computation**:
```python
scores = Q @ K^T / sqrt(d_k)
# Apply causal mask: set future positions to -inf
scores = scores.masked_fill(causal_mask, -inf)
attention_weights = softmax(scores)
output = attention_weights @ V
```

**Why?** Ensures model learns causal dependencies for autoregressive generation.

### 2. Bidirectional Self-Attention (BERT)

**Attention Mask**:
```
     t1   t2   t3   t4
t1 [ ✓    ✓    ✓    ✓  ]
t2 [ ✓    ✓    ✓    ✓  ]
t3 [ ✓    ✓    ✓    ✓  ]
t4 [ ✓    ✓    ✓    ✓  ]
```

Every token can attend to every other token (no restrictions).

**Computation**:
```python
scores = Q @ K^T / sqrt(d_k)
# No causal mask, all positions visible
attention_weights = softmax(scores)
output = attention_weights @ V
```

**Why?** Maximizes context understanding by seeing all tokens simultaneously.

### 3. Cross-Attention (T5 Decoder)

**Mechanism**:
```
Queries (Q):     From decoder hidden states
Keys (K):        From encoder outputs
Values (V):      From encoder outputs

Result: Decoder attends to encoder's understanding of input
```

**Attention Pattern**:
```
Decoder tokens can attend to ALL encoder tokens:

Decoder        Encoder Input
  t1    →    [ e1  e2  e3  e4 ]  ✓ all visible
  t2    →    [ e1  e2  e3  e4 ]  ✓ all visible
  t3    →    [ e1  e2  e3  e4 ]  ✓ all visible
```

**Computation**:
```python
# Q from decoder, K&V from encoder
Q = decoder_hidden_states @ W_q
K = encoder_outputs @ W_k
V = encoder_outputs @ W_v

scores = Q @ K^T / sqrt(d_k)
# No mask needed, decoder can see all encoder tokens
attention_weights = softmax(scores)
output = attention_weights @ V
```

**Why?** Allows decoder to dynamically focus on relevant parts of the input when generating each output token.

## Training Objectives

### GPT: Next Token Prediction

**Objective**: Predict the next token given previous tokens.

```
Training Example:
Input:  ["The", "cat", "sat", "on"]
Target: "the"

Loss: CrossEntropy(predicted_logits, target_token_id)
```

**Training Process**:
1. Feed tokens 1..N-1 to model
2. Model predicts token at each position
3. Compute loss for all predictions simultaneously (teacher forcing)
4. Backpropagate and update weights

**Advantages**:
- Simple, natural objective
- Directly aligned with generation task
- No special preprocessing needed

### BERT: Masked Language Modeling (MLM)

**Objective**: Predict randomly masked tokens using bidirectional context.

```
Training Example:
Original: ["The", "cat", "sat", "on", "the", "mat"]
Masked:   ["The", "[MASK]", "sat", "on", "[MASK]", "mat"]
Targets:  [       "cat",              "the"             ]

Loss: CrossEntropy for masked positions only
```

**Masking Strategy** (for each selected token):
- 80% chance: Replace with [MASK]
- 10% chance: Replace with random token
- 10% chance: Keep original token

**Training Process**:
1. Randomly select 15% of tokens for masking
2. Apply 80/10/10 masking strategy
3. Feed masked sequence to BERT
4. Predict original tokens at masked positions
5. Compute loss only for masked tokens

**Advantages**:
- Learns deep bidirectional representations
- Forces model to use context from both directions
- Effective for understanding tasks

**Disadvantages**:
- Mismatch between pretraining ([MASK]) and finetuning (no [MASK])
- Only 15% of tokens provide training signal per example

### T5: Span Corruption

**Objective**: Predict masked contiguous spans using encoder-decoder.

```
Training Example:
Original:  ["I", "love", "natural", "language", "processing", "tasks"]
Corrupted: ["I", "love", "<extra_id_0>", "processing", "<extra_id_1>"]
Target:    ["<extra_id_0>", "natural", "language", "<extra_id_1>", "tasks", "<eos>"]

Encoder Input:  Corrupted sequence
Decoder Output: Sentinel tokens + corrupted spans in order
```

**Why Span Corruption?**
1. **More Efficient**: Multiple consecutive tokens in one span (higher training signal per example)
2. **More Realistic**: Real-world understanding often involves phrases, not individual words
3. **Flexible Length**: Decoder can predict variable-length spans

**Training Process**:
1. Sample contiguous spans to corrupt (default: 15% of tokens)
2. Replace each span with unique sentinel token (<extra_id_0>, <extra_id_1>, etc.)
3. Encoder processes corrupted input (bidirectional)
4. Decoder autoregressively predicts: sentinel + span tokens
5. Compute cross-entropy loss on decoder predictions

**Text-to-Text Framework**:
All tasks formatted as text-to-text:
```
Translation:     "translate English to French: Hello" → "Bonjour"
Summarization:   "summarize: [long text]" → "[summary]"
Classification:  "cola sentence: The book is on table" → "unacceptable"
```

**Advantages**:
- Unified framework for all tasks
- More efficient than MLM (span-level vs token-level)
- Better generalization across tasks
- Relative positions work better for varying lengths

## Use Cases

### When to Use GPT (Decoder-Only)

✅ **Ideal For**:
- **Text Generation**: Stories, articles, creative writing
  - Autoregressive nature perfect for open-ended generation
  - Each token conditions on full history

- **Code Generation**: GitHub Copilot, code completion
  - Programming is sequential and autoregressive
  - Next token prediction aligns with how code is written

- **Dialogue/Chat**: Conversational AI, chatbots
  - Natural conversation flow is sequential
  - Can maintain context across turns

- **Few-Shot Learning**: In-context learning via prompting
  - Can learn tasks from examples in the prompt
  - No finetuning required for many tasks

❌ **Not Ideal For**:
- Classification tasks (no [CLS] token, though can prompt for it)
- Tasks requiring bidirectional context (sentiment analysis)
- Extractive QA (cannot see full context when predicting)

**Example Code**:
```go
// See transformer.go for GPT implementation
config := NewConfig()
model := NewModel(config)

// Generate text autoregressively
tokens := []int{10, 20, 30}  // Starting prompt
for i := 0; i < maxLength; i++ {
    logits := model.Forward(tokens)
    nextToken := sample(logits)
    tokens = append(tokens, nextToken)
}
```

### When to Use BERT (Encoder-Only)

✅ **Ideal For**:
- **Text Classification**: Sentiment analysis, topic classification, spam detection
  - [CLS] token provides sentence-level representation
  - Bidirectional context for deep understanding

- **Named Entity Recognition (NER)**: Identifying entities in text
  - Each token embedding contains full sentence context
  - Effective for token-level classification

- **Question Answering (Extractive)**: Finding answer spans in documents
  - Can see both question and passage bidirectionally
  - Excellent for span prediction

- **Semantic Similarity**: Sentence embeddings, paraphrase detection
  - Rich bidirectional representations
  - [CLS] token captures sentence meaning

- **Token-Level Tasks**: POS tagging, dependency parsing
  - Every token sees full sentence context

❌ **Not Ideal For**:
- Text generation (no autoregressive mechanism)
- Translation (no encoder-decoder structure)
- Tasks requiring sequential generation

**Example Code**:
```go
// See transformer_bert.go for BERT implementation
config := NewBERTConfig()
model := NewBERTForMaskedLM(config)

// Text classification (using [CLS] token)
tokens := []int{101, 10, 20, 30, 102}  // [CLS] ... [SEP]
hidden := model.Forward(tokens)
clsEmbedding := hidden[0]  // First token = [CLS]
logits := classificationHead.Forward(clsEmbedding)

// Token-level task (NER)
for i, token := range hidden {
    entityLabel := nerHead.Forward(token)
}
```

### When to Use T5 (Encoder-Decoder)

✅ **Ideal For**:
- **Machine Translation**: English to French, etc.
  - Encoder understands source language bidirectionally
  - Decoder generates target language autoregressively
  - Cross-attention connects source and target

- **Abstractive Summarization**: Generating summaries (not extracting)
  - Encoder understands full document
  - Decoder generates concise summary

- **Question Answering (Abstractive)**: Generating answers
  - Encoder processes question + context
  - Decoder generates answer (not just extracting spans)

- **Text-to-Text Tasks**: Any input-output transformation
  - Paraphrasing, style transfer, etc.
  - Unified framework for diverse tasks

- **Data-to-Text**: Generating text from structured data
  - Encoder processes structured input
  - Decoder generates fluent text

✅ **Can Also Do (Less Efficient)**:
- Classification (via text-to-text: input → "positive"/"negative")
- Understanding tasks (encoder half is similar to BERT)

❌ **Not Ideal For**:
- Simple classification (BERT is faster)
- Open-ended generation (GPT is simpler)
- Real-time applications (slower due to two-pass)

**Example Code**:
```go
// See transformer_t5.go for T5 implementation
config := NewT5Config()
model := NewT5ForConditionalGeneration(config)

// Translation example
inputTokens := []int{10, 11, 12, 13, 14}      // "translate English to French: Hello"
decoderInputs := []int{0}                     // Start with <pad> or <bos>

// Generate translation autoregressively
for i := 0; i < maxLength; i++ {
    logits := model.Forward(inputTokens, decoderInputs)
    nextToken := sample(logits[len(decoderInputs)-1])  // Last position
    if nextToken == eosTokenID { break }
    decoderInputs = append(decoderInputs, nextToken)
}
```

## Implementation Details

### Code Organization

```
transformer.go          → GPT implementation (decoder-only)
transformer_bert.go     → BERT implementation (encoder-only)
transformer_t5.go       → T5 implementation (encoder-decoder)

transformer_test.go     → GPT tests
transformer_bert_test.go → BERT tests
transformer_t5_test.go  → T5 tests
```

### Configuration Comparison

```go
// GPT Config
type Config struct {
    VocabSize   int   // 50257 (GPT-2)
    ContextSize int   // 1024 (max sequence length)
    EmbedDim    int   // 768
    NumHeads    int   // 12
    NumLayers   int   // 12
    // ... includes FFN, dropout, etc.
}

// BERT Config
type BERTConfig struct {
    VocabSize     int   // 30522 (BERT vocab)
    HiddenDim     int   // 768
    NumLayers     int   // 12
    NumHeads      int   // 12
    MaxSeqLen     int   // 512
    CLSTokenID    int   // 101 ([CLS])
    SEPTokenID    int   // 102 ([SEP])
    MASKTokenID   int   // 103 ([MASK])
    PADTokenID    int   // 0   ([PAD])
    // ...includes token type embeddings
}

// T5 Config
type T5Config struct {
    VocabSize       int   // 32128 (includes sentinel tokens)
    HiddenDim       int   // 768
    NumLayers       int   // 12 (encoder) + 12 (decoder)
    NumHeads        int   // 12
    MaxSeqLen       int   // 512
    SpanLength      int   // 3 (avg span for corruption)
    CorruptRate     float64 // 0.15 (15% corruption)
    SentinelStartID int   // 32000 (start of <extra_id_X>)
    // ... no absolute positions
}
```

### Memory and Compute Comparison

| Architecture | Parameters (Base) | Training Memory | Inference Speed | Generation Quality |
|--------------|-------------------|-----------------|-----------------|-------------------|
| GPT          | ~110M (12 layers) | Moderate | Fast (single pass) | Excellent for generation |
| BERT         | ~110M (12 layers) | Moderate | Fast (single pass) | N/A (no generation) |
| T5           | ~220M (12+12 layers) | High (encoder + decoder) | Slower (two passes) | Excellent for seq2seq |

**Inference Comparison**:
- **GPT**: One forward pass per token (autoregressive)
- **BERT**: One forward pass per input (parallel)
- **T5**: Encoder pass (once) + decoder pass (per output token)

### Positional Encoding Differences

**GPT**: Learned absolute positions
```go
posEmbeddings := NewTensor(config.ContextSize, config.EmbedDim)
// Add position i to token i
embeddings = tokenEmbeddings + posEmbeddings[position]
```

**BERT**: Learned absolute positions (same as GPT)
```go
posEmbeddings := NewTensor(config.MaxSeqLen, config.HiddenDim)
embeddings = tokenEmbeddings + posEmbeddings[position] + tokenTypeEmbeddings
```

**T5**: Relative position biases
```go
// No position embeddings added to tokens
// Instead, attention has position-dependent bias:
relativePosBias := NewTensor(config.NumHeads, config.MaxSeqLen, config.MaxSeqLen)
scores = (Q @ K^T) / sqrt(d_k) + relativePosBias[head][i][j]
```

Why relative positions?
- Better generalization to different sequence lengths
- Captures relative distances (more meaningful than absolute positions)
- Used in modern transformers (T5, PaLM, etc.)

## Performance Considerations

### Training Efficiency

**Sample Efficiency**:
1. **BERT** (Best): 15% of tokens provide signal per example
2. **T5** (Better): Span-level corruption is more efficient than token-level
3. **GPT** (Good): All tokens provide training signal (predict next at every position)

**Compute Efficiency** (FLOPs per training step):
1. **GPT**: Lowest (single attention stack)
2. **BERT**: Similar to GPT (single attention stack)
3. **T5**: Highest (encoder + decoder + cross-attention)

### Inference Latency

**For Understanding Tasks** (e.g., classification):
- **BERT**: Fastest (one pass, parallel processing)
- **T5**: Slower (encoder pass, then task-specific decoding)
- **GPT**: Moderate (can be adapted with prompting)

**For Generation Tasks**:
- **GPT**: Moderate (one decoder, autoregressive)
- **T5**: Slowest (encoder once + decoder autoregressive)
- **BERT**: N/A (cannot generate)

### Memory Footprint

**Model Size** (assuming same hidden dim and layer count):
- **GPT**: Base size (single stack)
- **BERT**: ~Same as GPT (similar architecture)
- **T5**: ~2× GPT (encoder + decoder stacks)

**Activation Memory** (during training):
- **GPT**: Moderate (one attention stack)
- **BERT**: Moderate (one attention stack)
- **T5**: High (two stacks + cross-attention activations)

**KV Cache** (for generation):
- **GPT**: Requires KV cache for fast generation
- **T5**: Requires KV cache for both encoder (cached once) and decoder (per step)
- **BERT**: N/A (no generation)

## Conclusion

### Quick Selection Guide

**Choose GPT if**:
- Primary task is text generation
- Need autoregressive modeling
- Want simpler architecture
- Few-shot learning via prompting is important

**Choose BERT if**:
- Primary task is text understanding/classification
- Need bidirectional context
- No text generation required
- Best performance on understanding tasks

**Choose T5 if**:
- Primary task is sequence-to-sequence (translation, summarization)
- Want unified text-to-text framework
- Need both understanding AND generation
- Can afford larger model and slower inference

### Modern Trends

**Current State (2024-2025)**:
- **Decoder-only (GPT-style)** dominates large-scale models (GPT-4, PaLM, LLaMA, Claude)
  - Can handle all tasks via prompting/in-context learning
  - Simpler to scale to billions/trillions of parameters
  - Emergent abilities at scale

- **Encoder-only (BERT-style)** still excellent for task-specific finetuning
  - More efficient for classification/understanding when no generation needed
  - Widely used in production for specific tasks

- **Encoder-decoder (T5-style)** remains strong for specific seq2seq tasks
  - Best for translation, summarization when optimized for those tasks
  - mT5, FLAN-T5 show continued relevance

### Further Reading

- **Original Papers**:
  - GPT: "Improving Language Understanding by Generative Pre-Training" (Radford et al., 2018)
  - GPT-2: "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
  - BERT: "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
  - T5: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (Raffel et al., 2019)

- **Implementation Guides**:
  - See test files for comprehensive examples
  - `transformer_test.go`, `transformer_bert_test.go`, `transformer_t5_test.go`
  - Each includes educational comparisons and workflow demonstrations

- **Related Documentation**:
  - `docs/attention-mechanism.md`: Deep dive into attention
  - `docs/backpropagation.md`: Training dynamics
  - `docs/training-dynamics.md`: Loss curves and optimization

---

*This document reflects the implementations in this codebase as of Phase 3.2. All architectures follow the same 30-50% documentation standards with educational focus.*
