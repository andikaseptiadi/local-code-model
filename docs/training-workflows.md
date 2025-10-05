# Training Workflows for Different Transformer Architectures

This guide demonstrates practical training workflows for GPT, BERT, and T5, showing you exactly how to train each architecture for real-world tasks.

## Table of Contents

1. [GPT Training Workflow](#gpt-training-workflow)
2. [BERT Training Workflow](#bert-training-workflow)
3. [T5 Training Workflow](#t5-training-workflow)
4. [Comparison and Selection](#comparison-and-selection)
5. [Performance Optimization](#performance-optimization)

---

## GPT Training Workflow

**Use Case**: Text generation (stories, code completion, chat)
**Training Objective**: Next token prediction
**Key Feature**: Autoregressive generation with causal attention

### 1. Data Preparation

```go
// Load and tokenize training data
func prepareGPTData(textFile string) ([][]int, error) {
    // Read raw text
    text, err := os.ReadFile(textFile)
    if err != nil {
        return nil, err
    }

    // Tokenize (character-level or BPE)
    tokenizer := NewTokenizer(TokenizerConfig{
        Type:      "char", // or "bpe"
        VocabSize: 256,    // ASCII
    })
    tokens := tokenizer.Encode(string(text))

    // Create training examples (sliding window)
    seqLen := 128
    examples := make([][]int, 0)
    for i := 0; i+seqLen+1 <= len(tokens); i += seqLen {
        examples = append(examples, tokens[i:i+seqLen+1])
    }

    return examples, nil
}
```

### 2. Model Configuration

```go
// Create GPT model
config := &TransformerConfig{
    VocabSize:    256,      // Character-level
    HiddenDim:    512,      // Embedding dimension
    NumLayers:    8,        // Transformer blocks
    NumHeads:     8,        // Attention heads
    FFNDim:       2048,     // Feed-forward dimension
    MaxSeqLen:    128,      // Context window
    DropoutRate:  0.1,      // Regularization
    LearningRate: 3e-4,     // Adam LR
}

// Optional: Enable modern features
config.UseRoPE = true         // Rotary position embeddings
config.UseSwiGLU = true       // Better activation
config.UseRMSNorm = true      // Faster normalization
config.UseExplicitMask = false // Dynamic masking is faster

model := NewTransformer(config)
```

### 3. Training Loop

```go
func trainGPT(model *Transformer, examples [][]int, config *TransformerConfig) {
    optimizer := NewAdamOptimizer(config.LearningRate)
    batchSize := 32
    numEpochs := 10

    for epoch := 0; epoch < numEpochs; epoch++ {
        totalLoss := 0.0
        numBatches := 0

        // Shuffle examples
        rand.Shuffle(len(examples), func(i, j int) {
            examples[i], examples[j] = examples[j], examples[i]
        })

        // Process batches
        for i := 0; i < len(examples); i += batchSize {
            end := i + batchSize
            if end > len(examples) {
                end = len(examples)
            }
            batch := examples[i:end]

            // Accumulate batch loss
            batchLoss := 0.0
            for _, example := range batch {
                // Input: all tokens except last
                input := example[:len(example)-1]
                // Target: all tokens except first (shifted by 1)
                target := example[1:]

                // Forward pass
                logits := model.Forward(input)

                // Compute cross-entropy loss for next token prediction
                loss := computeNextTokenLoss(logits, target)
                batchLoss += loss

                // Backward pass
                model.Backward(loss)
            }

            // Average gradients over batch
            batchLoss /= float64(len(batch))

            // Update parameters
            optimizer.Step(model)
            model.ZeroGrad()

            totalLoss += batchLoss
            numBatches++

            // Log progress
            if numBatches%100 == 0 {
                avgLoss := totalLoss / float64(numBatches)
                perplexity := math.Exp(avgLoss)
                fmt.Printf("Epoch %d, Batch %d, Loss: %.4f, Perplexity: %.2f\n",
                    epoch+1, numBatches, avgLoss, perplexity)
            }
        }

        // Epoch complete
        epochLoss := totalLoss / float64(numBatches)
        fmt.Printf("=== Epoch %d Complete: Loss %.4f, Perplexity %.2f ===\n\n",
            epoch+1, epochLoss, math.Exp(epochLoss))

        // Save checkpoint
        SaveModel(model, fmt.Sprintf("checkpoint_epoch_%d.bin", epoch+1))

        // Generate sample
        sample := generateSample(model, "The", 100)
        fmt.Printf("Sample generation: %s\n\n", sample)
    }
}

// Compute cross-entropy loss for next token prediction
func computeNextTokenLoss(logits *Tensor, targets []int) float64 {
    seqLen := len(targets)
    loss := 0.0

    for t := 0; t < seqLen; t++ {
        // Get logits for position t
        posLogits := make([]float64, logits.Shape()[1])
        for v := 0; v < len(posLogits); v++ {
            posLogits[v] = logits.At(t, v)
        }

        // Compute softmax probabilities
        probs := softmax(posLogits)

        // Cross-entropy: -log(P(correct_token))
        correctToken := targets[t]
        loss -= math.Log(probs[correctToken] + 1e-10)
    }

    return loss / float64(seqLen)
}
```

### 4. Generation

```go
// Generate text autoregressively
func generateSample(model *Transformer, prompt string, maxLen int) string {
    tokenizer := NewTokenizer(TokenizerConfig{Type: "char"})
    tokens := tokenizer.Encode(prompt)

    for len(tokens) < maxLen {
        // Forward pass
        logits := model.Forward(tokens)

        // Get logits for last position
        lastLogits := make([]float64, logits.Shape()[1])
        lastPos := len(tokens) - 1
        for v := 0; v < len(lastLogits); v++ {
            lastLogits[v] = logits.At(lastPos, v)
        }

        // Sample next token (temperature sampling)
        temperature := 0.8
        nextToken := sampleWithTemperature(lastLogits, temperature)

        // Append and continue
        tokens = append(tokens, nextToken)

        // Stop at end-of-sequence
        if nextToken == EOSToken {
            break
        }
    }

    return tokenizer.Decode(tokens)
}

func sampleWithTemperature(logits []float64, temp float64) int {
    // Scale by temperature
    scaled := make([]float64, len(logits))
    for i := range logits {
        scaled[i] = logits[i] / temp
    }

    // Softmax
    probs := softmax(scaled)

    // Sample from distribution
    r := rand.Float64()
    cumProb := 0.0
    for i, p := range probs {
        cumProb += p
        if r < cumProb {
            return i
        }
    }
    return len(probs) - 1
}
```

---

## BERT Training Workflow

**Use Case**: Text understanding (classification, NER, QA)
**Training Objective**: Masked language modeling (MLM)
**Key Feature**: Bidirectional context understanding

### 1. Data Preparation

```go
// Prepare BERT training data with masked tokens
func prepareBERTData(textFile string) ([]*BERTExample, error) {
    text, err := os.ReadFile(textFile)
    if err != nil {
        return nil, err
    }

    // Tokenize into sentences
    sentences := splitIntoSentences(string(text))
    tokenizer := NewTokenizer(TokenizerConfig{Type: "char"})

    examples := make([]*BERTExample, 0)
    for _, sentence := range sentences {
        tokens := tokenizer.Encode(sentence)

        // Add [CLS] and [SEP] tokens
        tokens = append([]int{CLSToken}, tokens...)
        tokens = append(tokens, SEPToken)

        // Apply random masking (15% of tokens)
        maskedTokens, labels := applyMasking(tokens, 0.15)

        examples = append(examples, &BERTExample{
            Tokens: maskedTokens,
            Labels: labels,
        })
    }

    return examples, nil
}

type BERTExample struct {
    Tokens []int // Input tokens (some are [MASK])
    Labels []int // Original tokens (for masked positions)
}

// Apply BERT-style masking
func applyMasking(tokens []int, maskProb float64) ([]int, []int) {
    masked := make([]int, len(tokens))
    labels := make([]int, len(tokens))
    copy(masked, tokens)

    for i := 1; i < len(tokens)-1; i++ { // Skip [CLS] and [SEP]
        if rand.Float64() < maskProb {
            labels[i] = tokens[i] // Save original token

            // BERT masking strategy:
            // 80% → [MASK]
            // 10% → random token
            // 10% → unchanged
            r := rand.Float64()
            if r < 0.8 {
                masked[i] = MASKToken
            } else if r < 0.9 {
                masked[i] = rand.Intn(VocabSize)
            }
            // else: keep original (10%)
        } else {
            labels[i] = -1 // Not masked (ignore in loss)
        }
    }

    return masked, labels
}
```

### 2. Model Configuration

```go
// Create BERT model
config := NewBERTConfig()
config.VocabSize = 256
config.HiddenDim = 512
config.NumLayers = 8
config.NumHeads = 8
config.FFNDim = 2048
config.MaxSeqLen = 128
config.DropoutRate = 0.1
config.LearningRate = 1e-4

model := NewBERTForMaskedLM(config)
```

### 3. Training Loop

```go
func trainBERT(model *BERTForMaskedLM, examples []*BERTExample, config *BERTConfig) {
    optimizer := NewAdamOptimizer(config.LearningRate)
    batchSize := 32
    numEpochs := 10

    for epoch := 0; epoch < numEpochs; epoch++ {
        totalLoss := 0.0
        totalCorrect := 0
        totalMasked := 0

        // Shuffle examples
        rand.Shuffle(len(examples), func(i, j int) {
            examples[i], examples[j] = examples[j], examples[i]
        })

        // Process batches
        numBatches := 0
        for i := 0; i < len(examples); i += batchSize {
            end := i + batchSize
            if end > len(examples) {
                end = len(examples)
            }
            batch := examples[i:end]

            batchLoss := 0.0
            batchCorrect := 0
            batchMasked := 0

            for _, example := range batch {
                // Forward pass
                logits := model.Forward(example.Tokens)

                // Compute loss only for masked positions
                loss := 0.0
                for pos, label := range example.Labels {
                    if label == -1 {
                        continue // Not masked
                    }

                    // Get logits for this position
                    posLogits := make([]float64, logits.Shape()[1])
                    for v := 0; v < len(posLogits); v++ {
                        posLogits[v] = logits.At(pos, v)
                    }

                    // Cross-entropy loss
                    probs := softmax(posLogits)
                    loss -= math.Log(probs[label] + 1e-10)

                    // Track accuracy
                    predicted := argmax(posLogits)
                    if predicted == label {
                        batchCorrect++
                    }
                    batchMasked++
                }

                batchLoss += loss
                model.Backward(loss)
            }

            // Average and update
            batchLoss /= float64(len(batch))
            optimizer.Step(model)
            model.ZeroGrad()

            totalLoss += batchLoss
            totalCorrect += batchCorrect
            totalMasked += batchMasked
            numBatches++

            // Log progress
            if numBatches%100 == 0 {
                accuracy := float64(totalCorrect) / float64(totalMasked)
                fmt.Printf("Epoch %d, Batch %d, Loss: %.4f, Accuracy: %.2f%%\n",
                    epoch+1, numBatches, totalLoss/float64(numBatches), accuracy*100)
            }
        }

        // Epoch complete
        epochLoss := totalLoss / float64(numBatches)
        epochAccuracy := float64(totalCorrect) / float64(totalMasked)
        fmt.Printf("=== Epoch %d Complete: Loss %.4f, Accuracy %.2f%% ===\n\n",
            epoch+1, epochLoss, epochAccuracy*100)

        // Save checkpoint
        SaveModel(model, fmt.Sprintf("bert_checkpoint_epoch_%d.bin", epoch+1))
    }
}
```

### 4. Fine-Tuning for Classification

```go
// Fine-tune BERT for sentiment classification
func fineTuneBERTForClassification(pretrainedModel *BERTForMaskedLM,
                                    labeledExamples []ClassificationExample) {
    // Add classification head
    classifier := &BERTClassifier{
        BERT:      pretrainedModel.BERT,
        ClassHead: NewLinearLayer(config.HiddenDim, numClasses),
    }

    optimizer := NewAdamOptimizer(1e-5) // Lower LR for fine-tuning
    batchSize := 16

    for epoch := 0; epoch < 3; epoch++ { // Fewer epochs for fine-tuning
        for i := 0; i < len(labeledExamples); i += batchSize {
            batch := labeledExamples[i:min(i+batchSize, len(labeledExamples))]

            for _, example := range batch {
                // Get [CLS] token embedding (represents whole sentence)
                bertOutput := classifier.BERT.Forward(example.Tokens)
                clsEmbedding := bertOutput[0] // [CLS] is first token

                // Classification head
                logits := classifier.ClassHead.Forward(clsEmbedding)

                // Cross-entropy loss
                loss := crossEntropyLoss(logits, example.Label)

                // Backward and update
                classifier.Backward(loss)
            }

            optimizer.Step(classifier)
            classifier.ZeroGrad()
        }
    }
}
```

---

## T5 Training Workflow

**Use Case**: Seq2seq tasks (translation, summarization, QA)
**Training Objective**: Span corruption
**Key Feature**: Encoder-decoder with cross-attention

### 1. Data Preparation

```go
// Prepare T5 training data with span corruption
func prepareT5Data(textFile string) ([]*T5Example, error) {
    text, err := os.ReadFile(textFile)
    if err != nil {
        return nil, err
    }

    sentences := splitIntoSentences(string(text))
    tokenizer := NewTokenizer(TokenizerConfig{Type: "char"})
    config := NewT5Config()

    examples := make([]*T5Example, 0)
    for _, sentence := range sentences {
        tokens := tokenizer.Encode(sentence)

        // Apply span corruption
        result := ApplySpanCorruption(tokens, config, rand.New(rand.NewSource(time.Now().UnixNano())))

        examples = append(examples, &T5Example{
            EncoderInput:  result.CorruptedTokens,  // "The cat <X> mat"
            DecoderInput:  result.DecoderInput,      // "<S> <X>"
            DecoderTarget: result.DecoderTarget,     // "<X> sat on <EOS>"
        })
    }

    return examples, nil
}

type T5Example struct {
    EncoderInput  []int // Corrupted input with sentinel tokens
    DecoderInput  []int // Decoder input (starts with <S>)
    DecoderTarget []int // Target output (sentinels + corrupted spans)
}
```

### 2. Model Configuration

```go
// Create T5 model
config := NewT5Config()
config.VocabSize = 256 + 100 // Regular vocab + 100 sentinel tokens
config.HiddenDim = 512
config.NumLayers = 12        // 12 encoder + 12 decoder layers
config.NumHeads = 8
config.FFNDim = 2048
config.MaxSeqLen = 128
config.DropoutRate = 0.1
config.SpanLength = 3        // Average span length
config.CorruptRate = 0.15    // 15% of tokens corrupted
config.SentinelStartID = 256 // Sentinels start after regular vocab

model := NewT5ForConditionalGeneration(config)
```

### 3. Training Loop

```go
func trainT5(model *T5ForConditionalGeneration, examples []*T5Example, config *T5Config) {
    optimizer := NewAdamOptimizer(1e-4)
    batchSize := 16 // Smaller batch due to encoder+decoder
    numEpochs := 10

    for epoch := 0; epoch < numEpochs; epoch++ {
        totalLoss := 0.0
        numBatches := 0

        // Shuffle examples
        rand.Shuffle(len(examples), func(i, j int) {
            examples[i], examples[j] = examples[j], examples[i]
        })

        for i := 0; i < len(examples); i += batchSize {
            end := i + batchSize
            if end > len(examples) {
                end = len(examples)
            }
            batch := examples[i:end]

            batchLoss := 0.0
            for _, example := range batch {
                // Step 1: Encode corrupted input (bidirectional)
                encoderOutputs := model.Encoder.Forward(example.EncoderInput)

                // Step 2: Decode target autoregressively (with cross-attention)
                decoderLogits := model.Decoder.Forward(example.DecoderInput, encoderOutputs)

                // Step 3: Compute seq2seq loss
                loss := computeSeq2SeqLoss(decoderLogits, example.DecoderTarget)
                batchLoss += loss

                // Backward pass
                model.Backward(loss)
            }

            // Average and update
            batchLoss /= float64(len(batch))
            optimizer.Step(model)
            model.ZeroGrad()

            totalLoss += batchLoss
            numBatches++

            if numBatches%100 == 0 {
                fmt.Printf("Epoch %d, Batch %d, Loss: %.4f\n",
                    epoch+1, numBatches, totalLoss/float64(numBatches))
            }
        }

        epochLoss := totalLoss / float64(numBatches)
        fmt.Printf("=== Epoch %d Complete: Loss %.4f ===\n\n", epoch+1, epochLoss)

        SaveModel(model, fmt.Sprintf("t5_checkpoint_epoch_%d.bin", epoch+1))
    }
}

func computeSeq2SeqLoss(logits *Tensor, targets []int) float64 {
    seqLen := len(targets)
    loss := 0.0

    for t := 0; t < seqLen; t++ {
        posLogits := make([]float64, logits.Shape()[1])
        for v := 0; v < len(posLogits); v++ {
            posLogits[v] = logits.At(t, v)
        }

        probs := softmax(posLogits)
        loss -= math.Log(probs[targets[t]] + 1e-10)
    }

    return loss / float64(seqLen)
}
```

### 4. Fine-Tuning for Translation

```go
// Fine-tune T5 for translation
func fineTuneT5ForTranslation(pretrainedModel *T5ForConditionalGeneration,
                               translationPairs []TranslationPair) {
    optimizer := NewAdamOptimizer(1e-5)
    batchSize := 8

    for epoch := 0; epoch < 5; epoch++ {
        for i := 0; i < len(translationPairs); i += batchSize {
            batch := translationPairs[i:min(i+batchSize, len(translationPairs))]

            for _, pair := range batch {
                // Encoder: source sentence
                encoderOutputs := pretrainedModel.Encoder.Forward(pair.SourceTokens)

                // Decoder: generate target sentence
                decoderLogits := pretrainedModel.Decoder.Forward(pair.TargetInputTokens, encoderOutputs)

                // Loss: cross-entropy against target
                loss := computeSeq2SeqLoss(decoderLogits, pair.TargetOutputTokens)

                pretrainedModel.Backward(loss)
            }

            optimizer.Step(pretrainedModel)
            pretrainedModel.ZeroGrad()
        }
    }
}

type TranslationPair struct {
    SourceTokens       []int // English: "Hello world"
    TargetInputTokens  []int // French decoder input: "<S> Bonjour"
    TargetOutputTokens []int // French target: "Bonjour monde <EOS>"
}
```

---

## Comparison and Selection

### Quick Reference

| Architecture | Best For | Data Format | Training Time |
|--------------|----------|-------------|---------------|
| **GPT** | Generation | Plain text | Fast (1× baseline) |
| **BERT** | Understanding | Plain text with masking | Fast (1× baseline) |
| **T5** | Transformation | Input-output pairs | Slow (2× baseline) |

### Training Time Comparison

```
Dataset: 1M tokens, 8-layer model, single GPU

GPT:  ~2 hours (causal attention only)
BERT: ~2 hours (bidirectional attention, no generation)
T5:   ~4 hours (encoder + decoder, 2× parameters)
```

### Memory Comparison

```
Sequence length: 128 tokens, Batch size: 32

GPT:  ~2GB GPU memory
BERT: ~2GB GPU memory
T5:   ~4GB GPU memory (encoder + decoder + cross-attention)
```

---

## Performance Optimization

### 1. Gradient Accumulation

Simulate larger batch sizes with limited memory:

```go
// Train with effective batch size of 128 (32 × 4)
gradAccumSteps := 4
actualBatchSize := 32

for i := 0; i < len(examples); i += actualBatchSize {
    batch := examples[i : i+actualBatchSize]

    // Accumulate gradients over multiple micro-batches
    for step := 0; step < gradAccumSteps; step++ {
        microBatch := batch[step*len(batch)/gradAccumSteps : (step+1)*len(batch)/gradAccumSteps]

        for _, example := range microBatch {
            loss := computeLoss(model, example)
            model.Backward(loss / float64(gradAccumSteps)) // Scale loss
        }
    }

    // Update after accumulation
    optimizer.Step(model)
    model.ZeroGrad()
}
```

### 2. Mixed Precision Training

Reduce memory usage by 50%:

```go
// Enable mixed precision
mixedPrecisionConfig := &MixedPrecisionConfig{
    Enabled:          true,
    LossScale:        1024.0, // Prevent gradient underflow
    ScaleGrowthRate:  2.0,
    ScaleBackoffRate: 0.5,
}

// Forward in float16
inputFP16 := TensorFloat32ToFloat16(input)
logitsFP16 := model.ForwardFP16(inputFP16)

// Loss in float32
logitsFP32 := TensorFloat16ToFloat32(logitsFP16)
loss := computeLoss(logitsFP32, target)

// Backward with loss scaling
scaledLoss := loss * mixedPrecisionConfig.LossScale
model.Backward(scaledLoss)

// Unscale gradients before optimizer step
optimizer.UnscaleGradients(mixedPrecisionConfig.LossScale)
optimizer.Step(model)
```

### 3. Gradient Checkpointing

Trade compute for memory (2-4× memory reduction):

```go
// Enable gradient checkpointing
checkpointConfig := &CheckpointConfig{
    Enabled:         true,
    CheckpointEveryN: 2, // Checkpoint every 2 layers
}

// During forward pass
layerOutputs := make([]*Tensor, model.NumLayers)
for i := 0; i < model.NumLayers; i++ {
    if i%checkpointConfig.CheckpointEveryN == 0 {
        // Checkpoint this layer (discard intermediate activations)
        layerOutputs[i] = model.ForwardLayerCheckpointed(i, layerOutputs[i-1])
    } else {
        // Normal forward pass
        layerOutputs[i] = model.ForwardLayer(i, layerOutputs[i-1])
    }
}

// During backward pass, recompute checkpointed activations
```

### 4. Flash Attention

Reduce attention memory from O(N²) to O(N):

```go
// Use Flash Attention for long sequences
flashConfig := &FlashAttentionConfig{
    BlockSize:    128, // Tile size for L2 cache
    CausalMask:   true,
    MultiHead:    true,
    NumHeads:     8,
}

// Standard attention: O(N²) memory
attentionOutput := StandardAttention(Q, K, V) // Stores full N×N matrix

// Flash Attention: O(N) memory, ~2-4× faster
attentionOutput := FlashAttentionForward(Q, K, V, flashConfig) // Tiled computation
```

---

## Summary

### When to Use Each Architecture

```
Generation Task?
  → Train GPT
  → Example: "Complete: The cat..." → "The cat sat on the mat"

Understanding Task?
  → Train BERT
  → Example: "Classify: This movie is great" → Positive

Transformation Task?
  → Train T5
  → Example: "Translate: Hello" → "Bonjour"
```

### Training Checklist

**GPT**:
- ✓ Plain text corpus
- ✓ Character or BPE tokenization
- ✓ Next token prediction loss
- ✓ Temperature sampling for generation

**BERT**:
- ✓ Plain text corpus
- ✓ Random masking (15%)
- ✓ Masked language modeling loss
- ✓ Fine-tune for downstream tasks

**T5**:
- ✓ Plain text or parallel corpus
- ✓ Span corruption for pretraining
- ✓ Seq2seq loss
- ✓ Fine-tune for specific tasks (translation, summarization, etc.)

---

## Further Reading

- Implementation details: `transformer.go`, `transformer_bert.go`, `transformer_t5.go`
- Architecture comparison: `docs/transformer-architectures.md`
- Interactive tutorial: `notebooks/architecture-comparison.md`
- Training dynamics: `docs/training-dynamics.md`
