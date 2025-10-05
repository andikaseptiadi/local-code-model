# Model Evaluation and Analysis Guide

This guide demonstrates how to evaluate transformer models and analyze their behavior for GPT, BERT, and T5 architectures.

## Table of Contents

1. [Evaluation Metrics](#evaluation-metrics)
2. [Perplexity Analysis](#perplexity-analysis)
3. [Attention Visualization](#attention-visualization)
4. [Embedding Analysis](#embedding-analysis)
5. [Generation Quality Metrics](#generation-quality-metrics)
6. [Model Comparison](#model-comparison)

---

## Evaluation Metrics

### 1. Loss and Perplexity

**Perplexity** measures how well a model predicts a sample. Lower is better.

```go
// Compute perplexity from cross-entropy loss
func computePerplexity(model *Transformer, testData [][]int) float64 {
    totalLoss := 0.0
    totalTokens := 0

    for _, sequence := range testData {
        input := sequence[:len(sequence)-1]
        target := sequence[1:]

        // Forward pass
        logits := model.Forward(input)

        // Compute loss per token
        for t := 0; t < len(target); t++ {
            posLogits := make([]float64, logits.Shape()[1])
            for v := 0; v < len(posLogits); v++ {
                posLogits[v] = logits.At(t, v)
            }

            probs := softmax(posLogits)
            loss := -math.Log(probs[target[t]] + 1e-10)
            totalLoss += loss
            totalTokens++
        }
    }

    avgLoss := totalLoss / float64(totalTokens)
    perplexity := math.Exp(avgLoss)

    return perplexity
}
```

**Interpretation:**
```
Perplexity = 1:    Perfect prediction (impossible in practice)
Perplexity = 10:   Excellent (very small vocabulary or domain-specific)
Perplexity = 50:   Good for character-level models
Perplexity = 100:  Reasonable for word-level models
Perplexity = 1000: Poor, model needs more training
```

**Example usage:**
```go
trainPerplexity := computePerplexity(model, trainData)
testPerplexity := computePerplexity(model, testData)

fmt.Printf("Train Perplexity: %.2f\n", trainPerplexity)
fmt.Printf("Test Perplexity:  %.2f\n", testPerplexity)

// Check for overfitting
if testPerplexity > trainPerplexity*1.5 {
    fmt.Println("Warning: Model may be overfitting")
    fmt.Println("Consider: regularization, more data, or smaller model")
}
```

### 2. Token-Level Accuracy

For BERT-style masked language modeling:

```go
func computeMLMAccuracy(model *BERTForMaskedLM, testData []*BERTExample) float64 {
    correct := 0
    total := 0

    for _, example := range testData {
        logits := model.Forward(example.Tokens)

        for pos, label := range example.Labels {
            if label == -1 {
                continue // Not masked
            }

            // Get prediction
            posLogits := make([]float64, logits.Shape()[1])
            for v := 0; v < len(posLogits); v++ {
                posLogits[v] = logits.At(pos, v)
            }

            predicted := argmax(posLogits)
            if predicted == label {
                correct++
            }
            total++
        }
    }

    return float64(correct) / float64(total)
}
```

**Example output:**
```
MLM Accuracy: 72.5%

Interpretation:
- Random guessing (vocab=50K): 0.002% accuracy
- 70-80%: Good BERT model
- 80-90%: Excellent BERT model
- >90%: May be overfitting or task is too easy
```

---

## Perplexity Analysis

### Track Perplexity During Training

```go
type TrainingMetrics struct {
    Epoch          int
    TrainLoss      float64
    TestLoss       float64
    TrainPPL       float64
    TestPPL        float64
    LearningRate   float64
}

func trainWithMetrics(model *Transformer, trainData, testData [][]int) []*TrainingMetrics {
    metrics := make([]*TrainingMetrics, 0)
    optimizer := NewAdamOptimizer(3e-4)
    scheduler := &WarmupScheduler{BaseLR: 3e-4, WarmupSteps: 500}

    for epoch := 0; epoch < numEpochs; epoch++ {
        // Training epoch
        trainLoss := trainEpoch(model, trainData, optimizer)

        // Evaluation
        testLoss := evaluateLoss(model, testData)
        trainPPL := math.Exp(trainLoss)
        testPPL := math.Exp(testLoss)

        m := &TrainingMetrics{
            Epoch:        epoch + 1,
            TrainLoss:    trainLoss,
            TestLoss:     testLoss,
            TrainPPL:     trainPPL,
            TestPPL:      testPPL,
            LearningRate: scheduler.GetLR(),
        }
        metrics = append(metrics, m)

        // Print progress
        fmt.Printf("Epoch %d: Train PPL=%.2f, Test PPL=%.2f, LR=%.6f\n",
            m.Epoch, m.TrainPPL, m.TestPPL, m.LearningRate)

        // Early stopping
        if epoch > 5 && m.TestPPL > metrics[epoch-5].TestPPL {
            fmt.Println("Early stopping: test perplexity not improving")
            break
        }
    }

    return metrics
}
```

### Visualize Training Progress

```go
func plotTrainingMetrics(metrics []*TrainingMetrics) {
    // Generate HTML visualization
    html := `<!DOCTYPE html>
<html>
<head>
    <title>Training Metrics</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div id="perplexity" style="width:800px;height:400px;"></div>
    <div id="loss" style="width:800px;height:400px;"></div>
    <div id="lr" style="width:800px;height:400px;"></div>
    <script>
`

    // Extract data
    epochs := make([]int, len(metrics))
    trainPPL := make([]float64, len(metrics))
    testPPL := make([]float64, len(metrics))
    trainLoss := make([]float64, len(metrics))
    testLoss := make([]float64, len(metrics))
    learningRates := make([]float64, len(metrics))

    for i, m := range metrics {
        epochs[i] = m.Epoch
        trainPPL[i] = m.TrainPPL
        testPPL[i] = m.TestPPL
        trainLoss[i] = m.TrainLoss
        testLoss[i] = m.TestLoss
        learningRates[i] = m.LearningRate
    }

    // Perplexity plot
    html += fmt.Sprintf(`
        var pplData = [
            {x: %v, y: %v, name: 'Train', type: 'scatter'},
            {x: %v, y: %v, name: 'Test', type: 'scatter'}
        ];
        Plotly.newPlot('perplexity', pplData, {title: 'Perplexity Over Time'});
    `, epochs, trainPPL, epochs, testPPL)

    // Loss plot
    html += fmt.Sprintf(`
        var lossData = [
            {x: %v, y: %v, name: 'Train', type: 'scatter'},
            {x: %v, y: %v, name: 'Test', type: 'scatter'}
        ];
        Plotly.newPlot('loss', lossData, {title: 'Loss Over Time'});
    `, epochs, trainLoss, epochs, testLoss)

    // Learning rate plot
    html += fmt.Sprintf(`
        var lrData = [{x: %v, y: %v, type: 'scatter'}];
        Plotly.newPlot('lr', lrData, {title: 'Learning Rate Schedule'});
    `, epochs, learningRates)

    html += `
    </script>
</body>
</html>`

    os.WriteFile("training_metrics.html", []byte(html), 0644)
    fmt.Println("Saved training_metrics.html")
}
```

---

## Attention Visualization

### Extract Attention Weights

```go
type AttentionWeights struct {
    Layer    int
    Head     int
    Weights  [][]float64  // [seq_len, seq_len]
    Tokens   []string
}

func extractAttentionWeights(model *Transformer, input []int,
                             tokenizer *Tokenizer) []*AttentionWeights {
    // Enable attention recording
    model.RecordAttention = true

    // Forward pass
    _ = model.Forward(input)

    // Extract weights from each layer and head
    weights := make([]*AttentionWeights, 0)
    tokens := tokenizer.Decode(input)

    for layerIdx, layer := range model.Layers {
        for headIdx := 0; headIdx < model.Config.NumHeads; headIdx++ {
            // Get attention weights for this head
            headWeights := layer.Attention.GetHeadWeights(headIdx)

            weights = append(weights, &AttentionWeights{
                Layer:   layerIdx,
                Head:    headIdx,
                Weights: headWeights,
                Tokens:  strings.Split(tokens, ""),
            })
        }
    }

    return weights
}
```

### Visualize Attention Heatmap

```go
func visualizeAttention(weights *AttentionWeights) string {
    seqLen := len(weights.Tokens)

    html := `<!DOCTYPE html>
<html>
<head>
    <title>Attention Visualization</title>
    <style>
        .heatmap { display: inline-block; margin: 10px; }
        .cell {
            display: inline-block;
            width: 30px;
            height: 30px;
            text-align: center;
            line-height: 30px;
            font-size: 10px;
        }
        .row { display: block; }
        .label { font-weight: bold; margin: 5px; }
    </style>
</head>
<body>
    <h2>Layer ` + fmt.Sprintf("%d", weights.Layer) + `, Head ` + fmt.Sprintf("%d", weights.Head) + `</h2>
    <div class="heatmap">
`

    // Header row
    html += `<div class="row"><div class="cell"></div>`
    for _, token := range weights.Tokens {
        html += `<div class="cell label">` + token + `</div>`
    }
    html += `</div>`

    // Data rows
    for i := 0; i < seqLen; i++ {
        html += `<div class="row">`
        html += `<div class="cell label">` + weights.Tokens[i] + `</div>`

        for j := 0; j < seqLen; j++ {
            weight := weights.Weights[i][j]

            // Color intensity based on weight
            intensity := int(weight * 255)
            color := fmt.Sprintf("rgb(%d, %d, 255)", 255-intensity, 255-intensity)

            html += fmt.Sprintf(`<div class="cell" style="background-color: %s;" title="%.3f">%.2f</div>`,
                color, weight, weight)
        }
        html += `</div>`
    }

    html += `
    </div>
</body>
</html>`

    return html
}
```

### Analyze Attention Patterns

```go
func analyzeAttentionPatterns(weights []*AttentionWeights) {
    fmt.Println("=== Attention Pattern Analysis ===\n")

    // 1. Self-attention strength
    for _, w := range weights {
        selfAttention := 0.0
        for i := 0; i < len(w.Tokens); i++ {
            selfAttention += w.Weights[i][i]
        }
        selfAttention /= float64(len(w.Tokens))

        fmt.Printf("Layer %d, Head %d: Self-attention = %.3f\n",
            w.Layer, w.Head, selfAttention)
    }

    fmt.Println()

    // 2. Attention span (how far tokens attend)
    for _, w := range weights {
        totalDistance := 0.0
        totalWeight := 0.0

        for i := 0; i < len(w.Tokens); i++ {
            for j := 0; j < len(w.Tokens); j++ {
                distance := float64(abs(i - j))
                totalDistance += distance * w.Weights[i][j]
                totalWeight += w.Weights[i][j]
            }
        }

        avgSpan := totalDistance / totalWeight
        fmt.Printf("Layer %d, Head %d: Avg attention span = %.2f tokens\n",
            w.Layer, w.Head, avgSpan)
    }

    fmt.Println()

    // 3. Attention concentration (entropy)
    for _, w := range weights {
        avgEntropy := 0.0

        for i := 0; i < len(w.Tokens); i++ {
            entropy := 0.0
            for j := 0; j < len(w.Tokens); j++ {
                if w.Weights[i][j] > 0 {
                    entropy -= w.Weights[i][j] * math.Log(w.Weights[i][j])
                }
            }
            avgEntropy += entropy
        }

        avgEntropy /= float64(len(w.Tokens))
        maxEntropy := math.Log(float64(len(w.Tokens)))

        fmt.Printf("Layer %d, Head %d: Entropy = %.3f (max=%.3f, concentrated=%.1f%%)\n",
            w.Layer, w.Head, avgEntropy, maxEntropy,
            (1.0-avgEntropy/maxEntropy)*100)
    }
}
```

**Example output:**
```
=== Attention Pattern Analysis ===

Layer 0, Head 0: Self-attention = 0.156
Layer 0, Head 1: Self-attention = 0.089
Layer 0, Head 2: Self-attention = 0.234

Layer 0, Head 0: Avg attention span = 2.3 tokens
Layer 0, Head 1: Avg attention span = 4.7 tokens
Layer 0, Head 2: Avg attention span = 1.2 tokens

Layer 0, Head 0: Entropy = 1.456 (max=2.079, concentrated=30%)
Layer 0, Head 1: Entropy = 1.892 (max=2.079, concentrated=9%)
Layer 0, Head 2: Entropy = 0.987 (max=2.079, concentrated=53%)

Interpretation:
- Head 0: Medium-range, moderately focused attention
- Head 1: Long-range, diffuse attention (contextual)
- Head 2: Short-range, highly focused (local patterns)
```

---

## Embedding Analysis

### Visualize Token Embeddings

```go
func analyzeEmbeddings(model *Transformer, tokenizer *Tokenizer) {
    embeddings := model.TokenEmbeddings  // [vocab_size, d_model]

    // 1. Compute pairwise similarities
    fmt.Println("=== Most Similar Token Pairs ===\n")

    type similarity struct {
        token1 string
        token2 string
        score  float64
    }

    similarities := make([]similarity, 0)

    for i := 0; i < min(embeddings.Shape()[0], 1000); i++ {
        for j := i + 1; j < min(embeddings.Shape()[0], 1000); j++ {
            // Cosine similarity
            dot := 0.0
            norm1 := 0.0
            norm2 := 0.0

            for d := 0; d < embeddings.Shape()[1]; d++ {
                v1 := embeddings.At(i, d)
                v2 := embeddings.At(j, d)
                dot += v1 * v2
                norm1 += v1 * v1
                norm2 += v2 * v2
            }

            score := dot / (math.Sqrt(norm1) * math.Sqrt(norm2))

            similarities = append(similarities, similarity{
                token1: tokenizer.IDToToken[i],
                token2: tokenizer.IDToToken[j],
                score:  score,
            })
        }
    }

    // Sort by similarity
    sort.Slice(similarities, func(i, j int) bool {
        return similarities[i].score > similarities[j].score
    })

    // Print top 20
    for i := 0; i < min(20, len(similarities)); i++ {
        fmt.Printf("%.3f: '%s' <-> '%s'\n",
            similarities[i].score,
            similarities[i].token1,
            similarities[i].token2)
    }
}
```

### PCA Visualization

```go
func visualizeEmbeddingsPCA(embeddings *Tensor, tokens []string) {
    // Simplified PCA (use proper library in production)
    // Project to 2D for visualization

    // 1. Center the data
    means := make([]float64, embeddings.Shape()[1])
    for d := 0; d < embeddings.Shape()[1]; d++ {
        sum := 0.0
        for i := 0; i < embeddings.Shape()[0]; i++ {
            sum += embeddings.At(i, d)
        }
        means[d] = sum / float64(embeddings.Shape()[0])
    }

    centered := NewTensor(embeddings.Shape()...)
    for i := 0; i < embeddings.Shape()[0]; i++ {
        for d := 0; d < embeddings.Shape()[1]; d++ {
            centered.Set(embeddings.At(i, d)-means[d], i, d)
        }
    }

    // 2. Compute covariance matrix (simplified)
    // 3. Compute eigenvectors
    // 4. Project onto top 2 components
    // (Implementation details omitted for brevity)

    projected := computePCA(centered, 2)

    // Generate scatter plot
    generateScatterPlot(projected, tokens, "embeddings_pca.html")
}
```

---

## Generation Quality Metrics

### Diversity Metrics

```go
func analyzeGenerationDiversity(model *Transformer, prompts []string, numSamples int) {
    fmt.Println("=== Generation Diversity Analysis ===\n")

    for _, prompt := range prompts {
        fmt.Printf("Prompt: \"%s\"\n", prompt)

        generations := make([]string, numSamples)
        for i := 0; i < numSamples; i++ {
            generations[i] = generateText(model, prompt, 50, 0.8)
        }

        // 1. Unique generations
        uniqueGens := make(map[string]bool)
        for _, gen := range generations {
            uniqueGens[gen] = true
        }
        uniqueRate := float64(len(uniqueGens)) / float64(numSamples)

        fmt.Printf("  Unique: %d/%d (%.1f%%)\n",
            len(uniqueGens), numSamples, uniqueRate*100)

        // 2. Average length
        avgLen := 0.0
        for _, gen := range generations {
            avgLen += float64(len(strings.Split(gen, " ")))
        }
        avgLen /= float64(numSamples)

        fmt.Printf("  Avg length: %.1f tokens\n", avgLen)

        // 3. Vocabulary diversity
        allTokens := make(map[string]int)
        for _, gen := range generations {
            tokens := strings.Split(gen, " ")
            for _, token := range tokens {
                allTokens[token]++
            }
        }

        fmt.Printf("  Unique tokens: %d\n", len(allTokens))
        fmt.Printf("  Vocabulary richness: %.3f\n\n",
            float64(len(allTokens))/avgLen)
    }
}
```

### Repetition Detection

```go
func detectRepetition(text string) float64 {
    tokens := strings.Split(text, " ")

    // Count repeated n-grams
    repeatedTokens := 0

    for n := 2; n <= 4; n++ {  // Check 2-grams, 3-grams, 4-grams
        ngrams := make(map[string]int)

        for i := 0; i <= len(tokens)-n; i++ {
            ngram := strings.Join(tokens[i:i+n], " ")
            ngrams[ngram]++

            if ngrams[ngram] > 1 {
                repeatedTokens += n
            }
        }
    }

    repetitionRate := float64(repeatedTokens) / float64(len(tokens))
    return repetitionRate
}
```

**Usage:**
```go
generated := generateText(model, "The cat", 100, 0.7)
repRate := detectRepetition(generated)

fmt.Printf("Generated: %s\n", generated)
fmt.Printf("Repetition rate: %.1f%%\n", repRate*100)

if repRate > 0.3 {
    fmt.Println("Warning: High repetition detected")
    fmt.Println("Try: higher temperature, repetition penalty, or top-p sampling")
}
```

---

## Model Comparison

### Compare Multiple Models

```go
type ModelEvaluation struct {
    Name         string
    Perplexity   float64
    Accuracy     float64
    Speed        time.Duration  // Inference time
    MemoryMB     int
}

func compareModels(models map[string]*Transformer, testData [][]int) {
    evaluations := make([]*ModelEvaluation, 0)

    for name, model := range models {
        fmt.Printf("Evaluating %s...\n", name)

        // 1. Perplexity
        start := time.Now()
        ppl := computePerplexity(model, testData)
        elapsed := time.Since(start)

        // 2. Speed (tokens/sec)
        totalTokens := 0
        for _, seq := range testData {
            totalTokens += len(seq)
        }
        tokensPerSec := float64(totalTokens) / elapsed.Seconds()

        // 3. Memory (approximate)
        params := countParameters(model)
        memoryMB := params * 4 / 1024 / 1024  // 4 bytes per float32

        eval := &ModelEvaluation{
            Name:       name,
            Perplexity: ppl,
            Speed:      elapsed,
            MemoryMB:   memoryMB,
        }
        evaluations = append(evaluations, eval)
    }

    // Print comparison table
    fmt.Println("\n=== Model Comparison ===\n")
    fmt.Printf("%-20s %12s %15s %10s\n", "Model", "Perplexity", "Time", "Memory")
    fmt.Println(strings.Repeat("-", 60))

    for _, eval := range evaluations {
        fmt.Printf("%-20s %12.2f %15s %8dMB\n",
            eval.Name, eval.Perplexity, eval.Speed, eval.MemoryMB)
    }
}
```

**Example output:**
```
=== Model Comparison ===

Model                 Perplexity            Time     Memory
------------------------------------------------------------
small-gpt                 48.23         125.3ms       50MB
medium-gpt                32.45         453.2ms      200MB
large-gpt                 28.91        1234.5ms      800MB

Interpretation:
- small-gpt:  Fast, low memory, higher perplexity
- medium-gpt: Balanced trade-off
- large-gpt:  Best quality, slowest, most memory
```

---

## Complete Evaluation Script

```go
func evaluateModel(model *Transformer, trainData, testData [][]int,
                   tokenizer *Tokenizer) {
    fmt.Println("=" * 60)
    fmt.Println("MODEL EVALUATION REPORT")
    fmt.Println("=" * 60)

    // 1. Basic metrics
    fmt.Println("\n1. PERPLEXITY")
    trainPPL := computePerplexity(model, trainData)
    testPPL := computePerplexity(model, testData)
    fmt.Printf("   Train: %.2f\n", trainPPL)
    fmt.Printf("   Test:  %.2f\n", testPPL)

    // 2. Attention analysis
    fmt.Println("\n2. ATTENTION PATTERNS")
    sampleInput := testData[0][:20]
    weights := extractAttentionWeights(model, sampleInput, tokenizer)
    analyzeAttentionPatterns(weights)

    // 3. Generation quality
    fmt.Println("\n3. GENERATION QUALITY")
    prompts := []string{"The", "Once upon a time", "In the"}
    analyzeGenerationDiversity(model, prompts, 10)

    // 4. Model statistics
    fmt.Println("\n4. MODEL STATISTICS")
    params := countParameters(model)
    fmt.Printf("   Parameters: %d (%.1fM)\n", params, float64(params)/1e6)

    fmt.Println("\n" + "="*60)
}
```

---

## Key Takeaways

### Evaluation Checklist

- ✓ **Perplexity**: Lower is better, track on train and test sets
- ✓ **Attention patterns**: Verify heads learn different patterns
- ✓ **Generation quality**: Check diversity and repetition
- ✓ **Comparison**: Benchmark against baselines

### Red Flags

- Test perplexity >> train perplexity → Overfitting
- All attention heads similar → Model not learning diverse features
- High repetition in generations → Need better sampling strategy
- Perplexity not improving → Learning rate or architecture issues

### Best Practices

1. **Track metrics over time**: Save and visualize training progress
2. **Qualitative analysis**: Generate samples and manually inspect
3. **Ablation studies**: Compare with/without features
4. **Error analysis**: Look at cases where model fails

---

## Further Reading

- Perplexity: Lower-bound estimate of model quality
- Attention visualization: Reveals what model learns
- Generation metrics: BLEU, ROUGE, METEOR for comparison
- Our implementations: `transformer.go`, `evaluation.go`
