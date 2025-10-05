# Transformer Training Troubleshooting Guide

This guide helps you diagnose and fix common issues when training transformer models.

## Table of Contents

1. [NaN Loss and Gradient Issues](#nan-loss-and-gradient-issues)
2. [Poor Convergence](#poor-convergence)
3. [Memory Issues](#memory-issues)
4. [Slow Training](#slow-training)
5. [Generation Issues](#generation-issues)
6. [Implementation Bugs](#implementation-bugs)

---

## NaN Loss and Gradient Issues

### Symptom: Loss becomes NaN after a few iterations

```
Epoch 1, Batch 1: Loss 3.245
Epoch 1, Batch 2: Loss 2.987
Epoch 1, Batch 3: Loss NaN  ← Problem!
```

### Causes and Solutions

#### 1. Exploding Gradients

**Diagnosis:**
```go
// Print gradient norms
for i, param := range model.Parameters() {
    gradNorm := computeNorm(param.Grad)
    if gradNorm > 10.0 {
        fmt.Printf("Large gradient at param %d: %.2f\n", i, gradNorm)
    }
}
```

**Solutions:**

a) **Gradient Clipping**
```go
func clipGradients(params []*Tensor, maxNorm float64) {
    // Compute total gradient norm
    totalNorm := 0.0
    for _, param := range params {
        for _, grad := range param.Grad.Data {
            totalNorm += grad * grad
        }
    }
    totalNorm = math.Sqrt(totalNorm)

    // Clip if necessary
    if totalNorm > maxNorm {
        scale := maxNorm / totalNorm
        for _, param := range params {
            for i := range param.Grad.Data {
                param.Grad.Data[i] *= scale
            }
        }
        fmt.Printf("Clipped gradients: %.2f → %.2f\n", totalNorm, maxNorm)
    }
}

// Usage in training loop
clipGradients(model.Parameters(), 1.0)  // Max norm = 1.0
optimizer.Step(model)
```

b) **Lower Learning Rate**
```go
// Instead of:
config.LearningRate = 3e-4  // Too high

// Try:
config.LearningRate = 1e-4  // More conservative
```

c) **Use Layer Normalization**
```go
// Already implemented in our transformer
// Ensures activations stay in reasonable ranges
output = LayerNorm(x + Attention(x))
```

#### 2. Numerical Instability in Softmax

**Problem:**
```go
// Large logits cause overflow in exp()
logits = [100.0, 101.0, 99.0]
exp(logits) = [overflow!, overflow!, overflow!]
```

**Solution: Softmax Numerical Stability**
```go
func softmax(logits []float64) []float64 {
    // Subtract max for numerical stability
    maxLogit := math.Inf(-1)
    for _, l := range logits {
        if l > maxLogit {
            maxLogit = l
        }
    }

    // Compute exp with shifted logits
    expSum := 0.0
    probs := make([]float64, len(logits))
    for i, l := range logits {
        probs[i] = math.Exp(l - maxLogit)  // Subtract max!
        expSum += probs[i]
    }

    // Normalize
    for i := range probs {
        probs[i] /= expSum
    }

    return probs
}
```

**Example:**
```
Before:
  logits = [100, 101, 99]
  exp([100, 101, 99]) = overflow!

After (subtract max=101):
  logits - max = [-1, 0, -2]
  exp([-1, 0, -2]) = [0.368, 1.0, 0.135]
  probs = [0.245, 0.665, 0.090]  ← Stable!
```

#### 3. Division by Zero in Attention Scaling

**Problem:**
```go
scores / math.Sqrt(float64(d_k))  // d_k must be > 0
```

**Solution:**
```go
func scaleScores(scores *Tensor, d_k int) *Tensor {
    if d_k <= 0 {
        panic("d_k must be positive")
    }
    scale := 1.0 / math.Sqrt(float64(d_k))
    return scores.Scale(scale)
}
```

#### 4. Incorrect Gradient Initialization

**Problem:**
```go
// Forgot to zero gradients
for iteration := 0; iteration < 1000; iteration++ {
    loss := forward()
    backward(loss)
    optimizer.Step()  // Gradients accumulate! ← Problem
}
```

**Solution:**
```go
for iteration := 0; iteration < 1000; iteration++ {
    model.ZeroGrad()  // Zero gradients first!

    loss := forward()
    backward(loss)
    optimizer.Step()
}
```

---

## Poor Convergence

### Symptom: Loss decreases very slowly or plateaus

```
Epoch 1: Loss 5.234
Epoch 2: Loss 5.198
Epoch 3: Loss 5.187
Epoch 4: Loss 5.181  ← Barely improving
...
Epoch 50: Loss 5.123  ← Still high
```

### Causes and Solutions

#### 1. Learning Rate Too Low

**Diagnosis:**
```go
// Monitor gradient updates
avgUpdate := 0.0
avgParam := 0.0
for _, param := range model.Parameters() {
    for i := range param.Data {
        avgUpdate += math.Abs(param.Grad.Data[i] * learningRate)
        avgParam += math.Abs(param.Data[i])
    }
}
updateRatio := avgUpdate / avgParam

// Rule of thumb: update ratio should be ~0.001 to 0.01
if updateRatio < 0.0001 {
    fmt.Printf("Warning: Updates too small (%.6f)\n", updateRatio)
}
```

**Solution: Learning Rate Schedule**
```go
type WarmupScheduler struct {
    BaseLR      float64
    WarmupSteps int
    CurrentStep int
}

func (s *WarmupScheduler) GetLR() float64 {
    s.CurrentStep++

    if s.CurrentStep < s.WarmupSteps {
        // Linear warmup
        return s.BaseLR * float64(s.CurrentStep) / float64(s.WarmupSteps)
    }

    // Cosine decay after warmup
    progress := float64(s.CurrentStep-s.WarmupSteps) / 1000.0
    return s.BaseLR * 0.5 * (1.0 + math.Cos(math.Pi*progress))
}

// Usage
scheduler := &WarmupScheduler{
    BaseLR:      3e-4,
    WarmupSteps: 500,
}

for iteration := 0; iteration < 10000; iteration++ {
    lr := scheduler.GetLR()
    optimizer.SetLearningRate(lr)

    // Training step...
}
```

#### 2. Model Too Small

**Problem:** Model lacks capacity to learn the data

**Solution:**
```go
// Increase model size
config.HiddenDim = 512   // Was 256
config.NumLayers = 8     // Was 4
config.NumHeads = 8      // Was 4
config.FFNDim = 2048     // Was 1024 (typically 4× hidden)
```

**Rule of thumb:**
- Small dataset (< 1M tokens): 256-512 hidden, 4-6 layers
- Medium dataset (1-10M tokens): 512-768 hidden, 6-12 layers
- Large dataset (> 10M tokens): 768-1024 hidden, 12-24 layers

#### 3. Insufficient Training Data

**Symptom:** Model memorizes training set but doesn't generalize

**Solution:**
```go
// Data augmentation for text
func augmentText(text string) string {
    // Random deletion (10% chance per word)
    // Random swap (swap adjacent words)
    // Back-translation (translate → back)
    return augmentedText
}

// More data collection
// Or use transfer learning from pretrained model
```

#### 4. Poor Initialization

**Problem:** Weights start in bad region of loss landscape

**Solution: Xavier/Glorot Initialization**
```go
func initializeWeights(tensor *Tensor, fanIn, fanOut int) {
    // Xavier initialization
    limit := math.Sqrt(6.0 / float64(fanIn+fanOut))

    for i := range tensor.Data {
        // Uniform distribution [-limit, limit]
        tensor.Data[i] = (rand.Float64()*2.0 - 1.0) * limit
    }
}

// For attention layers
initializeWeights(W_Q, d_model, d_k)
initializeWeights(W_K, d_model, d_k)
initializeWeights(W_V, d_model, d_v)
```

---

## Memory Issues

### Symptom: Out of memory errors

```
panic: runtime: out of memory
```

### Causes and Solutions

#### 1. Batch Size Too Large

**Solution: Reduce batch size or use gradient accumulation**
```go
// Instead of batch_size = 64 (OOM)
actualBatchSize := 16  // Fits in memory
gradAccumSteps := 4    // Accumulate 4 times
effectiveBatchSize := actualBatchSize * gradAccumSteps  // = 64

for step := 0; step < gradAccumSteps; step++ {
    batch := getBatch(actualBatchSize)
    loss := forward(batch)

    // Scale loss by accumulation steps
    loss = loss / float64(gradAccumSteps)
    backward(loss)

    // Don't zero gradients between accumulation steps!
}

optimizer.Step()
model.ZeroGrad()  // Zero after all accumulation
```

#### 2. Storing Too Many Activations

**Solution: Gradient Checkpointing**
```go
// Trade compute for memory
type CheckpointedLayer struct {
    layer *TransformerLayer
}

func (c *CheckpointedLayer) Forward(x *Tensor) *Tensor {
    // Don't store intermediate activations
    return c.layer.Forward(x)
}

func (c *CheckpointedLayer) Backward(gradOutput *Tensor) *Tensor {
    // Recompute forward pass to get activations
    x := c.savedInput
    _ = c.layer.Forward(x)  // Recompute

    // Now compute backward pass
    return c.layer.Backward(gradOutput)
}
```

**Memory savings:**
- Without checkpointing: O(N² × d × L) where L = num_layers
- With checkpointing: O(N² × d × √L)
- Trade-off: +33% compute time

#### 3. Large Vocabulary

**Problem:** Token embeddings table is huge

```go
// Embedding table size
vocabSize := 50000
d_model := 1024
embeddingParams := vocabSize * d_model  // 51M parameters!
memoryMB := embeddingParams * 4 / 1024 / 1024  // 195 MB just for embeddings
```

**Solution: Vocabulary Pruning**
```go
// Keep only top-k frequent tokens
func pruneVocabulary(vocab map[string]int, tokenCounts map[string]int, k int) map[string]int {
    // Sort by frequency
    type pair struct {
        token string
        count int
    }
    pairs := make([]pair, 0, len(tokenCounts))
    for token, count := range tokenCounts {
        pairs = append(pairs, pair{token, count})
    }
    sort.Slice(pairs, func(i, j int) bool {
        return pairs[i].count > pairs[j].count
    })

    // Keep top k
    newVocab := make(map[string]int)
    for i := 0; i < k && i < len(pairs); i++ {
        newVocab[pairs[i].token] = i
    }
    return newVocab
}
```

#### 4. Memory Leaks

**Solution: Proper tensor cleanup**
```go
// Use pooling for temporary tensors
func (m *Model) Forward(x *Tensor) *Tensor {
    // Get tensor from pool
    temp := GetPooledTensor(x.Shape()...)
    defer PutPooledTensor(temp)  // Return to pool when done

    // Use temp tensor
    result := computeWithTemp(x, temp)
    return result
}
```

---

## Slow Training

### Symptom: Training takes too long

```
Epoch 1, Batch 1: 5.2s
Epoch 1, Batch 2: 5.3s
Epoch 1, Batch 3: 5.1s
```

Expected: < 1s per batch on GPU, < 5s on CPU (small models)

### Causes and Solutions

#### 1. Inefficient Matrix Multiplication

**Solution: Use parallel/blocked matrix multiplication**
```go
// Instead of naive MatMul
result := A.MatMul(B)  // Slow

// Use optimized version
result := A.MatMulBlockedParallel(B, 128)  // 8-16× faster
```

**See:** `tensor_blocked.go` and `tensor_parallel.go`

#### 2. Not Using Batch Operations

**Problem:**
```go
// Processing one example at a time
for _, example := range batch {
    output := model.Forward(example)  // Slow! No vectorization
    loss += computeLoss(output, example.label)
}
```

**Solution: Batch processing**
```go
// Process entire batch at once
batchInput := stackExamples(batch)  // [batch_size, seq_len]
batchOutput := model.Forward(batchInput)  // Vectorized!
loss := computeLoss(batchOutput, batchLabels)
```

#### 3. Unnecessary Copies

**Problem:**
```go
func inefficientAdd(a, b *Tensor) *Tensor {
    result := NewTensor(a.Shape()...)
    copy(result.Data, a.Data)  // Copy 1
    for i := range result.Data {
        result.Data[i] += b.Data[i]
    }
    return result
}
```

**Solution: In-place operations when possible**
```go
func efficientAdd(a, b *Tensor) *Tensor {
    // Reuse a if it's not needed elsewhere
    for i := range a.Data {
        a.Data[i] += b.Data[i]
    }
    return a  // No copy!
}
```

#### 4. Small Batch Sizes

**Problem:** Not utilizing hardware efficiently

**Solution:**
```go
// Increase batch size until memory limit
batchSize := 8   // Too small, underutilizes GPU/CPU
batchSize := 32  // Better
batchSize := 64  // Even better (if memory allows)
```

Rule of thumb: Larger batches = better hardware utilization (up to memory limits)

---

## Generation Issues

### Symptom: Generated text is repetitive or nonsensical

```
Generated: "The cat cat cat cat cat cat cat..."
Or: "sdfjk asldkfj alskdjf alskjdf"  ← Gibberish
```

### Causes and Solutions

#### 1. Greedy Decoding → Repetition

**Problem:**
```go
// Always picks most likely token
nextToken := argmax(logits)  // Deterministic, repetitive
```

**Solution: Temperature Sampling**
```go
func sampleWithTemperature(logits []float64, temperature float64) int {
    // Higher temperature = more random
    // Lower temperature = more deterministic

    scaled := make([]float64, len(logits))
    for i := range logits {
        scaled[i] = logits[i] / temperature
    }

    probs := softmax(scaled)
    return sampleFromDistribution(probs)
}

// Usage
nextToken := sampleWithTemperature(logits, 0.8)  // Slightly random
```

**Temperature effects:**
- 0.1: Very focused, deterministic
- 0.7: Balanced, natural
- 1.0: Raw model probabilities
- 2.0: Very random, creative

#### 2. No Repetition Penalty

**Solution:**
```go
func applyRepetitionPenalty(logits []float64, generatedTokens []int, penalty float64) {
    // Penalize tokens that were already generated
    for _, token := range generatedTokens {
        if token < len(logits) {
            logits[token] /= penalty  // Reduce probability
        }
    }
}

// Usage
applyRepetitionPenalty(logits, generated, 1.2)  // 20% penalty
nextToken := sampleWithTemperature(logits, 0.8)
```

#### 3. Model Not Fully Trained

**Problem:** Model hasn't learned language patterns yet

**Solution:**
```go
// Train longer or check training metrics
if trainingLoss > 2.0 {
    fmt.Println("Model undertrained, continue training")
}

// For character-level models, expect loss ~1.5-2.0 when trained
// For BPE models, expect loss ~2.0-3.0 when trained
```

#### 4. Top-k and Top-p Sampling

**Solution: Nucleus (top-p) sampling**
```go
func topPSampling(logits []float64, p float64) int {
    // Sort by probability
    probs := softmax(logits)
    type pair struct {
        idx  int
        prob float64
    }
    pairs := make([]pair, len(probs))
    for i, prob := range probs {
        pairs[i] = pair{i, prob}
    }
    sort.Slice(pairs, func(i, j int) bool {
        return pairs[i].prob > pairs[j].prob
    })

    // Accumulate probability mass
    cumProb := 0.0
    cutoff := 0
    for i, pair := range pairs {
        cumProb += pair.prob
        if cumProb >= p {
            cutoff = i + 1
            break
        }
    }

    // Renormalize and sample from top-p
    topProbs := make([]float64, cutoff)
    topIndices := make([]int, cutoff)
    sum := 0.0
    for i := 0; i < cutoff; i++ {
        topProbs[i] = pairs[i].prob
        topIndices[i] = pairs[i].idx
        sum += topProbs[i]
    }
    for i := range topProbs {
        topProbs[i] /= sum
    }

    sampledIdx := sampleFromDistribution(topProbs)
    return topIndices[sampledIdx]
}

// Usage
nextToken := topPSampling(logits, 0.9)  // Sample from top 90% probability mass
```

---

## Implementation Bugs

### Common Bugs and How to Find Them

#### 1. Dimension Mismatches

**Symptom:**
```
panic: dimension mismatch: [128, 512] × [256, 512]
```

**Debug:**
```go
func debugShapes(name string, tensors ...*Tensor) {
    fmt.Printf("=== %s ===\n", name)
    for i, t := range tensors {
        fmt.Printf("  Tensor %d: %v\n", i, t.Shape())
    }
}

// Usage
debugShapes("Attention input", Q, K, V)
```

**Common issues:**
- Forgot to transpose: `K^T` not `K`
- Wrong dimension for multi-head: split by heads first
- Batch dimension issues: [batch, seq, hidden] vs [seq, hidden]

#### 2. Incorrect Attention Masking

**Test:**
```go
func TestCausalMask(t *testing.T) {
    seqLen := 3
    mask := createCausalMask(seqLen)

    // Token 0 should only see itself
    if mask.At(0, 1) != -math.Inf(1) {
        t.Error("Causal mask broken: position 0 sees future")
    }

    // Token 2 should see all previous
    if mask.At(2, 0) == -math.Inf(1) {
        t.Error("Causal mask broken: position 2 doesn't see past")
    }
}
```

#### 3. Gradient Not Flowing

**Debug:**
```go
func checkGradients(model *Model) {
    for i, param := range model.Parameters() {
        if param.Grad == nil {
            fmt.Printf("Warning: Param %d has no gradient\n", i)
            continue
        }

        gradNorm := 0.0
        for _, g := range param.Grad.Data {
            gradNorm += g * g
        }
        gradNorm = math.Sqrt(gradNorm)

        if gradNorm < 1e-10 {
            fmt.Printf("Warning: Param %d has near-zero gradient (%.2e)\n", i, gradNorm)
        }
    }
}
```

**Common issues:**
- Missing backward() call
- Forgot to connect computation graph
- Dead ReLU (all negative inputs → zero gradients)

---

## Quick Debugging Checklist

When training goes wrong, check these in order:

1. **Print shapes**
   ```go
   fmt.Printf("Input shape: %v\n", input.Shape())
   fmt.Printf("Output shape: %v\n", output.Shape())
   ```

2. **Check for NaN/Inf**
   ```go
   for i, val := range tensor.Data {
       if math.IsNaN(val) || math.IsInf(val, 0) {
           fmt.Printf("Invalid value at position %d: %v\n", i, val)
       }
   }
   ```

3. **Monitor gradient norms**
   ```go
   gradNorm := computeGradientNorm(model)
   fmt.Printf("Gradient norm: %.4f\n", gradNorm)
   ```

4. **Verify batch sizes**
   ```go
   fmt.Printf("Batch size: %d\n", len(batch))
   fmt.Printf("Expected: %d\n", config.BatchSize)
   ```

5. **Test on tiny data**
   ```go
   // Can your model overfit 10 examples?
   // If not, there's a bug
   tinyBatch := data[:10]
   for i := 0; i < 1000; i++ {
       loss := train(tinyBatch)
       if i % 100 == 0 {
           fmt.Printf("Iteration %d, Loss: %.4f\n", i, loss)
       }
   }
   // Loss should go near zero
   ```

---

## Getting Help

If you're still stuck:

1. **Check tests**: Run `go test -v` to see if any tests fail
2. **Compare with reference**: See `transformer_test.go` for working examples
3. **Enable debug mode**: Set `DEBUG=true` for verbose logging
4. **Profile performance**: Use `go test -bench . -cpuprofile=cpu.prof`
5. **Read the docs**: See `docs/` directory for detailed explanations

Remember: Most bugs are dimension mismatches or gradient issues. Print shapes liberally!
