# Model Deployment and Serving Guide

This guide covers deploying trained transformer models for production inference, with practical patterns for serving, optimization, and scaling.

## Table of Contents

1. [Deployment Patterns](#deployment-patterns)
2. [Inference Optimization](#inference-optimization)
3. [Batching Strategies](#batching-strategies)
4. [Model Quantization](#model-quantization)
5. [Serving Architectures](#serving-architectures)
6. [Monitoring and Observability](#monitoring-and-observability)
7. [Cost Optimization](#cost-optimization)

---

## Deployment Patterns

### Single-Instance Serving

**Best For:** Low-traffic applications, prototypes, development

```go
// Simple HTTP server for model inference
package main

import (
    "encoding/json"
    "net/http"
    "sync"
)

// InferenceServer wraps a model for HTTP serving
// This pattern works well for:
// - Development and testing
// - Low-traffic APIs (<100 req/s)
// - Internal tools and demos
type InferenceServer struct {
    model      *Transformer
    tokenizer  *Tokenizer
    mu         sync.RWMutex  // Protects model access
    maxTokens  int
}

// GenerateRequest defines the API contract
type GenerateRequest struct {
    Prompt      string  `json:"prompt"`
    MaxTokens   int     `json:"max_tokens,omitempty"`
    Temperature float64 `json:"temperature,omitempty"`
    TopP        float64 `json:"top_p,omitempty"`
}

type GenerateResponse struct {
    Text       string  `json:"text"`
    TokenCount int     `json:"token_count"`
    Latency    float64 `json:"latency_ms"`
}

func (s *InferenceServer) HandleGenerate(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }

    var req GenerateRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "Invalid request", http.StatusBadRequest)
        return
    }

    // Validate and apply defaults
    if req.MaxTokens == 0 || req.MaxTokens > s.maxTokens {
        req.MaxTokens = s.maxTokens
    }
    if req.Temperature == 0 {
        req.Temperature = 0.7
    }
    if req.TopP == 0 {
        req.TopP = 0.9
    }

    // Tokenize input
    start := time.Now()
    tokens := s.tokenizer.Encode(req.Prompt)

    // Thread-safe inference (allows concurrent reads)
    s.mu.RLock()
    defer s.mu.RUnlock()

    // Generate with sampling parameters
    generated := s.model.Generate(tokens, GenerateConfig{
        MaxTokens:   req.MaxTokens,
        Temperature: req.Temperature,
        TopP:        req.TopP,
    })

    // Decode output
    text := s.tokenizer.Decode(generated)
    latency := time.Since(start).Milliseconds()

    // Return response
    resp := GenerateResponse{
        Text:       text,
        TokenCount: len(generated),
        Latency:    float64(latency),
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(resp)
}

func main() {
    // Load model once at startup
    model := LoadModel("model.bin")
    tokenizer := LoadTokenizer("vocab.json")

    server := &InferenceServer{
        model:     model,
        tokenizer: tokenizer,
        maxTokens: 512,
    }

    http.HandleFunc("/generate", server.HandleGenerate)
    http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
        w.WriteHeader(http.StatusOK)
        w.Write([]byte("OK"))
    })

    log.Println("Starting server on :8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

**Pros:**
- Simple to deploy and debug
- Low infrastructure complexity
- Predictable latency (no batching delays)

**Cons:**
- Limited throughput (~10-50 req/s per instance)
- No automatic scaling
- Single point of failure

---

### Load-Balanced Multi-Instance

**Best For:** Medium traffic (100-1000 req/s), production APIs

```
┌─────────┐
│  Load   │
│ Balancer├─────┬──────┬──────┬──────┐
└─────────┘     │      │      │      │
                ▼      ▼      ▼      ▼
            ┌───────┬───────┬───────┬───────┐
            │ Pod 1 │ Pod 2 │ Pod 3 │ Pod 4 │
            │ Model │ Model │ Model │ Model │
            └───────┴───────┴───────┴───────┘
```

**Kubernetes Deployment:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: transformer-inference
spec:
  replicas: 4  # Horizontal scaling
  selector:
    matchLabels:
      app: transformer
  template:
    metadata:
      labels:
        app: transformer
    spec:
      containers:
      - name: inference
        image: transformer-server:latest
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
          limits:
            memory: "16Gi"
            cpu: "4"
        ports:
        - containerPort: 8080
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: transformer-service
spec:
  selector:
    app: transformer
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: transformer-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: transformer-inference
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

**Key Benefits:**
- Horizontal scaling (add more pods under load)
- High availability (load balancer routes around failures)
- Rolling updates (zero-downtime deployments)
- Auto-scaling based on CPU/memory metrics

---

## Inference Optimization

### KV Cache Implementation

KV caching eliminates redundant computation during autoregressive generation:

```go
// KVCache stores key/value tensors from previous tokens
// This is critical for transformer inference efficiency
type KVCache struct {
    Keys   []*Tensor  // [num_layers][batch, seq_len, hidden_dim]
    Values []*Tensor  // [num_layers][batch, seq_len, hidden_dim]
    SeqLen int        // Current cached sequence length
}

// GenerateWithCache uses KV cache for efficient autoregressive generation
// Without cache: O(N²) for N tokens (recompute all previous keys/values)
// With cache:    O(N) for N tokens (only compute new token's key/value)
func (m *Transformer) GenerateWithCache(
    prompt []int,
    maxTokens int,
    config GenerateConfig,
) []int {
    batchSize := 1
    seqLen := len(prompt)

    // Initialize cache (nil means empty)
    cache := &KVCache{
        Keys:   make([]*Tensor, m.Config.NumLayers),
        Values: make([]*Tensor, m.Config.NumLayers),
        SeqLen: 0,
    }

    // Encode prompt (builds initial KV cache)
    input := prompt
    output := make([]int, len(prompt))
    copy(output, prompt)

    for i := 0; i < maxTokens; i++ {
        // Forward pass with cache
        // Only processes the last token after initial pass
        startIdx := 0
        if cache.SeqLen > 0 {
            startIdx = cache.SeqLen  // Only process new tokens
            input = input[startIdx:]
        }

        logits := m.ForwardWithCache(input, cache)

        // Sample next token (only need last position logits)
        nextToken := m.Sample(logits, config)
        output = append(output, nextToken)

        // Stop on EOS token
        if nextToken == m.Tokenizer.EOSToken {
            break
        }

        // Update input for next iteration
        input = []int{nextToken}
    }

    return output
}

// ForwardWithCache performs inference with KV caching
func (m *Transformer) ForwardWithCache(
    tokens []int,
    cache *KVCache,
) *Tensor {
    batchSize := 1
    seqLen := len(tokens)

    // Embed tokens
    x := m.Embedding.Forward(tokens)  // [batch, seq_len, embed_dim]

    // Add positional encoding (account for cache position)
    pos := m.PositionEncoding(cache.SeqLen, cache.SeqLen+seqLen)
    x = x.Add(pos)

    // Process through transformer layers with cache
    for layerIdx := 0; layerIdx < m.Config.NumLayers; layerIdx++ {
        layer := m.Layers[layerIdx]

        // Self-attention with KV cache
        x = layer.AttentionWithCache(x, cache, layerIdx)

        // Feed-forward (no cache needed)
        x = layer.FeedForward(x)
    }

    // Final layer norm and output projection
    x = m.LayerNorm.Forward(x)
    logits := m.OutputProjection.Forward(x)

    // Return logits for last position only
    return logits.Slice([]int{0, seqLen-1, 0},
                        []int{batchSize, seqLen, m.VocabSize})
}

// AttentionWithCache computes attention using cached keys/values
func (l *TransformerLayer) AttentionWithCache(
    x *Tensor,
    cache *KVCache,
    layerIdx int,
) *Tensor {
    batchSize, seqLen, embedDim := x.Shape()[0], x.Shape()[1], x.Shape()[2]

    // Compute Q, K, V for new tokens
    Q := l.QueryProj.Forward(x)   // [batch, new_seq, embed_dim]
    K := l.KeyProj.Forward(x)     // [batch, new_seq, embed_dim]
    V := l.ValueProj.Forward(x)   // [batch, new_seq, embed_dim]

    // Concatenate with cached K, V
    if cache.Keys[layerIdx] != nil {
        K = Concat(cache.Keys[layerIdx], K, 1)    // Concat along seq dim
        V = Concat(cache.Values[layerIdx], V, 1)
    }

    // Update cache
    cache.Keys[layerIdx] = K
    cache.Values[layerIdx] = V
    cache.SeqLen = K.Shape()[1]

    // Compute attention (Q only for new tokens, K/V include cache)
    headDim := embedDim / l.NumHeads

    // Reshape for multi-head attention
    Q = Q.Reshape(batchSize, seqLen, l.NumHeads, headDim).Transpose(1, 2)
    K = K.Reshape(batchSize, cache.SeqLen, l.NumHeads, headDim).Transpose(1, 2)
    V = V.Reshape(batchSize, cache.SeqLen, l.NumHeads, headDim).Transpose(1, 2)

    // Scaled dot-product attention
    scores := MatMul(Q, K.Transpose(2, 3))  // [batch, heads, new_seq, cached_seq]
    scores = scores.MulScalar(1.0 / math.Sqrt(float64(headDim)))

    // Causal mask (only attend to past tokens)
    mask := CausalMask(seqLen, cache.SeqLen)
    scores = scores.Add(mask)

    attn := Softmax(scores, -1)
    output := MatMul(attn, V)  // [batch, heads, new_seq, head_dim]

    // Reshape back
    output = output.Transpose(1, 2).Reshape(batchSize, seqLen, embedDim)

    return l.OutputProj.Forward(output)
}
```

**Performance Impact:**
- **Without KV cache:** 100 tokens = 100² = 10,000 attention operations
- **With KV cache:** 100 tokens = 100 attention operations (100× faster)

**Memory Trade-off:**
```
KV cache memory = 2 × num_layers × batch × seq_len × embed_dim × sizeof(float32)

Example (GPT-2 small):
= 2 × 12 layers × 1 batch × 1024 tokens × 768 dim × 4 bytes
= 75 MB per sequence
```

---

### Speculative Decoding

Speculative decoding uses a small "draft" model to predict multiple tokens, then verifies with the large target model:

```go
// SpeculativeDecoder uses a small draft model to speed up generation
// Typical speedup: 2-3× for similar quality
type SpeculativeDecoder struct {
    DraftModel  *Transformer  // Small, fast model (e.g., 125M params)
    TargetModel *Transformer  // Large, accurate model (e.g., 7B params)
    NumDraft    int           // How many tokens to draft (typically 4-8)
}

// Generate uses speculative decoding for faster inference
func (sd *SpeculativeDecoder) Generate(
    prompt []int,
    maxTokens int,
) []int {
    output := make([]int, len(prompt))
    copy(output, prompt)

    drafted := 0
    accepted := 0

    for len(output) < len(prompt)+maxTokens {
        // Step 1: Draft model generates K candidate tokens
        draftTokens := sd.DraftModel.Generate(output, sd.NumDraft)

        // Step 2: Target model verifies all K+1 positions in parallel
        // This is key: batch verification is much faster than K serial calls
        candidates := append(output, draftTokens...)
        targetLogits := sd.TargetModel.ForwardBatch(candidates)

        // Step 3: Accept tokens while probabilities align
        accepted := 0
        for i := 0; i < len(draftTokens); i++ {
            pos := len(output) + i
            targetToken := Sample(targetLogits[pos])

            if targetToken == draftTokens[i] {
                output = append(output, targetToken)
                accepted++
            } else {
                // Mismatch: accept target model's token and restart
                output = append(output, targetToken)
                break
            }
        }

        drafted += sd.NumDraft

        // If all K tokens were accepted, we're doing well
        if accepted == sd.NumDraft {
            // Bonus: generate one more token from target model
            extraToken := Sample(targetLogits[len(output)])
            output = append(output, extraToken)
        }
    }

    acceptRate := float64(accepted) / float64(drafted)
    log.Printf("Speculative decoding: %.1f%% acceptance rate", acceptRate*100)

    return output[:len(prompt)+maxTokens]
}
```

**When to Use:**
- Target model is >10× larger than draft model
- Latency-sensitive applications
- Acceptable to train/deploy two models

**Speedup Calculation:**
```
Without speculation: Time = N × T_large
With speculation:    Time ≈ N/K × (T_small × K + T_large)

Example (K=4, T_large=100ms, T_small=10ms):
Without: 100 tokens × 100ms = 10,000ms
With:    100/4 × (4×10ms + 100ms) = 25 × 140ms = 3,500ms
Speedup: 2.9×
```

---

## Batching Strategies

### Dynamic Batching

Dynamic batching groups multiple requests to maximize throughput:

```go
// BatchProcessor accumulates requests and processes them together
// This is essential for GPU utilization: batch size 1 uses <10% of GPU
type BatchProcessor struct {
    model         *Transformer
    maxBatchSize  int
    maxWaitTime   time.Duration
    requestQueue  chan *InferenceRequest
    mu            sync.Mutex
}

type InferenceRequest struct {
    Input    []int
    Response chan *InferenceResponse
    EnqueueTime time.Time
}

type InferenceResponse struct {
    Output  []int
    Latency time.Duration
}

// Start begins the batch processing loop
func (bp *BatchProcessor) Start() {
    go bp.processBatches()
}

// processBatches is the core batching logic
// Trade-off: Wait longer for bigger batches vs. lower latency
func (bp *BatchProcessor) processBatches() {
    ticker := time.NewTicker(bp.maxWaitTime)
    defer ticker.Stop()

    var batch []*InferenceRequest

    for {
        select {
        case req := <-bp.requestQueue:
            batch = append(batch, req)

            // Process immediately if batch is full
            if len(batch) >= bp.maxBatchSize {
                bp.executeBatch(batch)
                batch = nil
            }

        case <-ticker.C:
            // Process partial batch after timeout
            if len(batch) > 0 {
                bp.executeBatch(batch)
                batch = nil
            }
        }
    }
}

// executeBatch processes multiple requests in parallel
func (bp *BatchProcessor) executeBatch(batch []*InferenceRequest) {
    start := time.Now()
    batchSize := len(batch)

    // Find max sequence length in batch
    maxLen := 0
    for _, req := range batch {
        if len(req.Input) > maxLen {
            maxLen = len(req.Input)
        }
    }

    // Pad all inputs to same length (required for batching)
    paddedInputs := make([][]int, batchSize)
    for i, req := range batch {
        paddedInputs[i] = make([]int, maxLen)
        copy(paddedInputs[i], req.Input)
        // Pad with special token (typically 0 or PAD token)
        for j := len(req.Input); j < maxLen; j++ {
            paddedInputs[i][j] = bp.model.Tokenizer.PadToken
        }
    }

    // Process entire batch in single forward pass
    // This is where GPU utilization goes from 10% → 80%+
    outputs := bp.model.ForwardBatch(paddedInputs)

    // Return results to individual requests
    batchLatency := time.Since(start)
    for i, req := range batch {
        queueLatency := start.Sub(req.EnqueueTime)
        totalLatency := time.Since(req.EnqueueTime)

        req.Response <- &InferenceResponse{
            Output:  outputs[i],
            Latency: totalLatency,
        }

        // Log metrics
        log.Printf("Request %d: queue=%dms batch=%dms total=%dms",
            i, queueLatency.Milliseconds(),
            batchLatency.Milliseconds(),
            totalLatency.Milliseconds())
    }
}

// Submit adds a request to the queue
func (bp *BatchProcessor) Submit(input []int) *InferenceResponse {
    respChan := make(chan *InferenceResponse, 1)
    req := &InferenceRequest{
        Input:       input,
        Response:    respChan,
        EnqueueTime: time.Now(),
    }

    bp.requestQueue <- req
    return <-respChan
}
```

**Batching Trade-offs:**

| Batch Size | GPU Util | Throughput | Latency | Use Case |
|-----------|----------|------------|---------|----------|
| 1 | 5-10% | 10 req/s | 50ms | Real-time chat |
| 4 | 30-40% | 40 req/s | 75ms | Interactive apps |
| 16 | 70-80% | 140 req/s | 150ms | Batch processing |
| 64 | 90-95% | 400 req/s | 500ms | Offline jobs |

**Key Insight:** Latency grows sub-linearly while throughput grows linearly with batch size, making batching highly effective for throughput optimization.

---

### Continuous Batching (PagedAttention)

Continuous batching removes the constraint that all sequences must finish together:

```
Traditional Batching:
Batch 1: [====]  [==]  [=======]  [===]
         Wait for longest sequence (7 tokens)
         GPU idle 50% of the time!

Continuous Batching:
Batch:   [====]
         [==] [new1]
         [====] [new1]
         [===] [new1] [new2]
         [new1] [new2] [new3]
         GPU utilization: 90%+
```

```go
// ContinuousBatcher maintains a dynamic batch of in-flight requests
// This is how production systems (vLLM, TensorRT-LLM) achieve high throughput
type ContinuousBatcher struct {
    model        *Transformer
    maxBatchSize int
    maxSeqLen    int
    activeReqs   map[string]*ActiveRequest
    mu           sync.Mutex
}

type ActiveRequest struct {
    ID          string
    Tokens      []int
    MaxTokens   int
    Response    chan int  // Stream tokens back
    KVCache     *KVCache
    IsFinished  bool
}

// ProcessStep runs one decoding step for all active requests
func (cb *ContinuousBatcher) ProcessStep() {
    cb.mu.Lock()
    defer cb.mu.Unlock()

    if len(cb.activeReqs) == 0 {
        return
    }

    // Collect all active requests
    batch := make([]*ActiveRequest, 0, len(cb.activeReqs))
    for _, req := range cb.activeReqs {
        if !req.IsFinished {
            batch = append(batch, req)
        }
    }

    if len(batch) == 0 {
        return
    }

    // Prepare batched input (each request has 1 token to process)
    batchInputs := make([][]int, len(batch))
    caches := make([]*KVCache, len(batch))
    for i, req := range batch {
        batchInputs[i] = []int{req.Tokens[len(req.Tokens)-1]}
        caches[i] = req.KVCache
    }

    // Single forward pass for entire batch
    logits := cb.model.ForwardBatchWithCache(batchInputs, caches)

    // Generate next token for each request
    for i, req := range batch {
        nextToken := Sample(logits[i])
        req.Tokens = append(req.Tokens, nextToken)

        // Stream token back to caller
        req.Response <- nextToken

        // Check termination conditions
        if nextToken == cb.model.Tokenizer.EOSToken ||
           len(req.Tokens) >= req.MaxTokens {
            req.IsFinished = true
            close(req.Response)
            delete(cb.activeReqs, req.ID)
        }
    }
}

// Add inserts a new request into the active batch
func (cb *ContinuousBatcher) Add(
    prompt []int,
    maxTokens int,
) chan int {
    cb.mu.Lock()
    defer cb.mu.Unlock()

    req := &ActiveRequest{
        ID:         generateID(),
        Tokens:     prompt,
        MaxTokens:  maxTokens,
        Response:   make(chan int, 10),  // Buffered for streaming
        KVCache:    &KVCache{},
        IsFinished: false,
    }

    cb.activeReqs[req.ID] = req
    return req.Response
}

// Run starts the continuous batching loop
func (cb *ContinuousBatcher) Run() {
    ticker := time.NewTicker(10 * time.Millisecond)
    defer ticker.Stop()

    for range ticker.C {
        cb.ProcessStep()
    }
}
```

**Benefits:**
- **Higher GPU utilization:** 80-95% vs 40-60% with static batching
- **Lower latency:** No waiting for batch to fill
- **Better throughput:** Can mix long and short sequences efficiently

**Real-world impact:**
- vLLM reports 2-4× higher throughput vs. HuggingFace Transformers
- TensorRT-LLM achieves 3-5× higher throughput vs. PyTorch

---

## Model Quantization

Quantization reduces model size and increases speed by using lower precision:

### INT8 Quantization

```go
// QuantizedLinear stores weights in INT8 for 4× memory reduction
type QuantizedLinear struct {
    // Original weights: float32 [out_features, in_features]
    // Quantized weights: int8 [out_features, in_features]
    WeightsInt8 []int8
    Scale       float32  // Quantization scale factor
    ZeroPoint   int8     // Quantization zero point
    InFeatures  int
    OutFeatures int
}

// Quantize converts float32 weights to int8
// Formula: Q = round(W / scale) + zero_point
func Quantize(weights *Tensor) *QuantizedLinear {
    data := weights.Data

    // Find min/max for asymmetric quantization
    min := data[0]
    max := data[0]
    for _, v := range data {
        if v < min {
            min = v
        }
        if v > max {
            max = v
        }
    }

    // Compute scale and zero point
    // Map float range [min, max] → int8 range [-128, 127]
    scale := (max - min) / 255.0
    zeroPoint := int8(-min/scale - 128)

    // Quantize each weight
    weightsInt8 := make([]int8, len(data))
    for i, w := range data {
        quantized := int32(w/scale) + int32(zeroPoint)
        // Clamp to int8 range
        if quantized < -128 {
            quantized = -128
        }
        if quantized > 127 {
            quantized = 127
        }
        weightsInt8[i] = int8(quantized)
    }

    return &QuantizedLinear{
        WeightsInt8: weightsInt8,
        Scale:       scale,
        ZeroPoint:   zeroPoint,
        InFeatures:  weights.Shape()[1],
        OutFeatures: weights.Shape()[0],
    }
}

// Forward performs quantized matrix multiplication
func (ql *QuantizedLinear) Forward(x *Tensor) *Tensor {
    batchSize := x.Shape()[0]
    seqLen := x.Shape()[1]

    // Dequantize on-the-fly during computation
    // Modern CPUs/GPUs have fast int8 matrix multiplication
    output := make([]float32, batchSize*seqLen*ql.OutFeatures)

    for b := 0; b < batchSize; b++ {
        for s := 0; s < seqLen; s++ {
            for o := 0; o < ql.OutFeatures; o++ {
                sum := int32(0)
                for i := 0; i < ql.InFeatures; i++ {
                    // INT8 multiplication (fast!)
                    xIdx := b*seqLen*ql.InFeatures + s*ql.InFeatures + i
                    wIdx := o*ql.InFeatures + i

                    xQuantized := int8(x.Data[xIdx] / ql.Scale)
                    sum += int32(xQuantized) * int32(ql.WeightsInt8[wIdx])
                }

                // Dequantize result
                outIdx := b*seqLen*ql.OutFeatures + s*ql.OutFeatures + o
                output[outIdx] = (float32(sum) - float32(ql.ZeroPoint)) * ql.Scale
            }
        }
    }

    return &Tensor{
        Data:  output,
        shape: []int{batchSize, seqLen, ql.OutFeatures},
    }
}
```

**Quantization Impact:**

| Precision | Memory | Speed | Perplexity | Use Case |
|-----------|--------|-------|------------|----------|
| FP32 | 100% | 1× | Baseline | Training |
| FP16 | 50% | 1.5-2× | +0.01 | Training/inference |
| INT8 | 25% | 2-3× | +0.1 | Inference |
| INT4 | 12.5% | 3-4× | +0.5 | Aggressive inference |

**Example (GPT-2 small):**
- FP32: 548 MB, 50 tokens/sec
- INT8: 137 MB, 120 tokens/sec
- INT4: 69 MB, 180 tokens/sec

---

## Serving Architectures

### Microservices Pattern

```
┌──────────────────────────────────────────────┐
│              API Gateway                      │
│         (Authentication, Rate Limiting)       │
└───────┬──────────────────────────────────────┘
        │
        ├─────────┬─────────┬─────────┬─────────┐
        ▼         ▼         ▼         ▼         ▼
   ┌─────────┐ ┌──────────┐ ┌──────┐ ┌──────┐ ┌──────┐
   │  Load   │ │ Embedding│ │Ranking│ │Safety│ │ Cache│
   │ Balancer│ │  Service │ │Service│ │Filter│ │(Redis)│
   └────┬────┘ └──────────┘ └──────┘ └──────┘ └──────┘
        │
        ├──────────┬──────────┬──────────┐
        ▼          ▼          ▼          ▼
   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
   │ Model  │ │ Model  │ │ Model  │ │ Model  │
   │ Pod 1  │ │ Pod 2  │ │ Pod 3  │ │ Pod 4  │
   │ (GPU)  │ │ (GPU)  │ │ (GPU)  │ │ (GPU)  │
   └────────┘ └────────┘ └────────┘ └────────┘
```

**Benefits:**
- Independent scaling of components
- Easier to update/replace individual services
- Can use different hardware for different services

**Challenges:**
- More complex networking
- Service discovery overhead
- Need for distributed tracing

---

## Monitoring and Observability

### Key Metrics to Track

```go
// MetricsCollector tracks inference metrics for monitoring
type MetricsCollector struct {
    // Latency metrics
    P50Latency prometheus.Histogram
    P95Latency prometheus.Histogram
    P99Latency prometheus.Histogram

    // Throughput metrics
    RequestsPerSecond prometheus.Counter
    TokensPerSecond   prometheus.Counter

    // Resource metrics
    GPUUtilization prometheus.Gauge
    GPUMemoryUsed  prometheus.Gauge
    BatchSize      prometheus.Histogram

    // Quality metrics
    TokenAcceptanceRate prometheus.Histogram  // For speculative decoding
    CacheHitRate        prometheus.Gauge
}

// RecordInference logs metrics for a single inference request
func (mc *MetricsCollector) RecordInference(
    latency time.Duration,
    inputTokens int,
    outputTokens int,
    batchSize int,
) {
    mc.P50Latency.Observe(latency.Seconds())
    mc.P95Latency.Observe(latency.Seconds())
    mc.P99Latency.Observe(latency.Seconds())

    tokensPerSec := float64(outputTokens) / latency.Seconds()
    mc.TokensPerSecond.Add(tokensPerSec)
    mc.BatchSize.Observe(float64(batchSize))
}
```

**Critical Alerts:**
- P99 latency > 2× P50 latency (tail latency issues)
- GPU utilization < 60% (underutilized hardware)
- Error rate > 1% (model or infrastructure problems)
- OOM crashes (need larger instances or quantization)

---

## Cost Optimization

### Spot Instances for Batch Workloads

```yaml
# Use spot instances for non-critical batch inference
# Can save 60-90% on compute costs
apiVersion: v1
kind: Pod
metadata:
  name: batch-inference
spec:
  nodeSelector:
    node.kubernetes.io/instance-type: g5.xlarge  # NVIDIA A10G
  tolerations:
  - key: "spot"
    operator: "Equal"
    value: "true"
    effect: "NoSchedule"
  containers:
  - name: inference
    image: transformer-batch:latest
    resources:
      requests:
        nvidia.com/gpu: 1
```

**Cost Comparison (AWS):**
- On-demand g5.xlarge: $1.01/hr
- Spot g5.xlarge: $0.30/hr (70% savings)
- Reserved (3-year): $0.60/hr (40% savings)

### Auto-scaling Guidelines

```python
# Scale based on queue depth, not CPU/GPU utilization
# Why? Model inference is latency-sensitive

if queue_depth > 100:
    scale_up()  # Add 2 pods
elif queue_depth < 20 and num_pods > min_pods:
    scale_down()  # Remove 1 pod
```

**Cost-Optimized Deployment:**
1. Use spot instances for 70% of capacity
2. Use on-demand instances for 30% (handle spot interruptions)
3. Scale to zero during off-hours (if traffic allows)
4. Cache common prompts (Redis)
5. Use quantized models for less-critical workloads

---

## Production Checklist

### Before Deploying to Production

- [ ] Load testing (sustained 2× expected peak traffic)
- [ ] Latency testing (P50, P95, P99 under load)
- [ ] Error handling (graceful degradation, retries)
- [ ] Monitoring (metrics, logs, alerts)
- [ ] Resource limits (prevent OOM crashes)
- [ ] Health checks (liveness, readiness probes)
- [ ] Security (input validation, rate limiting)
- [ ] Model versioning (A/B testing, rollbacks)
- [ ] Cost estimation (monthly spend projection)
- [ ] Disaster recovery (multi-region, backups)

### Common Pitfalls

**1. Not Using KV Cache:**
- Impact: 10-100× slower generation
- Fix: Implement KV caching (shown above)

**2. Batch Size = 1:**
- Impact: 5-10% GPU utilization, wasted money
- Fix: Implement dynamic batching (max wait time 50-100ms)

**3. No Request Timeout:**
- Impact: Stuck requests block resources
- Fix: Set timeouts (30-60 seconds for generation)

**4. Cold Start Issues:**
- Impact: First request takes 10-30 seconds (model loading)
- Fix: Pre-warm instances, keep min replicas > 0

**5. OOM Crashes:**
- Impact: Service unavailable, lost requests
- Fix: Set memory limits, implement quantization, monitor memory

---

## Related Documentation

- [NVIDIA GPU Selection Guide](./nvidia-gpu-guide.md) - Choose hardware for deployment
- [Optimization Guide](./optimization-guide.md) - Performance tuning techniques
- [Training Workflows](./training-workflows.md) - From training to deployment
- [GPU Acceleration Guide](./gpu-acceleration-guide.md) - Hardware-specific optimizations

