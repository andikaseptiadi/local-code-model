# Training Dynamics and Loss Curves

**Understanding How Models Learn**

## Table of Contents

1. [Introduction](#introduction)
2. [The Loss Curve](#the-loss-curve)
3. [Learning Rate Schedules](#learning-rate-schedules)
4. [Optimizer Dynamics](#optimizer-dynamics)
5. [Training Phases](#training-phases)
6. [Monitoring Training](#monitoring-training)
7. [Common Problems](#common-problems)
8. [Practical Exercises](#practical-exercises)

## Introduction

Training a neural network is an optimization problem: we're trying to find parameters that minimize loss on our training data. But the journey from random initialization to a trained model is complex and fascinating.

This tutorial explains what happens during training, how to interpret loss curves, and how to diagnose and fix common training problems.

## The Loss Curve

### What Is Loss?

Loss measures how wrong the model's predictions are. For language modeling, we use **cross-entropy loss**:

```
Loss = -log(P(correct_token))
```

Where `P(correct_token)` is the probability the model assigns to the actual next token.

**Example:**
```
Sentence: "The cat sat on the"
Model predicts: {"mat": 0.4, "floor": 0.3, "chair": 0.2, "car": 0.1}
Actual word: "mat"

Loss = -log(0.4) ≈ 0.916
```

If the model predicted perfectly (probability 1.0 for "mat"):
```
Loss = -log(1.0) = 0
```

### A Typical Loss Curve

Here's what training looks like for our tiny GPT model:

```
5.5 |
    |  *
5.0 |   **
    |     ***
4.5 |        ****
    |            *****
4.0 |                 ******
    |                       *******
3.5 |                              *******
    |                                     ********
3.0 |                                            ******
    +-------------------------------------------------->
    0        200       400       600       800      1000
                        Training Steps
```

**Key observations:**

1. **Initial spike**: Loss often increases briefly at the start (steps 0-50)
2. **Rapid descent**: Loss drops quickly in early training (steps 50-300)
3. **Plateau**: Loss decreases slowly later (steps 300-1000)
4. **Noisy**: Loss fluctuates batch-to-batch (not perfectly smooth)

### Why Does Loss Start High?

At initialization, the model assigns roughly equal probability to all tokens:

```
Vocabulary size: 259 characters
Initial prediction: ~1/259 ≈ 0.00386 for each token
Initial loss: -log(0.00386) ≈ 5.56
```

This matches what we see in the example output (starting around 5.47-5.56).

### Why Does Loss Sometimes Increase Initially?

The brief increase (5.47 → 5.62 in our example) happens because:

1. **Learning rate warmup**: The model starts with a very low learning rate (0.0001)
2. **Parameter instability**: Some parameters move in suboptimal directions initially
3. **Batch variance**: Early batches might be harder than average

This is normal and usually corrects itself within 50-100 steps.

## Learning Rate Schedules

The learning rate (LR) controls how large each parameter update is:

```
new_param = old_param - learning_rate × gradient
```

### Why Not Use Constant Learning Rate?

**Too high**: Parameters oscillate or diverge
```
4.0 |    *   *   *   *   *   *   *   *
    |   * * * * * * * * * * * * * * *
3.5 |  *  *  *  *  *  *  *  *  *  *  *
    +--------------------------------->
    Never converges!
```

**Too low**: Training is painfully slow
```
4.0 |*
    | *
3.5 |  *
    |   *
3.0 |    *
    |     *
2.5 |      * (still going after 10,000 steps...)
    +--------------------------------->
```

### The Three-Phase Schedule

Our implementation uses a sophisticated schedule:

#### Phase 1: Warmup (Steps 0-10)

```
LR: 0.0001 → 0.001 (linear increase)
```

**Why warmup?**
- Prevents large updates from disrupting random initialization
- Allows the model to "orient itself" in parameter space
- Stabilizes training dynamics

**Code:**
```go
if step < warmupSteps {
    lr = baseLR * float64(step) / float64(warmupSteps)
}
```

#### Phase 2: Training (Steps 10-80% of total)

```
LR: 0.001 (constant)
```

**Why constant?**
- Model makes rapid progress
- Loss decreases quickly
- Parameters explore the solution space

#### Phase 3: Cosine Decay (Steps 80%-100%)

```
LR: 0.001 → 0.0001 (smooth decrease)
```

**Why decay?**
- Allows model to fine-tune
- Prevents oscillation around minimum
- Produces smoother convergence

**Formula (cosine annealing):**
```go
progress := float64(step-warmupSteps) / float64(totalSteps-warmupSteps)
lr = minLR + (baseLR-minLR) * 0.5 * (1 + math.Cos(progress*math.Pi))
```

This creates a smooth curve from `baseLR` to `minLR`.

### Visualizing Our Learning Rate Schedule

For 1000 total steps with 10 warmup steps:

```
0.0010 |    ************************
       |   *                         ****
       |  *                              ****
       | *                                   ****
0.0005 |*                                        ****
       |                                             ****
       |                                                 ***
0.0001 |                                                    *
       +-------------------------------------------------->
       0    10   100   200   300   400   500   600   700   800   900   1000
            └─┬─┘ └──────────┬──────────┘ └─────────┬─────────┘
            Warmup         Constant              Cosine Decay
```

## Optimizer Dynamics

### SGD vs Adam

**Stochastic Gradient Descent (SGD)**:
```go
param = param - learning_rate × gradient
```

Simple but inefficient. Each parameter gets the same learning rate.

**Adam (Adaptive Moment Estimation)**:
```go
m = β₁ × m + (1-β₁) × gradient          // momentum
v = β₂ × v + (1-β₂) × gradient²         // variance
param = param - learning_rate × m / (√v + ε)
```

Adam adapts the learning rate per parameter:
- **Fast-changing parameters** (large gradients): Get smaller updates
- **Slow-changing parameters** (small gradients): Get larger updates

### Why Adam Converges Faster

Consider two parameters:
- Parameter A: Gradient = 10.0 (very noisy)
- Parameter B: Gradient = 0.01 (steady signal)

**SGD** treats them the same:
```
Update A: -0.001 × 10.0 = -0.01
Update B: -0.001 × 0.01 = -0.00001
```

**Adam** adapts:
```
Update A: -0.001 × 10.0 / √(100 + ε) ≈ -0.001    (dampened)
Update B: -0.001 × 0.01 / √(0.0001 + ε) ≈ -0.001  (amplified)
```

Both parameters make similar-sized updates!

### Adam Hyperparameters

From `optimizer.go`:

```go
β₁ = 0.9    // Momentum decay rate
β₂ = 0.999  // Variance decay rate
ε = 1e-8    // Numerical stability
```

**What they control:**

- **β₁ (momentum)**: How much to "remember" previous gradients
  - Higher (0.9-0.99): Smoother updates, less noise
  - Lower (0.5-0.8): More responsive to recent gradients

- **β₂ (variance)**: How much to "remember" gradient variance
  - Higher (0.999): More stable adaptation
  - Lower (0.99): Faster adaptation to changes

### Bias Correction

Adam applies bias correction in early training:

```go
m_hat = m / (1 - β₁^t)
v_hat = v / (1 - β₂^t)
```

**Why?** At step 1, `m` and `v` are initialized to zero, so they're biased toward zero. Correction compensates for this.

After ~100 steps, `β₁^t` and `β₂^t` approach zero, so correction has minimal effect.

## Training Phases

### Phase 1: Memorization (Steps 0-200)

**What's happening:**
- Model memorizes common patterns in training data
- Learns frequent tokens and sequences
- Loss drops rapidly

**Loss behavior:**
```
5.5 → 4.0 (rapid decrease)
```

**What to look for:**
- Steady decrease in loss
- Learning rate ramping up during warmup
- No NaN or exploding gradients

### Phase 2: Generalization (Steps 200-600)

**What's happening:**
- Model learns underlying structure
- Begins to compose patterns
- Loss decreases more slowly

**Loss behavior:**
```
4.0 → 3.5 (slower decrease)
```

**What to look for:**
- Continued but slower loss decrease
- Generated text shows more coherent patterns
- Model produces reasonable outputs on unseen prompts

### Phase 3: Refinement (Steps 600-1000)

**What's happening:**
- Model fine-tunes parameters
- Loss improvements become marginal
- Risk of overfitting increases

**Loss behavior:**
```
3.5 → 3.2 (slow decrease)
```

**What to look for:**
- Loss curves flattening
- Diminishing returns on training time
- Check validation loss to detect overfitting

### Understanding Perplexity

Perplexity is an alternative metric: `perplexity = exp(loss)`

```
Loss 5.5 → Perplexity 245
Loss 4.0 → Perplexity 55
Loss 3.0 → Perplexity 20
Loss 2.0 → Perplexity 7
Loss 1.0 → Perplexity 2.7
```

**Interpretation**: Perplexity ~20 means the model is "confused between about 20 equally likely tokens" on average.

For reference:
- **Random model**: Perplexity = vocabulary size (259 for us)
- **Perfect model**: Perplexity = 1
- **Good small model**: Perplexity = 20-50
- **GPT-2 small**: Perplexity = 29 (on WikiText)

## Monitoring Training

### Key Metrics to Track

1. **Training Loss**
   - Primary indicator of learning
   - Should decrease steadily
   - Watch for stalls or divergence

2. **Learning Rate**
   - Verify schedule is working correctly
   - Should show warmup → constant → decay

3. **Gradient Norms**
   - Average magnitude of gradients
   - Should be stable (not exploding or vanishing)

4. **Parameter Norms**
   - Average magnitude of parameters
   - Should increase slowly during training

### Example: Training Output Analysis

```
Step 1:   Loss: 5.4727, LR: 0.000100
Step 11:  Loss: 5.5216, LR: 0.001000  ← Warmup complete
Step 21:  Loss: 5.6214, LR: 0.001000  ← Brief spike (normal)
...
Step 231: Loss: 5.6322, LR: 0.000919  ← Decay begins (~80% through)
Step 241: Loss: 5.5906, LR: 0.000911
Step 251: Loss: 5.5779, LR: 0.000904  ← Steady decrease
```

**Analysis:**
1. ✅ Warmup working (LR: 0.0001 → 0.001 by step 11)
2. ✅ Brief loss spike at step 21 (expected, resolved quickly)
3. ✅ Loss decreasing by step 251 (learning is happening)
4. ✅ Cosine decay active (LR decreasing from 0.000919)

### When to Stop Training

**Signs training is complete:**
- Loss has plateaued for 100+ steps
- Validation loss starts increasing (overfitting)
- Generated text quality stops improving
- Diminishing returns on compute time

**For our tiny model:**
- 2 epochs (~1000 steps) is usually sufficient
- Loss should reach 3.0-4.0 range
- Model learns basic syntax patterns

**For production models:**
- May train for days or weeks
- Validation loss is critical
- Early stopping based on validation plateau

## Common Problems

### Problem 1: Loss is NaN

**Symptoms:**
```
Step 1:   Loss: 5.4727
Step 2:   Loss: 5.5123
Step 3:   Loss: NaN
```

**Causes:**
1. Learning rate too high
2. Exploding gradients
3. Numerical instability in attention

**Solutions:**
```go
// Add gradient clipping
maxNorm := 1.0
for _, param := range model.Parameters() {
    norm := param.Grad.Norm()
    if norm > maxNorm {
        param.Grad = param.Grad.MulScalar(maxNorm / norm)
    }
}

// Reduce learning rate
lr := 0.0001  // Instead of 0.001

// Add numerical stability to softmax
func Softmax(x *Tensor) *Tensor {
    maxVal := x.Max()
    e := x.Sub(maxVal).Exp()  // Subtract max prevents overflow
    return e.Div(e.Sum())
}
```

### Problem 2: Loss Not Decreasing

**Symptoms:**
```
Step 1:    Loss: 5.4727
Step 100:  Loss: 5.4689
Step 200:  Loss: 5.4701
Step 300:  Loss: 5.4673
```

**Causes:**
1. Learning rate too low
2. Model too small for the task
3. Insufficient training data
4. Bug in backpropagation

**Debugging checklist:**
```go
// 1. Verify gradients exist
for name, param := range model.Parameters() {
    if param.Grad == nil {
        fmt.Printf("Parameter %s has no gradient!\n", name)
    }
    gradNorm := param.Grad.Norm()
    fmt.Printf("%s gradient norm: %f\n", name, gradNorm)
}

// 2. Try a tiny test case
// Can the model overfit a single batch?
for i := 0; i < 1000; i++ {
    loss := model.Forward(singleBatch)
    loss.Backward()
    optimizer.Step()
    // Loss should go to nearly zero
}

// 3. Check learning rate schedule
fmt.Printf("Step %d: LR = %f\n", step, currentLR)

// 4. Increase model capacity
config.NumLayers = 4      // Was 2
config.EmbedDim = 128     // Was 64
```

### Problem 3: Overfitting

**Symptoms:**
```
Training Loss:   2.5 (still decreasing)
Validation Loss: 3.8 (increasing!)
```

**Cause:** Model memorizing training data instead of learning patterns.

**Solutions:**
```go
// 1. Early stopping
if validationLoss > bestValidationLoss {
    patienceCounter++
    if patienceCounter > patience {
        break  // Stop training
    }
}

// 2. Add dropout
type TransformerLayer struct {
    ...
    DropoutRate float64  // e.g., 0.1
}

func (l *TransformerLayer) Forward(x *Tensor, training bool) *Tensor {
    ...
    if training {
        x = x.Dropout(l.DropoutRate)
    }
    return x
}

// 3. Reduce model size or increase data
```

### Problem 4: Loss Oscillating

**Symptoms:**
```
Step 100: Loss: 4.2
Step 101: Loss: 3.8
Step 102: Loss: 4.5
Step 103: Loss: 3.9
Step 104: Loss: 4.3
```

**Causes:**
1. Learning rate too high
2. Batch size too small
3. Noisy gradients

**Solutions:**
```go
// Reduce learning rate
lr = lr * 0.5

// Increase batch size
batchSize = 8  // Was 4

// Add gradient accumulation
accumSteps := 4
for i := 0; i < accumSteps; i++ {
    loss := model.Forward(batch[i])
    loss.Backward()
}
optimizer.Step()  // Update once every 4 batches
```

### Problem 5: Training Too Slow

**Symptoms:**
- 10+ minutes per epoch
- Can't iterate quickly

**Solutions:**

```go
// 1. Reduce sequence length
seqLen = 32  // Was 64

// 2. Reduce batch size (less memory, faster)
batchSize = 2  // Was 4

// 3. Reduce model size
config.NumLayers = 1    // Was 2
config.EmbedDim = 32    // Was 64

// 4. Use smaller dataset for initial experiments
maxFiles := 10  // Only use 10 .go files

// 5. Profile and optimize bottlenecks
// See performance notes in tensor.go
```

## Practical Exercises

### Exercise 1: Analyzing a Loss Curve

Given this training output:

```
Step 1:   Loss: 5.472
Step 50:  Loss: 5.801
Step 100: Loss: 5.234
Step 200: Loss: 4.567
Step 300: Loss: 4.123
Step 400: Loss: 3.889
Step 500: Loss: 3.756
Step 600: Loss: 3.689
Step 700: Loss: 3.645
Step 800: Loss: 3.621
Step 900: Loss: 3.609
```

**Questions:**
1. When does the warmup period end?
2. Is there an initial loss spike? How long does it last?
3. At what step does learning rate decay likely begin?
4. Is the model still improving at step 900?
5. Should you train for more steps?

**Answers:**
1. Around step 50 (loss spike peaks, then rapid decrease begins)
2. Yes, from 5.472 → 5.801 (steps 1-50), resolves by step 100
3. Around step 640 (80% of 800 total steps, loss decrease slows)
4. Yes, but slowly (3.645 → 3.609 = -0.036 over 200 steps)
5. Maybe 100-200 more steps, but diminishing returns

### Exercise 2: Implementing Loss Logging

Modify the training loop to track and visualize loss:

```go
type TrainingMetrics struct {
    Steps          []int
    Losses         []float64
    LearningRates  []float64
}

func TrainWithMetrics(model *Transformer, data [][]int, config TrainConfig) *TrainingMetrics {
    metrics := &TrainingMetrics{}

    for step := 0; step < totalSteps; step++ {
        loss := /* compute loss */
        lr := /* compute learning rate */

        // Log every 10 steps
        if step % 10 == 0 {
            metrics.Steps = append(metrics.Steps, step)
            metrics.Losses = append(metrics.Losses, loss.At(0))
            metrics.LearningRates = append(metrics.LearningRates, lr)
        }

        /* backprop and optimizer step */
    }

    return metrics
}

// Then visualize (pseudo-code)
func PlotMetrics(m *TrainingMetrics) {
    // Plot loss curve
    // Plot learning rate schedule
    // Save to loss_curve.png
}
```

### Exercise 3: Gradient Monitoring

Add gradient monitoring to detect problems early:

```go
func ComputeGradientStats(model *Transformer) {
    var totalNorm float64
    var maxGrad float64
    var minGrad float64 = math.Inf(1)

    for _, param := range model.Parameters() {
        if param.Grad == nil {
            continue
        }

        norm := param.Grad.Norm()
        totalNorm += norm * norm

        max := param.Grad.Max()
        min := param.Grad.Min()

        if max > maxGrad {
            maxGrad = max
        }
        if min < minGrad {
            minGrad = min
        }
    }

    totalNorm = math.Sqrt(totalNorm)

    fmt.Printf("Gradient stats: norm=%.4f, max=%.4f, min=%.4f\n",
               totalNorm, maxGrad, minGrad)

    // Warning signs
    if totalNorm > 10.0 {
        fmt.Println("WARNING: Exploding gradients!")
    }
    if totalNorm < 1e-6 {
        fmt.Println("WARNING: Vanishing gradients!")
    }
}
```

## Key Takeaways

1. **Loss curves tell a story**: Initial spike, rapid descent, slow convergence
2. **Learning rate scheduling is crucial**: Warmup prevents instability, decay enables fine-tuning
3. **Adam adapts per-parameter**: Different parameters need different learning rates
4. **Training has phases**: Memorization → Generalization → Refinement
5. **Monitor multiple metrics**: Loss, LR, gradients, parameters
6. **Common problems have solutions**: NaN → clip gradients, slow → reduce LR, overfitting → regularize
7. **Know when to stop**: Plateau, validation loss divergence, diminishing returns

## Further Reading

- **Papers:**
  - ["Adam: A Method for Stochastic Optimization"](https://arxiv.org/abs/1412.6980) (Kingma & Ba, 2014)
  - ["Cyclical Learning Rates"](https://arxiv.org/abs/1506.01186) (Smith, 2015)
  - ["Accurate, Large Minibatch SGD"](https://arxiv.org/abs/1706.02677) (Goyal et al., 2017)

- **Books:**
  - "Deep Learning" (Goodfellow et al., 2016) - Chapter 8: Optimization

- **Blogs:**
  - [An Overview of Gradient Descent Optimization Algorithms](https://ruder.io/optimizing-gradient-descent/)
  - [The Marginal Value of Adaptive Gradient Methods](https://blog.ml.cmu.edu/2020/08/31/parameters-hyperparameters-and-the-importance-of-both/)

## Next Steps

Now that you understand training dynamics, explore:
1. [Attention Mechanism](attention-mechanism.md) - How attention computes context
2. [Backpropagation Through Transformers](backpropagation.md) - How gradients flow
3. Implementation details in `train.go` and `optimizer.go`

---

*This tutorial is part of the [Local Code Model](https://github.com/scttfrdmn/local-code-model) educational project.*
