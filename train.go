package main

// ===========================================================================
// WHAT'S GOING ON HERE
// ===========================================================================
//
// This file implements the training loop for the transformer model, including
// backpropagation, optimizers, and gradient descent.
//
// INTENTION:
// Create a complete training system that demonstrates:
//   - Forward pass: Input → Model → Loss
//   - Backward pass: Loss → Gradients → Parameter updates
//   - Optimization: SGD, Adam, learning rate schedules
//   - Training loop: Epochs, batches, logging
//
// WHERE THIS SITS:
// This is the LEARNING component - how the model improves from data.
// Everything before this was about INFERENCE (using a trained model).
//
// THE TRAINING PROCESS:
//
// 1. Forward Pass:
//    - Input tokens → Embeddings → Transformer blocks → Logits
//    - Logits → Cross-entropy loss (comparing to target tokens)
//
// 2. Backward Pass (Backpropagation):
//    - Loss → ∂Loss/∂Logits (gradient of loss)
//    - Chain rule: propagate gradients back through each layer
//    - Each parameter gets a gradient: ∂Loss/∂Parameter
//
// 3. Optimization:
//    - Update rule: Parameter -= LearningRate * Gradient
//    - Advanced: Adam uses momentum + adaptive learning rates
//
// 4. Iteration:
//    - Process batches of data
//    - Repeat for multiple epochs
//    - Track loss, validate performance
//
// KEY CONCEPTS:
//
// Gradient: Direction of steepest increase in loss
//   - Negative gradient = direction to decrease loss
//   - Magnitude = how much loss changes per parameter change
//
// Learning Rate: How big a step to take in gradient direction
//   - Too large: Unstable training, divergence
//   - Too small: Slow convergence
//   - Schedule: Start high, decay over time
//
// Batch Size: Number of examples processed together
//   - Larger: More stable gradients, better parallelism
//   - Smaller: More updates per epoch, more exploration
//   - Typical: 32-256 for transformers
//
// PERFORMANCE CHARACTERISTICS:
//
// Training is dominated by matrix multiplies (same as inference):
//   - Forward pass: Same cost as inference
//   - Backward pass: ~2x forward pass (gradient computation)
//   - Total: ~3x inference cost per training step
//
// Memory:
//   - Forward: Store activations for backprop
//   - Backward: Compute gradients
//   - Optimizer: Store momentum/variance (Adam: 2x parameters)
//   - Total: ~4x parameter memory
//
// For small model (256 dim, 4 layers, 1K vocab):
//   - Parameters: ~1M (~4 MB)
//   - Training memory: ~16 MB
//   - Training speed: ~100 tokens/sec (CPU), ~10K tokens/sec (GPU)
//
// WHY THIS IS HARD:
//
// Backpropagation requires:
//   - Careful gradient tracking
//   - Numerical stability (avoid NaN, overflow)
//   - Efficient memory management
//   - Correct chain rule application
//
// The reward: A model that learns from data!
//
// ===========================================================================

import (
	"fmt"
	"math"
	"math/rand"
)

// TrainingConfig holds hyperparameters for training.
type TrainingConfig struct {
	// Optimization
	LearningRate      float64
	WeightDecay       float64 // L2 regularization
	GradientClipValue float64 // Clip gradients to prevent explosion

	// Training
	BatchSize    int
	NumEpochs    int
	MaxSteps     int // Max training steps (overrides epochs if set)

	// Learning rate schedule
	WarmupSteps  int     // Linear warmup from 0 to LearningRate
	DecaySteps   int     // Cosine decay after warmup
	MinLR        float64 // Minimum learning rate

	// Optimization algorithm
	Optimizer    string // "sgd", "adam"
	AdamBeta1    float64
	AdamBeta2    float64
	AdamEpsilon  float64

	// Logging
	LogInterval  int // Log every N steps
	EvalInterval int // Evaluate every N steps

	// Hardware
	Backend BackendConfig
}

// DefaultTrainingConfig returns sensible defaults.
func DefaultTrainingConfig() TrainingConfig {
	return TrainingConfig{
		// Optimization
		LearningRate:      3e-4, // GPT-3 uses 6e-4, we use smaller for stability
		WeightDecay:       0.01,
		GradientClipValue: 1.0,

		// Training
		BatchSize:  32,
		NumEpochs:  10,
		MaxSteps:   0, // Unlimited

		// Learning rate schedule
		WarmupSteps: 2000,
		DecaySteps:  10000,
		MinLR:       1e-5,

		// Adam optimizer (industry standard for transformers)
		Optimizer:   "adam",
		AdamBeta1:   0.9,
		AdamBeta2:   0.999,
		AdamEpsilon: 1e-8,

		// Logging
		LogInterval:  100,
		EvalInterval: 500,

		// Hardware
		Backend: DefaultBackendConfig(),
	}
}

// Optimizer interface for different optimization algorithms.
type Optimizer interface {
	// Step performs a single optimization step.
	// Updates parameters using their gradients.
	Step(params []*Tensor, lr float64)

	// ZeroGrad clears all gradients.
	ZeroGrad(params []*Tensor)
}

// SGDOptimizer implements Stochastic Gradient Descent.
type SGDOptimizer struct {
	weightDecay float64
}

// NewSGDOptimizer creates an SGD optimizer.
func NewSGDOptimizer(weightDecay float64) *SGDOptimizer {
	return &SGDOptimizer{
		weightDecay: weightDecay,
	}
}

// Step updates parameters using SGD: param -= lr * (grad + weightDecay * param).
func (opt *SGDOptimizer) Step(params []*Tensor, lr float64) {
	for _, p := range params {
		for i := range p.data {
			// L2 regularization: add weight decay
			grad := p.grad[i] + opt.weightDecay*p.data[i]

			// Update: param -= lr * grad
			p.data[i] -= lr * grad
		}
	}
}

// ZeroGrad clears gradients.
func (opt *SGDOptimizer) ZeroGrad(params []*Tensor) {
	for _, p := range params {
		p.ZeroGrad()
	}
}

// AdamOptimizer implements Adam optimization algorithm.
//
// Adam combines:
//   - Momentum (moving average of gradients)
//   - RMSProp (moving average of squared gradients)
//   - Bias correction (accounts for initialization at zero)
//
// Update rule:
//   m_t = beta1 * m_{t-1} + (1 - beta1) * grad
//   v_t = beta2 * v_{t-1} + (1 - beta2) * grad²
//   m_hat = m_t / (1 - beta1^t)  // Bias correction
//   v_hat = v_t / (1 - beta2^t)
//   param -= lr * m_hat / (sqrt(v_hat) + epsilon)
type AdamOptimizer struct {
	beta1       float64
	beta2       float64
	epsilon     float64
	weightDecay float64

	// State (one per parameter)
	m []*Tensor // First moment (momentum)
	v []*Tensor // Second moment (variance)
	t int       // Time step (for bias correction)
}

// NewAdamOptimizer creates an Adam optimizer.
func NewAdamOptimizer(params []*Tensor, beta1, beta2, epsilon, weightDecay float64) *AdamOptimizer {
	// Initialize moment tensors (same shape as parameters)
	m := make([]*Tensor, len(params))
	v := make([]*Tensor, len(params))

	for i, p := range params {
		m[i] = NewTensor(p.shape...)
		v[i] = NewTensor(p.shape...)
	}

	return &AdamOptimizer{
		beta1:       beta1,
		beta2:       beta2,
		epsilon:     epsilon,
		weightDecay: weightDecay,
		m:           m,
		v:           v,
		t:           0,
	}
}

// Step performs Adam update.
func (opt *AdamOptimizer) Step(params []*Tensor, lr float64) {
	opt.t++

	// Bias correction factors
	bias1 := 1.0 - math.Pow(opt.beta1, float64(opt.t))
	bias2 := 1.0 - math.Pow(opt.beta2, float64(opt.t))

	for i, p := range params {
		for j := range p.data {
			// Gradient with weight decay
			grad := p.grad[j] + opt.weightDecay*p.data[j]

			// Update biased first moment
			opt.m[i].data[j] = opt.beta1*opt.m[i].data[j] + (1.0-opt.beta1)*grad

			// Update biased second moment
			opt.v[i].data[j] = opt.beta2*opt.v[i].data[j] + (1.0-opt.beta2)*grad*grad

			// Bias-corrected moments
			mHat := opt.m[i].data[j] / bias1
			vHat := opt.v[i].data[j] / bias2

			// Update parameter
			p.data[i] -= lr * mHat / (math.Sqrt(vHat) + opt.epsilon)
		}
	}
}

// ZeroGrad clears gradients.
func (opt *AdamOptimizer) ZeroGrad(params []*Tensor) {
	for _, p := range params {
		p.ZeroGrad()
	}
}

// LRScheduler implements learning rate scheduling.
type LRScheduler struct {
	baseLR      float64
	minLR       float64
	warmupSteps int
	decaySteps  int
	step        int
}

// NewLRScheduler creates a learning rate scheduler.
func NewLRScheduler(baseLR, minLR float64, warmupSteps, decaySteps int) *LRScheduler {
	return &LRScheduler{
		baseLR:      baseLR,
		minLR:       minLR,
		warmupSteps: warmupSteps,
		decaySteps:  decaySteps,
		step:        0,
	}
}

// GetLR returns the current learning rate.
// Uses linear warmup followed by cosine decay.
func (sched *LRScheduler) GetLR() float64 {
	sched.step++

	// Phase 1: Linear warmup
	if sched.step < sched.warmupSteps {
		return sched.baseLR * float64(sched.step) / float64(sched.warmupSteps)
	}

	// Phase 2: Cosine decay
	if sched.step < sched.decaySteps {
		progress := float64(sched.step-sched.warmupSteps) / float64(sched.decaySteps-sched.warmupSteps)
		cosine := 0.5 * (1.0 + math.Cos(math.Pi*progress))
		return sched.minLR + (sched.baseLR-sched.minLR)*cosine
	}

	// Phase 3: Constant minimum
	return sched.minLR
}

// CrossEntropyLoss computes the cross-entropy loss for language modeling.
//
// Given:
//   - logits: (batch, vocab_size) - unnormalized scores
//   - targets: (batch) - target token IDs
//
// Computes:
//   loss = -log(softmax(logits)[target])
//   averaged over batch
//
// This is the standard loss for classification and language modeling.
func CrossEntropyLoss(logits *Tensor, targets []int) float64 {
	if len(logits.shape) != 2 {
		panic("CrossEntropyLoss expects 2D logits")
	}

	batchSize := logits.shape[0]
	vocabSize := logits.shape[1]

	if len(targets) != batchSize {
		panic(fmt.Sprintf("target length %d != batch size %d", len(targets), batchSize))
	}

	totalLoss := 0.0

	for b := 0; b < batchSize; b++ {
		// Find max logit for numerical stability
		maxLogit := logits.At(b, 0)
		for v := 1; v < vocabSize; v++ {
			if logit := logits.At(b, v); logit > maxLogit {
				maxLogit = logit
			}
		}

		// Compute log-sum-exp
		sumExp := 0.0
		for v := 0; v < vocabSize; v++ {
			sumExp += math.Exp(logits.At(b, v) - maxLogit)
		}
		logSumExp := maxLogit + math.Log(sumExp)

		// Cross-entropy: -log(softmax(logits[target]))
		targetLogit := logits.At(b, targets[b])
		loss := logSumExp - targetLogit
		totalLoss += loss
	}

	return totalLoss / float64(batchSize)
}

// TrainStep performs a single training step.
func TrainStep(model *GPT, batch [][]int, targets [][]int, optimizer Optimizer, lr float64) float64 {
	// Zero gradients
	params := model.Parameters()
	optimizer.ZeroGrad(params)

	// Forward pass (accumulate loss over batch)
	totalLoss := 0.0

	for i := range batch {
		// Forward with caching (stores activations for backward)
		logits, cache := model.ForwardWithCache(batch[i])

		// Compute loss
		loss := CrossEntropyLoss(logits, targets[i])
		totalLoss += loss

		// Backward (compute gradients)
		gradLogits := CrossEntropyBackward(logits, targets[i])
		model.Backward(gradLogits, cache)
	}

	// Average loss
	avgLoss := totalLoss / float64(len(batch))

	// Gradient clipping (prevent explosion)
	clipGradients(params, 1.0)

	// Optimizer step
	optimizer.Step(params, lr)

	return avgLoss
}

// clipGradients clips gradients by global norm.
func clipGradients(params []*Tensor, maxNorm float64) {
	// Compute global gradient norm
	globalNorm := 0.0
	for _, p := range params {
		for _, g := range p.grad {
			globalNorm += g * g
		}
	}
	globalNorm = math.Sqrt(globalNorm)

	// Clip if necessary
	if globalNorm > maxNorm {
		scale := maxNorm / globalNorm
		for _, p := range params {
			for i := range p.grad {
				p.grad[i] *= scale
			}
		}
	}
}

// Parameters returns all trainable parameters in the model.
func (gpt *GPT) Parameters() []*Tensor {
	params := make([]*Tensor, 0)

	// Embeddings
	params = append(params, gpt.tokenEmbed, gpt.posEmbed)

	// Transformer blocks
	for _, block := range gpt.blocks {
		// Attention
		params = append(params, block.attn.wq, block.attn.wk, block.attn.wv, block.attn.wo)

		// LayerNorm (attention)
		params = append(params, block.ln1.gamma, block.ln1.beta)

		// Feed-forward
		params = append(params, block.ff.w1, block.ff.b1, block.ff.w2, block.ff.b2)

		// LayerNorm (feed-forward)
		params = append(params, block.ln2.gamma, block.ln2.beta)
	}

	// Final layer norm
	params = append(params, gpt.lnFinal.gamma, gpt.lnFinal.beta)

	// Output projection
	params = append(params, gpt.lmHead)

	return params
}

// Train trains the model on a dataset.
func Train(model *GPT, trainData, valData [][][]int, config TrainingConfig) {
	fmt.Println("=== Training Started ===")
	fmt.Printf("Batch size: %d\n", config.BatchSize)
	fmt.Printf("Learning rate: %.6f\n", config.LearningRate)
	fmt.Printf("Optimizer: %s\n", config.Optimizer)
	fmt.Println()

	// Initialize optimizer
	params := model.Parameters()
	var optimizer Optimizer

	if config.Optimizer == "adam" {
		optimizer = NewAdamOptimizer(params, config.AdamBeta1, config.AdamBeta2,
			config.AdamEpsilon, config.WeightDecay)
	} else {
		optimizer = NewSGDOptimizer(config.WeightDecay)
	}

	// Learning rate scheduler
	scheduler := NewLRScheduler(config.LearningRate, config.MinLR,
		config.WarmupSteps, config.DecaySteps)

	step := 0

	// Training loop
	for epoch := 0; epoch < config.NumEpochs; epoch++ {
		fmt.Printf("Epoch %d/%d\n", epoch+1, config.NumEpochs)

		// Shuffle training data
		shuffled := make([][][]int, len(trainData))
		copy(shuffled, trainData)
		rand.Shuffle(len(shuffled), func(i, j int) {
			shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
		})

		// Process batches
		for i := 0; i < len(shuffled); i += config.BatchSize {
			end := i + config.BatchSize
			if end > len(shuffled) {
				end = len(shuffled)
			}

			batch := shuffled[i:end]

			// Extract inputs and targets
			// (inputs and targets are the same for language modeling,
			//  just shifted by one token)
			inputs := make([][]int, len(batch))
			targets := make([][]int, len(batch))
			for j, seqs := range batch {
				// seqs is [][]int, take first sequence
				if len(seqs) > 0 && len(seqs[0]) > 1 {
					seq := seqs[0]
					inputs[j] = seq[0 : len(seq)-1]
					targets[j] = seq[1:]
				}
			}

			// Get current learning rate
			lr := scheduler.GetLR()

			// Training step
			loss := TrainStep(model, inputs, targets, optimizer, lr)
			step++

			// Logging
			if step%config.LogInterval == 0 {
				fmt.Printf("Step %d | Loss: %.4f | LR: %.6f\n", step, loss, lr)
			}

			// Evaluation
			if step%config.EvalInterval == 0 && valData != nil {
				valLoss := Evaluate(model, valData, config.BatchSize)
				fmt.Printf("Step %d | Val Loss: %.4f\n", step, valLoss)
			}

			// Max steps check
			if config.MaxSteps > 0 && step >= config.MaxSteps {
				fmt.Printf("Reached max steps (%d)\n", config.MaxSteps)
				return
			}
		}
	}

	fmt.Println("=== Training Complete ===")
}

// Evaluate computes validation loss.
func Evaluate(model *GPT, valData [][][]int, batchSize int) float64 {
	totalLoss := 0.0
	numBatches := 0

	for i := 0; i < len(valData); i += batchSize {
		end := i + batchSize
		if end > len(valData) {
			end = len(valData)
		}

		batch := valData[i:end]

		// Extract inputs and targets
		inputs := make([][]int, len(batch))
		targets := make([][]int, len(batch))
		for j, seqs := range batch {
			// seqs is [][]int, take first sequence
			if len(seqs) > 0 && len(seqs[0]) > 1 {
				seq := seqs[0]
				inputs[j] = seq[0 : len(seq)-1]
				targets[j] = seq[1:]
			}
		}

		// Forward pass only (no gradients)
		for j := range inputs {
			logits := model.Forward(inputs[j])
			loss := CrossEntropyLoss(logits, targets[j])
			totalLoss += loss
		}

		numBatches++
	}

	return totalLoss / float64(numBatches)
}
