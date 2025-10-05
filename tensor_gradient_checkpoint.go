package main

// ===========================================================================
// WHAT'S GOING ON HERE: Gradient Checkpointing
// ===========================================================================
//
// This file implements gradient checkpointing (also called "activation checkpointing"),
// a memory optimization technique that trades compute for memory during training.
//
// INTENTION:
// Demonstrate how modern deep learning frameworks enable training of very large
// models by strategically discarding and recomputing intermediate activations.
// This is an educational implementation showing the core concept.
//
// THE MEMORY PROBLEM:
//
// During neural network training, the forward pass must store all intermediate
// activations (layer outputs) because the backward pass needs them to compute
// gradients. For a transformer with N layers:
//
// Memory usage = O(N × B × S × D) where:
//   N = number of layers
//   B = batch size
//   S = sequence length
//   D = hidden dimension
//
// Example: GPT-3 with 96 layers, batch 8, sequence 2048, dimension 12288:
//   Memory ≈ 96 × 8 × 2048 × 12288 × 4 bytes ≈ 60 GB just for activations!
//
// THE GRADIENT CHECKPOINTING SOLUTION:
//
// Instead of storing all N layer activations, we:
//
// 1. FORWARD PASS: Store only checkpoints (e.g., every K layers)
//    - Keep inputs to checkpoint layers: layers 0, K, 2K, 3K, ...
//    - Discard intermediate activations within segments
//    - Memory: O(N/K × B × S × D) - reduced by factor of K
//
// 2. BACKWARD PASS: Recompute activations as needed
//    - When backprop reaches a checkpoint boundary, recompute forward
//      for that segment using the saved checkpoint input
//    - Now we have activations needed for backward pass
//    - Compute gradients and continue backprop
//    - Discard recomputed activations after use
//
// TRADE-OFF ANALYSIS:
//
// Memory savings: K× reduction (typically K=2-4)
// Compute overhead: ~33% more FLOPs (one extra forward pass per segment)
// Wall time overhead: ~10-20% in practice (memory bandwidth improvements)
//
// Why it's worth it:
// - Enables training 2-4× larger models on same hardware
// - Enables 2-4× larger batch sizes (better GPU utilization)
// - Often actually faster (less memory pressure, better caching)
//
// WHERE TO CHECKPOINT:
//
// Good checkpoint locations:
// - Between transformer blocks (coarse-grained, efficient)
// - Between major operations (QKV, attention, FFN)
// - Automatic: choose to minimize peak memory
//
// Bad checkpoint locations:
// - After cheap operations (overhead > savings)
// - Inside tight loops (recomputation too expensive)
//
// IMPLEMENTATION PATTERNS:
//
// Pattern 1: Explicit checkpointing
//   - Programmer manually marks checkpoint boundaries
//   - Full control, best for educational purposes
//   - Used in this implementation
//
// Pattern 2: Automatic checkpointing
//   - Framework analyzes computation graph
//   - Optimal checkpoint placement via dynamic programming
//   - Used in production (PyTorch, JAX)
//
// ===========================================================================
// RECOMMENDED READING:
//
// Gradient Checkpointing:
// - "Training Deep Nets with Sublinear Memory Cost" by Chen et al. (2016)
//   https://arxiv.org/abs/1604.06174
//   The original gradient checkpointing paper
//
// - "Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization"
//   by Jain et al. (2020) - optimal checkpoint placement algorithms
//
// - "Reducing Activation Recomputation in Large Transformer Models"
//   by Korthikanti et al. (2022) - NVIDIA's selective checkpointing
//
// Memory Optimization:
// - "Memory-Efficient Backpropagation Through Time" by Gruslys et al. (2016)
//   - Applies similar ideas to recurrent networks
//
// - "Gist: Efficient Data Encoding for Deep Neural Network Training"
//   by Jain et al. (2018) - lossy compression of activations
//
// ===========================================================================

// CheckpointFunction represents a forward computation that can be checkpointed.
// The function takes inputs and returns outputs. During checkpointing, only
// inputs are saved; outputs are recomputed during backward pass.
type CheckpointFunction func(inputs ...*Tensor) []*Tensor

// CheckpointSegment represents a checkpointed segment of computation.
// It stores the checkpoint input and the forward function, allowing
// recomputation during the backward pass.
type CheckpointSegment struct {
	// Forward is the function to execute for the forward pass
	Forward CheckpointFunction

	// Inputs are the checkpoint boundary inputs (saved for recomputation)
	Inputs []*Tensor

	// Outputs are the results (only kept during forward pass)
	// Set to nil after forward pass to free memory
	Outputs []*Tensor

	// Recomputed indicates whether activations have been recomputed
	// for backward pass (used to avoid double recomputation)
	Recomputed bool
}

// NewCheckpointSegment creates a new checkpoint segment with the given forward function.
func NewCheckpointSegment(forward CheckpointFunction) *CheckpointSegment {
	return &CheckpointSegment{
		Forward:    forward,
		Recomputed: false,
	}
}

// RunForward executes the forward pass and saves inputs (but not outputs).
// This is the memory-efficient version that discards intermediate activations.
//
// Usage during training forward pass:
//   segment := NewCheckpointSegment(forwardFunc)
//   outputs := segment.RunForward(inputs...)
func (cs *CheckpointSegment) RunForward(inputs ...*Tensor) []*Tensor {
	// Save inputs for potential recomputation during backward pass
	// This is the only thing we keep in memory!
	cs.Inputs = make([]*Tensor, len(inputs))
	copy(cs.Inputs, inputs)

	// Execute forward computation
	outputs := cs.Forward(inputs...)

	// DO NOT save outputs - this is the whole point of checkpointing!
	// Outputs will be recomputed during backward pass if needed
	cs.Outputs = nil

	return outputs
}

// RecomputeForward recomputes the forward pass using saved inputs.
// This is called during the backward pass when we need activations for gradient computation.
//
// The recomputation is typically much cheaper than storing activations because:
// 1. We save memory bandwidth (no need to read/write large activation tensors)
// 2. Modern GPUs have abundant compute but limited memory bandwidth
// 3. Recomputation can use faster algorithms (no need to save intermediates)
func (cs *CheckpointSegment) RecomputeForward() []*Tensor {
	if cs.Recomputed {
		// Already recomputed, return cached outputs
		return cs.Outputs
	}

	// Recompute forward pass using saved inputs
	cs.Outputs = cs.Forward(cs.Inputs...)
	cs.Recomputed = true

	return cs.Outputs
}

// ClearOutputs frees memory used by recomputed outputs after backward pass.
// Call this after computing gradients to minimize memory footprint.
func (cs *CheckpointSegment) ClearOutputs() {
	cs.Outputs = nil
	cs.Recomputed = false
}

// CheckpointConfig controls gradient checkpointing behavior for a model.
type CheckpointConfig struct {
	// Enabled turns checkpointing on/off
	Enabled bool

	// CheckpointEveryN layers means checkpoint every N transformer blocks
	// Typical values: 1 (checkpoint all), 2, 4
	// Higher values = more memory savings but more recomputation
	CheckpointEveryN int

	// Segments stores all checkpointed computation segments
	Segments []*CheckpointSegment

	// MemoryBudget is the maximum memory to use for activations (in MB)
	// If set, automatically determines CheckpointEveryN
	// Set to 0 to use CheckpointEveryN directly
	MemoryBudget float64

	// CurrentLayer tracks which layer we're currently processing
	// Used for checkpoint decisions
	CurrentLayer int
}

// NewCheckpointConfig creates a default checkpointing configuration.
func NewCheckpointConfig() *CheckpointConfig {
	return &CheckpointConfig{
		Enabled:          false, // Off by default (opt-in for performance)
		CheckpointEveryN: 2,     // Checkpoint every 2 layers (good default)
		Segments:         make([]*CheckpointSegment, 0),
		MemoryBudget:     0,     // No automatic budget management by default
		CurrentLayer:     0,
	}
}

// ShouldCheckpoint returns true if the current layer should be checkpointed.
// This implements the checkpointing policy.
func (cfg *CheckpointConfig) ShouldCheckpoint() bool {
	if !cfg.Enabled {
		return false
	}

	// Checkpoint every N layers
	return cfg.CurrentLayer%cfg.CheckpointEveryN == 0
}

// AddSegment adds a checkpointed segment and increments the layer counter.
func (cfg *CheckpointConfig) AddSegment(segment *CheckpointSegment) {
	cfg.Segments = append(cfg.Segments, segment)
	cfg.CurrentLayer++
}

// Reset clears all checkpoint segments and resets state.
// Call this at the start of each training step.
func (cfg *CheckpointConfig) Reset() {
	cfg.Segments = make([]*CheckpointSegment, 0)
	cfg.CurrentLayer = 0
}

// EstimateMemorySavings calculates approximate memory savings from checkpointing.
// Returns (memory_without_checkpointing, memory_with_checkpointing, savings_ratio).
func (cfg *CheckpointConfig) EstimateMemorySavings(numLayers, batchSize, seqLen, hiddenDim int) (float64, float64, float64) {
	if !cfg.Enabled {
		return 0, 0, 1.0
	}

	// Bytes per activation: float32 = 4 bytes
	bytesPerActivation := float64(4)

	// Memory for one layer's activations
	layerMemoryMB := float64(batchSize*seqLen*hiddenDim) * bytesPerActivation / (1024 * 1024)

	// Without checkpointing: store all N layers
	memoryWithout := float64(numLayers) * layerMemoryMB

	// With checkpointing: store every K-th layer
	// Plus overhead for recomputation scratch space (1 layer worth)
	checkpointLayers := (numLayers + cfg.CheckpointEveryN - 1) / cfg.CheckpointEveryN
	memoryWith := float64(checkpointLayers)*layerMemoryMB + layerMemoryMB

	// Savings ratio
	savingsRatio := memoryWithout / memoryWith

	return memoryWithout, memoryWith, savingsRatio
}

// ===========================================================================
// EXAMPLE USAGE
// ===========================================================================
//
// // Setup checkpointing for a transformer model
// func trainWithCheckpointing() {
//     cfg := NewCheckpointConfig()
//     cfg.Enabled = true
//     cfg.CheckpointEveryN = 2  // Checkpoint every 2 layers
//
//     // Estimate memory savings
//     memWithout, memWith, ratio := cfg.EstimateMemorySavings(
//         numLayers: 24,
//         batchSize: 8,
//         seqLen: 1024,
//         hiddenDim: 768,
//     )
//     fmt.Printf("Memory: %.1fMB → %.1fMB (%.1fx savings)\n",
//                memWithout, memWith, ratio)
//
//     for epoch := range epochs {
//         cfg.Reset()  // Start of training step
//
//         // Forward pass with checkpointing
//         x := input
//         for layer := 0; layer < numLayers; layer++ {
//             if cfg.ShouldCheckpoint() {
//                 // Checkpoint this layer
//                 segment := NewCheckpointSegment(func(inputs ...*Tensor) []*Tensor {
//                     return model.layers[layer].Forward(inputs[0])
//                 })
//                 x = segment.RunForward(x)[0]
//                 cfg.AddSegment(segment)
//             } else {
//                 // Normal forward (no checkpointing)
//                 x = model.layers[layer].Forward(x)
//             }
//         }
//
//         loss := computeLoss(x, target)
//
//         // Backward pass with recomputation
//         dLoss := 1.0
//         for i := len(cfg.Segments) - 1; i >= 0; i-- {
//             segment := cfg.Segments[i]
//
//             // Recompute activations for this segment
//             outputs := segment.RecomputeForward()
//
//             // Compute gradients (now we have the activations we need)
//             gradients := computeGradients(outputs, dLoss)
//
//             // Free recomputed activations
//             segment.ClearOutputs()
//
//             // Continue backprop
//             dLoss = gradients
//         }
//
//         // Update parameters
//         optimizer.Step(gradients)
//     }
// }
//
// // Example: Checkpointing a transformer block
// func checkpointTransformerBlock(block *TransformerBlock, input *Tensor) *Tensor {
//     segment := NewCheckpointSegment(func(inputs ...*Tensor) []*Tensor {
//         x := inputs[0]
//
//         // Attention sub-block
//         attnOut := block.SelfAttention(x)
//         x = block.LayerNorm1(x.Add(attnOut))  // Residual connection
//
//         // Feed-forward sub-block
//         ffOut := block.FeedForward(x)
//         x = block.LayerNorm2(x.Add(ffOut))    // Residual connection
//
//         return []*Tensor{x}
//     })
//
//     outputs := segment.RunForward(input)
//     return outputs[0]
// }
//
// ===========================================================================
