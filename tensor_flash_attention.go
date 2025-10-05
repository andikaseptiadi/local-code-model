package main

import (
	"math"
)

// ===========================================================================
// WHAT'S GOING ON HERE: Flash Attention
// ===========================================================================
//
// This file implements Flash Attention, a memory-efficient attention algorithm
// that uses tiling and recomputation to minimize HBM (High Bandwidth Memory)
// accesses while keeping intermediate results in fast on-chip SRAM.
//
// INTENTION:
// Demonstrate how modern attention mechanisms achieve 2-4x speedup and enable
// longer sequence lengths by carefully managing the memory hierarchy. This is
// a simplified educational implementation showing the core tiling concepts.
//
// THE STANDARD ATTENTION PROBLEM:
//
// Standard attention computes: softmax(QK^T)V
//
// For a sequence of length N with dimension d:
// 1. Compute S = QK^T                    [N×N matrix, requires N² memory]
// 2. Compute P = softmax(S)              [N×N matrix, requires N² memory]
// 3. Compute O = PV                      [N×d matrix, output]
//
// Memory bottleneck: The N×N attention matrix doesn't fit in fast SRAM for
// long sequences. We must read/write to slow HBM, which dominates runtime.
//
// Example: N=2048, d=64, float32
//   Attention matrix: 2048² × 4 bytes = 16 MB (doesn't fit in typical 20 MB SRAM)
//   This forces expensive HBM reads/writes
//
// THE FLASH ATTENTION SOLUTION:
//
// Key insight: We don't need to materialize the full N×N attention matrix!
// We can compute the output O in tiles, keeping everything in SRAM.
//
// Algorithm (simplified):
//
// 1. Split Q, K, V into blocks: Q = [Q₁, Q₂, ..., Qₜ]
//    Block size chosen to fit in SRAM (typically 32-128 tokens)
//
// 2. For each query block Qᵢ:
//    a) Load Qᵢ into SRAM
//    b) Initialize partial output Oᵢ = 0 and statistics
//    c) For each key-value block (Kⱼ, Vⱼ):
//       - Load Kⱼ, Vⱼ into SRAM
//       - Compute Sᵢⱼ = QᵢKⱼᵀ (block of attention matrix)
//       - Update softmax statistics incrementally
//       - Compute partial output contribution
//       - Discard Sᵢⱼ (don't store!)
//    d) Write final Oᵢ to HBM
//
// WHY THIS WORKS:
//
// Memory hierarchy:
// - SRAM: ~20 MB, ~100 GB/s, ~100 cycles latency
// - HBM:  ~40 GB, ~1.5 TB/s, ~400 cycles latency
//
// Standard attention: O(N²) HBM accesses (read/write attention matrix)
// Flash attention: O(N) HBM accesses (only read Q,K,V and write O)
//
// Speedup sources:
// 1. Fewer HBM accesses (memory bound → compute bound)
// 2. Better SRAM reuse (data stays on-chip)
// 3. More compute can be fused (no intermediate materialization)
//
// TRADE-OFFS:
//
// Standard attention:
//   + Simple to implement
//   + Materializes attention for analysis
//   - O(N²) memory
//   - O(N²) HBM accesses
//
// Flash attention:
//   + O(N) memory (only inputs/outputs)
//   + O(N) HBM accesses
//   + 2-4× faster in practice
//   - More complex (tiling, statistics tracking)
//   - Attention matrix not available (unless recomputed)
//
// INCREMENTAL SOFTMAX:
//
// Computing softmax in tiles requires careful statistics tracking:
//
// softmax(x)ᵢ = exp(xᵢ - max(x)) / Σⱼ exp(xⱼ - max(x))
//
// To compute this incrementally across tiles:
// 1. Track running max: mᵢ = max(m_{i-1}, max(xᵢ))
// 2. Track running sum: sᵢ = s_{i-1} × exp(m_{i-1} - mᵢ) + Σⱼ exp(xᵢⱼ - mᵢ)
// 3. Rescale previous outputs when max changes
//
// This "online softmax" allows us to compute softmax without storing the full
// attention matrix, which is the key to Flash Attention's memory efficiency.
//
// WHERE THIS SITS IN THE OPTIMIZATION HIERARCHY:
//
// Phase 1.5: SIMD/Assembly (COMPLETED)
//   - Vectorized operations, BLAS-style GEMM
//   - 10-20x speedup from better compute efficiency
//
// Phase 2.1: Mixed Precision (COMPLETED)
//   - Float16 forward, float32 gradients
//   - 2-3x speedup + 50% memory reduction
//
// Phase 2.2: Gradient Checkpointing (COMPLETED)
//   - Trade compute for memory
//   - Enables larger models/batches
//
// Phase 2.3: Flash Attention (THIS FILE)
//   - Tiled attention with on-chip SRAM
//   - 2-4x speedup + memory reduction for attention
//   - Enables much longer sequences (8K → 32K tokens)
//
// ===========================================================================
// RECOMMENDED READING:
//
// Flash Attention:
// - "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
//   by Dao et al. (2022) https://arxiv.org/abs/2205.14135
//   The original Flash Attention paper
//
// - "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
//   by Dao (2023) https://arxiv.org/abs/2307.08691
//   Improved version with better parallelism
//
// Memory Hierarchy:
// - "What Every Programmer Should Know About Memory" by Ulrich Drepper (2007)
//   Deep dive into memory hierarchies and cache behavior
//
// Online Algorithms:
// - "Streaming Algorithm" Wikipedia article
//   Background on incremental computation patterns
//
// ===========================================================================

// FlashAttentionConfig controls Flash Attention behavior.
type FlashAttentionConfig struct {
	// Enabled turns Flash Attention on/off
	Enabled bool

	// BlockSize is the tile size for Q, K, V (number of tokens per block)
	// Typical values: 32, 64, 128
	// Smaller = more tiling overhead, Larger = more memory per tile
	// Should be chosen to fit in SRAM: ~BlockSize × (3 × HeadDim + SeqLen) × 4 bytes
	BlockSize int

	// CausalMask applies causal masking (for autoregressive models)
	// When true, position i can only attend to positions ≤ i
	CausalMask bool

	// SoftmaxScale is the scaling factor applied before softmax
	// Typically 1/sqrt(head_dim) to prevent softmax saturation
	SoftmaxScale float64
}

// NewFlashAttentionConfig creates a default Flash Attention configuration.
func NewFlashAttentionConfig(headDim int) *FlashAttentionConfig {
	return &FlashAttentionConfig{
		Enabled:      true,
		BlockSize:    64, // Good default: fits in most SRAM
		CausalMask:   false,
		SoftmaxScale: 1.0 / math.Sqrt(float64(headDim)),
	}
}

// FlashAttentionForward computes attention using the Flash Attention algorithm.
//
// Inputs:
//   - Q: Query tensor [batch × num_heads × seq_len × head_dim]
//   - K: Key tensor   [batch × num_heads × seq_len × head_dim]
//   - V: Value tensor [batch × num_heads × seq_len × head_dim]
//   - config: Flash Attention configuration
//
// Output:
//   - O: Output tensor [batch × num_heads × seq_len × head_dim]
//
// This function implements the tiled attention algorithm, processing Q, K, V
// in blocks to minimize HBM accesses.
func FlashAttentionForward(Q, K, V *Tensor, config *FlashAttentionConfig) *Tensor {
	if !config.Enabled {
		// Fall back to standard attention if disabled
		return StandardAttention(Q, K, V, config.SoftmaxScale, config.CausalMask)
	}

	// Extract dimensions
	// Shape: [batch × num_heads × seq_len × head_dim]
	batch := Q.shape[0]
	numHeads := Q.shape[1]
	seqLen := Q.shape[2]
	headDim := Q.shape[3]

	// Create output tensor
	O := NewTensor(batch, numHeads, seqLen, headDim)

	// Process each batch and head independently
	for b := 0; b < batch; b++ {
		for h := 0; h < numHeads; h++ {
			// Compute attention for this (batch, head) pair
			flashAttentionForwardSingleHead(Q, K, V, O, b, h, config)
		}
	}

	return O
}

// flashAttentionForwardSingleHead computes Flash Attention for a single head.
//
// This is where the core tiling algorithm lives. We process Q in blocks of
// BlockSize tokens, and for each Q block, we iterate over all K/V blocks,
// computing attention contributions incrementally.
func flashAttentionForwardSingleHead(Q, K, V, O *Tensor, batch, head int, config *FlashAttentionConfig) {
	seqLen := Q.shape[2]
	headDim := Q.shape[3]
	blockSize := config.BlockSize

	// Number of blocks (round up)
	numBlocks := (seqLen + blockSize - 1) / blockSize

	// Process each query block Qᵢ
	for iBlock := 0; iBlock < numBlocks; iBlock++ {
		// Query block range: [iStart, iEnd)
		iStart := iBlock * blockSize
		iEnd := min(iStart+blockSize, seqLen)
		blockRows := iEnd - iStart

		// Initialize output block and statistics
		Oi := make([]float64, blockRows*headDim)           // Accumulated output
		mi := make([]float64, blockRows)                   // Max scores per row (for softmax)
		li := make([]float64, blockRows)                   // Sum of exponentials per row
		for i := 0; i < blockRows; i++ {
			mi[i] = math.Inf(-1) // Start with -∞
			li[i] = 0.0
		}

		// Process each key-value block Kⱼ, Vⱼ
		for jBlock := 0; jBlock < numBlocks; jBlock++ {
			// Key-value block range: [jStart, jEnd)
			jStart := jBlock * blockSize
			jEnd := min(jStart+blockSize, seqLen)
			blockCols := jEnd - jStart

			// Skip if causal mask excludes this block entirely
			if config.CausalMask && jStart > iEnd-1 {
				continue
			}

			// Compute block of attention scores: Sᵢⱼ = QᵢKⱼᵀ
			// Shape: [blockRows × blockCols]
			Sij := make([]float64, blockRows*blockCols)

			for i := 0; i < blockRows; i++ {
				for j := 0; j < blockCols; j++ {
					// Compute dot product: Q[iStart+i, :] · K[jStart+j, :]
					var score float64
					for d := 0; d < headDim; d++ {
						qIdx := ((batch*Q.shape[1]+head)*Q.shape[2]+(iStart+i))*headDim + d
						kIdx := ((batch*K.shape[1]+head)*K.shape[2]+(jStart+j))*headDim + d
						score += Q.data[qIdx] * K.data[kIdx]
					}

					// Apply scaling
					score *= config.SoftmaxScale

					// Apply causal mask if needed
					if config.CausalMask && (iStart+i) < (jStart+j) {
						score = math.Inf(-1) // Mask future positions
					}

					Sij[i*blockCols+j] = score
				}
			}

			// Update statistics and accumulate output using online softmax
			// This is the key to Flash Attention: we update running max and sum
			// without storing the full attention matrix
			for i := 0; i < blockRows; i++ {
				// Find max score in this row of current block
				var mjNew float64 = math.Inf(-1)
				for j := 0; j < blockCols; j++ {
					if Sij[i*blockCols+j] > mjNew {
						mjNew = Sij[i*blockCols+j]
					}
				}

				// Update running max
				miOld := mi[i]
				mi[i] = max(mi[i], mjNew)

				// Compute sum of exponentials for this block
				var ljNew float64
				for j := 0; j < blockCols; j++ {
					ljNew += math.Exp(Sij[i*blockCols+j] - mi[i])
				}

				// Update running sum, rescaling for new max
				li[i] = li[i]*math.Exp(miOld-mi[i]) + ljNew

				// Rescale previous output contribution for new max
				scale := math.Exp(miOld - mi[i])
				for d := 0; d < headDim; d++ {
					Oi[i*headDim+d] *= scale
				}

				// Add new contribution: exp(Sᵢⱼ - mᵢ) × Vⱼ
				for j := 0; j < blockCols; j++ {
					attn := math.Exp(Sij[i*blockCols+j] - mi[i])
					for d := 0; d < headDim; d++ {
						vIdx := ((batch*V.shape[1]+head)*V.shape[2]+(jStart+j))*headDim + d
						Oi[i*headDim+d] += attn * V.data[vIdx]
					}
				}
			}
		}

		// Normalize output by final sum
		for i := 0; i < blockRows; i++ {
			for d := 0; d < headDim; d++ {
				Oi[i*headDim+d] /= li[i]
			}
		}

		// Write output block to tensor
		for i := 0; i < blockRows; i++ {
			for d := 0; d < headDim; d++ {
				oIdx := ((batch*O.shape[1]+head)*O.shape[2]+(iStart+i))*headDim + d
				O.data[oIdx] = Oi[i*headDim+d]
			}
		}
	}
}

// StandardAttention computes attention using the standard algorithm.
// This materializes the full attention matrix and is used as a reference
// implementation for testing and when Flash Attention is disabled.
//
// Algorithm:
//   1. S = QK^T / sqrt(d)
//   2. Apply causal mask if needed
//   3. P = softmax(S)
//   4. O = PV
func StandardAttention(Q, K, V *Tensor, scale float64, causal bool) *Tensor {
	// Extract dimensions: [batch × num_heads × seq_len × head_dim]
	batch := Q.shape[0]
	numHeads := Q.shape[1]
	seqLen := Q.shape[2]
	headDim := Q.shape[3]

	O := NewTensor(batch, numHeads, seqLen, headDim)

	for b := 0; b < batch; b++ {
		for h := 0; h < numHeads; h++ {
			// Compute attention scores: S = QK^T
			S := make([]float64, seqLen*seqLen)
			for i := 0; i < seqLen; i++ {
				for j := 0; j < seqLen; j++ {
					var score float64
					for d := 0; d < headDim; d++ {
						qIdx := ((b*numHeads+h)*seqLen+i)*headDim + d
						kIdx := ((b*numHeads+h)*seqLen+j)*headDim + d
						score += Q.data[qIdx] * K.data[kIdx]
					}
					S[i*seqLen+j] = score * scale
				}
			}

			// Apply causal mask
			if causal {
				for i := 0; i < seqLen; i++ {
					for j := i + 1; j < seqLen; j++ {
						S[i*seqLen+j] = math.Inf(-1)
					}
				}
			}

			// Apply softmax row-wise
			for i := 0; i < seqLen; i++ {
				// Find max for numerical stability
				maxVal := math.Inf(-1)
				for j := 0; j < seqLen; j++ {
					if S[i*seqLen+j] > maxVal {
						maxVal = S[i*seqLen+j]
					}
				}

				// Compute exp and sum
				var sumExp float64
				for j := 0; j < seqLen; j++ {
					S[i*seqLen+j] = math.Exp(S[i*seqLen+j] - maxVal)
					sumExp += S[i*seqLen+j]
				}

				// Normalize
				for j := 0; j < seqLen; j++ {
					S[i*seqLen+j] /= sumExp
				}
			}

			// Compute output: O = PV
			for i := 0; i < seqLen; i++ {
				for d := 0; d < headDim; d++ {
					var out float64
					for j := 0; j < seqLen; j++ {
						vIdx := ((b*numHeads+h)*seqLen+j)*headDim + d
						out += S[i*seqLen+j] * V.data[vIdx]
					}
					oIdx := ((b*numHeads+h)*seqLen+i)*headDim + d
					O.data[oIdx] = out
				}
			}
		}
	}

	return O
}

// EstimateFlashAttentionSpeedup estimates the speedup from Flash Attention.
//
// This is a simplified model that considers:
// - Memory bandwidth bottleneck (HBM accesses)
// - SRAM capacity (what fits on-chip)
// - Compute vs memory bound ratio
//
// Returns (standard_time, flash_time, speedup_ratio).
func EstimateFlashAttentionSpeedup(seqLen, headDim, blockSize int) (float64, float64, float64) {
	// Memory access costs (arbitrary units, representing HBM vs SRAM)
	hbmAccessCost := 100.0  // HBM access is expensive
	sramAccessCost := 1.0   // SRAM access is cheap

	// Standard attention HBM accesses:
	// - Read Q, K: 2 × N × d
	// - Write S: N²
	// - Read S: N²
	// - Read V: N × d
	// - Write O: N × d
	standardHBM := float64(2*seqLen*headDim + 2*seqLen*seqLen + seqLen*headDim + seqLen*headDim)
	standardTime := standardHBM * hbmAccessCost

	// Flash attention HBM accesses:
	// - Read Q, K, V: 3 × N × d (once)
	// - Write O: N × d
	// All intermediate computation stays in SRAM
	flashHBM := float64(4 * seqLen * headDim)
	flashTime := flashHBM * hbmAccessCost

	// Add SRAM access costs (more accesses but much cheaper)
	// Flash attention does more SRAM reads/writes due to tiling
	numBlocks := (seqLen + blockSize - 1) / blockSize
	flashSRAM := float64(numBlocks * numBlocks * blockSize * blockSize)
	flashTime += flashSRAM * sramAccessCost

	speedup := standardTime / flashTime
	return standardTime, flashTime, speedup
}

// ===========================================================================
// EXAMPLE USAGE
// ===========================================================================
//
// // Example 1: Basic Flash Attention
// func exampleFlashAttention() {
//     seqLen := 512
//     headDim := 64
//     batch := 2
//     numHeads := 8
//
//     // Create Q, K, V tensors
//     Q := NewTensor(batch, numHeads, seqLen, headDim)
//     K := NewTensor(batch, numHeads, seqLen, headDim)
//     V := NewTensor(batch, numHeads, seqLen, headDim)
//     // ... fill with data ...
//
//     // Configure Flash Attention
//     config := NewFlashAttentionConfig(headDim)
//     config.BlockSize = 64       // Tile size
//     config.CausalMask = true    // For autoregressive generation
//
//     // Compute attention
//     output := FlashAttentionForward(Q, K, V, config)
//
//     // Estimate speedup
//     stdTime, flashTime, speedup := EstimateFlashAttentionSpeedup(
//         seqLen, headDim, config.BlockSize)
//     fmt.Printf("Speedup: %.2fx\n", speedup)
// }
//
// // Example 2: Comparing Flash vs Standard
// func exampleCompareAttention() {
//     seqLen := 128
//     headDim := 64
//     Q := NewTensor(1, 1, seqLen, headDim)
//     K := NewTensor(1, 1, seqLen, headDim)
//     V := NewTensor(1, 1, seqLen, headDim)
//
//     // Standard attention
//     config := NewFlashAttentionConfig(headDim)
//     config.Enabled = false
//     outputStd := FlashAttentionForward(Q, K, V, config)
//
//     // Flash attention
//     config.Enabled = true
//     outputFlash := FlashAttentionForward(Q, K, V, config)
//
//     // Results should be very close (within numerical precision)
//     maxDiff := 0.0
//     for i := range outputStd.data {
//         diff := abs(outputStd.data[i] - outputFlash.data[i])
//         if diff > maxDiff {
//             maxDiff = diff
//         }
//     }
//     fmt.Printf("Max difference: %e\n", maxDiff)
// }
//
// ===========================================================================
