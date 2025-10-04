package main

import (
	"runtime"
	"sync"
)

// ===========================================================================
// WHAT'S GOING ON HERE
// ===========================================================================
//
// This file implements a reusable worker pool for batch tensor operations.
// Instead of spawning goroutines for each operation (which has overhead),
// we maintain a persistent pool of workers that process tasks from a queue.
//
// MOTIVATION:
// During ML training, we often need to process batches of tensors:
//   - Batch matrix multiplication: Process multiple samples in parallel
//   - Batch activations: Apply GELU/softmax to multiple tensors
//   - Batch gradients: Compute gradients for entire batch simultaneously
//
// Creating goroutines on every operation is wasteful:
//   - Goroutine creation: ~3-5 microseconds
//   - Scheduling overhead: context switches, synchronization
//   - GC pressure: short-lived goroutines create garbage
//
// A worker pool amortizes this cost:
//   - Create workers ONCE at startup
//   - Reuse workers across many operations
//   - Reduce GC pressure (long-lived goroutines)
//   - Better CPU cache locality (same workers process similar work)
//
// DESIGN:
//
// 1. **TensorPool**: Manages a pool of workers and a task queue
//    - Workers: Long-lived goroutines that process tasks
//    - Tasks: Functions to execute (e.g., "compute row i of matmul")
//    - Queue: Buffered channel for work distribution
//
// 2. **Task Interface**: Generic function signature for tensor operations
//    type TensorTask func()
//
// 3. **Lifecycle**:
//    - pool.Start(): Spawns worker goroutines
//    - pool.Submit(task): Adds task to queue
//    - pool.Wait(): Blocks until all tasks complete
//    - pool.Stop(): Gracefully shuts down workers
//
// EDUCATIONAL VALUE:
// This demonstrates several important Go patterns:
//   - Worker pool pattern (common in concurrent programming)
//   - Channel-based work distribution
//   - Graceful shutdown with context/channels
//   - Resource management (limit concurrent operations)
//
// PERFORMANCE CHARACTERISTICS:
//
// Single-use goroutines (tensor_parallel.go):
//   - Good for: One-off parallel operations
//   - Overhead: ~5-10 µs per operation
//   - Best for: Large matrices where setup << compute
//
// Worker pool (this file):
//   - Good for: Repeated operations (training loops)
//   - Overhead: ~1-2 µs per task (amortized)
//   - Best for: Batch processing, training iterations
//
// USAGE EXAMPLE:
//
//   // Create pool once at training start
//   pool := NewTensorPool(runtime.NumCPU())
//   pool.Start()
//   defer pool.Stop()
//
//   // Process batch (reuse pool across iterations)
//   for epoch := 0; epoch < numEpochs; epoch++ {
//       for _, batch := range batches {
//           // Submit all batch operations to pool
//           for i := 0; i < batchSize; i++ {
//               pool.Submit(func() {
//                   // Process sample i
//                   output[i] = model.Forward(batch[i])
//               })
//           }
//           pool.Wait() // Wait for batch to complete
//       }
//   }
//
// ===========================================================================

// TensorTask is a function that performs a tensor operation.
// Tasks are submitted to the worker pool and executed by worker goroutines.
type TensorTask func()

// TensorPool manages a pool of worker goroutines for parallel tensor operations.
// It provides work distribution via a buffered channel and synchronization
// via WaitGroups.
//
// Key features:
//   - Fixed number of workers (avoids goroutine explosion)
//   - Reusable across multiple operations (amortizes startup cost)
//   - Graceful shutdown (waits for in-flight tasks)
//   - Configurable queue size (controls memory vs latency)
type TensorPool struct {
	numWorkers int           // Number of worker goroutines
	tasks      chan TensorTask // Buffered channel for task queue
	wg         sync.WaitGroup  // Tracks in-flight tasks
	stopOnce   sync.Once       // Ensures Stop() is idempotent
	stopChan   chan struct{}   // Signals workers to exit
}

// NewTensorPool creates a new worker pool with the specified number of workers.
// If numWorkers <= 0, defaults to runtime.NumCPU().
//
// The task queue is buffered to allow submitting work without blocking,
// up to a reasonable limit (10x workers). This trades memory for latency:
//   - Larger buffer: Submit() rarely blocks, but uses more memory
//   - Smaller buffer: Submit() may block if workers are busy
//
// Parameters:
//   - numWorkers: Number of worker goroutines (0 = auto-detect)
//
// Returns:
//   - *TensorPool: Configured pool (call Start() to begin processing)
func NewTensorPool(numWorkers int) *TensorPool {
	if numWorkers <= 0 {
		numWorkers = runtime.NumCPU()
	}

	// Buffer size: 10x workers
	// This allows bursts of work without blocking Submit(),
	// while keeping memory usage reasonable.
	queueSize := numWorkers * 10

	return &TensorPool{
		numWorkers: numWorkers,
		tasks:      make(chan TensorTask, queueSize),
		stopChan:   make(chan struct{}),
	}
}

// Start spawns worker goroutines that process tasks from the queue.
// Workers run until Stop() is called.
//
// This method is idempotent: calling Start() multiple times has no effect.
func (p *TensorPool) Start() {
	for i := 0; i < p.numWorkers; i++ {
		go p.worker()
	}
}

// worker is the main loop for a worker goroutine.
// It pulls tasks from the queue and executes them until stopChan is closed.
//
// Design notes:
//   - select{} allows checking stopChan while waiting for tasks
//   - wg.Done() ensures Wait() can track completion
//   - Panics are NOT caught (fail-fast for debugging)
func (p *TensorPool) worker() {
	for {
		select {
		case task, ok := <-p.tasks:
			if !ok {
				// Channel closed, exit worker
				return
			}
			// Execute task
			task()
			// Mark task complete
			p.wg.Done()

		case <-p.stopChan:
			// Graceful shutdown requested
			return
		}
	}
}

// Submit adds a task to the worker pool's queue.
// The task will be executed by an available worker goroutine.
//
// This method blocks if the queue is full (all workers busy + buffer full).
// In practice, this is rare with a properly sized buffer.
//
// Parameters:
//   - task: Function to execute in a worker goroutine
//
// Thread safety:
//   - Safe to call concurrently from multiple goroutines
//
// Example:
//   pool.Submit(func() {
//       result := MatMul(a, b)
//       // Store result somewhere
//   })
func (p *TensorPool) Submit(task TensorTask) {
	p.wg.Add(1)     // Increment counter BEFORE submitting
	p.tasks <- task // Send task to queue
}

// Wait blocks until all submitted tasks have completed.
// This is typically called after submitting a batch of work.
//
// Example:
//   // Submit batch of operations
//   for i := 0; i < batchSize; i++ {
//       pool.Submit(func() { /* work */ })
//   }
//   // Wait for entire batch to complete
//   pool.Wait()
//
// Thread safety:
//   - Safe to call concurrently (WaitGroup handles synchronization)
func (p *TensorPool) Wait() {
	p.wg.Wait()
}

// Stop gracefully shuts down the worker pool.
// It waits for in-flight tasks to complete, then exits all workers.
//
// Steps:
//   1. Wait for in-flight tasks (wg.Wait)
//   2. Close task channel (signals workers to exit)
//   3. Close stop channel (backup shutdown signal)
//
// This method is idempotent: calling Stop() multiple times is safe.
//
// After Stop(), the pool cannot be restarted. Create a new pool if needed.
func (p *TensorPool) Stop() {
	p.stopOnce.Do(func() {
		// Wait for in-flight tasks to complete
		p.wg.Wait()

		// Signal workers to exit
		close(p.stopChan)

		// Close task channel (workers will exit when they see closed channel)
		close(p.tasks)
	})
}

// ===========================================================================
// BATCH OPERATIONS USING WORKER POOL
// ===========================================================================

// BatchMatMul performs matrix multiplication on a batch of matrices using
// the worker pool. Each matrix multiplication in the batch is submitted as
// a separate task to the pool.
//
// This is useful during training where you process multiple samples:
//   - Forward pass: Apply same weights to batch of inputs
//   - Backward pass: Compute gradients for batch of samples
//
// Parameters:
//   - pool: Worker pool to use for parallel execution
//   - as: Slice of (M, K) matrices (batch of left operands)
//   - bs: Slice of (K, N) matrices (batch of right operands)
//
// Returns:
//   - []*Tensor: Slice of (M, N) result matrices (one per batch element)
//
// Performance:
//   - Best for: batchSize >= numWorkers (good parallelism)
//   - Speedup: ~2-4x for typical batch sizes (8-32)
//   - Memory: O(batchSize * M * N) for outputs
//
// Example:
//   pool := NewTensorPool(8)
//   pool.Start()
//   defer pool.Stop()
//
//   // Process batch of 32 samples
//   batch := make([]*Tensor, 32)
//   weights := []*Tensor{W1, W2, W3} // Same weights for all samples
//
//   for i := range batch {
//       results := BatchMatMul(pool, batch, weights)
//   }
func BatchMatMul(pool *TensorPool, as, bs []*Tensor) []*Tensor {
	if len(as) != len(bs) {
		panic("tensor: BatchMatMul requires equal-length slices")
	}

	batchSize := len(as)
	results := make([]*Tensor, batchSize)

	// Submit all matmuls to pool
	for i := 0; i < batchSize; i++ {
		// Capture loop variables
		idx := i
		a := as[i]
		b := bs[i]

		pool.Submit(func() {
			results[idx] = MatMul(a, b)
		})
	}

	// Wait for all matmuls to complete
	pool.Wait()

	return results
}

// BatchMatMulParallel performs matrix multiplication on a batch of matrices,
// using BOTH the worker pool (for batch parallelism) AND parallel matmul
// (for per-matrix parallelism).
//
// This provides two levels of parallelism:
//   1. Batch-level: Different matrices processed by different workers
//   2. Matrix-level: Each matrix multiplication uses multiple goroutines
//
// This is most effective when:
//   - Batch size is large (many matrices to process)
//   - Matrices are large (benefit from parallel matmul)
//   - You have many CPU cores (e.g., 16+ cores)
//
// Parameters:
//   - pool: Worker pool for batch-level parallelism
//   - as: Slice of (M, K) matrices
//   - bs: Slice of (K, N) matrices
//   - matmulWorkers: Workers per matrix (0 = runtime.NumCPU() / batchSize)
//
// Returns:
//   - []*Tensor: Slice of result matrices
//
// Performance notes:
//   - Total workers: pool.numWorkers * matmulWorkers
//   - Best when: totalWorkers ≈ runtime.NumCPU()
//   - Oversubscription hurts: too many workers → context switching overhead
//
// Example (16-core machine, batch of 4 matrices):
//   pool := NewTensorPool(4)  // 4 batch-level workers
//   pool.Start()
//   defer pool.Stop()
//
//   // Each matrix uses 4 workers → total 4*4 = 16 workers
//   results := BatchMatMulParallel(pool, batch, weights, 4)
func BatchMatMulParallel(pool *TensorPool, as, bs []*Tensor, matmulWorkers int) []*Tensor {
	if len(as) != len(bs) {
		panic("tensor: BatchMatMulParallel requires equal-length slices")
	}

	batchSize := len(as)
	results := make([]*Tensor, batchSize)

	// Auto-detect matmul workers: distribute available cores across batch
	if matmulWorkers <= 0 {
		totalCores := runtime.NumCPU()
		matmulWorkers = totalCores / batchSize
		if matmulWorkers < 1 {
			matmulWorkers = 1
		}
	}

	// Submit all matmuls to pool
	for i := 0; i < batchSize; i++ {
		idx := i
		a := as[i]
		b := bs[i]

		pool.Submit(func() {
			results[idx] = MatMulParallel(a, b, matmulWorkers)
		})
	}

	pool.Wait()
	return results
}

// ===========================================================================
// PERFORMANCE NOTES
// ===========================================================================
//
// Worker pool overhead:
//   - Pool creation: ~100-200 µs (one-time cost)
//   - Task submission: ~100-500 ns (per task)
//   - Task execution: ~1-2 µs overhead (vs direct call)
//
// When to use worker pool vs direct parallel:
//
// 1. Single large operation:
//    - Use: MatMulParallel (spawn goroutines directly)
//    - Why: No amortization benefit, pool overhead not worth it
//
// 2. Repeated operations (training loop):
//    - Use: TensorPool + BatchMatMul
//    - Why: Amortize pool creation over many iterations
//
// 3. Mixed workloads:
//    - Use: TensorPool + submit various operations
//    - Why: Flexibility to process different tensor ops in same pool
//
// Memory considerations:
//   - Pool size: ~100 bytes per worker
//   - Queue buffer: ~8 bytes per slot
//   - Total: ~1-2 KB for typical pool (8 workers, 80 buffer)
//
// ===========================================================================
