package main

import (
	"sync"
)

// ===========================================================================
// WHAT'S GOING ON HERE
// ===========================================================================
//
// This file implements object pooling using sync.Pool to reduce GC pressure
// during tensor operations. sync.Pool is Go's built-in mechanism for reusing
// objects, particularly useful for reducing allocation overhead in hot paths.
//
// MOTIVATION:
//
// Deep learning training creates many temporary tensors:
//   - Forward pass: Intermediate activations
//   - Backward pass: Gradients for each layer
//   - Optimizer: Momentum buffers, velocity, etc.
//
// Without pooling:
//   1. Allocate tensor → Use → GC collects
//   2. Repeat millions of times
//   3. GC pauses hurt performance (stop-the-world)
//   4. Memory thrashing from constant alloc/free
//
// With sync.Pool:
//   1. Get tensor from pool (reuse or allocate if empty)
//   2. Use tensor
//   3. Return to pool for next iteration
//   4. Much less GC pressure
//
// EXAMPLE WITHOUT POOLING:
//
//   func forward() {
//       temp1 := MatMul(a, b)     // Allocate
//       temp2 := Add(temp1, bias) // Allocate
//       return GELU(temp2)        // temp1, temp2 become garbage
//   }
//   // After 1000 iterations: 2000 allocations + GC overhead
//
// EXAMPLE WITH POOLING:
//
//   func forward() {
//       temp1 := GetTensorFromPool(m, n)
//       MatMulInto(temp1, a, b)   // Reuse memory
//       temp2 := GetTensorFromPool(m, n)
//       AddInto(temp2, temp1, bias)
//       result := GELU(temp2)
//       PutTensorInPool(temp1)
//       PutTensorInPool(temp2)
//       return result
//   }
//   // After 1000 iterations: ~2 allocations total
//
// SYNC.POOL CHARACTERISTICS:
//
// 1. **Thread-safe**: Safe for concurrent use
// 2. **Automatic cleanup**: GC can reclaim pooled objects if memory is tight
// 3. **Per-P pools**: Go runtime creates one pool per P (processor) to reduce contention
// 4. **No guarantees**: Objects may be removed from pool at any time
//
// WHEN TO USE:
//
// ✅ Good use cases:
//   - Training loops (repeated allocations of same size)
//   - Batch processing (temporary buffers)
//   - Forward/backward passes (intermediate activations)
//
// ❌ Bad use cases:
//   - Long-lived tensors (model parameters)
//   - Variable-sized tensors (pool can't reuse)
//   - Rarely-allocated objects (overhead > benefit)
//
// PERFORMANCE CHARACTERISTICS:
//
// Allocation overhead (approximate):
//   - malloc: 50-100 ns
//   - sync.Pool Get (cache hit): 5-10 ns (10x faster)
//   - sync.Pool Get (cache miss): 50-100 ns (same as malloc)
//
// GC impact (1000 iterations, 10 tensors/iteration):
//   Without pooling:
//     - Allocations: 10,000
//     - GC pauses: ~100ms total
//     - Memory: sawtooth pattern (allocate → GC → repeat)
//
//   With pooling:
//     - Allocations: ~10 (initial)
//     - GC pauses: ~5ms total
//     - Memory: stable (objects reused)
//
// EDUCATIONAL VALUE:
//
// This demonstrates:
//   1. Object pooling pattern
//   2. GC pressure reduction
//   3. Trade-offs: memory vs CPU vs complexity
//   4. sync.Pool API and semantics
//   5. When to optimize allocations
//
// COMPARISON WITH OTHER APPROACHES:
//
//   Approach                Pros                           Cons
//   -----------------------------------------------------------------------------
//   No pooling              Simple, safe                   GC pressure, slow
//   sync.Pool               Fast, automatic cleanup        No guarantees, complexity
//   Manual pooling          Full control                   Complex, error-prone
//   Arena allocator         Fastest, batch free           Unsafe, manual lifetime
//   Pre-allocation          Predictable                    Fixed size, waste memory
//
// CAVEATS:
//
// 1. **Don't pool long-lived objects**: Defeats the purpose
// 2. **Clear sensitive data**: Pooled objects are reused
// 3. **Size matters**: Pools work best for fixed sizes
// 4. **Measure first**: Profile before optimizing
//
// ===========================================================================

// TensorAllocPool wraps sync.Pool for tensor-specific pooling.
// It maintains separate pools for different tensor sizes to maximize reuse.
type TensorAllocPool struct {
	// pools maps tensor sizes to sync.Pool instances
	// Key: size (rows * cols), Value: *sync.Pool
	pools map[int]*sync.Pool
	mu    sync.RWMutex
}

// GlobalAllocPool is the default tensor allocation pool for the entire program.
// Most operations should use this global pool for maximum reuse.
var GlobalAllocPool = NewTensorAllocPool()

// NewTensorAllocPool creates a new tensor allocation pool.
//
// Returns:
//   - *TensorAllocPool: A new tensor allocation pool with no pre-allocated pools
//
// Usage:
//   pool := NewTensorAllocPool()
//   tensor := pool.Get(10, 10)
//   defer pool.Put(tensor)
func NewTensorAllocPool() *TensorAllocPool {
	return &TensorAllocPool{
		pools: make(map[int]*sync.Pool),
	}
}

// getPoolForSize returns the sync.Pool for tensors of the given size.
// Creates a new pool if one doesn't exist.
//
// This uses read-write locks to minimize contention:
//   - Fast path: RLock for reading (common case)
//   - Slow path: Lock for writing (pool creation, rare)
func (tp *TensorAllocPool) getPoolForSize(size int) *sync.Pool {
	// Fast path: pool already exists
	tp.mu.RLock()
	pool, exists := tp.pools[size]
	tp.mu.RUnlock()

	if exists {
		return pool
	}

	// Slow path: create new pool
	tp.mu.Lock()
	defer tp.mu.Unlock()

	// Double-check: another goroutine may have created it
	if pool, exists := tp.pools[size]; exists {
		return pool
	}

	// Create new pool with factory function
	pool = &sync.Pool{
		New: func() interface{} {
			// This is called when pool is empty
			// We don't know the shape here, so we allocate just the slice
			return make([]float64, size)
		},
	}

	tp.pools[size] = pool
	return pool
}

// Get retrieves a tensor from the pool or allocates a new one.
//
// The returned tensor is NOT zeroed. Caller must initialize it.
//
// Parameters:
//   - rows, cols: Tensor dimensions
//
// Returns:
//   - *Tensor: A tensor with the requested dimensions
//
// Performance:
//   - Pool hit: ~10ns
//   - Pool miss: ~100ns (allocates)
//
// Example:
//   t := pool.Get(10, 20)
//   defer pool.Put(t)
//   // Use t...
func (tp *TensorAllocPool) Get(rows, cols int) *Tensor {
	size := rows * cols
	pool := tp.getPoolForSize(size)

	// Try to get from pool
	if obj := pool.Get(); obj != nil {
		data := obj.([]float64)
		return &Tensor{
			data:  data[:size], // Reslice to exact size
			shape: []int{rows, cols},
		}
	}

	// Pool was empty, allocate new
	return NewTensor(rows, cols)
}

// Put returns a tensor to the pool for reuse.
//
// IMPORTANT: After calling Put(), the caller must not use the tensor.
// The memory may be reused by another goroutine immediately.
//
// Parameters:
//   - t: Tensor to return to pool
//
// Performance:
//   - ~10ns (just adds to pool)
//
// Example:
//   t := pool.Get(10, 20)
//   // Use t...
//   pool.Put(t) // Return for reuse
//   // DO NOT use t after this point!
func (tp *TensorAllocPool) Put(t *Tensor) {
	if t == nil || len(t.data) == 0 {
		return
	}

	size := len(t.data)
	pool := tp.getPoolForSize(size)

	// Optional: Clear sensitive data
	// Uncomment if tensors contain sensitive information
	// for i := range t.data {
	// 	t.data[i] = 0
	// }

	// Return underlying slice to pool
	pool.Put(t.data)
}

// GetZeroed retrieves a zeroed tensor from the pool.
//
// This is slower than Get() but ensures the tensor is initialized.
//
// Parameters:
//   - rows, cols: Tensor dimensions
//
// Returns:
//   - *Tensor: A zero-initialized tensor
//
// Performance:
//   - Pool hit: ~10ns + O(size) for zeroing
//   - Pool miss: ~100ns + O(size) for zeroing
func (tp *TensorAllocPool) GetZeroed(rows, cols int) *Tensor {
	t := tp.Get(rows, cols)

	// Zero the data
	for i := range t.data {
		t.data[i] = 0
	}

	return t
}

// ===========================================================================
// CONVENIENCE FUNCTIONS FOR GLOBAL POOL
// ===========================================================================

// GetPooledTensor gets a tensor from the global allocation pool.
//
// This is the recommended way to get pooled tensors in most code.
//
// Example:
//   t := GetPooledTensor(10, 20)
//   defer PutPooledTensor(t)
func GetPooledTensor(rows, cols int) *Tensor {
	return GlobalAllocPool.Get(rows, cols)
}

// PutPooledTensor returns a tensor to the global allocation pool.
//
// Example:
//   t := GetPooledTensor(10, 20)
//   defer PutPooledTensor(t)
func PutPooledTensor(t *Tensor) {
	GlobalAllocPool.Put(t)
}

// GetPooledTensorZeroed gets a zeroed tensor from the global allocation pool.
//
// Example:
//   t := GetPooledTensorZeroed(10, 20)
//   defer PutPooledTensor(t)
func GetPooledTensorZeroed(rows, cols int) *Tensor {
	return GlobalAllocPool.GetZeroed(rows, cols)
}

// ===========================================================================
// HELPER: WITH PATTERN
// ===========================================================================

// WithPooledTensor executes a function with a pooled tensor, automatically
// returning it to the pool when done.
//
// This ensures the tensor is always returned, even if the function panics.
//
// Parameters:
//   - rows, cols: Tensor dimensions
//   - fn: Function to execute with the tensor
//
// Returns:
//   - error: Any error from fn
//
// Example:
//   err := WithPooledTensor(10, 20, func(t *Tensor) error {
//       // Use t...
//       return nil
//   })
func WithPooledTensor(rows, cols int, fn func(*Tensor) error) error {
	t := GetPooledTensor(rows, cols)
	defer PutPooledTensor(t)
	return fn(t)
}

// ===========================================================================
// USAGE PATTERNS
// ===========================================================================
//
// PATTERN 1: Manual Get/Put
//
//   func processData() {
//       temp := GetPooledTensor(100, 100)
//       defer PutPooledTensor(temp)
//       // Use temp...
//   }
//
// PATTERN 2: With helper
//
//   func processData() error {
//       return WithPooledTensor(100, 100, func(temp *Tensor) error {
//           // Use temp...
//           return nil
//       })
//   }
//
// PATTERN 3: Training loop
//
//   for epoch := 0; epoch < epochs; epoch++ {
//       for _, batch := range batches {
//           // Forward pass (reuses tensors each iteration)
//           hidden := GetPooledTensor(batchSize, hiddenSize)
//           forward(batch, hidden)
//
//           // Backward pass
//           grad := GetPooledTensor(batchSize, hiddenSize)
//           backward(hidden, grad)
//
//           // Return to pool
//           PutPooledTensor(hidden)
//           PutPooledTensor(grad)
//       }
//   }
//
// PATTERN 4: Batch processing with worker pool
//
//   pool := NewTensorPool() // Local pool for workers
//   for i := 0; i < numWorkers; i++ {
//       go func() {
//           // Each worker reuses tensors
//           temp := pool.Get(size, size)
//           defer pool.Put(temp)
//           // Process batches...
//       }()
//   }
//
// ===========================================================================
// BENCHMARKING NOTES
// ===========================================================================
//
// To measure impact of pooling:
//
//   go test -bench=BenchmarkPooling -benchmem
//
// Expected results (approximate):
//
//   BenchmarkWithoutPooling-8    1000000    1200 ns/op    800 B/op   10 allocs/op
//   BenchmarkWithPooling-8      10000000     120 ns/op      0 B/op    0 allocs/op
//
// Key metrics:
//   - ns/op: 10x faster with pooling
//   - B/op: 0 bytes allocated (reusing memory)
//   - allocs/op: 0 allocations (reusing objects)
//
// ===========================================================================
