package main

// ===========================================================================
// KV CACHE - Speed up autoregressive generation
// ===========================================================================
//
// WHAT IS KV CACHING?
//
// In autoregressive generation (generating text one token at a time), the
// naive approach recomputes attention for ALL previous tokens every time we
// generate a new token. This is extremely wasteful!
//
// Example: Generating 100 tokens
//   - Token 1: Compute attention for 1 token
//   - Token 2: Compute attention for 1+2 = 3 tokens (wasteful! token 1 repeated)
//   - Token 3: Compute attention for 1+2+3 = 6 tokens
//   - ...
//   - Token 100: Compute attention for 1+2+...+100 = 5050 tokens
//
// Total: ~5000 redundant computations for just 100 tokens!
//
// THE SOLUTION: KV Caching
//
// In attention, we compute three matrices: Query (Q), Key (K), Value (V).
// The KEY INSIGHT: For tokens we've already seen, their K and V matrices
// never change! Only the Q matrix for the NEW token needs to be computed.
//
// So we CACHE the K and V matrices for all previous tokens, and only compute
// Q, K, V for the new token. Then we can compute attention between the new Q
// and all cached K/V pairs.
//
// With KV caching:
//   - Token 1: Compute Q, K, V for 1 token, cache K and V
//   - Token 2: Compute Q, K, V for 1 NEW token, use cached K/V from token 1
//   - Token 3: Compute Q, K, V for 1 NEW token, use cached K/V from tokens 1-2
//   - ...
//   - Token 100: Compute Q, K, V for 1 NEW token, use cached K/V from tokens 1-99
//
// Total: Only 100 new token computations! ~50x faster for 100 tokens.
//
// SPEEDUP ANALYSIS:
//
// Without KV cache: O(n²) where n is sequence length
//   - Generating 100 tokens: ~5000 attention computations
//   - Generating 1000 tokens: ~500,000 attention computations
//
// With KV cache: O(n) where n is sequence length
//   - Generating 100 tokens: ~100 attention computations
//   - Generating 1000 tokens: ~1000 attention computations
//
// The speedup factor grows linearly with sequence length!
//   - 100 tokens: ~50x faster
//   - 1000 tokens: ~500x faster
//
// TRADE-OFFS:
//
// Memory: KV cache stores 2 × num_layers × seq_len × embed_dim floats
//   - For a 12-layer, 768-dim model with 2048 context: ~75MB per sequence
//   - This is totally worth it for the 50-500x speedup!
//
// Complexity: Code becomes slightly more complex because attention needs to
//   handle two modes: (1) normal forward pass for training, and (2) incremental
//   forward pass with cache for generation.
//
// WHY THIS MATTERS:
//
// This is THE key optimization that makes interactive chat with LLMs practical.
// Without KV caching, generating responses would take minutes instead of seconds.
// Every production LLM inference system (GPT-4, Claude, etc.) uses KV caching.
//
// RECOMMENDED READING:
//   - "Generating Long Sequences with Sparse Transformers" (Child et al., 2019)
//     https://arxiv.org/abs/1904.10509
//
//   - Flash Attention paper discusses KV cache optimizations:
//     https://arxiv.org/abs/2205.14135
//
// ===========================================================================

// KVCache stores cached Key and Value matrices for efficient generation.
//
// During autoregressive generation, we generate tokens one at a time.
// For each new token, we need to compute attention with all previous tokens.
// But the K and V matrices for previous tokens never change! So we cache them.
//
// Structure:
//   - keys[layer_idx] = tensor of shape (cached_len, embed_dim)
//   - values[layer_idx] = tensor of shape (cached_len, embed_dim)
//
// The cache grows as we generate more tokens:
//   - Initially empty (cached_len = 0)
//   - After token 1: cached_len = 1
//   - After token 2: cached_len = 2
//   - etc.
type KVCache struct {
	// Separate cache for each transformer layer
	keys   []*Tensor // keys[layer_idx] shape: (cached_len, embed_dim)
	values []*Tensor // values[layer_idx] shape: (cached_len, embed_dim)

	// Current number of cached tokens
	cachedLen int

	// Maximum cache size (usually equals model's max sequence length)
	maxLen int
}

// NewKVCache creates a new KV cache for a model.
//
// Parameters:
//   - numLayers: number of transformer layers in the model
//   - maxLen: maximum sequence length the model can handle
//   - embedDim: embedding dimension of the model
//
// The cache is initially empty (cachedLen = 0).
func NewKVCache(numLayers, maxLen, embedDim int) *KVCache {
	cache := &KVCache{
		keys:      make([]*Tensor, numLayers),
		values:    make([]*Tensor, numLayers),
		cachedLen: 0,
		maxLen:    maxLen,
	}

	// Pre-allocate tensors for maximum size
	// This avoids repeated allocations during generation
	for i := 0; i < numLayers; i++ {
		cache.keys[i] = NewTensor(maxLen, embedDim)
		cache.values[i] = NewTensor(maxLen, embedDim)
	}

	return cache
}

// Append adds new K and V matrices to the cache for a specific layer.
//
// This is called after computing K and V for a new token during generation.
//
// Parameters:
//   - layerIdx: which transformer layer this cache entry is for
//   - newKeys: K matrix for the new token(s), shape: (new_len, embed_dim)
//   - newValues: V matrix for the new token(s), shape: (new_len, embed_dim)
//
// The new K/V are appended to the existing cache.
func (kv *KVCache) Append(layerIdx int, newKeys, newValues *Tensor) {
	if len(newKeys.shape) != 2 || len(newValues.shape) != 2 {
		panic("kv_cache: new keys and values must be 2D tensors")
	}

	newLen := newKeys.shape[0]
	embedDim := newKeys.shape[1]

	if newLen+kv.cachedLen > kv.maxLen {
		panic("kv_cache: cache overflow - sequence too long")
	}

	// Copy new K/V into the cache at the current position
	for i := 0; i < newLen; i++ {
		for j := 0; j < embedDim; j++ {
			kv.keys[layerIdx].Set(newKeys.At(i, j), kv.cachedLen+i, j)
			kv.values[layerIdx].Set(newValues.At(i, j), kv.cachedLen+i, j)
		}
	}

	// Update cached length (only once after appending to all layers)
	// Note: This assumes Append is called sequentially for all layers
	// with the same newLen before incrementing cachedLen
	if layerIdx == len(kv.keys)-1 {
		kv.cachedLen += newLen
	}
}

// GetKeys returns the cached K matrix for a specific layer.
//
// Returns a view of shape (cachedLen, embedDim) containing all cached keys.
func (kv *KVCache) GetKeys(layerIdx int) *Tensor {
	if kv.cachedLen == 0 {
		return nil
	}

	// Return a slice of the cache up to cachedLen
	embedDim := kv.keys[layerIdx].shape[1]
	result := NewTensor(kv.cachedLen, embedDim)

	for i := 0; i < kv.cachedLen; i++ {
		for j := 0; j < embedDim; j++ {
			result.Set(kv.keys[layerIdx].At(i, j), i, j)
		}
	}

	return result
}

// GetValues returns the cached V matrix for a specific layer.
//
// Returns a view of shape (cachedLen, embedDim) containing all cached values.
func (kv *KVCache) GetValues(layerIdx int) *Tensor {
	if kv.cachedLen == 0 {
		return nil
	}

	// Return a slice of the cache up to cachedLen
	embedDim := kv.values[layerIdx].shape[1]
	result := NewTensor(kv.cachedLen, embedDim)

	for i := 0; i < kv.cachedLen; i++ {
		for j := 0; j < embedDim; j++ {
			result.Set(kv.values[layerIdx].At(i, j), i, j)
		}
	}

	return result
}

// Reset clears the cache, preparing it for a new generation sequence.
//
// This should be called:
//   - Before starting generation for a new prompt
//   - When switching between different generation tasks
func (kv *KVCache) Reset() {
	kv.cachedLen = 0
	// Note: We don't need to zero out the tensors, just reset the length.
	// The old data will be overwritten as new data is appended.
}

// CachedLen returns the number of tokens currently cached.
func (kv *KVCache) CachedLen() int {
	return kv.cachedLen
}
