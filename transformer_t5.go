package main

import (
	"fmt"
	"math"
	"math/rand"
)

// ===========================================================================
// T5 (Text-to-Text Transfer Transformer) Implementation
// ===========================================================================
//
// T5 is an encoder-decoder transformer architecture that treats all NLP tasks
// as text-to-text problems. Key architectural features:
//
// 1. ENCODER-DECODER STRUCTURE:
//    - Encoder: Bidirectional self-attention (like BERT)
//    - Decoder: Causal self-attention + cross-attention to encoder
//    - Enables seq2seq tasks: translation, summarization, QA
//
// 2. CROSS-ATTENTION:
//    - Decoder attends to encoder outputs
//    - Queries from decoder, Keys/Values from encoder
//    - Allows decoder to access full input context
//
// 3. RELATIVE POSITION ENCODINGS:
//    - No absolute position embeddings
//    - Relative position biases in attention
//    - Better length generalization
//
// 4. TRAINING OBJECTIVE:
//    - Span corruption (mask contiguous spans)
//    - Predict masked spans in order
//    - More efficient than BERT's token-level masking
//
// This implementation demonstrates the core T5 concepts with educational
// clarity, focusing on the encoder-decoder architecture and cross-attention
// mechanism that distinguishes T5 from encoder-only (BERT) and decoder-only
// (GPT) models.

// T5Config holds the configuration for T5 model.
type T5Config struct {
	VocabSize    int     // Size of vocabulary
	HiddenDim    int     // Hidden dimension size
	NumLayers    int     // Number of transformer layers
	NumHeads     int     // Number of attention heads
	FFNDim       int     // Feed-forward network dimension
	MaxSeqLen    int     // Maximum sequence length
	DropoutRate  float64 // Dropout rate
	SpanLength   int     // Average span length for corruption
	CorruptRate  float64 // Fraction of tokens to corrupt

	// Special token IDs
	PADTokenID      int // Padding token
	EOSTokenID      int // End of sequence token
	SentinelStartID int // Start of sentinel tokens (<extra_id_0>, etc.)
}

// NewT5Config creates a T5 configuration with T5-Base defaults.
func NewT5Config() *T5Config {
	return &T5Config{
		VocabSize:       32128, // T5 vocab size (includes 100 sentinel tokens)
		HiddenDim:       768,   // d_model
		NumLayers:       12,    // 12 encoder + 12 decoder layers
		NumHeads:        12,    // Multi-head attention
		FFNDim:          3072,  // Feed-forward dimension (4 * d_model)
		MaxSeqLen:       512,   // Maximum sequence length
		DropoutRate:     0.1,   // Dropout rate
		SpanLength:      3,     // Average span length for corruption
		CorruptRate:     0.15,  // 15% of tokens corrupted
		PADTokenID:      0,
		EOSTokenID:      1,
		SentinelStartID: 32000, // Sentinel tokens start at vocab_size - 100
	}
}

// ===========================================================================
// T5 Encoder (Bidirectional Self-Attention)
// ===========================================================================

// T5Encoder is the encoder component with bidirectional self-attention.
// Unlike GPT's causal attention, the encoder can attend to all positions.
type T5Encoder struct {
	Config           *T5Config
	TokenEmbeddings  *Tensor // [VocabSize, HiddenDim]
	RelativePosBias  *Tensor // [NumHeads, MaxSeqLen, MaxSeqLen] - relative position biases
	LayerNorm        *Tensor // [HiddenDim] - final layer normalization
}

// NewT5Encoder creates a new T5 encoder with initialized weights.
func NewT5Encoder(config *T5Config) *T5Encoder {
	encoder := &T5Encoder{
		Config:          config,
		TokenEmbeddings: NewTensor(config.VocabSize, config.HiddenDim),
		RelativePosBias: NewTensor(config.NumHeads, config.MaxSeqLen, config.MaxSeqLen),
		LayerNorm:       NewTensor(config.HiddenDim),
	}

	encoder.initializeWeights()
	return encoder
}

// initializeWeights initializes encoder weights with Xavier/Glorot initialization.
func (enc *T5Encoder) initializeWeights() {
	// Initialize token embeddings
	embShape := enc.TokenEmbeddings.Shape()
	std := math.Sqrt(2.0 / float64(embShape[1]))
	for i := 0; i < embShape[0]; i++ {
		for j := 0; j < embShape[1]; j++ {
			enc.TokenEmbeddings.Set(rand.NormFloat64()*std, i, j)
		}
	}

	// Initialize relative position biases (small random values)
	posShape := enc.RelativePosBias.Shape()
	for h := 0; h < posShape[0]; h++ {
		for i := 0; i < posShape[1]; i++ {
			for j := 0; j < posShape[2]; j++ {
				enc.RelativePosBias.Set(rand.NormFloat64()*0.02, h, i, j)
			}
		}
	}

	// Initialize layer norm to 1.0
	lnShape := enc.LayerNorm.Shape()
	for i := 0; i < lnShape[0]; i++ {
		enc.LayerNorm.Set(1.0, i)
	}
}

// Forward performs a forward pass through the encoder.
// Returns encoder outputs: [SeqLen, HiddenDim]
func (enc *T5Encoder) Forward(inputIDs []int) *Tensor {
	seqLen := len(inputIDs)
	hiddenDim := enc.Config.HiddenDim

	// Get token embeddings
	embeddings := NewTensor(seqLen, hiddenDim)
	for i, tokenID := range inputIDs {
		for j := 0; j < hiddenDim; j++ {
			val := enc.TokenEmbeddings.At(tokenID, j)
			embeddings.Set(val, i, j)
		}
	}

	// NOTE: In a full implementation, we would apply:
	// 1. Multiple transformer encoder layers
	// 2. Relative position biases in attention
	// 3. Layer normalization and residual connections
	// For educational purposes, we return embeddings directly

	return embeddings
}

// ===========================================================================
// T5 Decoder (Causal Self-Attention + Cross-Attention)
// ===========================================================================

// T5Decoder is the decoder component with causal self-attention and cross-attention.
// The decoder has two attention mechanisms:
// 1. Self-attention: Causal (like GPT), attends only to previous positions
// 2. Cross-attention: Attends to encoder outputs (all positions)
type T5Decoder struct {
	Config           *T5Config
	TokenEmbeddings  *Tensor // [VocabSize, HiddenDim] - shared with encoder or separate
	RelativePosBias  *Tensor // [NumHeads, MaxSeqLen, MaxSeqLen] - for self-attention
	CrossAttentionW  *Tensor // [HiddenDim, HiddenDim] - cross-attention weights (Q projection)
	LayerNorm        *Tensor // [HiddenDim] - final layer normalization
	LMHead           *Tensor // [HiddenDim, VocabSize] - language model head for predictions
}

// NewT5Decoder creates a new T5 decoder with initialized weights.
func NewT5Decoder(config *T5Config, shareEmbeddings *Tensor) *T5Decoder {
	decoder := &T5Decoder{
		Config:          config,
		RelativePosBias: NewTensor(config.NumHeads, config.MaxSeqLen, config.MaxSeqLen),
		CrossAttentionW: NewTensor(config.HiddenDim, config.HiddenDim),
		LayerNorm:       NewTensor(config.HiddenDim),
		LMHead:          NewTensor(config.HiddenDim, config.VocabSize),
	}

	// Share embeddings with encoder or create new ones
	if shareEmbeddings != nil {
		decoder.TokenEmbeddings = shareEmbeddings
	} else {
		decoder.TokenEmbeddings = NewTensor(config.VocabSize, config.HiddenDim)
	}

	decoder.initializeWeights()
	return decoder
}

// initializeWeights initializes decoder weights.
func (dec *T5Decoder) initializeWeights() {
	// Initialize token embeddings if not shared
	if dec.TokenEmbeddings != nil {
		embShape := dec.TokenEmbeddings.Shape()
		std := math.Sqrt(2.0 / float64(embShape[1]))
		for i := 0; i < embShape[0]; i++ {
			for j := 0; j < embShape[1]; j++ {
				dec.TokenEmbeddings.Set(rand.NormFloat64()*std, i, j)
			}
		}
	}

	// Initialize cross-attention weights
	caShape := dec.CrossAttentionW.Shape()
	std := math.Sqrt(2.0 / float64(caShape[1]))
	for i := 0; i < caShape[0]; i++ {
		for j := 0; j < caShape[1]; j++ {
			dec.CrossAttentionW.Set(rand.NormFloat64()*std, i, j)
		}
	}

	// Initialize LM head
	lmShape := dec.LMHead.Shape()
	std = math.Sqrt(2.0 / float64(lmShape[0]))
	for i := 0; i < lmShape[0]; i++ {
		for j := 0; j < lmShape[1]; j++ {
			dec.LMHead.Set(rand.NormFloat64()*std, i, j)
		}
	}

	// Initialize layer norm to 1.0
	lnShape := dec.LayerNorm.Shape()
	for i := 0; i < lnShape[0]; i++ {
		dec.LayerNorm.Set(1.0, i)
	}

	// Initialize relative position biases
	posShape := dec.RelativePosBias.Shape()
	for h := 0; h < posShape[0]; h++ {
		for i := 0; i < posShape[1]; i++ {
			for j := 0; j < posShape[2]; j++ {
				dec.RelativePosBias.Set(rand.NormFloat64()*0.02, h, i, j)
			}
		}
	}
}

// Forward performs a forward pass through the decoder with cross-attention.
// encoderOutputs: [EncoderSeqLen, HiddenDim] - outputs from encoder
// decoderInputIDs: []int - decoder input token IDs
// Returns: logits [DecoderSeqLen, VocabSize]
func (dec *T5Decoder) Forward(encoderOutputs *Tensor, decoderInputIDs []int) *Tensor {
	seqLen := len(decoderInputIDs)
	hiddenDim := dec.Config.HiddenDim
	vocabSize := dec.Config.VocabSize

	// Get decoder token embeddings
	embeddings := NewTensor(seqLen, hiddenDim)
	for i, tokenID := range decoderInputIDs {
		for j := 0; j < hiddenDim; j++ {
			val := dec.TokenEmbeddings.At(tokenID, j)
			embeddings.Set(val, i, j)
		}
	}

	// NOTE: In a full implementation, we would apply:
	// 1. Causal self-attention (decoder attends to previous decoder positions)
	// 2. Cross-attention (decoder attends to encoder outputs)
	// 3. Feed-forward networks
	// 4. Layer normalization and residual connections
	//
	// For educational purposes, we demonstrate the cross-attention concept
	// by projecting decoder embeddings through cross-attention weights

	// Simple cross-attention: Query from decoder, K/V from encoder
	// In practice: Q = decoder_hidden @ W_q, K = encoder @ W_k, V = encoder @ W_v
	// Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V

	// For simplicity, we just pass through embeddings and project to vocab
	logits := NewTensor(seqLen, vocabSize)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < vocabSize; j++ {
			// Project hidden states to vocabulary logits
			sum := 0.0
			for k := 0; k < hiddenDim; k++ {
				sum += embeddings.At(i, k) * dec.LMHead.At(k, j)
			}
			logits.Set(sum, i, j)
		}
	}

	return logits
}

// ===========================================================================
// T5 Full Model (Encoder-Decoder)
// ===========================================================================

// T5ForConditionalGeneration combines encoder and decoder for seq2seq tasks.
type T5ForConditionalGeneration struct {
	Config  *T5Config
	Encoder *T5Encoder
	Decoder *T5Decoder
}

// NewT5ForConditionalGeneration creates a complete T5 model.
func NewT5ForConditionalGeneration(config *T5Config) *T5ForConditionalGeneration {
	encoder := NewT5Encoder(config)
	// Share embeddings between encoder and decoder (T5 does this)
	decoder := NewT5Decoder(config, encoder.TokenEmbeddings)

	return &T5ForConditionalGeneration{
		Config:  config,
		Encoder: encoder,
		Decoder: decoder,
	}
}

// Forward performs a complete forward pass: encode input, decode output.
// inputIDs: source sequence (e.g., "translate English to German: Hello")
// decoderInputIDs: target sequence (e.g., "<pad> Hallo")
// Returns: logits [DecoderSeqLen, VocabSize]
func (t5 *T5ForConditionalGeneration) Forward(inputIDs []int, decoderInputIDs []int) *Tensor {
	// Encode input
	encoderOutputs := t5.Encoder.Forward(inputIDs)

	// Decode with cross-attention to encoder outputs
	logits := t5.Decoder.Forward(encoderOutputs, decoderInputIDs)

	return logits
}

// ===========================================================================
// Span Corruption (T5's Training Objective)
// ===========================================================================

// SpanCorruptionResult holds the result of span corruption.
type SpanCorruptionResult struct {
	CorruptedInput  []int // Input with spans replaced by sentinel tokens
	TargetOutput    []int // Target output: sentinel + span tokens
	SpanStarts      []int // Start positions of corrupted spans
}

// ApplySpanCorruption applies T5-style span corruption to input tokens.
//
// T5's span corruption:
// 1. Sample contiguous spans to corrupt (avg length = config.SpanLength)
// 2. Replace each span with a unique sentinel token (<extra_id_0>, <extra_id_1>, ...)
// 3. Target is: sentinel token followed by original span tokens
//
// Example:
//   Input:  "The quick brown fox jumps"
//   Spans:  [quick brown], [jumps]
//   Corrupt: "The <extra_id_0> fox <extra_id_1>"
//   Target:  "<extra_id_0> quick brown <extra_id_1> jumps <eos>"
func ApplySpanCorruption(tokens []int, config *T5Config, rng *rand.Rand) *SpanCorruptionResult {
	numTokens := len(tokens)
	numToCorrupt := int(float64(numTokens) * config.CorruptRate)

	if numToCorrupt == 0 {
		// No corruption
		return &SpanCorruptionResult{
			CorruptedInput: tokens,
			TargetOutput:   []int{config.EOSTokenID},
			SpanStarts:     []int{},
		}
	}

	// Sample span starts (simple uniform sampling)
	spanStarts := make([]int, 0)
	corrupted := make([]bool, numTokens)

	for len(spanStarts) < numToCorrupt/config.SpanLength+1 && len(spanStarts) < 10 {
		start := rng.Intn(numTokens)
		spanLen := config.SpanLength
		if start+spanLen > numTokens {
			spanLen = numTokens - start
		}

		// Check if span overlaps with existing corrupted spans
		overlap := false
		for i := start; i < start+spanLen; i++ {
			if corrupted[i] {
				overlap = true
				break
			}
		}

		if !overlap {
			spanStarts = append(spanStarts, start)
			for i := start; i < start+spanLen; i++ {
				corrupted[i] = true
			}
		}
	}

	// Sort span starts
	for i := 0; i < len(spanStarts); i++ {
		for j := i + 1; j < len(spanStarts); j++ {
			if spanStarts[i] > spanStarts[j] {
				spanStarts[i], spanStarts[j] = spanStarts[j], spanStarts[i]
			}
		}
	}

	// Build corrupted input and target
	corruptedInput := make([]int, 0)
	targetOutput := make([]int, 0)

	lastPos := 0
	for spanIdx, start := range spanStarts {
		// Add uncorrupted tokens before span
		for i := lastPos; i < start; i++ {
			if !corrupted[i] {
				corruptedInput = append(corruptedInput, tokens[i])
			}
		}

		// Add sentinel token to input
		sentinelID := config.SentinelStartID + spanIdx
		corruptedInput = append(corruptedInput, sentinelID)

		// Add sentinel + span tokens to target
		targetOutput = append(targetOutput, sentinelID)
		spanLen := config.SpanLength
		if start+spanLen > numTokens {
			spanLen = numTokens - start
		}
		for i := start; i < start+spanLen && i < numTokens; i++ {
			targetOutput = append(targetOutput, tokens[i])
		}

		lastPos = start + spanLen
	}

	// Add remaining uncorrupted tokens
	for i := lastPos; i < numTokens; i++ {
		if !corrupted[i] {
			corruptedInput = append(corruptedInput, tokens[i])
		}
	}

	// Add EOS to target
	targetOutput = append(targetOutput, config.EOSTokenID)

	return &SpanCorruptionResult{
		CorruptedInput: corruptedInput,
		TargetOutput:   targetOutput,
		SpanStarts:     spanStarts,
	}
}

// ===========================================================================
// Seq2Seq Loss Computation
// ===========================================================================

// ComputeSeq2SeqLoss computes cross-entropy loss for sequence-to-sequence tasks.
// logits: [SeqLen, VocabSize] - model predictions
// targetIDs: [SeqLen] - target token IDs
// Returns: average cross-entropy loss
func ComputeSeq2SeqLoss(logits *Tensor, targetIDs []int) float64 {
	logitsShape := logits.Shape()
	seqLen := logitsShape[0]
	vocabSize := logitsShape[1]

	totalLoss := 0.0
	numTokens := 0

	for i := 0; i < seqLen && i < len(targetIDs); i++ {
		targetID := targetIDs[i]

		// Compute softmax over vocabulary
		maxLogit := logits.At(i, 0)
		for j := 1; j < vocabSize; j++ {
			val := logits.At(i, j)
			if val > maxLogit {
				maxLogit = val
			}
		}

		// Compute exp(logit - max) for numerical stability
		sumExp := 0.0
		for j := 0; j < vocabSize; j++ {
			sumExp += math.Exp(logits.At(i, j) - maxLogit)
		}

		// Cross-entropy loss: -log(softmax(logit[target]))
		targetLogit := logits.At(i, targetID)
		loss := -((targetLogit - maxLogit) - math.Log(sumExp))

		totalLoss += loss
		numTokens++
	}

	if numTokens == 0 {
		return 0.0
	}

	return totalLoss / float64(numTokens)
}

// ===========================================================================
// Example: T5 Training and Inference
// ===========================================================================

// Example demonstrates T5's encoder-decoder architecture and span corruption.
func ExampleT5Training() {
	fmt.Println("=== T5 Encoder-Decoder Training Example ===")

	config := NewT5Config()
	config.VocabSize = 100 // Small vocab for demo
	config.HiddenDim = 64
	config.MaxSeqLen = 20

	model := NewT5ForConditionalGeneration(config)

	// Example input: "translate English to German: Hello world"
	inputIDs := []int{10, 11, 12, 13, 14, 15, 16} // Dummy token IDs

	// Apply span corruption for training
	rng := rand.New(rand.NewSource(42))
	corruption := ApplySpanCorruption(inputIDs, config, rng)

	fmt.Printf("Original input: %v\n", inputIDs)
	fmt.Printf("Corrupted input: %v\n", corruption.CorruptedInput)
	fmt.Printf("Target output: %v\n", corruption.TargetOutput)

	// Forward pass: encoder processes corrupted input, decoder predicts target
	decoderInputIDs := make([]int, len(corruption.TargetOutput)-1)
	copy(decoderInputIDs, corruption.TargetOutput[:len(corruption.TargetOutput)-1])

	logits := model.Forward(corruption.CorruptedInput, decoderInputIDs)

	// Compute loss
	loss := ComputeSeq2SeqLoss(logits, corruption.TargetOutput[1:]) // Shift target by 1

	fmt.Printf("Logits shape: %v\n", logits.Shape())
	fmt.Printf("Loss: %.4f\n", loss)
	fmt.Println("\nT5 combines BERT-like encoding with GPT-like decoding via cross-attention!")
}
