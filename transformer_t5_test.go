package main

import (
	"math/rand"
	"testing"
)

// TestT5Config tests T5 configuration creation and defaults.
func TestT5Config(t *testing.T) {
	config := NewT5Config()

	// Check default values match T5-Base
	if config.VocabSize != 32128 {
		t.Errorf("Expected VocabSize 32128, got %d", config.VocabSize)
	}

	if config.HiddenDim != 768 {
		t.Errorf("Expected HiddenDim 768, got %d", config.HiddenDim)
	}

	if config.NumLayers != 12 {
		t.Errorf("Expected NumLayers 12, got %d", config.NumLayers)
	}

	if config.NumHeads != 12 {
		t.Errorf("Expected NumHeads 12, got %d", config.NumHeads)
	}

	if config.SpanLength != 3 {
		t.Errorf("Expected SpanLength 3, got %d", config.SpanLength)
	}

	if config.CorruptRate != 0.15 {
		t.Errorf("Expected CorruptRate 0.15, got %f", config.CorruptRate)
	}

	t.Logf("T5 config created successfully with T5-Base defaults")
}

// TestT5EncoderCreation tests encoder initialization.
func TestT5EncoderCreation(t *testing.T) {
	config := NewT5Config()
	config.VocabSize = 1000
	config.HiddenDim = 256

	encoder := NewT5Encoder(config)

	// Check embedding dimensions
	embShape := encoder.TokenEmbeddings.Shape()
	if embShape[0] != config.VocabSize || embShape[1] != config.HiddenDim {
		t.Errorf("Token embeddings shape mismatch: got [%d,%d], expected [%d,%d]",
			embShape[0], embShape[1], config.VocabSize, config.HiddenDim)
	}

	// Check relative position bias dimensions
	posShape := encoder.RelativePosBias.Shape()
	if posShape[0] != config.NumHeads || posShape[1] != config.MaxSeqLen || posShape[2] != config.MaxSeqLen {
		t.Errorf("Relative position bias shape mismatch")
	}

	t.Logf("T5 encoder created successfully")
}

// TestT5EncoderForward tests encoder forward pass.
func TestT5EncoderForward(t *testing.T) {
	config := NewT5Config()
	config.VocabSize = 100
	config.HiddenDim = 64

	encoder := NewT5Encoder(config)

	// Input sequence
	inputIDs := []int{10, 20, 30, 40, 50}

	// Forward pass
	outputs := encoder.Forward(inputIDs)

	// Check output shape
	outShape := outputs.Shape()
	if outShape[0] != len(inputIDs) || outShape[1] != config.HiddenDim {
		t.Errorf("Encoder output shape mismatch: got [%d,%d], expected [%d,%d]",
			outShape[0], outShape[1], len(inputIDs), config.HiddenDim)
	}

	t.Logf("Encoder forward pass successful: %v -> [%d,%d]", inputIDs, outShape[0], outShape[1])
}

// TestT5DecoderCreation tests decoder initialization.
func TestT5DecoderCreation(t *testing.T) {
	config := NewT5Config()
	config.VocabSize = 1000
	config.HiddenDim = 256

	// Create encoder first to share embeddings
	encoder := NewT5Encoder(config)
	decoder := NewT5Decoder(config, encoder.TokenEmbeddings)

	// Check that embeddings are shared
	if decoder.TokenEmbeddings != encoder.TokenEmbeddings {
		t.Error("Decoder should share embeddings with encoder")
	}

	// Check cross-attention weights
	caShape := decoder.CrossAttentionW.Shape()
	if caShape[0] != config.HiddenDim || caShape[1] != config.HiddenDim {
		t.Errorf("Cross-attention weight shape mismatch")
	}

	// Check LM head dimensions
	lmShape := decoder.LMHead.Shape()
	if lmShape[0] != config.HiddenDim || lmShape[1] != config.VocabSize {
		t.Errorf("LM head shape mismatch")
	}

	t.Logf("T5 decoder created successfully with shared embeddings")
}

// TestT5DecoderForward tests decoder forward pass with cross-attention.
func TestT5DecoderForward(t *testing.T) {
	config := NewT5Config()
	config.VocabSize = 100
	config.HiddenDim = 64

	encoder := NewT5Encoder(config)
	decoder := NewT5Decoder(config, encoder.TokenEmbeddings)

	// Encoder input
	inputIDs := []int{10, 20, 30, 40, 50}
	encoderOutputs := encoder.Forward(inputIDs)

	// Decoder input
	decoderInputIDs := []int{1, 2, 3}

	// Decoder forward pass
	logits := decoder.Forward(encoderOutputs, decoderInputIDs)

	// Check logits shape
	logitsShape := logits.Shape()
	if logitsShape[0] != len(decoderInputIDs) || logitsShape[1] != config.VocabSize {
		t.Errorf("Decoder logits shape mismatch: got [%d,%d], expected [%d,%d]",
			logitsShape[0], logitsShape[1], len(decoderInputIDs), config.VocabSize)
	}

	t.Logf("Decoder forward pass with cross-attention successful")
}

// TestT5FullModel tests complete encoder-decoder model.
func TestT5FullModel(t *testing.T) {
	config := NewT5Config()
	config.VocabSize = 100
	config.HiddenDim = 64

	model := NewT5ForConditionalGeneration(config)

	// Encoder input: "translate English to German: Hello"
	inputIDs := []int{10, 11, 12, 13, 14}

	// Decoder input: "<pad> Hallo"
	decoderInputIDs := []int{0, 20}

	// Forward pass
	logits := model.Forward(inputIDs, decoderInputIDs)

	// Check logits shape
	logitsShape := logits.Shape()
	if logitsShape[0] != len(decoderInputIDs) || logitsShape[1] != config.VocabSize {
		t.Errorf("Model logits shape mismatch")
	}

	t.Logf("T5 full model forward pass successful")
	t.Logf("  Input seq len: %d", len(inputIDs))
	t.Logf("  Output seq len: %d", len(decoderInputIDs))
	t.Logf("  Logits shape: [%d, %d]", logitsShape[0], logitsShape[1])
}

// TestSpanCorruption tests T5-style span corruption.
func TestSpanCorruption(t *testing.T) {
	config := NewT5Config()
	config.SpanLength = 2
	config.CorruptRate = 0.4 // Corrupt 40% for testing

	// Input tokens
	tokens := []int{10, 20, 30, 40, 50, 60, 70, 80, 90, 100}

	rng := rand.New(rand.NewSource(42))
	result := ApplySpanCorruption(tokens, config, rng)

	// Check that corruption happened
	if len(result.SpanStarts) == 0 {
		t.Error("Expected at least one span to be corrupted")
	}

	// Check that corrupted input is shorter than original
	if len(result.CorruptedInput) >= len(tokens) {
		t.Error("Corrupted input should be shorter than original (spans replaced by sentinels)")
	}

	// Check that target contains sentinel tokens
	hasSentinel := false
	for _, id := range result.TargetOutput {
		if id >= config.SentinelStartID {
			hasSentinel = true
			break
		}
	}
	if !hasSentinel {
		t.Error("Target output should contain sentinel tokens")
	}

	// Check that target ends with EOS
	if result.TargetOutput[len(result.TargetOutput)-1] != config.EOSTokenID {
		t.Error("Target output should end with EOS token")
	}

	t.Logf("Span corruption successful:")
	t.Logf("  Original length: %d", len(tokens))
	t.Logf("  Corrupted length: %d", len(result.CorruptedInput))
	t.Logf("  Target length: %d", len(result.TargetOutput))
	t.Logf("  Num spans: %d", len(result.SpanStarts))
}

// TestSpanCorruptionExample demonstrates span corruption clearly.
func TestSpanCorruptionExample(t *testing.T) {
	config := NewT5Config()
	config.SpanLength = 2
	config.CorruptRate = 0.3

	// Simple token sequence
	tokens := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

	rng := rand.New(rand.NewSource(123))
	result := ApplySpanCorruption(tokens, config, rng)

	t.Log("=== Span Corruption Example ===")
	t.Logf("Original tokens: %v", tokens)
	t.Logf("Corrupted input: %v", result.CorruptedInput)
	t.Logf("Target output:   %v", result.TargetOutput)
	t.Logf("Span starts:     %v", result.SpanStarts)
	t.Log("")
	t.Log("Interpretation:")
	t.Log("  - Spans in original are replaced by sentinel tokens (<extra_id_N>)")
	t.Log("  - Target shows sentinel followed by original span tokens")
	t.Log("  - Model learns to predict masked spans given corrupted context")
}

// TestSeq2SeqLoss tests sequence-to-sequence loss computation.
func TestSeq2SeqLoss(t *testing.T) {
	seqLen := 5
	vocabSize := 100

	// Create dummy logits
	logits := NewTensor(seqLen, vocabSize)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < vocabSize; j++ {
			logits.Set(rand.NormFloat64(), i, j)
		}
	}

	// Target IDs
	targetIDs := []int{10, 20, 30, 40, 50}

	// Compute loss
	loss := ComputeSeq2SeqLoss(logits, targetIDs)

	if loss <= 0 {
		t.Errorf("Expected positive loss, got %f", loss)
	}

	// Loss should be roughly log(vocabSize) for random logits
	expectedLoss := 4.6 // log(100) ≈ 4.6
	if loss < expectedLoss*0.5 || loss > expectedLoss*1.5 {
		t.Logf("Warning: Loss %f far from expected %f (random logits)", loss, expectedLoss)
	}

	t.Logf("Seq2Seq loss computed: %.4f", loss)
}

// TestSeq2SeqLossPerfectPrediction tests loss with perfect predictions.
func TestSeq2SeqLossPerfectPrediction(t *testing.T) {
	seqLen := 3
	vocabSize := 10

	// Create logits favoring correct predictions
	logits := NewTensor(seqLen, vocabSize)
	targetIDs := []int{5, 7, 2}

	for i := 0; i < seqLen; i++ {
		for j := 0; j < vocabSize; j++ {
			if j == targetIDs[i] {
				logits.Set(10.0, i, j) // High score for correct token
			} else {
				logits.Set(0.0, i, j)
			}
		}
	}

	// Compute loss
	loss := ComputeSeq2SeqLoss(logits, targetIDs)

	// Loss should be near zero for perfect predictions
	if loss > 0.1 {
		t.Errorf("Expected near-zero loss for perfect predictions, got %f", loss)
	}

	t.Logf("Loss with perfect predictions: %.6f", loss)
}

// TestT5VsGPTVsBERT compares the three architectures.
func TestT5VsGPTVsBERT(t *testing.T) {
	t.Log("=== Comparing GPT vs BERT vs T5 ===\n")

	t.Log("1. ARCHITECTURE:")
	t.Log("   GPT:  Decoder-only (causal self-attention)")
	t.Log("   BERT: Encoder-only (bidirectional self-attention)")
	t.Log("   T5:   Encoder-Decoder (bidirectional encoder + causal decoder + cross-attention)\n")

	t.Log("2. ATTENTION MECHANISMS:")
	t.Log("   GPT:  Causal self-attention only")
	t.Log("   BERT: Bidirectional self-attention only")
	t.Log("   T5:   Encoder: bidirectional, Decoder: causal + cross-attention to encoder\n")

	t.Log("3. TRAINING OBJECTIVE:")
	t.Log("   GPT:  Next token prediction (autoregressive)")
	t.Log("   BERT: Masked Language Modeling (MLM) - predict individual masked tokens")
	t.Log("   T5:   Span corruption - predict masked spans in sequence\n")

	t.Log("4. USE CASES:")
	t.Log("   GPT:  Text generation, completion, few-shot learning")
	t.Log("   BERT: Classification, QA, NER, understanding tasks")
	t.Log("   T5:   Translation, summarization, QA, any seq2seq task\n")

	t.Log("5. INPUT/OUTPUT:")
	t.Log("   GPT:  Input -> Output (continuation)")
	t.Log("   BERT: Input with [MASK] -> Predict masked tokens")
	t.Log("   T5:   Input (source) -> Output (target) with cross-attention\n")

	t.Log("6. GENERATION CAPABILITY:")
	t.Log("   GPT:  Can generate autoregressively")
	t.Log("   BERT: Cannot generate (no autoregressive mechanism)")
	t.Log("   T5:   Can generate via decoder (with encoder context)\n")

	t.Log("7. POSITION ENCODING:")
	t.Log("   GPT:  Learned or RoPE")
	t.Log("   BERT: Learned absolute positions")
	t.Log("   T5:   Relative position biases (no absolute positions)\n")

	t.Log("8. KEY INNOVATION:")
	t.Log("   GPT:  Scalable autoregressive pretraining")
	t.Log("   BERT: Bidirectional context via MLM")
	t.Log("   T5:   Unified text-to-text framework + encoder-decoder architecture")
}

// TestT5EncoderDecoderWorkflow demonstrates complete workflow.
func TestT5EncoderDecoderWorkflow(t *testing.T) {
	t.Log("=== T5 Encoder-Decoder Workflow ===\n")

	config := NewT5Config()
	config.VocabSize = 100
	config.HiddenDim = 64
	config.SpanLength = 2
	config.CorruptRate = 0.2
	config.SentinelStartID = 90 // Sentinel tokens in range [90-99]

	// Step 1: Create model
	t.Log("Step 1: Create T5 model with shared encoder-decoder embeddings")
	model := NewT5ForConditionalGeneration(config)

	// Step 2: Prepare training data with span corruption
	t.Log("\nStep 2: Apply span corruption for training")
	inputTokens := []int{10, 11, 12, 13, 14, 15, 16, 17, 18, 19}
	rng := rand.New(rand.NewSource(42))
	corruption := ApplySpanCorruption(inputTokens, config, rng)

	t.Logf("  Original: %v", inputTokens)
	t.Logf("  Corrupted input: %v", corruption.CorruptedInput)
	t.Logf("  Target output: %v", corruption.TargetOutput)

	// Step 3: Forward pass
	t.Log("\nStep 3: Forward pass through encoder-decoder")
	t.Log("  a. Encoder processes corrupted input (bidirectional attention)")
	t.Log("  b. Decoder generates target (causal self-attention + cross-attention)")

	// Prepare decoder input (shift target right, start with PAD)
	decoderInputIDs := make([]int, len(corruption.TargetOutput)-1)
	decoderInputIDs[0] = config.PADTokenID
	copy(decoderInputIDs[1:], corruption.TargetOutput[:len(corruption.TargetOutput)-2])

	logits := model.Forward(corruption.CorruptedInput, decoderInputIDs)

	// Step 4: Compute loss
	t.Log("\nStep 4: Compute seq2seq loss")
	targetIDs := corruption.TargetOutput[1:] // Shift target for loss computation
	loss := ComputeSeq2SeqLoss(logits, targetIDs)
	t.Logf("  Loss: %.4f", loss)

	t.Log("\nStep 5: Training loop (conceptual)")
	t.Log("  a. Backpropagate loss through decoder and encoder")
	t.Log("  b. Update weights with optimizer")
	t.Log("  c. Model learns to predict masked spans")

	t.Log("\nKey T5 Features Demonstrated:")
	t.Log("  ✓ Encoder-decoder architecture")
	t.Log("  ✓ Span corruption objective")
	t.Log("  ✓ Cross-attention from decoder to encoder")
	t.Log("  ✓ Shared embeddings between encoder/decoder")
	t.Log("  ✓ Text-to-text framework")
}

// TestCrossAttentionConcept demonstrates cross-attention.
func TestCrossAttentionConcept(t *testing.T) {
	t.Log("=== Cross-Attention in T5 ===\n")

	t.Log("Cross-attention is the key mechanism that allows the decoder to")
	t.Log("attend to the encoder's output.\n")

	t.Log("Self-Attention (GPT, BERT):")
	t.Log("  Q, K, V all come from the same sequence")
	t.Log("  Attention(Q, K, V) where Q = K = V = input\n")

	t.Log("Cross-Attention (T5 Decoder):")
	t.Log("  Q comes from decoder, K and V come from encoder")
	t.Log("  Attention(Q_decoder, K_encoder, V_encoder)")
	t.Log("  Allows decoder to look at encoder's understanding of input\n")

	t.Log("T5 Decoder has TWO attention mechanisms:")
	t.Log("  1. Self-attention: Decoder attends to previous decoder positions (causal)")
	t.Log("  2. Cross-attention: Decoder attends to ALL encoder positions\n")

	t.Log("Example: Translation")
	t.Log("  Encoder input: 'Hello world' (English)")
	t.Log("  Encoder output: Rich representations of English tokens")
	t.Log("  Decoder generates: 'Hallo Welt' (German)")
	t.Log("  - Decoder self-attention: Each German token sees previous German tokens")
	t.Log("  - Decoder cross-attention: Each German token sees ALL English tokens")
	t.Log("  This allows decoder to align German output with English input!")
}

// BenchmarkT5EncoderForward benchmarks encoder forward pass.
func BenchmarkT5EncoderForward(b *testing.B) {
	config := NewT5Config()
	config.VocabSize = 32000
	config.HiddenDim = 768

	encoder := NewT5Encoder(config)
	inputIDs := make([]int, 512)
	for i := range inputIDs {
		inputIDs[i] = rand.Intn(config.VocabSize)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		encoder.Forward(inputIDs)
	}
}

// BenchmarkT5FullModel benchmarks full encoder-decoder forward pass.
func BenchmarkT5FullModel(b *testing.B) {
	config := NewT5Config()
	config.VocabSize = 32000
	config.HiddenDim = 768

	model := NewT5ForConditionalGeneration(config)

	inputIDs := make([]int, 256)
	for i := range inputIDs {
		inputIDs[i] = rand.Intn(config.VocabSize)
	}

	decoderInputIDs := make([]int, 128)
	for i := range decoderInputIDs {
		decoderInputIDs[i] = rand.Intn(config.VocabSize)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model.Forward(inputIDs, decoderInputIDs)
	}
}

// BenchmarkSpanCorruption benchmarks span corruption.
func BenchmarkSpanCorruption(b *testing.B) {
	config := NewT5Config()
	tokens := make([]int, 512)
	for i := range tokens {
		tokens[i] = rand.Intn(config.VocabSize)
	}

	rng := rand.New(rand.NewSource(42))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ApplySpanCorruption(tokens, config, rng)
	}
}
