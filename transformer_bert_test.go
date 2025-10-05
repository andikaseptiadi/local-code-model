package main

import (
	"math/rand"
	"testing"
)

// TestBERTConfig tests BERT configuration creation and defaults.
func TestBERTConfig(t *testing.T) {
	config := NewBERTConfig()

	// Check default values match BERT-Base
	if config.VocabSize != 30522 {
		t.Errorf("Expected VocabSize 30522, got %d", config.VocabSize)
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

	if config.MaskProb != 0.15 {
		t.Errorf("Expected MaskProb 0.15, got %f", config.MaskProb)
	}

	if !config.UseSegmentEmbeddings {
		t.Error("Expected UseSegmentEmbeddings to be true")
	}

	t.Logf("BERT config created successfully with BERT-Base defaults")
}

// TestApplyMLMMasking tests the MLM masking strategy.
func TestApplyMLMMasking(t *testing.T) {
	config := NewBERTConfig()
	config.VocabSize = 1000
	config.MaskProb = 0.5 // High mask probability for testing

	// Input: [CLS] token1 token2 token3 token4 [SEP]
	inputIDs := []int{config.CLSTokenID, 150, 200, 250, 300, config.SEPTokenID}

	rng := rand.New(rand.NewSource(42))
	result := ApplyMLMMasking(inputIDs, config, rng)

	// Check that special tokens are not masked
	if result.MaskedInputIDs[0] != config.CLSTokenID {
		t.Error("[CLS] token should not be masked")
	}
	if result.MaskedInputIDs[5] != config.SEPTokenID {
		t.Error("[SEP] token should not be masked")
	}

	// Check that some non-special tokens are masked
	numMasked := 0
	for i := 1; i < 5; i++ { // Check middle tokens
		if result.MaskedInputIDs[i] != inputIDs[i] {
			numMasked++
		}
	}

	if numMasked == 0 {
		t.Error("Expected at least some tokens to be masked with MaskProb=0.5")
	}

	// Check that labels are set correctly
	labelCount := 0
	for _, label := range result.Labels {
		if label != -100 {
			labelCount++
		}
	}

	if labelCount != len(result.MaskedPositions) {
		t.Errorf("Expected %d non-ignore labels, got %d",
			len(result.MaskedPositions), labelCount)
	}

	t.Logf("MLM masking applied: %d/%d tokens masked",
		len(result.MaskedPositions), len(inputIDs))
	t.Logf("Original: %v", result.OriginalInputIDs)
	t.Logf("Masked:   %v", result.MaskedInputIDs)
	t.Logf("Positions: %v", result.MaskedPositions)
}

// TestMLMMaskingStrategies tests the 80/10/10 masking strategy.
func TestMLMMaskingStrategies(t *testing.T) {
	config := NewBERTConfig()
	config.VocabSize = 1000
	config.MaskProb = 1.0 // Mask all non-special tokens

	// Create input with only non-special tokens
	inputIDs := []int{config.CLSTokenID, 150, 200, 250, 300, 350, config.SEPTokenID}

	// Run masking multiple times to check strategy distribution
	rng := rand.New(rand.NewSource(42))
	numTrials := 1000
	maskTokenCount := 0
	randomTokenCount := 0
	keepTokenCount := 0

	for i := 0; i < numTrials; i++ {
		result := ApplyMLMMasking(inputIDs, config, rng)

		// Check middle tokens (skip [CLS] and [SEP])
		for j := 1; j < 6; j++ {
			if result.MaskedInputIDs[j] == config.MaskTokenID {
				maskTokenCount++
			} else if result.MaskedInputIDs[j] == inputIDs[j] {
				keepTokenCount++
			} else {
				randomTokenCount++
			}
		}
	}

	totalMasked := maskTokenCount + randomTokenCount + keepTokenCount
	maskRatio := float64(maskTokenCount) / float64(totalMasked)
	randomRatio := float64(randomTokenCount) / float64(totalMasked)
	keepRatio := float64(keepTokenCount) / float64(totalMasked)

	t.Logf("Masking strategy distribution over %d trials:", numTrials)
	t.Logf("  [MASK] token: %.1f%% (expected ~80%%)", maskRatio*100)
	t.Logf("  Random token: %.1f%% (expected ~10%%)", randomRatio*100)
	t.Logf("  Keep original: %.1f%% (expected ~10%%)", keepRatio*100)

	// Check that ratios are roughly correct (with some tolerance)
	if maskRatio < 0.75 || maskRatio > 0.85 {
		t.Errorf("Mask ratio %.2f outside expected range [0.75, 0.85]", maskRatio)
	}
	if randomRatio < 0.05 || randomRatio > 0.15 {
		t.Errorf("Random ratio %.2f outside expected range [0.05, 0.15]", randomRatio)
	}
	if keepRatio < 0.05 || keepRatio > 0.15 {
		t.Errorf("Keep ratio %.2f outside expected range [0.05, 0.15]", keepRatio)
	}
}

// TestCreateAttentionMask tests bidirectional attention mask creation.
func TestCreateAttentionMask(t *testing.T) {
	config := NewBERTConfig()

	// Input with padding: [CLS] token1 token2 [PAD] [PAD]
	inputIDs := []int{config.CLSTokenID, 150, 200, config.PADTokenID, config.PADTokenID}

	mask := CreateAttentionMask(inputIDs, config.PADTokenID)

	seqLen := len(inputIDs)
	shape := mask.Shape()
	if shape[0] != seqLen || shape[1] != seqLen {
		t.Errorf("Expected mask shape [%d, %d], got [%d, %d]",
			seqLen, seqLen, shape[0], shape[1])
	}

	// Check that attention is allowed to all non-padding positions
	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			expected := 0.0
			if inputIDs[j] != config.PADTokenID {
				expected = 1.0
			}

			actual := mask.At(i, j)
			if actual != expected {
				t.Errorf("Mask[%d,%d]: expected %.0f, got %.0f", i, j, expected, actual)
			}
		}
	}

	t.Log("Bidirectional attention mask created successfully")
	t.Log("All positions can attend to non-padding positions")
}

// TestBidirectionalVsCausalMask compares BERT and GPT attention patterns.
func TestBidirectionalVsCausalMask(t *testing.T) {
	config := NewBERTConfig()
	seqLen := 4

	// BERT mask: all 1s (bidirectional)
	inputIDs := []int{config.CLSTokenID, 150, 200, config.SEPTokenID}
	bertMask := CreateAttentionMask(inputIDs, config.PADTokenID)

	// Check that BERT mask is full (all positions attend to all)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			if bertMask.At(i, j) != 1.0 {
				t.Errorf("BERT mask[%d,%d] should be 1.0 (bidirectional)", i, j)
			}
		}
	}

	// GPT mask would be lower triangular (causal)
	// Here we just demonstrate the difference conceptually
	t.Log("=== Attention Mask Comparison ===")
	t.Log("BERT (Bidirectional):")
	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			t.Logf("%.0f ", bertMask.At(i, j))
		}
		t.Log()
	}

	t.Log("\nGPT (Causal) would be:")
	t.Log("1 0 0 0")
	t.Log("1 1 0 0")
	t.Log("1 1 1 0")
	t.Log("1 1 1 1")
	t.Log("\nKey: BERT sees full context, GPT sees only past")
}

// TestComputeMLMLoss tests MLM loss computation.
func TestComputeMLMLoss(t *testing.T) {
	seqLen := 5
	vocabSize := 100

	// Create dummy predictions (logits)
	predictions := NewTensor(seqLen, vocabSize)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < vocabSize; j++ {
			predictions.Set(rand.NormFloat64(), i, j)
		}
	}

	// Labels: -100 for ignore, token IDs for masked positions
	labels := []int{-100, 10, -100, 25, -100}

	loss := ComputeMLMLoss(predictions, labels)

	if loss <= 0 {
		t.Errorf("Expected positive loss, got %f", loss)
	}

	// Check that loss is only computed on non-ignore positions
	// With random logits, expected loss ≈ log(vocabSize)
	expectedLoss := 4.6 // log(100) ≈ 4.6
	if loss < expectedLoss*0.5 || loss > expectedLoss*1.5 {
		t.Logf("Warning: Loss %f far from expected %f (random logits)", loss, expectedLoss)
	}

	t.Logf("MLM loss computed: %.4f (on 2 masked positions)", loss)
}

// TestMLMLossIgnoresNonMasked tests that loss ignores non-masked positions.
func TestMLMLossIgnoresNonMasked(t *testing.T) {
	seqLen := 3
	vocabSize := 10

	predictions := NewTensor(seqLen, vocabSize)
	// Set predictions to favor token 5 at all positions
	for i := 0; i < seqLen; i++ {
		for j := 0; j < vocabSize; j++ {
			predictions.Set(0.0, i, j)
		}
		predictions.Set(10.0, i, 5) // High score for token 5
	}

	// Only mask position 1 with label 5
	labels1 := []int{-100, 5, -100}
	loss1 := ComputeMLMLoss(predictions, labels1)

	// Loss should be near zero since prediction matches label
	if loss1 > 0.1 {
		t.Errorf("Expected near-zero loss, got %f", loss1)
	}

	// Now mask position 1 with different label
	labels2 := []int{-100, 3, -100}
	loss2 := ComputeMLMLoss(predictions, labels2)

	// Loss should be higher since prediction doesn't match label
	if loss2 <= loss1 {
		t.Errorf("Expected higher loss for wrong prediction: %f <= %f", loss2, loss1)
	}

	// Check that labels at ignored positions don't affect loss
	labels3 := []int{3, 5, 8} // Different labels at ignored positions
	labels3[1] = 5             // Same label at masked position
	loss3 := ComputeMLMLoss(predictions, labels3)

	// Loss at position 1 should be same as loss1
	// (other positions should be ignored despite having labels)
	tolerance := 0.01
	if loss3 < loss1-tolerance || loss3 > loss1+tolerance {
		t.Errorf("Loss should ignore non-masked positions: got %f, expected ~%f", loss3, loss1)
	}

	t.Logf("Loss correctly ignores non-masked positions")
}

// TestBERTForMLMCreation tests BERT model creation.
func TestBERTForMLMCreation(t *testing.T) {
	config := NewBERTConfig()
	config.VocabSize = 1000
	config.MaxSeqLen = 128
	config.HiddenDim = 256

	model := NewBERTForMLM(config)

	// Check embedding dimensions
	if model.TokenEmbeddings.Shape()[0] != config.VocabSize ||
		model.TokenEmbeddings.Shape()[1] != config.HiddenDim {
		t.Errorf("Token embeddings shape mismatch: got [%d,%d], expected [%d,%d]",
			model.TokenEmbeddings.Shape()[0], model.TokenEmbeddings.Shape()[1],
			config.VocabSize, config.HiddenDim)
	}

	if model.PositionEmbeddings.Shape()[0] != config.MaxSeqLen ||
		model.PositionEmbeddings.Shape()[1] != config.HiddenDim {
		t.Errorf("Position embeddings shape mismatch")
	}

	if model.SegmentEmbeddings == nil {
		t.Error("Segment embeddings should be initialized")
	}

	if model.MLMHead.Shape()[0] != config.HiddenDim ||
		model.MLMHead.Shape()[1] != config.VocabSize {
		t.Errorf("MLM head shape mismatch")
	}

	// Check that weights are initialized (non-zero)
	nonZero := 0
	shape := model.TokenEmbeddings.Shape()
	checkLimit := 100
	if shape[0]*shape[1] < checkLimit {
		checkLimit = shape[0] * shape[1]
	}
	for idx := 0; idx < checkLimit; idx++ {
		i := idx / shape[1]
		j := idx % shape[1]
		if model.TokenEmbeddings.At(i, j) != 0.0 {
			nonZero++
		}
	}

	if nonZero == 0 {
		t.Error("Token embeddings should be initialized with non-zero values")
	}

	t.Logf("BERT model created with:")
	t.Logf("  Vocab size: %d", config.VocabSize)
	t.Logf("  Hidden dim: %d", config.HiddenDim)
	t.Logf("  Max seq len: %d", config.MaxSeqLen)
}

// TestBERTVsGPTArchitecture documents key architectural differences.
func TestBERTVsGPTArchitecture(t *testing.T) {
	t.Log("=== BERT vs GPT: Key Architectural Differences ===\n")

	t.Log("1. ATTENTION PATTERN:")
	t.Log("   GPT (Causal):         Token i attends to positions <= i")
	t.Log("   BERT (Bidirectional): Token i attends to all positions\n")

	t.Log("2. TRAINING OBJECTIVE:")
	t.Log("   GPT:  Predict next token (autoregressive)")
	t.Log("   BERT: Predict masked tokens (MLM)\n")

	t.Log("3. TRAINING SIGNAL:")
	t.Log("   GPT:  All tokens provide training signal")
	t.Log("   BERT: Only masked tokens (~15%) provide signal\n")

	t.Log("4. USE CASES:")
	t.Log("   GPT:  Text generation, completion")
	t.Log("   BERT: Classification, QA, NER, understanding\n")

	t.Log("5. INFERENCE:")
	t.Log("   GPT:  Autoregressive (sequential, slow)")
	t.Log("   BERT: Parallel (all positions at once, fast)\n")

	t.Log("6. GENERATION:")
	t.Log("   GPT:  Can generate text autoregressively")
	t.Log("   BERT: Cannot generate (bidirectional attention)\n")

	t.Log("7. CONTEXT:")
	t.Log("   GPT:  Left-to-right only")
	t.Log("   BERT: Full bidirectional context\n")

	t.Log("8. SPECIAL TOKENS:")
	t.Log("   GPT:  Minimal (mostly for control)")
	t.Log("   BERT: [CLS], [SEP], [MASK], [PAD]\n")

	t.Log("9. EMBEDDINGS:")
	t.Log("   GPT:  Token + position (often RoPE)")
	t.Log("   BERT: Token + position + segment\n")

	t.Log("10. TYPICAL ARCHITECTURE:")
	t.Log("    GPT:  Decoder-only (causal)")
	t.Log("    BERT: Encoder-only (bidirectional)")
}

// TestMLMTrainingWorkflow demonstrates a complete MLM training step.
func TestMLMTrainingWorkflow(t *testing.T) {
	t.Log("=== Demonstrating Complete MLM Training Step ===\n")

	config := NewBERTConfig()
	config.VocabSize = 100
	config.MaxSeqLen = 10
	config.HiddenDim = 64
	config.MaskProb = 0.3

	// Step 1: Prepare input
	t.Log("Step 1: Prepare input sequence")
	inputIDs := []int{config.CLSTokenID, 10, 20, 30, 40, 50, config.SEPTokenID}
	t.Logf("  Original: %v\n", inputIDs)

	// Step 2: Apply MLM masking
	t.Log("Step 2: Apply MLM masking (15% of tokens)")
	rng := rand.New(rand.NewSource(42))
	masked := ApplyMLMMasking(inputIDs, config, rng)
	t.Logf("  Masked:   %v", masked.MaskedInputIDs)
	t.Logf("  Positions: %v", masked.MaskedPositions)
	t.Logf("  Labels:    %v\n", masked.Labels)

	// Step 3: Create attention mask
	t.Log("Step 3: Create bidirectional attention mask")
	attentionMask := CreateAttentionMask(masked.MaskedInputIDs, config.PADTokenID)
	t.Logf("  Shape: [%d, %d] (all 1s for bidirectional)\n",
		attentionMask.Shape()[0], attentionMask.Shape()[1])

	// Step 4: Forward pass (conceptual)
	t.Log("Step 4: Forward pass through BERT")
	t.Log("  a. Get token embeddings")
	t.Log("  b. Add position embeddings")
	t.Log("  c. Add segment embeddings (if using)")
	t.Log("  d. Pass through N transformer layers")
	t.Log("  e. Apply MLM prediction head\n")

	// Step 5: Compute loss (simulated)
	t.Log("Step 5: Compute MLM loss")
	seqLen := len(inputIDs)
	predictions := NewTensor(seqLen, config.VocabSize)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < config.VocabSize; j++ {
			predictions.Set(rand.NormFloat64(), i, j)
		}
	}
	loss := ComputeMLMLoss(predictions, masked.Labels)
	t.Logf("  Loss: %.4f (computed only on masked positions)\n", loss)

	// Step 6: Backward pass (conceptual)
	t.Log("Step 6: Backward pass and optimization")
	t.Log("  a. Compute gradients via backpropagation")
	t.Log("  b. Update model parameters (embeddings, layers, head)")
	t.Log("  c. Loss decreases over time\n")

	t.Log("=== Training complete for one batch ===")
}

// BenchmarkMLMMasking benchmarks MLM masking application.
func BenchmarkMLMMasking(b *testing.B) {
	config := NewBERTConfig()
	config.VocabSize = 30000

	inputIDs := make([]int, 512)
	for i := range inputIDs {
		inputIDs[i] = rand.Intn(config.VocabSize)
	}
	inputIDs[0] = config.CLSTokenID
	inputIDs[511] = config.SEPTokenID

	rng := rand.New(rand.NewSource(42))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ApplyMLMMasking(inputIDs, config, rng)
	}
}

// BenchmarkMLMLoss benchmarks MLM loss computation.
func BenchmarkMLMLoss(b *testing.B) {
	seqLen := 512
	vocabSize := 30000

	predictions := NewTensor(seqLen, vocabSize)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < vocabSize; j++ {
			predictions.Set(rand.NormFloat64(), i, j)
		}
	}

	labels := make([]int, seqLen)
	for i := range labels {
		if i%5 == 0 { // Mask 20% of tokens
			labels[i] = rand.Intn(vocabSize)
		} else {
			labels[i] = -100
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ComputeMLMLoss(predictions, labels)
	}
}
