package main

// ===========================================================================
// WHAT'S GOING ON HERE: BERT-Style Bidirectional Transformer
// ===========================================================================
//
// This file implements a BERT-style bidirectional transformer architecture,
// which differs from GPT-style autoregressive models in fundamental ways.
// This is an educational implementation demonstrating the core BERT concepts.
//
// INTENTION:
// Show how bidirectional attention enables models to learn richer representations
// by attending to both past and future context, and how Masked Language Modeling
// (MLM) trains these models effectively. Demonstrate key differences from GPT.
//
// GPT VS BERT: KEY ARCHITECTURAL DIFFERENCES
//
// | Aspect              | GPT (Autoregressive)     | BERT (Bidirectional)      |
// |---------------------|--------------------------|---------------------------|
// | Attention           | Causal (past only)       | Full (past + future)      |
// | Training Objective  | Next token prediction    | Masked token prediction   |
// | Use Case            | Text generation          | Understanding/encoding    |
// | Inference           | Autoregressive (slow)    | Parallel (fast)          |
// | Context             | Left-to-right            | Full bidirectional        |
//
// BIDIRECTIONAL ATTENTION:
//
// GPT uses causal masking: position i can only attend to positions ≤ i
//   Attention mask: [[1, 0, 0],    (position 0 sees only position 0)
//                    [1, 1, 0],    (position 1 sees 0,1)
//                    [1, 1, 1]]    (position 2 sees all)
//
// BERT uses full attention: position i attends to ALL positions
//   Attention mask: [[1, 1, 1],    (position 0 sees all)
//                    [1, 1, 1],    (position 1 sees all)
//                    [1, 1, 1]]    (position 2 sees all)
//
// This enables richer representations but prevents autoregressive generation.
//
// MASKED LANGUAGE MODELING (MLM):
//
// Since BERT sees full context, we can't train it with next-token prediction
// (it would just "cheat" by looking ahead). Instead, we use MLM:
//
// 1. MASKING STRATEGY (15% of tokens):
//    - 80% replaced with [MASK]: "The cat sat" → "The [MASK] sat"
//    - 10% replaced with random token: "The cat sat" → "The dog sat"
//    - 10% left unchanged: "The cat sat" → "The cat sat"
//
//    Why this strategy?
//    - 80% [MASK]: Main training signal, forces model to predict
//    - 10% random: Prevents model from only learning to copy [MASK]
//    - 10% unchanged: Prevents mismatch between training and inference
//      (real data doesn't have [MASK] tokens)
//
// 2. PREDICTION OBJECTIVE:
//    - Model predicts original tokens only for masked positions
//    - Loss computed only on masked tokens (ignore unmasked)
//    - Cross-entropy between predictions and original tokens
//
// Example training step:
//   Original:  "The quick brown fox jumps"
//   Masked:    "The [MASK] brown [MASK] jumps"
//   Predict:   "The quick brown fox jumps"
//   Loss:      Only on positions 1 and 3 (the masked tokens)
//
// SPECIAL TOKENS:
//
// BERT uses several special tokens:
// - [CLS]: Added at start, final hidden state used for classification
// - [SEP]: Separates segments (e.g., question from passage)
// - [MASK]: Placeholder for masked tokens during training
// - [PAD]: Padding for variable-length sequences
//
// Example: "What is AI? [SEP] AI is machine learning."
// Becomes:  "[CLS] What is AI ? [SEP] AI is machine learning . [SEP]"
//
// POSITION EMBEDDINGS:
//
// BERT uses learned absolute position embeddings (not RoPE):
// - Each position 0..MaxSeqLen has a learned embedding vector
// - Added to token embeddings before first layer
// - Simple but effective for sequences up to MaxSeqLen
//
// SEGMENT EMBEDDINGS:
//
// For tasks with multiple segments (e.g., question answering):
// - Segment A embedding added to first segment
// - Segment B embedding added to second segment
// - Helps model distinguish between segments
//
// Example: "[CLS] Q1 Q2 [SEP] A1 A2 [SEP]"
//          Segments:  A  A  A   A   B  B  B
//
// TRAINING VS INFERENCE:
//
// Training:
// - Random masking applied to input
// - Model predicts masked tokens
// - Loss computed only on masked positions
// - Typically trained on large corpora (Wikipedia, books)
//
// Inference (downstream tasks):
// - No masking applied
// - Use [CLS] representation for classification
// - Use token representations for token-level tasks
// - Fine-tune entire model on task-specific data
//
// DOWNSTREAM TASKS:
//
// 1. Text Classification:
//    - Input: "[CLS] text [SEP]"
//    - Output: Classification from [CLS] hidden state
//    - Example: Sentiment analysis, topic classification
//
// 2. Token Classification:
//    - Input: "[CLS] tokens [SEP]"
//    - Output: Label for each token
//    - Example: Named entity recognition, POS tagging
//
// 3. Question Answering:
//    - Input: "[CLS] question [SEP] passage [SEP]"
//    - Output: Start/end positions of answer span
//    - Example: SQuAD dataset
//
// 4. Sentence Pair Tasks:
//    - Input: "[CLS] sentence1 [SEP] sentence2 [SEP]"
//    - Output: Relationship classification
//    - Example: Natural language inference, paraphrase detection
//
// ADVANTAGES OVER GPT:
//
// 1. Richer representations: Sees full context, not just past
// 2. Better for understanding tasks: Classification, QA, NER
// 3. Parallel inference: No autoregressive bottleneck
// 4. More efficient fine-tuning: Already trained on bidirectional context
//
// DISADVANTAGES:
//
// 1. Cannot generate text autoregressively
// 2. [MASK] token mismatch between training and inference
// 3. Only 15% of tokens provide training signal per example
// 4. Requires more sophisticated masking strategies
//
// ===========================================================================
// RECOMMENDED READING:
//
// BERT Papers:
// - "BERT: Pre-training of Deep Bidirectional Transformers" by Devlin et al. (2018)
//   https://arxiv.org/abs/1810.04805
//   The original BERT paper
//
// - "RoBERTa: A Robustly Optimized BERT Pretraining Approach" by Liu et al. (2019)
//   https://arxiv.org/abs/1907.11692
//   Improved BERT training (no NSP, dynamic masking, larger batches)
//
// - "ALBERT: A Lite BERT for Self-supervised Learning" by Lan et al. (2019)
//   https://arxiv.org/abs/1909.11942
//   Parameter sharing and factorized embeddings
//
// Masking Strategies:
// - "SpanBERT: Improving Pre-training by Representing and Predicting Spans"
//   by Joshi et al. (2019) - mask contiguous spans instead of random tokens
//
// - "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators"
//   by Clark et al. (2020) - replaced token detection instead of MLM
//
// Bidirectional Models:
// - "Attention Is All You Need" by Vaswani et al. (2017)
//   - Original transformer (encoder-decoder, both bidirectional encoder)
//
// - "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
//   by Raffel et al. (2019) - T5 model (encoder-decoder)
//
// ===========================================================================

import (
	"fmt"
	"math"
	"math/rand"
)

// BERTConfig holds configuration for a BERT-style model.
type BERTConfig struct {
	VocabSize      int     // Size of vocabulary
	MaxSeqLen      int     // Maximum sequence length
	HiddenDim      int     // Hidden dimension (d_model)
	NumLayers      int     // Number of transformer layers
	NumHeads       int     // Number of attention heads
	IntermediateDim int    // Feed-forward intermediate dimension
	DropoutProb    float64 // Dropout probability

	// MLM-specific
	MaskTokenID    int     // Token ID for [MASK]
	CLSTokenID     int     // Token ID for [CLS]
	SEPTokenID     int     // Token ID for [SEP]
	PADTokenID     int     // Token ID for [PAD]
	MaskProb       float64 // Probability of masking a token (default 0.15)
	RandomProb     float64 // Prob of replacing with random token (default 0.10 of masked)
	KeepProb       float64 // Prob of keeping original token (default 0.10 of masked)

	// Segment embeddings
	UseSegmentEmbeddings bool // Whether to use segment embeddings
	NumSegments          int  // Number of segments (usually 2)
}

// NewBERTConfig creates a default BERT configuration (similar to BERT-Base).
func NewBERTConfig() *BERTConfig {
	return &BERTConfig{
		VocabSize:            30522, // WordPiece vocab size
		MaxSeqLen:            512,   // Maximum sequence length
		HiddenDim:            768,   // d_model
		NumLayers:            12,    // Transformer layers
		NumHeads:             12,    // Attention heads
		IntermediateDim:      3072,  // FFN intermediate (4 * hidden)
		DropoutProb:          0.1,   // Dropout
		MaskTokenID:          103,   // [MASK] token
		CLSTokenID:           101,   // [CLS] token
		SEPTokenID:           102,   // [SEP] token
		PADTokenID:           0,     // [PAD] token
		MaskProb:             0.15,  // Mask 15% of tokens
		RandomProb:           0.10,  // 10% of masked → random
		KeepProb:             0.10,  // 10% of masked → keep
		UseSegmentEmbeddings: true,  // Use segment embeddings
		NumSegments:          2,     // Two segments (A and B)
	}
}

// MaskingResult holds the result of applying MLM masking to input.
type MaskingResult struct {
	// MaskedInputIDs are the input IDs after applying masking
	MaskedInputIDs []int

	// OriginalInputIDs are the original input IDs before masking
	OriginalInputIDs []int

	// MaskedPositions are the positions that were masked
	MaskedPositions []int

	// Labels are the target labels (-100 for non-masked, original ID for masked)
	// -100 is used to indicate positions that should be ignored in loss computation
	Labels []int
}

// ApplyMLMMasking applies Masked Language Modeling masking strategy to input IDs.
//
// For each token (except special tokens):
// - With probability maskProb (default 15%):
//   - 80% of the time: replace with [MASK]
//   - 10% of the time: replace with random token
//   - 10% of the time: keep original
//
// Returns MaskingResult with masked inputs, positions, and labels.
func ApplyMLMMasking(inputIDs []int, config *BERTConfig, rng *rand.Rand) *MaskingResult {
	seqLen := len(inputIDs)
	result := &MaskingResult{
		MaskedInputIDs:   make([]int, seqLen),
		OriginalInputIDs: make([]int, seqLen),
		MaskedPositions:  make([]int, 0),
		Labels:           make([]int, seqLen),
	}

	// Copy original IDs
	copy(result.MaskedInputIDs, inputIDs)
	copy(result.OriginalInputIDs, inputIDs)

	// Initialize all labels to -100 (ignore in loss)
	for i := range result.Labels {
		result.Labels[i] = -100
	}

	// Apply masking
	for i, tokenID := range inputIDs {
		// Skip special tokens
		if tokenID == config.CLSTokenID || tokenID == config.SEPTokenID || tokenID == config.PADTokenID {
			continue
		}

		// Randomly decide whether to mask this token
		if rng.Float64() < config.MaskProb {
			result.MaskedPositions = append(result.MaskedPositions, i)
			result.Labels[i] = tokenID // Set label to original token

			// Decide how to mask
			maskDecision := rng.Float64()

			if maskDecision < 0.8 {
				// 80%: Replace with [MASK]
				result.MaskedInputIDs[i] = config.MaskTokenID
			} else if maskDecision < 0.9 {
				// 10%: Replace with random token (not special token)
				// Sample from range [vocab - special_tokens]
				randomToken := rng.Intn(config.VocabSize - 4) + 4 // Skip first 4 special tokens
				result.MaskedInputIDs[i] = randomToken
			}
			// Else 10%: Keep original (already copied)
		}
	}

	return result
}

// CreateAttentionMask creates a bidirectional attention mask (all 1s for valid tokens).
//
// Unlike GPT's causal mask, BERT allows each position to attend to all positions.
// The mask only prevents attention to padding tokens.
//
// Returns: [seqLen, seqLen] attention mask (1 for valid, 0 for padding)
func CreateAttentionMask(inputIDs []int, padTokenID int) *Tensor {
	seqLen := len(inputIDs)
	mask := NewTensor(seqLen, seqLen)

	// Create mask: 1 for valid positions, 0 for padding
	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			// Attention is allowed if target position j is not padding
			if inputIDs[j] != padTokenID {
				mask.Set(1.0, i, j)
			} else {
				mask.Set(0.0, i, j)
			}
		}
	}

	return mask
}

// ComputeMLMLoss computes Masked Language Modeling loss.
//
// Only computes loss on masked positions (where labels != -100).
// Uses cross-entropy loss between predictions and labels.
//
// Args:
//   predictions: [seqLen, vocabSize] logits from model
//   labels:      [seqLen] target labels (-100 for ignore, tokenID for predict)
//
// Returns: scalar loss value
func ComputeMLMLoss(predictions *Tensor, labels []int) float64 {
	shape := predictions.Shape()
	seqLen := shape[0]
	vocabSize := shape[1]

	totalLoss := 0.0
	numMasked := 0

	for i := 0; i < seqLen; i++ {
		if labels[i] == -100 {
			continue // Skip non-masked positions
		}

		// Get logits for this position
		logits := make([]float64, vocabSize)
		for j := 0; j < vocabSize; j++ {
			logits[j] = predictions.At(i, j)
		}

		// Compute log-softmax
		maxLogit := logits[0]
		for _, v := range logits {
			if v > maxLogit {
				maxLogit = v
			}
		}

		sumExp := 0.0
		for j := 0; j < vocabSize; j++ {
			logits[j] -= maxLogit
			sumExp += math.Exp(logits[j])
		}

		logSumExp := math.Log(sumExp)

		// Cross-entropy: -log(p(correct_token))
		correctToken := labels[i]
		loss := -(logits[correctToken] - logSumExp)

		totalLoss += loss
		numMasked++
	}

	// Average loss over masked positions
	if numMasked > 0 {
		return totalLoss / float64(numMasked)
	}
	return 0.0
}

// BERTForMLM represents a BERT model for Masked Language Modeling.
//
// Architecture:
// - Token embeddings (vocab_size → hidden_dim)
// - Position embeddings (max_seq_len → hidden_dim)
// - Optional segment embeddings (num_segments → hidden_dim)
// - N transformer layers (bidirectional attention)
// - MLM prediction head (hidden_dim → vocab_size)
type BERTForMLM struct {
	Config *BERTConfig

	// Embeddings
	TokenEmbeddings    *Tensor // [vocabSize, hiddenDim]
	PositionEmbeddings *Tensor // [maxSeqLen, hiddenDim]
	SegmentEmbeddings  *Tensor // [numSegments, hiddenDim] (optional)

	// Transformer layers would go here (reuse TransformerBlock from main transformer)
	// For educational purposes, we'll implement the key concepts without full training

	// MLM head
	MLMHead *Tensor // [hiddenDim, vocabSize]
}

// NewBERTForMLM creates a new BERT model for Masked Language Modeling.
func NewBERTForMLM(config *BERTConfig) *BERTForMLM {
	bert := &BERTForMLM{
		Config: config,
	}

	// Initialize embeddings
	bert.TokenEmbeddings = NewTensor(config.VocabSize, config.HiddenDim)
	bert.PositionEmbeddings = NewTensor(config.MaxSeqLen, config.HiddenDim)

	if config.UseSegmentEmbeddings {
		bert.SegmentEmbeddings = NewTensor(config.NumSegments, config.HiddenDim)
	}

	// Initialize MLM head
	bert.MLMHead = NewTensor(config.HiddenDim, config.VocabSize)

	// Initialize with small random values (would use proper initialization in practice)
	bert.initializeWeights()

	return bert
}

// initializeWeights initializes model weights with small random values.
func (bert *BERTForMLM) initializeWeights() {
	// Token embeddings: normal distribution with std = 1/sqrt(hidden_dim)
	std := 1.0 / math.Sqrt(float64(bert.Config.HiddenDim))

	// Initialize token embeddings
	shape := bert.TokenEmbeddings.Shape()
	for i := 0; i < shape[0]; i++ {
		for j := 0; j < shape[1]; j++ {
			bert.TokenEmbeddings.Set(rand.NormFloat64()*std, i, j)
		}
	}

	// Initialize position embeddings
	shape = bert.PositionEmbeddings.Shape()
	for i := 0; i < shape[0]; i++ {
		for j := 0; j < shape[1]; j++ {
			bert.PositionEmbeddings.Set(rand.NormFloat64()*std, i, j)
		}
	}

	// Initialize segment embeddings
	if bert.SegmentEmbeddings != nil {
		shape = bert.SegmentEmbeddings.Shape()
		for i := 0; i < shape[0]; i++ {
			for j := 0; j < shape[1]; j++ {
				bert.SegmentEmbeddings.Set(rand.NormFloat64()*std, i, j)
			}
		}
	}

	// Initialize MLM head
	shape = bert.MLMHead.Shape()
	for i := 0; i < shape[0]; i++ {
		for j := 0; j < shape[1]; j++ {
			bert.MLMHead.Set(rand.NormFloat64()*std, i, j)
		}
	}
}

// ===========================================================================
// EXAMPLE USAGE
// ===========================================================================
//
// // Create BERT configuration
// config := NewBERTConfig()
// config.VocabSize = 10000
// config.MaxSeqLen = 128
// config.HiddenDim = 256
//
// // Create BERT model
// model := NewBERTForMLM(config)
//
// // Prepare input (e.g., "The cat sat on the mat")
// inputIDs := []int{101, 150, 200, 250, 300, 350, 400, 102} // [CLS] + tokens + [SEP]
//
// // Apply MLM masking
// rng := rand.New(rand.NewSource(42))
// masked := ApplyMLMMasking(inputIDs, config, rng)
//
// fmt.Printf("Original: %v\n", masked.OriginalInputIDs)
// fmt.Printf("Masked:   %v\n", masked.MaskedInputIDs)
// fmt.Printf("Positions: %v\n", masked.MaskedPositions)
// fmt.Printf("Labels:    %v\n", masked.Labels)
//
// // Create attention mask (bidirectional, no causal masking)
// attentionMask := CreateAttentionMask(inputIDs, config.PADTokenID)
//
// // Forward pass (simplified, without full transformer)
// // embeddings := model.GetEmbeddings(masked.MaskedInputIDs, segmentIDs)
// // hidden := model.TransformerLayers(embeddings, attentionMask)
// // logits := model.MLMHead(hidden)
// // loss := ComputeMLMLoss(logits, masked.Labels)
//
// // Downstream task: Text classification
// // - Use [CLS] token representation
// // - Add classification head
// // - Fine-tune on task data
//
// ===========================================================================

// Example: Comparing BERT and GPT masking
func ExampleCompareAttentionMasks() {
	fmt.Println("=== GPT vs BERT Attention Masking ===")

	// Example sequence: "[CLS] The cat sat [SEP]"
	seqLen := 5

	// GPT: Causal mask (lower triangular)
	fmt.Println("GPT (Causal) Attention Mask:")
	fmt.Println("Position i can attend to positions j where j <= i")
	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			if j <= i {
				fmt.Print("1 ")
			} else {
				fmt.Print("0 ")
			}
		}
		fmt.Println()
	}

	fmt.Println("\nBERT (Bidirectional) Attention Mask:")
	fmt.Println("Position i can attend to all positions j")
	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			fmt.Print("1 ")
		}
		fmt.Println()
	}

	fmt.Println("\nKey difference:")
	fmt.Println("- GPT sees only past context (autoregressive generation)")
	fmt.Println("- BERT sees full context (better understanding, no generation)")
}
