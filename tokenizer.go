package main

import (
	"bufio"
	"encoding/hex"
	"fmt"
	"os"
	"sort"
	"strings"
	"unicode/utf8"
)

// RECOMMENDED READING:
//
// Tokenization:
// - "Neural Machine Translation of Rare Words with Subword Units" (BPE paper)
//   Sennrich, Haddow, Birch (2016)
//   https://arxiv.org/abs/1508.07909
//
// - "SentencePiece: A simple and language independent approach to subword"
//   Kudo, Richardson (2018)
//   https://arxiv.org/abs/1808.06226

// TokenizerInterface defines the common interface for all tokenizers.
// This allows both BPE and character-level tokenizers to be used interchangeably.
type TokenizerInterface interface {
	Encode(text string) []int
	Decode(ids []int) string
	VocabSize() int
	Save(filename string) error
	Load(filename string) error
}

// Tokenizer implements Byte-Pair Encoding (BPE) for text tokenization.
//
// BPE is a data compression algorithm adapted for NLP:
// 1. Start with a vocabulary of individual characters (or bytes)
// 2. Iteratively merge the most frequent pair of symbols
// 3. Build a vocabulary of subword units
//
// Benefits:
// - No unknown tokens (can represent any text)
// - Efficient encoding of common words
// - Handles rare words via subword composition
type Tokenizer struct {
	vocab       map[string]int // token -> ID
	vocabInv    map[int]string // ID -> token
	merges      []pair         // ordered list of merge rules
	specialToks map[string]int // special tokens (PAD, UNK, etc.)
}

// pair represents a bigram for BPE merging.
type pair struct {
	first  string
	second string
}

func (p pair) String() string {
	return p.first + " " + p.second
}

// Special token constants
const (
	PadToken = "<|pad|>"
	UnkToken = "<|unk|>"
	EosToken = "<|endoftext|>"
)

// NewTokenizer creates a new tokenizer with special tokens.
func NewTokenizer() *Tokenizer {
	specialToks := map[string]int{
		PadToken: 0,
		UnkToken: 1,
		EosToken: 2,
	}

	vocab := make(map[string]int)
	vocabInv := make(map[int]string)

	// Add special tokens to vocab
	for tok, id := range specialToks {
		vocab[tok] = id
		vocabInv[id] = tok
	}

	return &Tokenizer{
		vocab:       vocab,
		vocabInv:    vocabInv,
		merges:      []pair{},
		specialToks: specialToks,
	}
}

// Train builds a BPE vocabulary from a corpus.
//
// Algorithm:
// 1. Initialize vocabulary with individual bytes/characters
// 2. Count all adjacent pairs
// 3. Merge most frequent pair, add to vocabulary
// 4. Repeat until target vocabulary size reached
func (t *Tokenizer) Train(corpus []string, targetVocabSize int) error {
	if targetVocabSize <= len(t.specialToks) {
		return fmt.Errorf("tokenizer: target vocab size must be > %d (special tokens)", len(t.specialToks))
	}

	// Start with byte-level vocabulary (256 bytes)
	currentVocabSize := len(t.specialToks)
	for i := 0; i < 256; i++ {
		token := string(rune(i))
		if _, exists := t.vocab[token]; !exists {
			t.vocab[token] = currentVocabSize
			t.vocabInv[currentVocabSize] = token
			currentVocabSize++
		}
	}

	// Tokenize corpus into bytes
	words := make([][]string, 0)
	for _, text := range corpus {
		// Convert to byte tokens
		tokens := make([]string, 0)
		for _, b := range []byte(text) {
			tokens = append(tokens, string(rune(b)))
		}
		if len(tokens) > 0 {
			words = append(words, tokens)
		}
	}

	// Iteratively merge most frequent pairs
	for currentVocabSize < targetVocabSize {
		// Count all pairs
		pairCounts := make(map[pair]int)
		for _, word := range words {
			for i := 0; i < len(word)-1; i++ {
				p := pair{word[i], word[i+1]}
				pairCounts[p]++
			}
		}

		if len(pairCounts) == 0 {
			break // No more pairs to merge
		}

		// Find most frequent pair
		var bestPair pair
		maxCount := 0
		for p, count := range pairCounts {
			if count > maxCount {
				maxCount = count
				bestPair = p
			}
		}

		// Add merged token to vocabulary
		mergedToken := bestPair.first + bestPair.second
		t.vocab[mergedToken] = currentVocabSize
		t.vocabInv[currentVocabSize] = mergedToken
		t.merges = append(t.merges, bestPair)
		currentVocabSize++

		// Apply merge to all words
		for i, word := range words {
			words[i] = t.applyMerge(word, bestPair)
		}
	}

	return nil
}

// applyMerge applies a merge rule to a word.
func (t *Tokenizer) applyMerge(word []string, merge pair) []string {
	if len(word) < 2 {
		return word
	}

	merged := make([]string, 0, len(word))
	i := 0
	for i < len(word) {
		if i < len(word)-1 && word[i] == merge.first && word[i+1] == merge.second {
			merged = append(merged, merge.first+merge.second)
			i += 2
		} else {
			merged = append(merged, word[i])
			i++
		}
	}

	return merged
}

// Encode converts text to token IDs.
func (t *Tokenizer) Encode(text string) []int {
	// Convert text to byte tokens
	tokens := make([]string, 0)
	for _, b := range []byte(text) {
		tokens = append(tokens, string(rune(b)))
	}

	// Apply all merge rules in order
	for _, merge := range t.merges {
		tokens = t.applyMerge(tokens, merge)
	}

	// Convert tokens to IDs
	ids := make([]int, 0, len(tokens))
	for _, tok := range tokens {
		if id, exists := t.vocab[tok]; exists {
			ids = append(ids, id)
		} else {
			// Unknown token (shouldn't happen with BPE, but handle gracefully)
			ids = append(ids, t.specialToks[UnkToken])
		}
	}

	return ids
}

// Decode converts token IDs back to text.
func (t *Tokenizer) Decode(ids []int) string {
	tokens := make([]string, 0, len(ids))
	for _, id := range ids {
		if tok, exists := t.vocabInv[id]; exists {
			// Skip special tokens in output
			if _, isSpecial := t.specialToks[tok]; !isSpecial {
				tokens = append(tokens, tok)
			}
		}
	}

	// Concatenate and convert back from bytes
	byteStr := strings.Join(tokens, "")
	result := make([]byte, 0, len(byteStr))
	for _, r := range byteStr {
		if r < 256 {
			result = append(result, byte(r))
		}
	}

	return string(result)
}

// VocabSize returns the current vocabulary size.
func (t *Tokenizer) VocabSize() int {
	return len(t.vocab)
}

// Save writes the tokenizer to a file.
func (t *Tokenizer) Save(filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("tokenizer: failed to create file: %w", err)
	}
	defer func() {
		if cerr := f.Close(); cerr != nil && err == nil {
			err = fmt.Errorf("tokenizer: failed to close file: %w", cerr)
		}
	}()

	w := bufio.NewWriter(f)
	defer func() {
		if ferr := w.Flush(); ferr != nil && err == nil {
			err = fmt.Errorf("tokenizer: failed to flush writer: %w", ferr)
		}
	}()

	// Write special tokens
	if _, err = fmt.Fprintf(w, "SPECIAL_TOKENS\n"); err != nil {
		return fmt.Errorf("tokenizer: failed to write special tokens header: %w", err)
	}
	for tok, id := range t.specialToks {
		if _, err = fmt.Fprintf(w, "%s\t%d\n", tok, id); err != nil {
			return fmt.Errorf("tokenizer: failed to write special token: %w", err)
		}
	}

	// Write merges
	// Use hex encoding to handle any byte values safely
	if _, err = fmt.Fprintf(w, "MERGES\n"); err != nil {
		return fmt.Errorf("tokenizer: failed to write merges header: %w", err)
	}
	for _, merge := range t.merges {
		firstHex := hex.EncodeToString([]byte(merge.first))
		secondHex := hex.EncodeToString([]byte(merge.second))
		if _, err = fmt.Fprintf(w, "%s %s\n", firstHex, secondHex); err != nil {
			return fmt.Errorf("tokenizer: failed to write merge: %w", err)
		}
	}

	return nil
}

// Load reads a tokenizer from a file.
func (t *Tokenizer) Load(filename string) error {
	// Reset tokenizer state
	t.vocab = make(map[string]int)
	t.vocabInv = make(map[int]string)
	t.merges = []pair{}

	f, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("tokenizer: failed to open file: %w", err)
	}
	defer func() {
		if cerr := f.Close(); cerr != nil && err == nil {
			err = fmt.Errorf("tokenizer: failed to close file: %w", cerr)
		}
	}()

	scanner := bufio.NewScanner(f)
	section := ""

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		// Check for section headers
		if line == "SPECIAL_TOKENS" {
			section = "special"
			continue
		} else if line == "MERGES" {
			section = "merges"
			continue
		}

		switch section {
		case "special":
			parts := strings.Split(line, "\t")
			if len(parts) == 2 {
				var id int
				if _, err = fmt.Sscanf(parts[1], "%d", &id); err != nil {
					return fmt.Errorf("tokenizer: failed to parse token ID: %w", err)
				}
				t.specialToks[parts[0]] = id
				t.vocab[parts[0]] = id
				t.vocabInv[id] = parts[0]
			}

		case "merges":
			// Parse hex-encoded tokens
			parts := strings.Split(line, " ")
			if len(parts) == 2 {
				firstBytes, err1 := hex.DecodeString(parts[0])
				secondBytes, err2 := hex.DecodeString(parts[1])
				if err1 == nil && err2 == nil {
					merge := pair{string(firstBytes), string(secondBytes)}
					t.merges = append(t.merges, merge)
				}
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("tokenizer: error reading file: %w", err)
	}

	// Rebuild vocabulary from merges
	// Start with byte-level vocab
	currentVocabSize := len(t.specialToks)
	for i := 0; i < 256; i++ {
		token := string(rune(i))
		if _, exists := t.vocab[token]; !exists {
			t.vocab[token] = currentVocabSize
			t.vocabInv[currentVocabSize] = token
			currentVocabSize++
		}
	}

	// Apply merges to build subword vocab
	for _, merge := range t.merges {
		mergedToken := merge.first + merge.second
		t.vocab[mergedToken] = currentVocabSize
		t.vocabInv[currentVocabSize] = mergedToken
		currentVocabSize++
	}

	return nil
}

// SimpleTokenizer is a character-level tokenizer for testing.
// It's simpler than BPE but less efficient.
type SimpleTokenizer struct {
	charToID map[rune]int
	idToChar map[int]rune
	nextID   int
}

// NewSimpleTokenizer creates a character-level tokenizer.
func NewSimpleTokenizer() *SimpleTokenizer {
	return &SimpleTokenizer{
		charToID: make(map[rune]int),
		idToChar: make(map[int]rune),
		nextID:   0,
	}
}

// BuildVocab builds vocabulary from corpus.
func (st *SimpleTokenizer) BuildVocab(corpus []string) {
	// Collect unique characters
	chars := make(map[rune]bool)
	for _, text := range corpus {
		for _, r := range text {
			chars[r] = true
		}
	}

	// Sort for deterministic ordering
	sortedChars := make([]rune, 0, len(chars))
	for r := range chars {
		sortedChars = append(sortedChars, r)
	}
	sort.Slice(sortedChars, func(i, j int) bool {
		return sortedChars[i] < sortedChars[j]
	})

	// Assign IDs
	for _, r := range sortedChars {
		st.charToID[r] = st.nextID
		st.idToChar[st.nextID] = r
		st.nextID++
	}
}

// Encode converts text to token IDs.
func (st *SimpleTokenizer) Encode(text string) []int {
	ids := make([]int, 0, utf8.RuneCountInString(text))
	for _, r := range text {
		if id, exists := st.charToID[r]; exists {
			ids = append(ids, id)
		}
		// Skip unknown characters
	}
	return ids
}

// Decode converts token IDs to text.
func (st *SimpleTokenizer) Decode(ids []int) string {
	runes := make([]rune, 0, len(ids))
	for _, id := range ids {
		if r, exists := st.idToChar[id]; exists {
			runes = append(runes, r)
		}
	}
	return string(runes)
}

// VocabSize returns vocabulary size.
func (st *SimpleTokenizer) VocabSize() int {
	return len(st.charToID)
}

// Save writes the simple tokenizer to a file.
func (st *SimpleTokenizer) Save(filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("simpletokenizer: failed to create file: %w", err)
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	defer w.Flush()

	// Write header
	if _, err = fmt.Fprintf(w, "SIMPLE_TOKENIZER\n"); err != nil {
		return fmt.Errorf("simpletokenizer: failed to write header: %w", err)
	}

	// Write vocabulary (sorted by ID for deterministic output)
	type entry struct {
		char rune
		id   int
	}
	entries := make([]entry, 0, len(st.charToID))
	for char, id := range st.charToID {
		entries = append(entries, entry{char, id})
	}
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].id < entries[j].id
	})

	for _, e := range entries {
		if _, err = fmt.Fprintf(w, "%d\\t%s\n", e.id, hex.EncodeToString([]byte(string(e.char)))); err != nil {
			return fmt.Errorf("simpletokenizer: failed to write entry: %w", err)
		}
	}

	return nil
}

// Load reads a simple tokenizer from a file.
func (st *SimpleTokenizer) Load(filename string) error {
	f, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("simpletokenizer: failed to open file: %w", err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)

	// Read header
	if !scanner.Scan() {
		return fmt.Errorf("simpletokenizer: empty file")
	}
	if scanner.Text() != "SIMPLE_TOKENIZER" {
		return fmt.Errorf("simpletokenizer: invalid header")
	}

	// Reset state
	st.charToID = make(map[rune]int)
	st.idToChar = make(map[int]rune)
	st.nextID = 0

	// Read vocabulary
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		parts := strings.Split(line, "\\t")
		if len(parts) != 2 {
			continue
		}

		var id int
		if _, err = fmt.Sscanf(parts[0], "%d", &id); err != nil {
			return fmt.Errorf("simpletokenizer: failed to parse ID: %w", err)
		}

		charBytes, err := hex.DecodeString(parts[1])
		if err != nil {
			return fmt.Errorf("simpletokenizer: failed to decode char: %w", err)
		}

		char := []rune(string(charBytes))[0]
		st.charToID[char] = id
		st.idToChar[id] = char

		if id >= st.nextID {
			st.nextID = id + 1
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("simpletokenizer: error reading file: %w", err)
	}

	return nil
}
