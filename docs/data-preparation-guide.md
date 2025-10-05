# Data Preparation and Preprocessing Guide

A comprehensive guide to preparing and preprocessing training data for transformer models (GPT, BERT, T5), with practical Go implementations and production best practices.

## Table of Contents

1. [Data Collection and Sourcing](#data-collection-and-sourcing)
2. [Text Cleaning and Normalization](#text-cleaning-and-normalization)
3. [Tokenization Strategies](#tokenization-strategies)
4. [Dataset Construction](#dataset-construction)
5. [Data Quality Filtering](#data-quality-filtering)
6. [Deduplication](#deduplication)
7. [Data Augmentation](#data-augmentation)
8. [Train/Validation/Test Splits](#trainvalidationtest-splits)
9. [Data Loading and Batching](#data-loading-and-batching)
10. [Production Pipeline](#production-pipeline)

---

## Data Collection and Sourcing

### Common Data Sources

**Public Datasets:**
- **The Pile** (825 GB): Diverse text corpus (academic papers, books, web text)
- **Common Crawl**: Web-scale corpus (petabytes of text)
- **Wikipedia dumps**: High-quality encyclopedic text
- **Books corpora**: BookCorpus, Project Gutenberg
- **Code**: GitHub, StackOverflow (for code models)
- **Conversational**: OpenSubtitles, Reddit, Twitter archives

**Domain-Specific:**
- Legal: court opinions, contracts, regulations
- Medical: PubMed, clinical notes (with proper PHI handling)
- Financial: SEC filings, earnings calls, news
- Scientific: ArXiv, PubMed, research papers

### Data Collection in Go

```go
package main

import (
	"bufio"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

// DataSource represents a source of training data.
type DataSource struct {
	Name        string   // Source identifier (e.g., "wikipedia", "github")
	URLs        []string // URLs to download from (if applicable)
	LocalPaths  []string // Local file paths
	FilePattern string   // Glob pattern for files (e.g., "*.txt")
}

// DataCollector manages data collection from multiple sources.
type DataCollector struct {
	Sources   []DataSource
	OutputDir string // Directory to save collected data
}

// CollectFromURL downloads text data from a URL.
// Returns the number of bytes downloaded.
func (dc *DataCollector) CollectFromURL(url string, outputPath string) (int64, error) {
	// Download with proper error handling and retries
	resp, err := http.Get(url)
	if err != nil {
		return 0, fmt.Errorf("failed to download %s: %w", url, err)
	}
	defer resp.Body.Close()

	// Check status code
	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("bad status: %s", resp.Status)
	}

	// Create output file
	out, err := os.Create(outputPath)
	if err != nil {
		return 0, fmt.Errorf("failed to create file: %w", err)
	}
	defer out.Close()

	// Copy with progress tracking
	n, err := io.Copy(out, resp.Body)
	if err != nil {
		return n, fmt.Errorf("failed to save data: %w", err)
	}

	return n, nil
}

// CollectFromLocal collects data from local files matching a pattern.
// Returns the total number of files collected.
func (dc *DataCollector) CollectFromLocal(pattern string) ([]string, error) {
	// Use filepath.Glob to find matching files
	// This is useful for collecting data from a directory tree
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return nil, fmt.Errorf("glob pattern failed: %w", err)
	}

	// Filter out directories, keep only files
	var files []string
	for _, match := range matches {
		info, err := os.Stat(match)
		if err != nil {
			continue // Skip files we can't stat
		}
		if !info.IsDir() {
			files = append(files, match)
		}
	}

	return files, nil
}

// Example: Collecting Wikipedia dumps
func collectWikipedia(outputDir string) error {
	// Wikipedia provides dumps in multiple languages
	// Example: https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

	collector := &DataCollector{
		OutputDir: outputDir,
	}

	// Download latest English Wikipedia dump (simplified)
	url := "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles1.xml.bz2"
	outputPath := filepath.Join(outputDir, "wikipedia_en.xml.bz2")

	fmt.Printf("Downloading Wikipedia dump to %s...\n", outputPath)
	n, err := collector.CollectFromURL(url, outputPath)
	if err != nil {
		return err
	}

	fmt.Printf("Downloaded %d bytes\n", n)
	return nil
}

// Example: Collecting local code files
func collectCodeFiles(rootDir string, extensions []string) ([]string, error) {
	collector := &DataCollector{
		OutputDir: ".",
	}

	var allFiles []string

	// Collect files for each extension
	for _, ext := range extensions {
		pattern := filepath.Join(rootDir, "**", "*"+ext)
		files, err := collector.CollectFromLocal(pattern)
		if err != nil {
			return nil, fmt.Errorf("failed to collect %s files: %w", ext, err)
		}
		allFiles = append(allFiles, files...)
	}

	fmt.Printf("Collected %d code files\n", len(allFiles))
	return allFiles, nil
}
```

**Key Considerations:**
- **Licensing**: Ensure you have rights to use the data (check licenses)
- **Privacy**: Remove PII (personally identifiable information) and PHI (protected health information)
- **Bias**: Diverse data sources reduce model bias
- **Scale**: Aim for 10-100× more tokens than model parameters as a rule of thumb
  - GPT-2 small (117M params): ~10B tokens minimum
  - GPT-3 (175B params): ~300B tokens in training set

---

## Text Cleaning and Normalization

Raw text data requires cleaning before tokenization. Common issues: HTML tags, special characters, encoding errors, excessive whitespace.

### Text Cleaning Pipeline

```go
package main

import (
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"
)

// TextCleaner handles text normalization and cleaning.
type TextCleaner struct {
	RemoveHTML        bool // Strip HTML tags
	NormalizeUnicode  bool // Normalize Unicode (NFC/NFKC)
	RemoveURLs        bool // Remove URLs
	RemoveEmails      bool // Remove email addresses
	NormalizeWhitespace bool // Collapse multiple spaces/newlines
	LowercaseOnly     bool // Convert to lowercase (use with caution)
	RemoveNonPrintable bool // Remove non-printable characters

	// Compiled regex patterns (cached for performance)
	htmlTagRegex  *regexp.Regexp
	urlRegex      *regexp.Regexp
	emailRegex    *regexp.Regexp
}

// NewTextCleaner creates a TextCleaner with default settings.
func NewTextCleaner() *TextCleaner {
	return &TextCleaner{
		RemoveHTML:         true,
		NormalizeUnicode:   true,
		RemoveURLs:         false, // Keep URLs for some applications
		RemoveEmails:       true,
		NormalizeWhitespace: true,
		RemoveNonPrintable: true,
		LowercaseOnly:      false, // Preserve case for most LLM training

		// Pre-compile regex patterns for performance
		htmlTagRegex: regexp.MustCompile(`<[^>]*>`),
		urlRegex:     regexp.MustCompile(`https?://\S+`),
		emailRegex:   regexp.MustCompile(`\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b`),
	}
}

// Clean applies all enabled cleaning operations to text.
func (tc *TextCleaner) Clean(text string) string {
	// Order matters! Each step builds on the previous one.

	// 1. Remove HTML tags (before other processing)
	if tc.RemoveHTML {
		text = tc.htmlTagRegex.ReplaceAllString(text, " ")
	}

	// 2. Remove URLs (can contain special characters)
	if tc.RemoveURLs {
		text = tc.urlRegex.ReplaceAllString(text, "")
	}

	// 3. Remove email addresses (privacy concern)
	if tc.RemoveEmails {
		text = tc.emailRegex.ReplaceAllString(text, "")
	}

	// 4. Remove non-printable characters
	if tc.RemoveNonPrintable {
		text = tc.removeNonPrintable(text)
	}

	// 5. Normalize whitespace (collapse multiple spaces/newlines)
	if tc.NormalizeWhitespace {
		text = tc.normalizeWhitespace(text)
	}

	// 6. Lowercase (optional, usually NOT recommended for LLMs)
	if tc.LowercaseOnly {
		text = strings.ToLower(text)
	}

	// 7. Trim leading/trailing whitespace
	text = strings.TrimSpace(text)

	return text
}

// removeNonPrintable removes control characters and non-printable Unicode.
func (tc *TextCleaner) removeNonPrintable(text string) string {
	// Keep: letters, numbers, punctuation, spaces, newlines, tabs
	// Remove: control characters (except \n, \t, \r), zero-width chars, etc.

	var builder strings.Builder
	builder.Grow(len(text))

	for _, r := range text {
		// Keep printable characters and common whitespace
		if r == '\n' || r == '\t' || r == '\r' {
			builder.WriteRune(r)
		} else if unicode.IsPrint(r) {
			builder.WriteRune(r)
		}
		// Skip non-printable characters
	}

	return builder.String()
}

// normalizeWhitespace collapses multiple spaces/newlines to single instances.
func (tc *TextCleaner) normalizeWhitespace(text string) string {
	// Replace multiple spaces with single space
	spaceRegex := regexp.MustCompile(`[ \t]+`)
	text = spaceRegex.ReplaceAllString(text, " ")

	// Replace multiple newlines with double newline (preserve paragraph breaks)
	newlineRegex := regexp.MustCompile(`\n{3,}`)
	text = newlineRegex.ReplaceAllString(text, "\n\n")

	return text
}

// ValidateUTF8 checks if text is valid UTF-8 and repairs if not.
// Invalid UTF-8 can cause tokenization errors.
func ValidateUTF8(text string) string {
	if utf8.ValidString(text) {
		return text
	}

	// Repair by replacing invalid bytes with replacement character
	var builder strings.Builder
	builder.Grow(len(text))

	for _, r := range text {
		if r == utf8.RuneError {
			builder.WriteRune('�') // Unicode replacement character
		} else {
			builder.WriteRune(r)
		}
	}

	return builder.String()
}

// Example usage
func cleanTextExample() {
	rawText := `
		<html><body>
		Hello World!   This is a test.


		Contact me at user@example.com or visit https://example.com

		Lots    of     spaces.
		</body></html>
	`

	cleaner := NewTextCleaner()
	cleaned := cleaner.Clean(rawText)

	fmt.Println("Cleaned:")
	fmt.Println(cleaned)
	// Output:
	// Hello World! This is a test.
	//
	// Contact me at  or visit
	//
	// Lots of spaces.
}
```

**Cleaning Best Practices:**
- **Preserve structure**: Keep paragraph breaks (double newlines) for document structure
- **Don't over-clean**: Modern tokenizers handle punctuation, emojis, Unicode well
- **Document-specific**: Adapt cleaning to your domain (keep HTML for web data, keep code formatting for programming)
- **Validation**: Always validate UTF-8 encoding before tokenization

---

## Tokenization Strategies

Tokenization converts text into tokens (subwords, characters, or words) that the model can process.

### Tokenization Approaches

**1. Character-level** (simplest, but longest sequences):
- Vocab size: ~256-300 (ASCII + special chars)
- Use case: Small models, multilingual, educational

**2. Word-level** (rare in modern LLMs):
- Vocab size: 10K-100K
- Problem: Out-of-vocabulary (OOV) words, large vocab

**3. Subword (BPE, WordPiece, Unigram)** (most common):
- Vocab size: 32K-50K (GPT), 30K (BERT)
- Best trade-off: balance between sequence length and vocab size
- Handles rare words by decomposing into subwords

**4. SentencePiece** (language-agnostic, used by T5, LLaMA):
- Vocab size: 32K+
- Works on raw text (no pre-tokenization)
- Good for multilingual models

### BPE Tokenizer Implementation

```go
package main

import (
	"bufio"
	"fmt"
	"os"
	"sort"
	"strings"
	"unicode/utf8"
)

// BPETokenizer implements Byte-Pair Encoding tokenization.
// BPE is used in GPT-2, GPT-3, RoBERTa, and many modern LLMs.
type BPETokenizer struct {
	Vocab      map[string]int // Token -> ID mapping
	Merges     []Pair         // Ordered list of merge rules
	VocabSize  int           // Target vocabulary size
	SpecialTokens map[string]int // Special tokens (<pad>, <unk>, <bos>, <eos>)
}

// Pair represents a token pair for BPE merging.
type Pair struct {
	First  string
	Second string
}

// Train builds a BPE vocabulary from a corpus.
// This is the training phase that learns merge rules.
func (bpe *BPETokenizer) Train(corpus []string, vocabSize int) {
	bpe.VocabSize = vocabSize
	bpe.Vocab = make(map[string]int)
	bpe.Merges = make([]Pair, 0)
	bpe.SpecialTokens = make(map[string]int)

	// 1. Initialize with special tokens
	bpe.addSpecialToken("<pad>")  // Padding token
	bpe.addSpecialToken("<unk>")  // Unknown token
	bpe.addSpecialToken("<bos>")  // Beginning of sequence
	bpe.addSpecialToken("<eos>")  // End of sequence

	// 2. Initialize with character-level vocabulary
	// Start with individual bytes/characters as base vocabulary
	charFreq := make(map[string]int)
	for _, text := range corpus {
		for _, r := range text {
			char := string(r)
			charFreq[char]++
		}
	}

	// Add characters to vocabulary
	for char := range charFreq {
		if _, exists := bpe.Vocab[char]; !exists {
			bpe.Vocab[char] = len(bpe.Vocab)
		}
	}

	fmt.Printf("Initial vocab size (characters): %d\n", len(bpe.Vocab))

	// 3. Iteratively merge most frequent pairs until vocab size reached
	// Start with words split into characters
	words := make([][]string, len(corpus))
	for i, text := range corpus {
		words[i] = bpe.splitIntoChars(text)
	}

	numMerges := vocabSize - len(bpe.Vocab)
	for i := 0; i < numMerges; i++ {
		// Count pair frequencies across all words
		pairCounts := make(map[Pair]int)
		for _, word := range words {
			for j := 0; j < len(word)-1; j++ {
				pair := Pair{First: word[j], Second: word[j+1]}
				pairCounts[pair]++
			}
		}

		// Find most frequent pair
		if len(pairCounts) == 0 {
			break // No more pairs to merge
		}

		bestPair := bpe.findMostFrequentPair(pairCounts)

		// Record this merge rule
		bpe.Merges = append(bpe.Merges, bestPair)

		// Add merged token to vocabulary
		mergedToken := bestPair.First + bestPair.Second
		bpe.Vocab[mergedToken] = len(bpe.Vocab)

		// Apply merge to all words
		words = bpe.applyMerge(words, bestPair)

		if (i+1)%1000 == 0 {
			fmt.Printf("Completed %d merges, vocab size: %d\n", i+1, len(bpe.Vocab))
		}
	}

	fmt.Printf("BPE training complete. Final vocab size: %d\n", len(bpe.Vocab))
}

// addSpecialToken adds a special token to the vocabulary.
func (bpe *BPETokenizer) addSpecialToken(token string) {
	if _, exists := bpe.Vocab[token]; !exists {
		tokenID := len(bpe.Vocab)
		bpe.Vocab[token] = tokenID
		bpe.SpecialTokens[token] = tokenID
	}
}

// splitIntoChars splits text into individual characters.
func (bpe *BPETokenizer) splitIntoChars(text string) []string {
	chars := make([]string, 0, utf8.RuneCountInString(text))
	for _, r := range text {
		chars = append(chars, string(r))
	}
	return chars
}

// findMostFrequentPair returns the most frequent pair from counts.
func (bpe *BPETokenizer) findMostFrequentPair(pairCounts map[Pair]int) Pair {
	var bestPair Pair
	maxCount := 0

	for pair, count := range pairCounts {
		if count > maxCount {
			bestPair = pair
			maxCount = count
		}
	}

	return bestPair
}

// applyMerge applies a merge rule to all words.
func (bpe *BPETokenizer) applyMerge(words [][]string, pair Pair) [][]string {
	merged := pair.First + pair.Second

	for i, word := range words {
		j := 0
		for j < len(word)-1 {
			if word[j] == pair.First && word[j+1] == pair.Second {
				// Merge this pair
				word = append(word[:j], append([]string{merged}, word[j+2:]...)...)
			} else {
				j++
			}
		}
		words[i] = word
	}

	return words
}

// Encode converts text to token IDs using trained BPE vocabulary.
func (bpe *BPETokenizer) Encode(text string) []int {
	// 1. Start with character-level split
	tokens := bpe.splitIntoChars(text)

	// 2. Apply merge rules in order
	for _, pair := range bpe.Merges {
		merged := pair.First + pair.Second

		// Apply this merge rule across all tokens
		j := 0
		for j < len(tokens)-1 {
			if tokens[j] == pair.First && tokens[j+1] == pair.Second {
				// Merge these two tokens
				tokens = append(tokens[:j], append([]string{merged}, tokens[j+2:]...)...)
			} else {
				j++
			}
		}
	}

	// 3. Convert tokens to IDs
	ids := make([]int, 0, len(tokens))
	unkID := bpe.SpecialTokens["<unk>"]

	for _, token := range tokens {
		if id, exists := bpe.Vocab[token]; exists {
			ids = append(ids, id)
		} else {
			ids = append(ids, unkID) // Unknown token
		}
	}

	return ids
}

// Decode converts token IDs back to text.
func (bpe *BPETokenizer) Decode(ids []int) string {
	// Create reverse vocab mapping (ID -> token)
	reverseVocab := make(map[int]string, len(bpe.Vocab))
	for token, id := range bpe.Vocab {
		reverseVocab[id] = token
	}

	// Convert IDs to tokens
	tokens := make([]string, 0, len(ids))
	for _, id := range ids {
		if token, exists := reverseVocab[id]; exists {
			// Skip special tokens in output
			if _, isSpecial := bpe.SpecialTokens[token]; !isSpecial {
				tokens = append(tokens, token)
			}
		}
	}

	// Join tokens into text
	return strings.Join(tokens, "")
}

// SaveVocab saves the BPE vocabulary and merges to files.
func (bpe *BPETokenizer) SaveVocab(vocabPath, mergesPath string) error {
	// Save vocabulary (token -> ID mapping)
	vocabFile, err := os.Create(vocabPath)
	if err != nil {
		return err
	}
	defer vocabFile.Close()

	// Sort vocab by ID for consistent ordering
	type vocabEntry struct {
		Token string
		ID    int
	}
	entries := make([]vocabEntry, 0, len(bpe.Vocab))
	for token, id := range bpe.Vocab {
		entries = append(entries, vocabEntry{Token: token, ID: id})
	}
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].ID < entries[j].ID
	})

	writer := bufio.NewWriter(vocabFile)
	for _, entry := range entries {
		fmt.Fprintf(writer, "%s\t%d\n", entry.Token, entry.ID)
	}
	writer.Flush()

	// Save merge rules
	mergesFile, err := os.Create(mergesPath)
	if err != nil {
		return err
	}
	defer mergesFile.Close()

	mergeWriter := bufio.NewWriter(mergesFile)
	for _, pair := range bpe.Merges {
		fmt.Fprintf(mergeWriter, "%s\t%s\n", pair.First, pair.Second)
	}
	mergeWriter.Flush()

	return nil
}

// Example: Training a BPE tokenizer
func trainBPEExample() {
	// Small corpus for demonstration
	corpus := []string{
		"the quick brown fox",
		"the lazy dog",
		"quick brown foxes",
		"lazy dogs",
	}

	tokenizer := &BPETokenizer{}
	tokenizer.Train(corpus, 100) // Target vocab size of 100

	// Test encoding
	text := "the quick fox"
	ids := tokenizer.Encode(text)
	decoded := tokenizer.Decode(ids)

	fmt.Printf("Original: %s\n", text)
	fmt.Printf("Token IDs: %v\n", ids)
	fmt.Printf("Decoded: %s\n", decoded)

	// Save vocabulary
	tokenizer.SaveVocab("vocab.txt", "merges.txt")
}
```

**Tokenization Trade-offs:**
- **Character-level**: Longest sequences (bad for attention O(n²) complexity), but handles any text
- **Word-level**: Shortest sequences, but huge vocabulary and OOV problems
- **BPE/Subword**: Best balance - vocabulary size vs sequence length

**Practical Recommendations:**
- **GPT models**: BPE with 50K vocab (GPT-2/3 standard)
- **BERT models**: WordPiece with 30K vocab
- **Multilingual**: SentencePiece with 32K+ vocab
- **Code models**: Byte-level BPE (handles any byte sequence)

---

## Dataset Construction

### Creating Training Examples

Different architectures require different data formats:

**GPT (Causal LM)**: Next-token prediction
```
Input:  "The quick brown"
Target: "quick brown fox"
```

**BERT (Masked LM)**: Masked token prediction
```
Input:  "The [MASK] brown fox"
Target: "quick"
```

**T5 (Seq2seq)**: Input → output pairs
```
Input:  "translate English to French: Hello world"
Target: "Bonjour le monde"
```

### Dataset Builder Implementation

```go
package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
)

// DatasetBuilder creates training examples from preprocessed text.
type DatasetBuilder struct {
	Tokenizer     *BPETokenizer
	SeqLen        int     // Maximum sequence length
	Overlap       int     // Overlap between sequences (for continuity)
	MinLength     int     // Minimum sequence length (filter short sequences)
	Architecture  string  // "gpt", "bert", or "t5"
	MaskProb      float64 // Probability of masking tokens (for BERT)
}

// TrainingExample represents a single training example.
type TrainingExample struct {
	InputIDs  []int  // Input token IDs
	TargetIDs []int  // Target token IDs (shifted for GPT, masked for BERT)
	Mask      []bool // Attention mask (1 for real tokens, 0 for padding)
}

// BuildDataset creates training examples from text files.
func (db *DatasetBuilder) BuildDataset(inputPaths []string, outputPath string) error {
	outputFile, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create output file: %w", err)
	}
	defer outputFile.Close()

	writer := bufio.NewWriter(outputFile)
	defer writer.Flush()

	totalExamples := 0

	// Process each input file
	for _, path := range inputPaths {
		file, err := os.Open(path)
		if err != nil {
			fmt.Printf("Warning: could not open %s: %v\n", path, err)
			continue
		}

		scanner := bufio.NewScanner(file)
		scanner.Buffer(make([]byte, 1024*1024), 1024*1024) // 1MB buffer for long lines

		// Read file line by line
		for scanner.Scan() {
			text := scanner.Text()

			// Tokenize text
			tokens := db.Tokenizer.Encode(text)

			// Create training examples from tokens
			examples := db.createExamples(tokens)

			// Write examples to output file
			for _, example := range examples {
				db.writeExample(writer, example)
				totalExamples++
			}
		}

		file.Close()

		if err := scanner.Err(); err != nil {
			fmt.Printf("Warning: error reading %s: %v\n", path, err)
		}
	}

	fmt.Printf("Created %d training examples\n", totalExamples)
	return nil
}

// createExamples splits tokens into training examples with overlap.
func (db *DatasetBuilder) createExamples(tokens []int) []TrainingExample {
	// Filter short sequences
	if len(tokens) < db.MinLength {
		return nil
	}

	var examples []TrainingExample
	stride := db.SeqLen - db.Overlap

	// Create overlapping windows
	for i := 0; i+db.SeqLen <= len(tokens); i += stride {
		window := tokens[i : i+db.SeqLen]

		var example TrainingExample

		switch db.Architecture {
		case "gpt":
			// GPT: predict next token
			// Input: [0:n-1], Target: [1:n]
			example = TrainingExample{
				InputIDs:  window[:len(window)-1],
				TargetIDs: window[1:],
				Mask:      make([]bool, len(window)-1),
			}
			for i := range example.Mask {
				example.Mask[i] = true
			}

		case "bert":
			// BERT: masked language modeling
			example = db.createMaskedExample(window)

		case "t5":
			// T5: span corruption (simplified)
			// For T5, you'd typically need paired input/output
			// This is a simplified version
			example = db.createSpanCorruptionExample(window)
		}

		examples = append(examples, example)
	}

	return examples
}

// createMaskedExample creates a BERT-style masked language modeling example.
func (db *DatasetBuilder) createMaskedExample(tokens []int) TrainingExample {
	// BERT masking: 15% of tokens are selected
	// Of those: 80% replaced with [MASK], 10% random, 10% unchanged

	maskID := db.Tokenizer.SpecialTokens["<unk>"] // Use <unk> as [MASK] for simplicity

	input := make([]int, len(tokens))
	target := make([]int, len(tokens))
	mask := make([]bool, len(tokens))

	copy(input, tokens)
	copy(target, tokens)

	// Mask random tokens
	for i := range tokens {
		mask[i] = true

		if rand.Float64() < db.MaskProb {
			// This token will be masked
			roll := rand.Float64()
			if roll < 0.8 {
				// 80%: replace with [MASK]
				input[i] = maskID
			} else if roll < 0.9 {
				// 10%: replace with random token
				input[i] = rand.Intn(len(db.Tokenizer.Vocab))
			}
			// 10%: keep original (else case, no change)
		} else {
			// Not masked - target is -1 (ignore in loss)
			target[i] = -1
		}
	}

	return TrainingExample{
		InputIDs:  input,
		TargetIDs: target,
		Mask:      mask,
	}
}

// createSpanCorruptionExample creates a T5-style span corruption example.
func (db *DatasetBuilder) createSpanCorruptionExample(tokens []int) TrainingExample {
	// Simplified T5 span corruption:
	// Mask contiguous spans and predict them

	// For simplicity, we'll just use the same approach as GPT
	// Real T5 would have separate encoder/decoder inputs
	return TrainingExample{
		InputIDs:  tokens[:len(tokens)-1],
		TargetIDs: tokens[1:],
		Mask:      make([]bool, len(tokens)-1),
	}
}

// writeExample writes a training example to file in binary format.
func (db *DatasetBuilder) writeExample(writer *bufio.Writer, example TrainingExample) error {
	// Format: length | input_ids | target_ids | mask
	// This is a simplified format - production systems use TFRecord, HDF5, etc.

	fmt.Fprintf(writer, "%d", len(example.InputIDs))
	for _, id := range example.InputIDs {
		fmt.Fprintf(writer, " %d", id)
	}
	for _, id := range example.TargetIDs {
		fmt.Fprintf(writer, " %d", id)
	}
	fmt.Fprintf(writer, "\n")

	return nil
}
```

**Dataset Construction Best Practices:**
- **Sequence overlap**: Use 10-20% overlap to provide context across boundaries
- **Minimum length**: Filter sequences shorter than 32-64 tokens (too short for learning)
- **Shuffling**: Shuffle examples during training to prevent overfitting to document order
- **Packing**: Pack multiple short documents into single sequence to maximize GPU utilization

---

## Data Quality Filtering

Not all data is high quality. Filtering improves model performance and reduces training time.

### Quality Filters

```go
package main

import (
	"math"
	"regexp"
	"strings"
	"unicode"
)

// QualityFilter filters low-quality text based on heuristics.
type QualityFilter struct {
	MinLength          int     // Minimum document length (characters)
	MaxLength          int     // Maximum document length (filter very long docs)
	MinWordLength      float64 // Minimum average word length
	MaxWordLength      float64 // Maximum average word length
	MinAlphaRatio      float64 // Minimum ratio of alphabetic characters
	MaxSymbolRatio     float64 // Maximum ratio of symbols (detect spam/gibberish)
	MaxRepetitionRatio float64 // Maximum line/word repetition (detect spam)
	BannedWords        []string // List of banned words/phrases
	RequireMinWords    int      // Minimum number of words
}

// NewDefaultQualityFilter creates a filter with sensible defaults.
func NewDefaultQualityFilter() *QualityFilter {
	return &QualityFilter{
		MinLength:          100,   // At least 100 characters
		MaxLength:          100000, // Max 100K characters per document
		MinWordLength:      2.5,    // Average word should be > 2.5 chars
		MaxWordLength:      15.0,   // Average word should be < 15 chars (detect gibberish)
		MinAlphaRatio:      0.6,    // At least 60% alphabetic
		MaxSymbolRatio:     0.2,    // At most 20% symbols
		MaxRepetitionRatio: 0.3,    // At most 30% repetitive lines
		BannedWords:        []string{}, // Add your banned words here
		RequireMinWords:    20,     // At least 20 words
	}
}

// ShouldKeep returns true if the document passes quality filters.
func (qf *QualityFilter) ShouldKeep(text string) bool {
	// 1. Length filters
	if len(text) < qf.MinLength || len(text) > qf.MaxLength {
		return false
	}

	// 2. Word count filter
	words := strings.Fields(text)
	if len(words) < qf.RequireMinWords {
		return false
	}

	// 3. Average word length filter (detect gibberish)
	avgWordLen := qf.averageWordLength(words)
	if avgWordLen < qf.MinWordLength || avgWordLen > qf.MaxWordLength {
		return false
	}

	// 4. Character type ratios (detect spam/gibberish)
	alphaRatio, symbolRatio := qf.characterRatios(text)
	if alphaRatio < qf.MinAlphaRatio || symbolRatio > qf.MaxSymbolRatio {
		return false
	}

	// 5. Repetition filter (detect spam)
	repetitionRatio := qf.lineRepetitionRatio(text)
	if repetitionRatio > qf.MaxRepetitionRatio {
		return false
	}

	// 6. Banned words filter
	textLower := strings.ToLower(text)
	for _, banned := range qf.BannedWords {
		if strings.Contains(textLower, strings.ToLower(banned)) {
			return false
		}
	}

	return true
}

// averageWordLength computes the average word length.
func (qf *QualityFilter) averageWordLength(words []string) float64 {
	if len(words) == 0 {
		return 0
	}

	totalChars := 0
	for _, word := range words {
		totalChars += len(word)
	}

	return float64(totalChars) / float64(len(words))
}

// characterRatios computes the ratio of alphabetic and symbol characters.
func (qf *QualityFilter) characterRatios(text string) (alphaRatio, symbolRatio float64) {
	if len(text) == 0 {
		return 0, 0
	}

	alphaCount := 0
	symbolCount := 0

	for _, r := range text {
		if unicode.IsLetter(r) {
			alphaCount++
		} else if unicode.IsPunct(r) || unicode.IsSymbol(r) {
			symbolCount++
		}
	}

	alphaRatio = float64(alphaCount) / float64(len(text))
	symbolRatio = float64(symbolCount) / float64(len(text))

	return alphaRatio, symbolRatio
}

// lineRepetitionRatio computes the ratio of repeated lines.
func (qf *QualityFilter) lineRepetitionRatio(text string) float64 {
	lines := strings.Split(text, "\n")
	if len(lines) <= 1 {
		return 0
	}

	// Count unique lines
	seen := make(map[string]int)
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed != "" {
			seen[trimmed]++
		}
	}

	// Count repeated lines (appeared more than once)
	repeatedCount := 0
	for _, count := range seen {
		if count > 1 {
			repeatedCount += count
		}
	}

	return float64(repeatedCount) / float64(len(lines))
}

// FilterCorpus filters a corpus and returns high-quality documents.
func FilterCorpus(documents []string, filter *QualityFilter) []string {
	var filtered []string

	for _, doc := range documents {
		if filter.ShouldKeep(doc) {
			filtered = append(filtered, doc)
		}
	}

	fmt.Printf("Filtered corpus: %d -> %d documents (kept %.1f%%)\n",
		len(documents), len(filtered), 100.0*float64(len(filtered))/float64(len(documents)))

	return filtered
}
```

**Additional Quality Signals:**
- **Perplexity filtering**: Use a language model to score documents, filter high-perplexity (low-quality) text
- **Classifier filtering**: Train a quality classifier (high-quality vs low-quality) on labeled data
- **Deduplication**: Remove duplicate or near-duplicate documents (see next section)

---

## Deduplication

Duplicate data hurts model performance and training efficiency. Remove exact and near-duplicates.

### Deduplication Approaches

**1. Exact deduplication** (fast, O(n)):
- Hash each document (MD5, SHA256)
- Keep only one copy of each hash

**2. Near-deduplication** (slower, O(n²) naive or O(n log n) with LSH):
- MinHash + LSH (Locality-Sensitive Hashing)
- Find documents with Jaccard similarity > threshold

### MinHash Deduplication

```go
package main

import (
	"crypto/md5"
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"math"
	"strings"
)

// MinHashDeduplicator finds near-duplicate documents using MinHash.
type MinHashDeduplicator struct {
	NumHashes    int     // Number of hash functions (more = higher precision)
	ShingleSize  int     // Size of character n-grams (shingles)
	Threshold    float64 // Jaccard similarity threshold (0.0-1.0)
	signatures   [][]uint32 // MinHash signatures for each document
}

// NewMinHashDeduplicator creates a new deduplicator.
func NewMinHashDeduplicator(numHashes, shingleSize int, threshold float64) *MinHashDeduplicator {
	return &MinHashDeduplicator{
		NumHashes:   numHashes,
		ShingleSize: shingleSize,
		Threshold:   threshold,
		signatures:  make([][]uint32, 0),
	}
}

// CreateShingles splits text into character n-grams (shingles).
func (mhd *MinHashDeduplicator) CreateShingles(text string) []string {
	// Character-level shingles (e.g., 3-grams)
	// Example: "hello" with shingleSize=3 -> ["hel", "ell", "llo"]

	if len(text) < mhd.ShingleSize {
		return []string{text}
	}

	shingles := make([]string, 0, len(text)-mhd.ShingleSize+1)
	for i := 0; i <= len(text)-mhd.ShingleSize; i++ {
		shingle := text[i : i+mhd.ShingleSize]
		shingles = append(shingles, shingle)
	}

	return shingles
}

// ComputeMinHash computes the MinHash signature for a document.
func (mhd *MinHashDeduplicator) ComputeMinHash(text string) []uint32 {
	shingles := mhd.CreateShingles(text)

	// Initialize signature with maximum values
	signature := make([]uint32, mhd.NumHashes)
	for i := range signature {
		signature[i] = math.MaxUint32
	}

	// For each shingle, compute hash values and update signature
	for _, shingle := range shingles {
		// Use multiple hash functions (simulate with seeds)
		for i := 0; i < mhd.NumHashes; i++ {
			hash := mhd.hashWithSeed(shingle, uint32(i))
			if hash < signature[i] {
				signature[i] = hash
			}
		}
	}

	return signature
}

// hashWithSeed computes a hash of the shingle with a seed.
func (mhd *MinHashDeduplicator) hashWithSeed(shingle string, seed uint32) uint32 {
	h := fnv.New32a()
	h.Write([]byte(shingle))
	binary.Write(h, binary.LittleEndian, seed)
	return h.Sum32()
}

// JaccardSimilarity estimates Jaccard similarity between two signatures.
func (mhd *MinHashDeduplicator) JaccardSimilarity(sig1, sig2 []uint32) float64 {
	if len(sig1) != len(sig2) {
		return 0
	}

	matches := 0
	for i := range sig1 {
		if sig1[i] == sig2[i] {
			matches++
		}
	}

	return float64(matches) / float64(len(sig1))
}

// Deduplicate removes near-duplicate documents from a corpus.
func (mhd *MinHashDeduplicator) Deduplicate(documents []string) []string {
	fmt.Printf("Computing MinHash signatures for %d documents...\n", len(documents))

	// 1. Compute MinHash signatures for all documents
	signatures := make([][]uint32, len(documents))
	for i, doc := range documents {
		signatures[i] = mhd.ComputeMinHash(doc)

		if (i+1)%1000 == 0 {
			fmt.Printf("  Processed %d documents\n", i+1)
		}
	}

	// 2. Find duplicates using pairwise comparison
	// Note: This is O(n²) - for large datasets, use LSH for O(n)
	keep := make([]bool, len(documents))
	for i := range keep {
		keep[i] = true
	}

	fmt.Println("Finding near-duplicates...")
	duplicatesFound := 0

	for i := 0; i < len(documents); i++ {
		if !keep[i] {
			continue // Already marked as duplicate
		}

		for j := i + 1; j < len(documents); j++ {
			if !keep[j] {
				continue
			}

			// Compute similarity
			sim := mhd.JaccardSimilarity(signatures[i], signatures[j])

			if sim >= mhd.Threshold {
				// Documents are near-duplicates, keep first one only
				keep[j] = false
				duplicatesFound++
			}
		}

		if (i+1)%100 == 0 {
			fmt.Printf("  Compared %d documents, found %d duplicates\n", i+1, duplicatesFound)
		}
	}

	// 3. Return deduplicated corpus
	var deduplicated []string
	for i, doc := range documents {
		if keep[i] {
			deduplicated = append(deduplicated, doc)
		}
	}

	fmt.Printf("Deduplication complete: %d -> %d documents (removed %.1f%%)\n",
		len(documents), len(deduplicated),
		100.0*float64(len(documents)-len(deduplicated))/float64(len(documents)))

	return deduplicated
}

// ExactDeduplicate removes exact duplicates using hashing (faster).
func ExactDeduplicate(documents []string) []string {
	seen := make(map[string]bool)
	var deduplicated []string

	for _, doc := range documents {
		// Compute hash of document
		hash := fmt.Sprintf("%x", md5.Sum([]byte(doc)))

		if !seen[hash] {
			seen[hash] = true
			deduplicated = append(deduplicated, doc)
		}
	}

	fmt.Printf("Exact deduplication: %d -> %d documents (removed %.1f%%)\n",
		len(documents), len(deduplicated),
		100.0*float64(len(documents)-len(deduplicated))/float64(len(documents)))

	return deduplicated
}
```

**Deduplication Best Practices:**
- **Run exact deduplication first**: Fast and removes obvious duplicates
- **Use MinHash for near-duplicates**: Set threshold to 0.8-0.9 Jaccard similarity
- **Cross-dataset deduplication**: Deduplicate test set against training set to prevent data leakage
- **Scale with LSH**: For very large corpora (>1M documents), use Locality-Sensitive Hashing

---

## Data Augmentation

Data augmentation increases training data diversity, especially useful for small datasets.

### Augmentation Techniques

**For NLP:**
1. **Synonym replacement**: Replace words with synonyms
2. **Back-translation**: Translate to another language and back
3. **Random insertion**: Insert random words
4. **Random swap**: Swap word positions
5. **Random deletion**: Delete random words
6. **Paraphrasing**: Use a model to generate paraphrases

```go
package main

import (
	"math/rand"
	"strings"
)

// DataAugmenter provides text augmentation methods.
type DataAugmenter struct {
	RandomSeed int
	SynonymMap map[string][]string // Simple synonym dictionary
}

// NewDataAugmenter creates a new augmenter.
func NewDataAugmenter(seed int) *DataAugmenter {
	rand.Seed(int64(seed))

	// Simple synonym map (in practice, use WordNet or similar)
	synonyms := map[string][]string{
		"good":  {"great", "excellent", "fine", "nice"},
		"bad":   {"poor", "terrible", "awful", "horrible"},
		"big":   {"large", "huge", "enormous", "gigantic"},
		"small": {"tiny", "little", "mini", "petite"},
	}

	return &DataAugmenter{
		RandomSeed: seed,
		SynonymMap: synonyms,
	}
}

// SynonymReplacement replaces n random words with synonyms.
func (da *DataAugmenter) SynonymReplacement(text string, n int) string {
	words := strings.Fields(text)
	if len(words) == 0 {
		return text
	}

	// Find replaceable words
	var replaceableIndices []int
	for i, word := range words {
		word = strings.ToLower(strings.Trim(word, ".,!?;:"))
		if _, exists := da.SynonymMap[word]; exists {
			replaceableIndices = append(replaceableIndices, i)
		}
	}

	if len(replaceableIndices) == 0 {
		return text
	}

	// Replace n random words
	numReplacements := min(n, len(replaceableIndices))
	rand.Shuffle(len(replaceableIndices), func(i, j int) {
		replaceableIndices[i], replaceableIndices[j] = replaceableIndices[j], replaceableIndices[i]
	})

	for i := 0; i < numReplacements; i++ {
		idx := replaceableIndices[i]
		word := strings.ToLower(strings.Trim(words[idx], ".,!?;:"))

		if synonyms, exists := da.SynonymMap[word]; exists && len(synonyms) > 0 {
			synonym := synonyms[rand.Intn(len(synonyms))]
			words[idx] = synonym
		}
	}

	return strings.Join(words, " ")
}

// RandomInsertion inserts n random words at random positions.
func (da *DataAugmenter) RandomInsertion(text string, n int) string {
	words := strings.Fields(text)
	if len(words) == 0 {
		return text
	}

	// Collect all possible insertion words from synonym map
	var insertWords []string
	for word, synonyms := range da.SynonymMap {
		insertWords = append(insertWords, word)
		insertWords = append(insertWords, synonyms...)
	}

	if len(insertWords) == 0 {
		return text
	}

	// Insert n random words
	for i := 0; i < n; i++ {
		insertWord := insertWords[rand.Intn(len(insertWords))]
		insertPos := rand.Intn(len(words) + 1)

		words = append(words[:insertPos], append([]string{insertWord}, words[insertPos:]...)...)
	}

	return strings.Join(words, " ")
}

// RandomSwap swaps n random pairs of words.
func (da *DataAugmenter) RandomSwap(text string, n int) string {
	words := strings.Fields(text)
	if len(words) < 2 {
		return text
	}

	for i := 0; i < n; i++ {
		idx1 := rand.Intn(len(words))
		idx2 := rand.Intn(len(words))
		words[idx1], words[idx2] = words[idx2], words[idx1]
	}

	return strings.Join(words, " ")
}

// RandomDeletion randomly deletes words with probability p.
func (da *DataAugmenter) RandomDeletion(text string, p float64) string {
	words := strings.Fields(text)
	if len(words) == 0 {
		return text
	}

	// Keep at least one word
	var kept []string
	for _, word := range words {
		if rand.Float64() > p {
			kept = append(kept, word)
		}
	}

	if len(kept) == 0 {
		return words[rand.Intn(len(words))]
	}

	return strings.Join(kept, " ")
}

// Augment applies random augmentation techniques to text.
func (da *DataAugmenter) Augment(text string) string {
	// Randomly choose an augmentation technique
	techniques := []func(string) string{
		func(t string) string { return da.SynonymReplacement(t, 2) },
		func(t string) string { return da.RandomInsertion(t, 1) },
		func(t string) string { return da.RandomSwap(t, 1) },
		func(t string) string { return da.RandomDeletion(t, 0.1) },
	}

	technique := techniques[rand.Intn(len(techniques))]
	return technique(text)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
```

**Augmentation Cautions:**
- **Use sparingly**: Over-augmentation can hurt model performance
- **Validation set**: Never augment validation/test sets
- **Label preservation**: Ensure augmentations don't change meaning (for supervised tasks)
- **For large models**: Often unnecessary - large models trained on massive corpora don't benefit much

---

## Train/Validation/Test Splits

Proper splitting prevents data leakage and enables reliable evaluation.

### Splitting Strategies

```go
package main

import (
	"fmt"
	"math/rand"
)

// DataSplitter splits data into train/validation/test sets.
type DataSplitter struct {
	TrainRatio float64
	ValRatio   float64
	TestRatio  float64
	Seed       int
	Shuffle    bool
}

// Split splits documents into train/val/test sets.
func (ds *DataSplitter) Split(documents []string) (train, val, test []string) {
	// Validate ratios
	total := ds.TrainRatio + ds.ValRatio + ds.TestRatio
	if math.Abs(total-1.0) > 0.001 {
		panic(fmt.Sprintf("ratios must sum to 1.0, got %.3f", total))
	}

	// Shuffle if requested
	if ds.Shuffle {
		rand.Seed(int64(ds.Seed))
		rand.Shuffle(len(documents), func(i, j int) {
			documents[i], documents[j] = documents[j], documents[i]
		})
	}

	// Calculate split indices
	n := len(documents)
	trainEnd := int(float64(n) * ds.TrainRatio)
	valEnd := trainEnd + int(float64(n)*ds.ValRatio)

	// Split
	train = documents[:trainEnd]
	val = documents[trainEnd:valEnd]
	test = documents[valEnd:]

	fmt.Printf("Split %d documents: train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%)\n",
		n, len(train), 100*ds.TrainRatio, len(val), 100*ds.ValRatio, len(test), 100*ds.TestRatio)

	return train, val, test
}

// TemporalSplit splits documents by time (for time-series data).
// Assumes documents are sorted chronologically.
func (ds *DataSplitter) TemporalSplit(documents []string) (train, val, test []string) {
	// No shuffling for temporal splits
	n := len(documents)
	trainEnd := int(float64(n) * ds.TrainRatio)
	valEnd := trainEnd + int(float64(n)*ds.ValRatio)

	train = documents[:trainEnd]
	val = documents[trainEnd:valEnd]
	test = documents[valEnd:]

	fmt.Printf("Temporal split: train (oldest %d), val (%d), test (newest %d)\n",
		len(train), len(val), len(test))

	return train, val, test
}
```

**Split Best Practices:**
- **Standard ratios**: 80% train, 10% validation, 10% test (or 90/5/5 for large datasets)
- **Stratified splitting**: For supervised tasks, ensure balanced labels across splits
- **Temporal splitting**: For time-series data, use chronological splits (no shuffling)
- **Validation usage**: Use validation for hyperparameter tuning, test for final evaluation only

---

## Data Loading and Batching

Efficient data loading is critical for GPU utilization and training speed.

### Efficient Data Loader

```go
package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"sync"
)

// DataLoader loads training batches efficiently.
type DataLoader struct {
	FilePath   string
	BatchSize  int
	SeqLen     int
	Shuffle    bool
	NumWorkers int

	// Internal state
	examples   []TrainingExample
	indices    []int
	currentIdx int
	mu         sync.Mutex
}

// NewDataLoader creates a new data loader.
func NewDataLoader(filePath string, batchSize, seqLen int, shuffle bool) *DataLoader {
	return &DataLoader{
		FilePath:  filePath,
		BatchSize: batchSize,
		SeqLen:    seqLen,
		Shuffle:   shuffle,
		examples:  make([]TrainingExample, 0),
		currentIdx: 0,
	}
}

// Load loads all training examples into memory.
// For very large datasets, use streaming instead.
func (dl *DataLoader) Load() error {
	file, err := os.Open(dl.FilePath)
	if err != nil {
		return fmt.Errorf("failed to open data file: %w", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	for scanner.Scan() {
		line := scanner.Text()

		// Parse training example from line
		// Format: length input_ids... target_ids...
		example := dl.parseExample(line)
		dl.examples = append(dl.examples, example)
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading data file: %w", err)
	}

	// Initialize indices
	dl.indices = make([]int, len(dl.examples))
	for i := range dl.indices {
		dl.indices[i] = i
	}

	if dl.Shuffle {
		rand.Shuffle(len(dl.indices), func(i, j int) {
			dl.indices[i], dl.indices[j] = dl.indices[j], dl.indices[i]
		})
	}

	fmt.Printf("Loaded %d training examples\n", len(dl.examples))
	return nil
}

// parseExample parses a training example from a text line.
func (dl *DataLoader) parseExample(line string) TrainingExample {
	// Simplified parser - production code would use binary format
	var length int
	fmt.Sscanf(line, "%d", &length)

	// Parse token IDs (simplified)
	tokens := make([]int, 0, length)
	// ... parse logic ...

	return TrainingExample{
		InputIDs:  tokens[:len(tokens)-1],
		TargetIDs: tokens[1:],
		Mask:      make([]bool, len(tokens)-1),
	}
}

// NextBatch returns the next batch of training examples.
func (dl *DataLoader) NextBatch() ([]TrainingExample, bool) {
	dl.mu.Lock()
	defer dl.mu.Unlock()

	// Check if we've exhausted the dataset
	if dl.currentIdx >= len(dl.indices) {
		return nil, false // End of epoch
	}

	// Determine batch size (handle remainder)
	batchSize := dl.BatchSize
	if dl.currentIdx+batchSize > len(dl.indices) {
		batchSize = len(dl.indices) - dl.currentIdx
	}

	// Collect batch
	batch := make([]TrainingExample, batchSize)
	for i := 0; i < batchSize; i++ {
		idx := dl.indices[dl.currentIdx+i]
		batch[i] = dl.examples[idx]
	}

	dl.currentIdx += batchSize

	return batch, true
}

// Reset resets the data loader for a new epoch.
func (dl *DataLoader) Reset() {
	dl.mu.Lock()
	defer dl.mu.Unlock()

	dl.currentIdx = 0

	if dl.Shuffle {
		rand.Shuffle(len(dl.indices), func(i, j int) {
			dl.indices[i], dl.indices[j] = dl.indices[j], dl.indices[i]
		})
	}
}

// NumBatches returns the total number of batches per epoch.
func (dl *DataLoader) NumBatches() int {
	return (len(dl.examples) + dl.BatchSize - 1) / dl.BatchSize
}
```

**Data Loading Optimizations:**
- **Memory mapping**: Use mmap for large datasets that don't fit in RAM
- **Prefetching**: Load next batch while current batch is training (overlap I/O and compute)
- **Binary format**: Use binary formats (TFRecord, HDF5) instead of text for faster loading
- **Distributed loading**: For multi-GPU training, shard data across workers

---

## Production Pipeline

Putting it all together: a complete data preparation pipeline.

### End-to-End Pipeline

```go
package main

import (
	"fmt"
	"os"
	"path/filepath"
)

// DataPipeline orchestrates the entire data preparation workflow.
type DataPipeline struct {
	// Components
	Collector    *DataCollector
	Cleaner      *TextCleaner
	QualityFilter *QualityFilter
	Deduplicator *MinHashDeduplicator
	Tokenizer    *BPETokenizer
	DatasetBuilder *DatasetBuilder
	Splitter     *DataSplitter

	// Configuration
	Config PipelineConfig
}

// PipelineConfig holds pipeline configuration.
type PipelineConfig struct {
	InputDir        string
	OutputDir       string
	VocabSize       int
	SeqLen          int
	Overlap         int
	Architecture    string
	TrainRatio      float64
	ValRatio        float64
	TestRatio       float64
	Seed            int
}

// NewDataPipeline creates a new data preparation pipeline.
func NewDataPipeline(config PipelineConfig) *DataPipeline {
	return &DataPipeline{
		Collector:      &DataCollector{OutputDir: config.OutputDir},
		Cleaner:        NewTextCleaner(),
		QualityFilter:  NewDefaultQualityFilter(),
		Deduplicator:   NewMinHashDeduplicator(128, 3, 0.85),
		Tokenizer:      &BPETokenizer{},
		DatasetBuilder: &DatasetBuilder{
			SeqLen:       config.SeqLen,
			Overlap:      config.Overlap,
			MinLength:    32,
			Architecture: config.Architecture,
			MaskProb:     0.15,
		},
		Splitter: &DataSplitter{
			TrainRatio: config.TrainRatio,
			ValRatio:   config.ValRatio,
			TestRatio:  config.TestRatio,
			Seed:       config.Seed,
			Shuffle:    true,
		},
		Config: config,
	}
}

// Run executes the full data preparation pipeline.
func (dp *DataPipeline) Run() error {
	fmt.Println("========================================")
	fmt.Println("Data Preparation Pipeline")
	fmt.Println("========================================")

	// Step 1: Collect raw data
	fmt.Println("\n[1/8] Collecting raw data...")
	files, err := dp.Collector.CollectFromLocal(filepath.Join(dp.Config.InputDir, "*.txt"))
	if err != nil {
		return fmt.Errorf("data collection failed: %w", err)
	}
	fmt.Printf("Collected %d files\n", len(files))

	// Step 2: Clean and normalize text
	fmt.Println("\n[2/8] Cleaning text...")
	var documents []string
	for _, file := range files {
		content, err := os.ReadFile(file)
		if err != nil {
			fmt.Printf("Warning: failed to read %s: %v\n", file, err)
			continue
		}

		cleaned := dp.Cleaner.Clean(string(content))
		cleaned = ValidateUTF8(cleaned)
		documents = append(documents, cleaned)
	}
	fmt.Printf("Cleaned %d documents\n", len(documents))

	// Step 3: Quality filtering
	fmt.Println("\n[3/8] Filtering low-quality documents...")
	documents = FilterCorpus(documents, dp.QualityFilter)

	// Step 4: Deduplication
	fmt.Println("\n[4/8] Deduplicating documents...")
	documents = ExactDeduplicate(documents)
	documents = dp.Deduplicator.Deduplicate(documents)

	// Step 5: Train tokenizer
	fmt.Println("\n[5/8] Training tokenizer...")
	dp.Tokenizer.Train(documents, dp.Config.VocabSize)
	dp.Tokenizer.SaveVocab(
		filepath.Join(dp.Config.OutputDir, "vocab.txt"),
		filepath.Join(dp.Config.OutputDir, "merges.txt"),
	)
	dp.DatasetBuilder.Tokenizer = dp.Tokenizer

	// Step 6: Split data
	fmt.Println("\n[6/8] Splitting data...")
	train, val, test := dp.Splitter.Split(documents)

	// Step 7: Build training datasets
	fmt.Println("\n[7/8] Building training datasets...")

	// Write documents to temporary files
	trainFile := filepath.Join(dp.Config.OutputDir, "train_raw.txt")
	valFile := filepath.Join(dp.Config.OutputDir, "val_raw.txt")
	testFile := filepath.Join(dp.Config.OutputDir, "test_raw.txt")

	dp.writeDocuments(train, trainFile)
	dp.writeDocuments(val, valFile)
	dp.writeDocuments(test, testFile)

	// Create training examples
	dp.DatasetBuilder.BuildDataset(
		[]string{trainFile},
		filepath.Join(dp.Config.OutputDir, "train.txt"),
	)
	dp.DatasetBuilder.BuildDataset(
		[]string{valFile},
		filepath.Join(dp.Config.OutputDir, "val.txt"),
	)
	dp.DatasetBuilder.BuildDataset(
		[]string{testFile},
		filepath.Join(dp.Config.OutputDir, "test.txt"),
	)

	// Step 8: Generate statistics
	fmt.Println("\n[8/8] Generating statistics...")
	dp.generateStatistics(train, val, test)

	fmt.Println("\n========================================")
	fmt.Println("Pipeline complete!")
	fmt.Println("========================================")

	return nil
}

// writeDocuments writes documents to a file (one per line).
func (dp *DataPipeline) writeDocuments(documents []string, outputPath string) error {
	file, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	defer writer.Flush()

	for _, doc := range documents {
		fmt.Fprintln(writer, doc)
	}

	return nil
}

// generateStatistics computes and prints dataset statistics.
func (dp *DataPipeline) generateStatistics(train, val, test []string) {
	fmt.Println("\nDataset Statistics:")
	fmt.Println("-------------------")

	// Document counts
	fmt.Printf("Documents:\n")
	fmt.Printf("  Train: %d\n", len(train))
	fmt.Printf("  Val:   %d\n", len(val))
	fmt.Printf("  Test:  %d\n", len(test))

	// Token counts (approximate)
	trainTokens := dp.countTokens(train)
	valTokens := dp.countTokens(val)
	testTokens := dp.countTokens(test)

	fmt.Printf("\nTokens (approximate):\n")
	fmt.Printf("  Train: %d (%.1fM)\n", trainTokens, float64(trainTokens)/1e6)
	fmt.Printf("  Val:   %d (%.1fM)\n", valTokens, float64(valTokens)/1e6)
	fmt.Printf("  Test:  %d (%.1fM)\n", testTokens, float64(testTokens)/1e6)

	// Vocabulary
	fmt.Printf("\nVocabulary:\n")
	fmt.Printf("  Size: %d tokens\n", len(dp.Tokenizer.Vocab))
	fmt.Printf("  Special tokens: %d\n", len(dp.Tokenizer.SpecialTokens))
}

// countTokens counts tokens in documents (approximate).
func (dp *DataPipeline) countTokens(documents []string) int {
	total := 0
	for _, doc := range documents {
		tokens := dp.Tokenizer.Encode(doc)
		total += len(tokens)
	}
	return total
}

// Example: Running the pipeline
func runPipelineExample() {
	config := PipelineConfig{
		InputDir:     "./raw_data",
		OutputDir:    "./processed_data",
		VocabSize:    10000,
		SeqLen:       512,
		Overlap:      64,
		Architecture: "gpt",
		TrainRatio:   0.8,
		ValRatio:     0.1,
		TestRatio:    0.1,
		Seed:         42,
	}

	pipeline := NewDataPipeline(config)

	if err := pipeline.Run(); err != nil {
		fmt.Printf("Pipeline failed: %v\n", err)
		os.Exit(1)
	}
}
```

---

## Summary

This guide covered the complete data preparation workflow for training transformer models:

1. **Data Collection**: Gathering data from multiple sources (public datasets, web crawls, domain-specific)
2. **Text Cleaning**: Removing noise, normalizing Unicode, handling encodings
3. **Tokenization**: BPE, WordPiece, character-level approaches and trade-offs
4. **Dataset Construction**: Creating training examples for GPT, BERT, T5 architectures
5. **Quality Filtering**: Heuristics and ML-based approaches to filter low-quality data
6. **Deduplication**: Exact and near-duplicate detection using MinHash
7. **Data Augmentation**: Techniques for increasing data diversity (use sparingly)
8. **Splitting**: Train/val/test splits with stratification and temporal considerations
9. **Data Loading**: Efficient batching and streaming for training
10. **Production Pipeline**: End-to-end orchestration and best practices

**Key Takeaways:**
- **Data quality > quantity**: Clean, deduplicated data trains better models faster
- **Architecture matters**: Different models (GPT/BERT/T5) need different data formats
- **Scale considerations**: Small datasets benefit from augmentation, large datasets from filtering
- **Reproducibility**: Always set random seeds and document data preparation steps
- **Iterate**: Data preparation is iterative - inspect samples, measure quality, refine pipeline

**Next Steps:**
- See `docs/training-workflows.md` for training with prepared data
- See `docs/optimization-guide.md` for training performance optimization
- See `docs/evaluation-guide.md` for evaluating trained models
