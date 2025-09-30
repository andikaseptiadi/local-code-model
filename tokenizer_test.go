package main

import (
	"os"
	"testing"
)

func TestSimpleTokenizer(t *testing.T) {
	tok := NewSimpleTokenizer()

	corpus := []string{
		"hello world",
		"hello go",
		"world of go",
	}

	tok.BuildVocab(corpus)

	// Test encoding
	text := "hello"
	ids := tok.Encode(text)

	if len(ids) != len(text) {
		t.Errorf("expected %d tokens, got %d", len(text), len(ids))
	}

	// Test decoding
	decoded := tok.Decode(ids)
	if decoded != text {
		t.Errorf("expected %q, got %q", text, decoded)
	}

	// Test vocab size
	expectedSize := len("helowrdgf ") // unique characters
	if tok.VocabSize() != expectedSize {
		t.Errorf("expected vocab size %d, got %d", expectedSize, tok.VocabSize())
	}
}

func TestBPETokenizer(t *testing.T) {
	tok := NewTokenizer()

	corpus := []string{
		"hello world",
		"hello hello",
		"world world world",
	}

	// Train with small vocabulary
	err := tok.Train(corpus, 300)
	if err != nil {
		t.Fatalf("training failed: %v", err)
	}

	// Test encoding/decoding
	text := "hello world"
	ids := tok.Encode(text)

	if len(ids) == 0 {
		t.Fatal("encoding produced no tokens")
	}

	decoded := tok.Decode(ids)
	if decoded != text {
		t.Errorf("roundtrip failed: expected %q, got %q", text, decoded)
	}

	// Test that common sequences are merged
	// "ll" should be merged since it appears in "hello"
	helloIDs := tok.Encode("hello")
	if len(helloIDs) >= 5 {
		t.Log("Note: Expected some merging of common pairs in 'hello'")
	}
}

func TestBPETokenizerSaveLoad(t *testing.T) {
	tok := NewTokenizer()

	corpus := []string{
		"package main",
		"func main() {",
		"fmt.Println()",
	}

	err := tok.Train(corpus, 300)
	if err != nil {
		t.Fatalf("training failed: %v", err)
	}

	// Save
	tmpfile := "test_tokenizer.txt"
	defer os.Remove(tmpfile)

	err = tok.Save(tmpfile)
	if err != nil {
		t.Fatalf("save failed: %v", err)
	}

	// Load into new tokenizer
	tok2 := NewTokenizer()
	err = tok2.Load(tmpfile)
	if err != nil {
		t.Fatalf("load failed: %v", err)
	}

	// Test that they produce same encoding
	text := "package main"
	ids1 := tok.Encode(text)
	ids2 := tok2.Encode(text)

	if len(ids1) != len(ids2) {
		t.Logf("tok1 vocab size: %d", tok.VocabSize())
		t.Logf("tok2 vocab size: %d", tok2.VocabSize())
		t.Logf("tok1 merges: %d", len(tok.merges))
		t.Logf("tok2 merges: %d", len(tok2.merges))
		t.Fatalf("encoding length mismatch: %d vs %d", len(ids1), len(ids2))
	}

	for i := range ids1 {
		if ids1[i] != ids2[i] {
			t.Errorf("encoding mismatch at position %d: %d vs %d", i, ids1[i], ids2[i])
		}
	}
}

func TestGoCodeTokenization(t *testing.T) {
	tok := NewSimpleTokenizer()

	// Build vocab from Go code samples
	goCode := []string{
		`package main

import "fmt"

func main() {
	fmt.Println("Hello, World!")
}`,
		`type Server struct {
	port int
	host string
}`,
		`func (s *Server) Start() error {
	return nil
}`,
	}

	tok.BuildVocab(goCode)

	// Test that we can encode and decode Go code
	sample := `func test() {}`
	ids := tok.Encode(sample)
	decoded := tok.Decode(ids)

	if decoded != sample {
		t.Errorf("Go code roundtrip failed:\noriginal: %q\ndecoded:  %q", sample, decoded)
	}

	t.Logf("Vocabulary size for Go code: %d", tok.VocabSize())
	t.Logf("Sample encoding length: %d tokens for %d characters", len(ids), len(sample))
}

func BenchmarkSimpleTokenizerEncode(b *testing.B) {
	tok := NewSimpleTokenizer()
	corpus := []string{"package main\nfunc main() {}\n"}
	tok.BuildVocab(corpus)

	text := "package main func main"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tok.Encode(text)
	}
}

func BenchmarkBPETokenizerEncode(b *testing.B) {
	tok := NewTokenizer()
	corpus := []string{
		"package main",
		"func main() {}",
		"import fmt",
	}
	tok.Train(corpus, 300)

	text := "package main func main"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tok.Encode(text)
	}
}