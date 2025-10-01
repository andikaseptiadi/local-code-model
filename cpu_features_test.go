//go:build linux && arm64

package main

import (
	"fmt"
	"testing"
)

func TestCPUFeatureDetection(t *testing.T) {
	features := DetectCPUFeatures()

	t.Logf("CPU Name: %s", GetCPUName())
	t.Logf("Has NEON: %v", features.HasNEON)
	t.Logf("Has SVE: %v", features.HasSVE)
	t.Logf("Has SVE2: %v", features.HasSVE2)

	graviton := GetGravitonGeneration()
	if graviton > 0 {
		t.Logf("Graviton Generation: %d", graviton)
	} else {
		t.Logf("Not running on AWS Graviton")
	}

	// NEON should always be available on ARM64
	if !features.HasNEON {
		t.Error("NEON should be available on all ARM64 processors")
	}
}

func TestGravitonDetection(t *testing.T) {
	generation := GetGravitonGeneration()
	cpuName := GetCPUName()

	t.Logf("CPU: %s", cpuName)
	t.Logf("Graviton Generation: %d", generation)

	// Print expected features based on generation
	switch generation {
	case 2:
		t.Log("Graviton2: NEON only (no SVE)")
	case 3:
		t.Log("Graviton3/3E: NEON + SVE (256-bit)")
	case 4:
		t.Log("Graviton4: NEON + SVE + SVE2 (up to 512-bit)")
	default:
		t.Log("Not Graviton (Apple Silicon or other ARM64)")
	}
}

func ExampleDetectCPUFeatures() {
	features := DetectCPUFeatures()
	fmt.Printf("NEON: %v, SVE: %v, SVE2: %v\n",
		features.HasNEON, features.HasSVE, features.HasSVE2)
}

func ExampleGetCPUName() {
	fmt.Println(GetCPUName())
}

// BenchmarkCPUDetection ensures detection is fast
func BenchmarkCPUDetection(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = DetectCPUFeatures()
	}
}
