//go:build darwin && arm64

package main

import (
	"os/exec"
	"strings"
)

// CPUFeatures represents available CPU SIMD features on macOS ARM64
type CPUFeatures struct {
	HasNEON bool
	HasSVE  bool
	HasSVE2 bool
}

// DetectCPUFeatures detects available SIMD features on macOS ARM64
func DetectCPUFeatures() CPUFeatures {
	return CPUFeatures{
		HasNEON: true,  // Apple Silicon always has NEON
		HasSVE:  false, // Apple Silicon doesn't support SVE
		HasSVE2: false, // Apple Silicon doesn't support SVE2
	}
}

// GetCPUName returns the CPU model name from sysctl
func GetCPUName() string {
	cmd := exec.Command("sysctl", "-n", "machdep.cpu.brand_string")
	output, err := cmd.Output()
	if err != nil {
		return "Apple Silicon"
	}

	name := strings.TrimSpace(string(output))
	if name == "" {
		return "Apple Silicon"
	}

	return name
}

// GetGravitonGeneration returns 0 on macOS (not Graviton)
func GetGravitonGeneration() int {
	return 0 // Not Graviton
}
