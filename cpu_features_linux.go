//go:build linux && arm64 && !darwin

package main

import (
	"os"
	"strings"
)

// CPUFeatures represents available CPU SIMD features on Linux ARM64
type CPUFeatures struct {
	HasNEON bool
	HasSVE  bool
	HasSVE2 bool
}

// DetectCPUFeatures detects available SIMD features on Linux ARM64
func DetectCPUFeatures() CPUFeatures {
	features := CPUFeatures{
		HasNEON: true, // NEON is mandatory on ARM64
	}

	// Read /proc/cpuinfo to detect SVE/SVE2
	data, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return features
	}

	cpuinfo := string(data)

	// Look for "Features:" line which lists CPU capabilities
	for _, line := range strings.Split(cpuinfo, "\n") {
		if strings.HasPrefix(line, "Features") {
			// Check for SVE
			if strings.Contains(line, "sve") {
				features.HasSVE = true
			}
			// Check for SVE2
			if strings.Contains(line, "sve2") {
				features.HasSVE2 = true
			}
			break
		}
	}

	return features
}

// GetCPUName returns the CPU model name from /proc/cpuinfo
func GetCPUName() string {
	data, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return "Unknown ARM64"
	}

	cpuinfo := string(data)

	// Look for CPU implementer and part
	var implementer, part string

	for _, line := range strings.Split(cpuinfo, "\n") {
		if strings.HasPrefix(line, "CPU implementer") {
			implementer = strings.TrimSpace(strings.Split(line, ":")[1])
		} else if strings.HasPrefix(line, "CPU part") {
			part = strings.TrimSpace(strings.Split(line, ":")[1])
		}
	}

	// AWS Graviton detection
	if implementer == "0x41" { // ARM
		switch part {
		case "0xd0c": // Neoverse N1
			return "AWS Graviton2 (ARM Neoverse N1)"
		case "0xd40": // Neoverse V1
			return "AWS Graviton3/3E (ARM Neoverse V1)"
		case "0xd4f": // Neoverse V2
			return "AWS Graviton4 (ARM Neoverse V2)"
		}
	}

	// Fallback to generic description
	return "ARM64 CPU (implementer: " + implementer + ", part: " + part + ")"
}

// GetGravitonGeneration returns the Graviton generation (2, 3, or 4)
func GetGravitonGeneration() int {
	cpuName := GetCPUName()

	if strings.Contains(cpuName, "Graviton2") {
		return 2
	} else if strings.Contains(cpuName, "Graviton3") {
		return 3
	} else if strings.Contains(cpuName, "Graviton4") {
		return 4
	}

	return 0 // Not Graviton
}
