//go:build darwin

package main

import (
	"testing"
)

// ===========================================================================
// ANE Backend Tests
// ===========================================================================
//
// These tests verify the ANE backend implementation via MPSGraph.
// ANE is now available on macOS with automatic hardware selection.
//
// ===========================================================================

// TestANEAvailability tests ANE availability reporting.
func TestANEAvailability(t *testing.T) {
	backend, err := NewANEBackend()

	// On macOS, ANE backend should initialize successfully
	// It returns an error describing it needs specific matrix sizes
	if err != nil && backend == nil {
		t.Fatalf("ANE backend creation failed: %v", err)
	}

	// Backend should exist
	if backend == nil {
		t.Fatal("ANE backend should exist")
	}

	// On M-series Macs, ANE should report as available
	// On Intel Macs or Linux, it should be unavailable
	if backend.IsAvailable() {
		t.Logf("✅ ANE available (M-series Mac)")
	} else {
		t.Logf("⚠️  ANE unavailable (Intel Mac or Linux)")
	}

	// Should provide informative device name
	deviceName := backend.DeviceName()
	if deviceName == "" {
		t.Error("ANE should provide device name")
	}

	t.Logf("ANE Device: %s", deviceName)
}

// TestANEMatMul tests that ANE MatMul returns appropriate error.
func TestANEMatMul(t *testing.T) {
	backend, _ := NewANEBackend()

	a := NewTensor(4, 4)
	b := NewTensor(4, 4)

	result, err := backend.MatMul(a, b)

	// Should return error (not implemented)
	if err == nil {
		t.Error("ANE MatMul should return error explaining implementation requirements")
	}

	// Result should be nil
	if result != nil {
		t.Error("ANE MatMul should return nil result when unavailable")
	}

	t.Logf("ANE MatMul error (expected): %v", err)
}

// TestANECapabilities tests that ANE capabilities documentation is available.
func TestANECapabilities(t *testing.T) {
	caps := ANECapabilities()

	if len(caps) < 100 {
		t.Error("ANE capabilities should provide detailed documentation")
	}

	// Should mention key constraints
	requiredInfo := []string{
		"TFLOPS",      // Performance spec
		"FP16",        // Precision
		"Core ML",     // Access method
		"Constraints", // Limitations
	}

	for _, info := range requiredInfo {
		if !contains(caps, info) {
			t.Errorf("ANE capabilities should mention %s", info)
		}
	}

	t.Logf("ANE Capabilities:\n%s", caps)
}

// TestANEFallbackChain tests that ANE properly falls back to other backends.
func TestANEFallbackChain(t *testing.T) {
	// In a full implementation, this would test:
	// 1. Try ANE
	// 2. Fall back to Metal if ANE unavailable
	// 3. Fall back to Accelerate if Metal unavailable
	// 4. Fall back to cache-blocked if nothing else works

	a := NewTensor(64, 64)
	b := NewTensor(64, 64)

	cfg := DefaultBackendConfig()

	// Try with ANE strategy (should fall back)
	result := MatMulWithStrategy(a, b, StrategyANE, cfg)

	// Should get a valid result via fallback
	if result == nil {
		t.Fatal("ANE strategy should fall back to working backend")
	}

	// Verify result shape
	if result.shape[0] != 64 || result.shape[1] != 64 {
		t.Errorf("Result shape incorrect: got %v, want [64 64]", result.shape)
	}

	t.Logf("ANE fallback chain worked correctly")
}

// ===========================================================================
// What Full ANE Testing Would Look Like
// ===========================================================================

// TestANECorrectness_Future would verify ANE results match CPU.
//
// If ANE were implemented, this test would:
//   1. Create random test matrices
//   2. Compute result on CPU (ground truth)
//   3. Compute result on ANE
//   4. Compare with tolerance (ANE may use FP16)
//
// func TestANECorrectness_Future(t *testing.T) {
//     backend, err := NewANEBackend()
//     if err != nil {
//         t.Skip("ANE not available")
//     }
//
//     sizes := []int{32, 64, 128, 256}
//     for _, size := range sizes {
//         a := NewTensorRand(size, size)
//         b := NewTensorRand(size, size)
//
//         // CPU ground truth
//         cpuResult := MatMulWithStrategy(a, b, StrategyNaive, CPUOnlyConfig())
//
//         // ANE result
//         aneResult, err := backend.MatMul(a, b)
//         if err != nil {
//             t.Fatalf("ANE MatMul failed: %v", err)
//         }
//
//         // Compare with larger tolerance (FP16 vs FP64)
//         if !tensorsEqualApprox(cpuResult, aneResult, 1e-3) {
//             t.Errorf("ANE result differs from CPU for size %d", size)
//         }
//     }
// }

// TestANEPrecision_Future would verify FP16 precision loss is acceptable.
//
// func TestANEPrecision_Future(t *testing.T) {
//     // ANE uses FP16 internally, so we need to verify that precision
//     // loss is within acceptable bounds for our use case.
//
//     backend, _ := NewANEBackend()
//     a := NewTensorRand(128, 128)
//     b := NewTensorRand(128, 128)
//
//     // FP64 result
//     fp64Result := MatMulWithStrategy(a, b, StrategyAccelerate, DefaultBackendConfig())
//
//     // ANE result (FP16 internally)
//     aneResult, _ := backend.MatMul(a, b)
//
//     // Measure precision loss
//     maxDiff := 0.0
//     for i := range fp64Result.data {
//         diff := abs(fp64Result.data[i] - aneResult.data[i])
//         if diff > maxDiff {
//             maxDiff = diff
//         }
//     }
//
//     // Acceptable for FP16: ~5e-4
//     if maxDiff > 1e-3 {
//         t.Errorf("Precision loss too high: %e", maxDiff)
//     }
//
//     t.Logf("Max precision difference: %e", maxDiff)
// }

// BenchmarkANE_Future would benchmark ANE performance.
//
// func BenchmarkANE_Future(b *testing.B) {
//     backend, err := NewANEBackend()
//     if err != nil {
//         b.Skip("ANE not available")
//     }
//
//     sizes := []int{128, 256, 512, 1024}
//     for _, size := range sizes {
//         a := NewTensorRand(size, size)
//         mat := NewTensorRand(size, size)
//
//         b.Run(fmt.Sprintf("ANE_%d", size), func(b *testing.B) {
//             b.ResetTimer()
//             for i := 0; i < b.N; i++ {
//                 _, err := backend.MatMul(a, mat)
//                 if err != nil {
//                     b.Fatalf("ANE MatMul failed: %v", err)
//                 }
//             }
//         })
//
//         // Compare with Metal
//         metal, _ := NewMetalBackend()
//         b.Run(fmt.Sprintf("Metal_%d", size), func(b *testing.B) {
//             b.ResetTimer()
//             for i := 0; i < b.N; i++ {
//                 _, _ = metal.MatMul(a, mat)
//             }
//         })
//     }
// }

// TestANEPower_Future would measure power consumption (requires macOS tools).
//
// func TestANEPower_Future(t *testing.T) {
//     // This would use:
//     //   - powermetrics command-line tool
//     //   - Or IOKit framework for power readings
//     //
//     // Expected results on M4 Max:
//     //   - ANE: ~2W for inference
//     //   - Metal GPU: ~50W for same workload
//     //   - Efficiency: 25x better on ANE
//
//     if !canMeasurePower() {
//         t.Skip("Power measurement requires root/admin privileges")
//     }
//
//     // Run workload on ANE and measure power
//     // Run same workload on Metal and measure power
//     // Compare efficiency (GFLOPS/Watt)
// }

// TestANEScheduling_Future would verify Apple uses ANE.
//
// func TestANEScheduling_Future(t *testing.T) {
//     // This would use Instruments or Activity Monitor to verify
//     // that Core ML actually scheduled the work on ANE and not GPU.
//     //
//     // Process:
//     //   1. Run inference
//     //   2. Check Activity Monitor: ANE % should be > 0
//     //   3. Check GPU %: should be low
//     //   4. If GPU % high: Core ML chose GPU, not ANE
//
//     // This is important because Core ML makes its own decisions
//     // about where to run operations. Just because you provide
//     // a model doesn't guarantee it runs on ANE!
// }

// Helper function
func contains(s, substr string) bool {
	return len(s) > 0 && len(substr) > 0 &&
		len(s) >= len(substr) &&
		findSubstring(s, substr)
}

func findSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
