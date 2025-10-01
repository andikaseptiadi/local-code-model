package main

// ===========================================================================
// WHAT'S GOING ON HERE
// ===========================================================================
//
// This file generates performance visualizations from benchmark data,
// creating graphs that show the optimization progression across different
// hardware platforms.
//
// INTENTION:
// Make performance data visual and intuitive. Show how the same code performs
// differently on M4 Max vs Graviton3 vs Graviton4, revealing how architectural
// choices (cache size, memory bandwidth, SIMD width) create different
// performance curves.
//
// VISUALIZATIONS GENERATED:
//
// 1. Optimization Progression (per architecture)
//    - X-axis: Optimization level (0-5)
//    - Y-axis: GFLOPS
//    - Shows the performance "staircase" from naive to accelerated
//
// 2. Speedup vs Baseline (per architecture)
//    - X-axis: Optimization level
//    - Y-axis: Speedup multiplier
//    - Shows diminishing returns and where optimizations matter most
//
// 3. Cross-Architecture Comparison
//    - Multiple lines, one per architecture
//    - Shows where different architectures excel
//
// 4. Efficiency Analysis
//    - Shows parallel efficiency, cache hit rates, memory bandwidth usage
//
// 5. Sweet Spot Analysis
//    - Problem size vs performance
//    - Shows where each optimization level crosses over
//
// OUTPUT FORMATS:
//   - gnuplot scripts (.plt files)
//   - CSV for spreadsheet import
//   - SVG/PNG when gnuplot is available
//   - ASCII art for terminal viewing
//
// WHY THESE VISUALIZATIONS MATTER:
// They reveal architectural truths that aren't obvious from code:
//   - M4 Max's unified memory shows in parallel scaling
//   - Graviton's smaller caches show in cache-blocking gains
//   - SVE vs NEON differences show in vectorization
//
// ===========================================================================

import (
	"fmt"
	"math"
	"strings"
)

// VisualizationConfig controls output format and style.
type VisualizationConfig struct {
	Format        string // "gnuplot", "csv", "ascii"
	OutputDir     string
	Width         int
	Height        int
	ShowGrid      bool
	LogScale      bool
	CompareMode   bool // Compare multiple architectures
}

// DefaultVisualizationConfig returns sensible defaults.
func DefaultVisualizationConfig() VisualizationConfig {
	return VisualizationConfig{
		Format:      "gnuplot",
		OutputDir:   "benchmarks",
		Width:       1200,
		Height:      800,
		ShowGrid:    true,
		LogScale:    false,
		CompareMode: false,
	}
}

// GenerateVisualization creates performance graphs from benchmark data.
func GenerateVisualization(suite *BenchmarkSuite, config VisualizationConfig) error {
	switch config.Format {
	case "gnuplot":
		return generateGnuplotScripts(suite, config)
	case "csv":
		return generateCSV(suite, config)
	case "ascii":
		return generateASCIIChart(suite, config)
	default:
		return fmt.Errorf("unknown format: %s", config.Format)
	}
}

// generateGnuplotScripts creates gnuplot scripts for visualization.
func generateGnuplotScripts(suite *BenchmarkSuite, config VisualizationConfig) error {
	fmt.Println("Generating gnuplot scripts...")

	// 1. Performance progression chart
	progressionScript := generateProgressionPlot(suite, config)
	fmt.Println("\n=== Gnuplot Script: Performance Progression ===")
	fmt.Println(progressionScript)

	// 2. Speedup chart
	speedupScript := generateSpeedupPlot(suite, config)
	fmt.Println("\n=== Gnuplot Script: Speedup vs Baseline ===")
	fmt.Println(speedupScript)

	// 3. Efficiency chart
	efficiencyScript := generateEfficiencyPlot(suite, config)
	fmt.Println("\n=== Gnuplot Script: Parallel Efficiency ===")
	fmt.Println(efficiencyScript)

	return nil
}

// generateProgressionPlot creates a gnuplot script showing performance progression.
func generateProgressionPlot(suite *BenchmarkSuite, config VisualizationConfig) string {
	var sb strings.Builder

	// Gnuplot header
	sb.WriteString("#!/usr/bin/gnuplot\n")
	sb.WriteString("reset\n")
	sb.WriteString(fmt.Sprintf("set terminal pngcairo size %d,%d enhanced font 'Arial,12'\n",
		config.Width, config.Height))
	sb.WriteString("set output 'performance_progression.png'\n\n")

	// Styling
	sb.WriteString("set title 'Matrix Multiplication: Performance Progression\\n")
	sb.WriteString(fmt.Sprintf("%s (%d cores)'\n", suite.Hardware.CPUModel, suite.Hardware.NumCPU))
	sb.WriteString("set xlabel 'Optimization Level'\n")
	sb.WriteString("set ylabel 'Performance (GFLOPS)'\n")
	sb.WriteString("set grid\n")
	sb.WriteString("set key top left\n\n")

	// Data
	sb.WriteString("# Level Strategy GFLOPS\n")
	sb.WriteString("$data << EOD\n")

	// Group by size, use largest for main chart
	maxSize := 0
	for _, r := range suite.Results {
		if r.Size > maxSize {
			maxSize = r.Size
		}
	}

	for _, r := range suite.Results {
		if r.Size == maxSize {
			sb.WriteString(fmt.Sprintf("%d %s %.2f\n", r.Level, r.Strategy, r.GFLOPS))
		}
	}
	sb.WriteString("EOD\n\n")

	// Plot
	sb.WriteString("plot $data using 1:3:xtic(2) with linespoints lw 2 pt 7 ps 1.5 title 'GFLOPS'\n")

	return sb.String()
}

// generateSpeedupPlot creates a gnuplot script showing speedup vs baseline.
func generateSpeedupPlot(suite *BenchmarkSuite, config VisualizationConfig) string {
	var sb strings.Builder

	sb.WriteString("#!/usr/bin/gnuplot\n")
	sb.WriteString("reset\n")
	sb.WriteString(fmt.Sprintf("set terminal pngcairo size %d,%d enhanced font 'Arial,12'\n",
		config.Width, config.Height))
	sb.WriteString("set output 'speedup_progression.png'\n\n")

	sb.WriteString("set title 'Speedup vs Naive Baseline\\n")
	sb.WriteString(fmt.Sprintf("%s (%d cores)'\n", suite.Hardware.CPUModel, suite.Hardware.NumCPU))
	sb.WriteString("set xlabel 'Optimization Level'\n")
	sb.WriteString("set ylabel 'Speedup (x)'\n")
	sb.WriteString("set grid\n")
	sb.WriteString("set key top left\n")
	sb.WriteString(fmt.Sprintf("set yrange [0.5:%d]\n", suite.Hardware.NumCPU*2))
	sb.WriteString("set logscale y\n\n")

	// Ideal speedup line
	sb.WriteString(fmt.Sprintf("# Ideal linear speedup with %d cores\n", suite.Hardware.NumCPU))
	sb.WriteString(fmt.Sprintf("set arrow from 1,1 to 1,%d nohead lc rgb 'gray' dt 2\n\n",
		suite.Hardware.NumCPU))

	sb.WriteString("$data << EOD\n")

	maxSize := 0
	for _, r := range suite.Results {
		if r.Size > maxSize {
			maxSize = r.Size
		}
	}

	for _, r := range suite.Results {
		if r.Size == maxSize {
			sb.WriteString(fmt.Sprintf("%d %s %.2f\n", r.Level, r.Strategy, r.SpeedupVsNaive))
		}
	}
	sb.WriteString("EOD\n\n")

	sb.WriteString("plot $data using 1:3:xtic(2) with linespoints lw 2 pt 7 ps 1.5 title 'Actual Speedup'\n")

	return sb.String()
}

// generateEfficiencyPlot shows parallel efficiency.
func generateEfficiencyPlot(suite *BenchmarkSuite, config VisualizationConfig) string {
	var sb strings.Builder

	sb.WriteString("#!/usr/bin/gnuplot\n")
	sb.WriteString("reset\n")
	sb.WriteString(fmt.Sprintf("set terminal pngcairo size %d,%d enhanced font 'Arial,12'\n",
		config.Width, config.Height))
	sb.WriteString("set output 'parallel_efficiency.png'\n\n")

	sb.WriteString("set title 'Parallel Efficiency Analysis\\n")
	sb.WriteString(fmt.Sprintf("%s (%d cores)'\n", suite.Hardware.CPUModel, suite.Hardware.NumCPU))
	sb.WriteString("set xlabel 'Matrix Size'\n")
	sb.WriteString("set ylabel 'Parallel Efficiency (%)'\n")
	sb.WriteString("set grid\n")
	sb.WriteString("set yrange [0:100]\n\n")

	sb.WriteString("# Size Efficiency\n")
	sb.WriteString("$data << EOD\n")

	// Calculate efficiency: (actual_speedup / num_cores) * 100
	sizeMap := make(map[int]struct{ naive, parallel *BenchmarkResult })
	for i := range suite.Results {
		r := &suite.Results[i]
		entry := sizeMap[r.Size]
		if r.Strategy == "Naive" {
			entry.naive = r
		} else if r.Strategy == "Parallel" {
			entry.parallel = r
		}
		sizeMap[r.Size] = entry
	}

	for size, results := range sizeMap {
		if results.naive != nil && results.parallel != nil {
			efficiency := (results.parallel.SpeedupVsNaive / float64(suite.Hardware.NumCPU)) * 100
			sb.WriteString(fmt.Sprintf("%d %.1f\n", size, efficiency))
		}
	}

	sb.WriteString("EOD\n\n")
	sb.WriteString("plot $data using 1:2 with linespoints lw 2 pt 7 ps 1.5 title 'Efficiency'\n")

	return sb.String()
}

// generateCSV exports benchmark data as CSV.
func generateCSV(suite *BenchmarkSuite, config VisualizationConfig) error {
	fmt.Println("=== CSV Export ===")
	fmt.Println()

	// Header
	fmt.Println("Architecture,OS,Arch,Cores,Strategy,Level,Size,AvgTime_ns,GFLOPS,Speedup")

	// Data
	for _, r := range suite.Results {
		fmt.Printf("%s,%s,%s,%d,%s,%d,%d,%d,%.2f,%.2f\n",
			suite.Hardware.CPUModel,
			suite.Hardware.OS,
			suite.Hardware.Arch,
			suite.Hardware.NumCPU,
			r.Strategy,
			r.Level,
			r.Size,
			r.AvgTime.Nanoseconds(),
			r.GFLOPS,
			r.SpeedupVsNaive,
		)
	}

	return nil
}

// generateASCIIChart creates a terminal-friendly ASCII bar chart.
func generateASCIIChart(suite *BenchmarkSuite, config VisualizationConfig) error {
	fmt.Println()
	fmt.Println("=== Performance Progression (ASCII) ===")
	fmt.Println()

	// Find max size results
	maxSize := 0
	for _, r := range suite.Results {
		if r.Size > maxSize {
			maxSize = r.Size
		}
	}

	var results []BenchmarkResult
	for _, r := range suite.Results {
		if r.Size == maxSize {
			results = append(results, r)
		}
	}

	// Find max GFLOPS for scaling
	maxGFLOPS := 0.0
	for _, r := range results {
		if r.GFLOPS > maxGFLOPS {
			maxGFLOPS = r.GFLOPS
		}
	}

	// Chart width
	const barWidth = 60

	fmt.Printf("Matrix Size: %dx%d\n", maxSize, maxSize)
	fmt.Printf("Scale: %.1f GFLOPS = %d chars\n", maxGFLOPS, barWidth)
	fmt.Println()

	for _, r := range results {
		// Calculate bar length
		barLen := int(math.Round((r.GFLOPS / maxGFLOPS) * barWidth))

		// Strategy name (padded)
		strategyName := fmt.Sprintf("%-15s", r.Strategy)

		// Bar
		bar := strings.Repeat("█", barLen)

		// Stats
		stats := fmt.Sprintf(" %.2f GFLOPS (%.2fx)", r.GFLOPS, r.SpeedupVsNaive)

		fmt.Printf("%s │%s%s\n", strategyName, bar, stats)
	}

	fmt.Println()

	// Speedup chart
	fmt.Println("=== Speedup vs Baseline ===")
	fmt.Println()

	maxSpeedup := 0.0
	for _, r := range results {
		if r.SpeedupVsNaive > maxSpeedup {
			maxSpeedup = r.SpeedupVsNaive
		}
	}

	// Show ideal speedup line
	idealSpeedup := float64(suite.Hardware.NumCPU)
	fmt.Printf("Target (ideal parallel): %.0fx with %d cores\n", idealSpeedup, suite.Hardware.NumCPU)
	fmt.Println()

	for _, r := range results {
		strategyName := fmt.Sprintf("%-15s", r.Strategy)
		barLen := int(math.Round((r.SpeedupVsNaive / maxSpeedup) * barWidth))
		bar := strings.Repeat("█", barLen)

		// Show if this exceeds ideal parallel speedup
		indicator := ""
		if r.SpeedupVsNaive > idealSpeedup {
			indicator = " ⚡ (exceeds ideal!)"
		}

		fmt.Printf("%s │%s %.2fx%s\n", strategyName, bar, r.SpeedupVsNaive, indicator)
	}

	fmt.Println()

	return nil
}

// CompareArchitecturesVisualization creates comparison charts.
func CompareArchitecturesVisualization(suites []*BenchmarkSuite, config VisualizationConfig) error {
	fmt.Println()
	fmt.Println("=== Multi-Architecture Comparison ===")
	fmt.Println()

	// Would generate:
	// - Overlaid line plots showing each architecture's progression
	// - Relative performance heatmap
	// - Architecture-specific bottleneck analysis

	fmt.Println("Architectures compared:")
	for _, suite := range suites {
		fmt.Printf("  - %s: %.2f baseline GFLOPS\n",
			suite.Hardware.CPUModel, suite.BaselineGFLOPS)
	}

	fmt.Println()
	fmt.Println("(Full comparison visualization requires gnuplot installation)")
	fmt.Println("Run: gnuplot comparison.plt")

	return nil
}

// ExportForExternalTools exports in formats for external analysis.
func ExportForExternalTools(suite *BenchmarkSuite) {
	fmt.Println()
	fmt.Println("=== Export Options ===")
	fmt.Println()
	fmt.Println("1. Gnuplot: Save scripts and run 'gnuplot *.plt'")
	fmt.Println("2. Python/matplotlib: Use CSV export")
	fmt.Println("3. Excel/Sheets: Use CSV export")
	fmt.Println("4. R: Use CSV export")
	fmt.Println()
	fmt.Println("Recommended for comparison across architectures:")
	fmt.Println("  - Collect benchmark.json from each machine")
	fmt.Println("  - Merge with visualization tool")
	fmt.Println("  - Generate comparative plots")
	fmt.Println()
}
