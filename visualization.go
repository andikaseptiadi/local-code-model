package main

/*
WHAT'S GOING ON HERE?

This file implements visualization tools for understanding model training and behavior.
We prioritize simplicity and educational value over fancy graphics.

KEY CONCEPTS:
- TrainingMetrics: Track loss, learning rate, and other metrics during training
- HTML-based visualizations: Self-contained files that open in any browser
- No external plotting libraries: Everything generated from pure Go

VISUALIZATION TYPES:
1. Loss curves: Show how loss decreases during training
2. Learning rate schedule: Visualize warmup → constant → decay
3. Attention heatmaps: (future) Show which tokens attend to which
4. Embedding plots: (future) 2D visualization of token embeddings

WHY HTML?
- Works everywhere (just open in browser)
- Self-contained (no server needed)
- Can include interactive elements with pure JavaScript
- Easy to share and archive training runs
*/

import (
	"fmt"
	"math"
	"os"
	"strings"
)

// TrainingMetrics stores metrics collected during training
type TrainingMetrics struct {
	Steps          []int     // Step numbers
	Losses         []float64 // Loss at each step
	LearningRates  []float64 // Learning rate at each step
	Epochs         []int     // Epoch number for each step
	BatchIndices   []int     // Batch index within epoch
}

// NewTrainingMetrics creates a new metrics tracker
func NewTrainingMetrics() *TrainingMetrics {
	return &TrainingMetrics{
		Steps:         make([]int, 0),
		Losses:        make([]float64, 0),
		LearningRates: make([]float64, 0),
		Epochs:        make([]int, 0),
		BatchIndices:  make([]int, 0),
	}
}

// Record adds a new data point to the metrics
func (m *TrainingMetrics) Record(step int, loss float64, lr float64, epoch int, batchIdx int) {
	m.Steps = append(m.Steps, step)
	m.Losses = append(m.Losses, loss)
	m.LearningRates = append(m.LearningRates, lr)
	m.Epochs = append(m.Epochs, epoch)
	m.BatchIndices = append(m.BatchIndices, batchIdx)
}

// SaveHTML saves training metrics as an interactive HTML file
//
// This creates a self-contained HTML file with:
// - Loss curve chart
// - Learning rate schedule chart
// - Summary statistics
//
// The chart uses basic HTML/CSS/JS (no external dependencies)
func (m *TrainingMetrics) SaveHTML(filename string) error {
	if len(m.Steps) == 0 {
		return fmt.Errorf("no metrics to save")
	}

	// Compute summary statistics
	finalLoss := m.Losses[len(m.Losses)-1]
	minLoss := m.Losses[0]
	maxLoss := m.Losses[0]
	for _, loss := range m.Losses {
		if loss < minLoss {
			minLoss = loss
		}
		if loss > maxLoss {
			maxLoss = loss
		}
	}
	avgLoss := 0.0
	for _, loss := range m.Losses {
		avgLoss += loss
	}
	avgLoss /= float64(len(m.Losses))

	// Generate HTML
	html := fmt.Sprintf(`<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Training Metrics - Local Code Model</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            padding: 20px;
            line-height: 1.6;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            font-size: 28px;
            margin-bottom: 10px;
            color: #58a6ff;
        }
        .subtitle {
            color: #8b949e;
            margin-bottom: 30px;
            font-size: 14px;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 15px;
        }
        .stat-label {
            font-size: 12px;
            color: #8b949e;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }
        .stat-value {
            font-size: 24px;
            font-weight: 600;
            color: #58a6ff;
        }
        .chart-container {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .chart-title {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 15px;
            color: #c9d1d9;
        }
        canvas {
            width: 100%% !important;
            height: 300px !important;
        }
        .footer {
            text-align: center;
            color: #8b949e;
            font-size: 12px;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #30363d;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Training Metrics</h1>
        <div class="subtitle">Local Code Model - Generated from training run</div>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Total Steps</div>
                <div class="stat-value">%d</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Final Loss</div>
                <div class="stat-value">%.4f</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Min Loss</div>
                <div class="stat-value">%.4f</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Average Loss</div>
                <div class="stat-value">%.4f</div>
            </div>
        </div>

        <div class="chart-container">
            <div class="chart-title">Loss Curve</div>
            <canvas id="lossChart"></canvas>
        </div>

        <div class="chart-container">
            <div class="chart-title">Learning Rate Schedule</div>
            <canvas id="lrChart"></canvas>
        </div>

        <div class="footer">
            Generated by Local Code Model | Pure Go Implementation
        </div>
    </div>

    <script>
        // Data from Go
        const steps = %s;
        const losses = %s;
        const learningRates = %s;

        // Simple chart rendering function
        function drawChart(canvasId, data, color, yLabel) {
            const canvas = document.getElementById(canvasId);
            const ctx = canvas.getContext('2d');
            const dpr = window.devicePixelRatio || 1;

            // Set canvas size accounting for device pixel ratio
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            ctx.scale(dpr, dpr);

            const width = rect.width;
            const height = rect.height;
            const padding = 50;
            const chartWidth = width - 2 * padding;
            const chartHeight = height - 2 * padding;

            // Find data range
            const minVal = Math.min(...data);
            const maxVal = Math.max(...data);
            const range = maxVal - minVal;
            const minStep = Math.min(...steps);
            const maxStep = Math.max(...steps);
            const stepRange = maxStep - minStep;

            // Draw axes
            ctx.strokeStyle = '#30363d';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(padding, padding);
            ctx.lineTo(padding, height - padding);
            ctx.lineTo(width - padding, height - padding);
            ctx.stroke();

            // Draw grid lines
            ctx.strokeStyle = '#21262d';
            ctx.lineWidth = 1;
            for (let i = 1; i < 5; i++) {
                const y = padding + (chartHeight * i / 5);
                ctx.beginPath();
                ctx.moveTo(padding, y);
                ctx.lineTo(width - padding, y);
                ctx.stroke();

                // Y-axis labels
                const val = maxVal - (range * i / 5);
                ctx.fillStyle = '#8b949e';
                ctx.font = '11px monospace';
                ctx.textAlign = 'right';
                ctx.fillText(val.toFixed(4), padding - 10, y + 4);
            }

            // Draw line chart
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            for (let i = 0; i < data.length; i++) {
                const x = padding + (chartWidth * (steps[i] - minStep) / stepRange);
                const y = height - padding - (chartHeight * (data[i] - minVal) / range);
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.stroke();

            // X-axis labels
            ctx.fillStyle = '#8b949e';
            ctx.font = '11px monospace';
            ctx.textAlign = 'center';
            for (let i = 0; i <= 4; i++) {
                const step = minStep + (stepRange * i / 4);
                const x = padding + (chartWidth * i / 4);
                ctx.fillText(Math.round(step).toString(), x, height - padding + 20);
            }

            // Axis labels
            ctx.fillStyle = '#c9d1d9';
            ctx.font = '12px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Training Step', width / 2, height - 10);

            ctx.save();
            ctx.translate(15, height / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.fillText(yLabel, 0, 0);
            ctx.restore();
        }

        // Draw charts when page loads
        window.onload = function() {
            drawChart('lossChart', losses, '#58a6ff', 'Loss');
            drawChart('lrChart', learningRates, '#56d364', 'Learning Rate');
        };

        // Redraw on window resize
        window.onresize = function() {
            drawChart('lossChart', losses, '#58a6ff', 'Loss');
            drawChart('lrChart', learningRates, '#56d364', 'Learning Rate');
        };
    </script>
</body>
</html>`, len(m.Steps), finalLoss, minLoss, avgLoss,
		formatJSArray(m.Steps),
		formatJSArrayFloat(m.Losses),
		formatJSArrayFloat(m.LearningRates))

	// Write to file
	return os.WriteFile(filename, []byte(html), 0644)
}

// formatJSArray formats an int slice as a JavaScript array
func formatJSArray(arr []int) string {
	if len(arr) == 0 {
		return "[]"
	}
	var sb strings.Builder
	sb.WriteString("[")
	for i, v := range arr {
		if i > 0 {
			sb.WriteString(",")
		}
		sb.WriteString(fmt.Sprintf("%d", v))
	}
	sb.WriteString("]")
	return sb.String()
}

// formatJSArrayFloat formats a float64 slice as a JavaScript array
func formatJSArrayFloat(arr []float64) string {
	if len(arr) == 0 {
		return "[]"
	}
	var sb strings.Builder
	sb.WriteString("[")
	for i, v := range arr {
		if i > 0 {
			sb.WriteString(",")
		}
		// Handle NaN and Inf
		if math.IsNaN(v) {
			sb.WriteString("null")
		} else if math.IsInf(v, 1) {
			sb.WriteString("1e308")
		} else if math.IsInf(v, -1) {
			sb.WriteString("-1e308")
		} else {
			sb.WriteString(fmt.Sprintf("%.6f", v))
		}
	}
	sb.WriteString("]")
	return sb.String()
}
