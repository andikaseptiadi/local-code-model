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

// SaveAttentionHTML generates an HTML visualization of attention weights.
//
// This creates an interactive heatmap showing which tokens attend to which other tokens.
// Each layer and head is visualized separately.
//
// Parameters:
// - filename: Output HTML file path
// - collector: AttentionCollector with captured weights
// - tokens: The input tokens (as strings) that were processed
// - seqLen: Sequence length (needed to reshape flattened matrices)
func SaveAttentionHTML(filename string, collector *AttentionCollector, tokens []string, seqLen int) error {
	if collector == nil || len(collector.Weights) == 0 {
		return fmt.Errorf("no attention weights to visualize")
	}

	// Generate JavaScript data for all layers and heads
	var jsData strings.Builder
	jsData.WriteString("const attentionData = {\n")
	jsData.WriteString("  layers: [\n")

	for layerIdx, layerWeights := range collector.Weights {
		jsData.WriteString(fmt.Sprintf("    { // Layer %d\n", layerIdx))
		jsData.WriteString("      heads: [\n")

		for headIdx, headWeights := range layerWeights {
			jsData.WriteString(fmt.Sprintf("        { // Head %d\n", headIdx))
			jsData.WriteString("          weights: ")
			jsData.WriteString(formatJSArrayFloat(headWeights))
			jsData.WriteString("\n        }")
			if headIdx < len(layerWeights)-1 {
				jsData.WriteString(",")
			}
			jsData.WriteString("\n")
		}

		jsData.WriteString("      ]\n    }")
		if layerIdx < len(collector.Weights)-1 {
			jsData.WriteString(",")
		}
		jsData.WriteString("\n")
	}

	jsData.WriteString("  ],\n")
	jsData.WriteString(fmt.Sprintf("  seqLen: %d,\n", seqLen))
	jsData.WriteString("  tokens: " + formatJSArrayString(tokens) + "\n")
	jsData.WriteString("};\n")

	// Generate HTML
	html := fmt.Sprintf(`<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Attention Visualization - Local Code Model</title>
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
            max-width: 1400px;
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
        .controls {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .control-group {
            display: inline-block;
            margin-right: 30px;
        }
        .control-label {
            font-size: 12px;
            color: #8b949e;
            display: block;
            margin-bottom: 5px;
        }
        select {
            background: #0d1117;
            color: #c9d1d9;
            border: 1px solid #30363d;
            border-radius: 4px;
            padding: 8px 12px;
            font-size: 14px;
            cursor: pointer;
        }
        select:hover {
            border-color: #58a6ff;
        }
        .heatmap-container {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .heatmap-title {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 15px;
            color: #c9d1d9;
        }
        canvas {
            width: 100%% !important;
            max-width: 800px;
            height: 600px !important;
            display: block;
            margin: 0 auto;
        }
        .explanation {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .explanation h3 {
            color: #58a6ff;
            margin-bottom: 10px;
            font-size: 16px;
        }
        .explanation p {
            color: #8b949e;
            font-size: 14px;
            line-height: 1.8;
        }
        .legend {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-top: 20px;
        }
        .legend-gradient {
            width: 300px;
            height: 20px;
            background: linear-gradient(to right, #0d1117, #58a6ff);
            border: 1px solid #30363d;
            border-radius: 4px;
        }
        .legend-labels {
            display: flex;
            justify-content: space-between;
            width: 300px;
            margin-top: 5px;
            font-size: 11px;
            color: #8b949e;
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
        <h1>Attention Weights Visualization</h1>
        <div class="subtitle">Local Code Model - Attention Pattern Analysis</div>

        <div class="explanation">
            <h3>Understanding Attention Heatmaps</h3>
            <p>
                This visualization shows which tokens (rows) attend to which other tokens (columns).
                Brighter colors indicate stronger attention weights. Each layer can learn different attention patterns:
                early layers often focus on local syntax, while deeper layers capture long-range dependencies.
            </p>
        </div>

        <div class="controls">
            <div class="control-group">
                <label class="control-label">Layer</label>
                <select id="layerSelect"></select>
            </div>
            <div class="control-group">
                <label class="control-label">Attention Head</label>
                <select id="headSelect"></select>
            </div>
        </div>

        <div class="heatmap-container">
            <div class="heatmap-title">Attention Heatmap - <span id="currentSelection"></span></div>
            <canvas id="heatmapCanvas"></canvas>
            <div class="legend">
                <div>
                    <div class="legend-gradient"></div>
                    <div class="legend-labels">
                        <span>Low</span>
                        <span>Attention Weight</span>
                        <span>High</span>
                    </div>
                </div>
            </div>
        </div>

        <div class="footer">
            Generated by Local Code Model | Pure Go Implementation
        </div>
    </div>

    <script>
        %s

        // Current selection
        let currentLayer = 0;
        let currentHead = 0;

        // Initialize dropdowns
        function initControls() {
            const layerSelect = document.getElementById('layerSelect');
            const headSelect = document.getElementById('headSelect');

            // Populate layer dropdown
            for (let i = 0; i < attentionData.layers.length; i++) {
                const option = document.createElement('option');
                option.value = i;
                option.textContent = 'Layer ' + i;
                layerSelect.appendChild(option);
            }

            // Populate head dropdown for first layer
            updateHeadDropdown();

            // Add event listeners
            layerSelect.addEventListener('change', (e) => {
                currentLayer = parseInt(e.target.value);
                currentHead = 0;
                updateHeadDropdown();
                drawHeatmap();
            });

            headSelect.addEventListener('change', (e) => {
                currentHead = parseInt(e.target.value);
                drawHeatmap();
            });
        }

        function updateHeadDropdown() {
            const headSelect = document.getElementById('headSelect');
            headSelect.innerHTML = '';

            const numHeads = attentionData.layers[currentLayer].heads.length;
            for (let i = 0; i < numHeads; i++) {
                const option = document.createElement('option');
                option.value = i;
                option.textContent = 'Head ' + i;
                headSelect.appendChild(option);
            }
            headSelect.value = currentHead;
        }

        // Draw attention heatmap
        function drawHeatmap() {
            const canvas = document.getElementById('heatmapCanvas');
            const ctx = canvas.getContext('2d');
            const dpr = window.devicePixelRatio || 1;

            // Update title
            document.getElementById('currentSelection').textContent =
                'Layer ' + currentLayer + ', Head ' + currentHead;

            // Set canvas size
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            ctx.scale(dpr, dpr);

            const width = rect.width;
            const height = rect.height;

            // Get attention weights for current layer/head
            const weights = attentionData.layers[currentLayer].heads[currentHead].weights;
            const seqLen = attentionData.seqLen;
            const tokens = attentionData.tokens;

            // Calculate cell size
            const padding = 80;
            const heatmapSize = Math.min(width - 2 * padding, height - 2 * padding);
            const cellSize = heatmapSize / seqLen;

            // Clear canvas
            ctx.fillStyle = '#0d1117';
            ctx.fillRect(0, 0, width, height);

            // Find min/max for normalization
            let minVal = weights[0];
            let maxVal = weights[0];
            for (let i = 0; i < weights.length; i++) {
                if (weights[i] < minVal) minVal = weights[i];
                if (weights[i] > maxVal) maxVal = weights[i];
            }
            const range = maxVal - minVal;

            // Draw heatmap cells
            const offsetX = (width - heatmapSize) / 2;
            const offsetY = (height - heatmapSize) / 2;

            for (let row = 0; row < seqLen; row++) {
                for (let col = 0; col < seqLen; col++) {
                    const idx = row * seqLen + col;
                    const val = weights[idx];
                    const normalized = range > 0 ? (val - minVal) / range : 0;

                    // Color: dark to blue gradient
                    const brightness = Math.floor(normalized * 255);
                    ctx.fillStyle = 'rgb(' + Math.floor(brightness * 0.35) + ', ' +
                                             Math.floor(brightness * 0.65) + ', ' +
                                             brightness + ')';

                    const x = offsetX + col * cellSize;
                    const y = offsetY + row * cellSize;
                    ctx.fillRect(x, y, cellSize, cellSize);
                }
            }

            // Draw grid lines
            ctx.strokeStyle = '#30363d';
            ctx.lineWidth = 0.5;
            for (let i = 0; i <= seqLen; i++) {
                // Vertical lines
                ctx.beginPath();
                ctx.moveTo(offsetX + i * cellSize, offsetY);
                ctx.lineTo(offsetX + i * cellSize, offsetY + heatmapSize);
                ctx.stroke();

                // Horizontal lines
                ctx.beginPath();
                ctx.moveTo(offsetX, offsetY + i * cellSize);
                ctx.lineTo(offsetX + heatmapSize, offsetY + i * cellSize);
                ctx.stroke();
            }

            // Draw token labels (if not too many)
            if (seqLen <= 32 && tokens && tokens.length === seqLen) {
                ctx.fillStyle = '#8b949e';
                ctx.font = '10px monospace';
                ctx.textAlign = 'center';

                for (let i = 0; i < seqLen; i++) {
                    // Truncate long tokens
                    let label = tokens[i];
                    if (label.length > 3) {
                        label = label.substring(0, 3);
                    }

                    // Column labels (top)
                    ctx.save();
                    ctx.translate(offsetX + i * cellSize + cellSize / 2, offsetY - 10);
                    ctx.rotate(-Math.PI / 4);
                    ctx.fillText(label, 0, 0);
                    ctx.restore();

                    // Row labels (left)
                    ctx.textAlign = 'right';
                    ctx.fillText(label, offsetX - 10, offsetY + i * cellSize + cellSize / 2 + 4);
                    ctx.textAlign = 'center';
                }
            }

            // Draw axis labels
            ctx.fillStyle = '#c9d1d9';
            ctx.font = '12px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Key Tokens (attended to)', width / 2, offsetY - 40);

            ctx.save();
            ctx.translate(offsetX - 50, height / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.fillText('Query Tokens (attending from)', 0, 0);
            ctx.restore();
        }

        // Initialize on page load
        window.onload = function() {
            initControls();
            drawHeatmap();
        };

        // Redraw on window resize
        window.onresize = function() {
            drawHeatmap();
        };
    </script>
</body>
</html>`, jsData.String())

	// Write to file
	return os.WriteFile(filename, []byte(html), 0644)
}

// SaveEmbeddingHTML generates an HTML visualization of token embeddings in 2D.
//
// The visualization shows a scatter plot where each point represents a token,
// and proximity indicates semantic similarity.
func SaveEmbeddingHTML(filename string, coords2D *Tensor, labels []string, method string) error {
	if coords2D == nil || len(coords2D.shape) != 2 || coords2D.shape[1] != 2 {
		return fmt.Errorf("coords2D must be an (n, 2) tensor")
	}

	n := coords2D.shape[0]
	if len(labels) != n {
		return fmt.Errorf("number of labels (%d) must match number of points (%d)", len(labels), n)
	}

	// Find bounds for scaling
	minX, maxX := coords2D.At(0, 0), coords2D.At(0, 0)
	minY, maxY := coords2D.At(0, 1), coords2D.At(0, 1)
	for i := 0; i < n; i++ {
		x, y := coords2D.At(i, 0), coords2D.At(i, 1)
		if x < minX {
			minX = x
		}
		if x > maxX {
			maxX = x
		}
		if y < minY {
			minY = y
		}
		if y > maxY {
			maxY = y
		}
	}

	// Generate JavaScript data
	var jsData strings.Builder
	jsData.WriteString("const embeddingData = {\n")
	jsData.WriteString(fmt.Sprintf("  method: \"%s\",\n", method))
	jsData.WriteString("  points: [\n")
	for i := 0; i < n; i++ {
		jsData.WriteString(fmt.Sprintf("    {x: %.6f, y: %.6f, label: %s}",
			coords2D.At(i, 0), coords2D.At(i, 1), jsonString(labels[i])))
		if i < n-1 {
			jsData.WriteString(",")
		}
		jsData.WriteString("\n")
	}
	jsData.WriteString("  ],\n")
	jsData.WriteString(fmt.Sprintf("  bounds: {minX: %.6f, maxX: %.6f, minY: %.6f, maxY: %.6f}\n", minX, maxX, minY, maxY))
	jsData.WriteString("};\n")

	// Generate HTML with embedded JavaScript
	html := fmt.Sprintf(`<!DOCTYPE html>
<html>
<head>
    <title>Token Embedding Visualization (%s)</title>
    <meta charset="utf-8">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }
        h1 {
            color: #333;
            margin-bottom: 5px;
        }
        .subtitle {
            color: #666;
            margin-bottom: 20px;
            font-size: 14px;
        }
        #canvas {
            background: white;
            border: 1px solid #ddd;
            cursor: crosshair;
            display: block;
            margin: 20px auto;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .controls {
            background: white;
            padding: 15px;
            border: 1px solid #ddd;
            margin: 20px auto;
            max-width: 800px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .info {
            font-size: 12px;
            color: #666;
            margin-top: 10px;
        }
        .legend {
            font-size: 12px;
            color: #333;
            margin-top: 15px;
        }
        label {
            margin-right: 10px;
            font-weight: 500;
        }
        input[type="checkbox"] {
            margin-right: 5px;
        }
    </style>
</head>
<body>
    <h1>Token Embedding Visualization</h1>
    <div class="subtitle">Method: %s | %d tokens visualized</div>

    <canvas id="canvas" width="800" height="600"></canvas>

    <div class="controls">
        <label>
            <input type="checkbox" id="showLabels" checked> Show labels
        </label>
        <label>
            <input type="checkbox" id="showGrid" checked> Show grid
        </label>
        <div class="legend">
            <strong>How to read this:</strong> Points that are close together represent tokens with similar embeddings.
            Hover over a point to see the token. %s preserves %s structure.
        </div>
        <div class="info" id="info">Hover over a point to see the token</div>
    </div>

    <script>
        %s

        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let hoveredPoint = null;

        // UI controls
        const showLabelsCheckbox = document.getElementById('showLabels');
        const showGridCheckbox = document.getElementById('showGrid');
        showLabelsCheckbox.addEventListener('change', drawScatter);
        showGridCheckbox.addEventListener('change', drawScatter);

        function drawScatter() {
            const showLabels = showLabelsCheckbox.checked;
            const showGrid = showGridCheckbox.checked;

            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Margins for axis labels
            const margin = 50;
            const plotWidth = canvas.width - 2 * margin;
            const plotHeight = canvas.height - 2 * margin;

            const data = embeddingData;
            const {minX, maxX, minY, maxY} = data.bounds;

            // Add padding to bounds
            const paddingX = (maxX - minX) * 0.1;
            const paddingY = (maxY - minY) * 0.1;
            const xMin = minX - paddingX;
            const xMax = maxX + paddingX;
            const yMin = minY - paddingY;
            const yMax = maxY + paddingY;

            // Coordinate transformation functions
            function toCanvasX(x) {
                return margin + ((x - xMin) / (xMax - xMin)) * plotWidth;
            }
            function toCanvasY(y) {
                return margin + ((yMax - y) / (yMax - yMin)) * plotHeight;
            }

            // Draw grid
            if (showGrid) {
                ctx.strokeStyle = '#e0e0e0';
                ctx.lineWidth = 1;
                for (let i = 0; i <= 10; i++) {
                    const x = margin + (i / 10) * plotWidth;
                    const y = margin + (i / 10) * plotHeight;

                    // Vertical lines
                    ctx.beginPath();
                    ctx.moveTo(x, margin);
                    ctx.lineTo(x, canvas.height - margin);
                    ctx.stroke();

                    // Horizontal lines
                    ctx.beginPath();
                    ctx.moveTo(margin, y);
                    ctx.lineTo(canvas.width - margin, y);
                    ctx.stroke();
                }
            }

            // Draw axes
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(margin, canvas.height - margin);
            ctx.lineTo(canvas.width - margin, canvas.height - margin);
            ctx.moveTo(margin, margin);
            ctx.lineTo(margin, canvas.height - margin);
            ctx.stroke();

            // Draw points
            data.points.forEach((point, idx) => {
                const cx = toCanvasX(point.x);
                const cy = toCanvasY(point.y);

                // Point
                ctx.fillStyle = hoveredPoint === idx ? '#ff6b6b' : '#4CAF50';
                ctx.beginPath();
                ctx.arc(cx, cy, hoveredPoint === idx ? 6 : 4, 0, 2 * Math.PI);
                ctx.fill();

                // Label (if enabled or hovered)
                if (showLabels || hoveredPoint === idx) {
                    ctx.fillStyle = '#333';
                    ctx.font = hoveredPoint === idx ? 'bold 12px monospace' : '10px monospace';
                    ctx.fillText(point.label, cx + 8, cy - 8);
                }
            });

            // Axis labels
            ctx.fillStyle = '#333';
            ctx.font = '12px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Principal Component 1', canvas.width / 2, canvas.height - 10);
            ctx.save();
            ctx.translate(15, canvas.height / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.fillText('Principal Component 2', 0, 0);
            ctx.restore();
        }

        // Mouse interaction
        canvas.addEventListener('mousemove', function(e) {
            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;

            const margin = 50;
            const plotWidth = canvas.width - 2 * margin;
            const plotHeight = canvas.height - 2 * margin;

            const data = embeddingData;
            const {minX, maxX, minY, maxY} = data.bounds;
            const paddingX = (maxX - minX) * 0.1;
            const paddingY = (maxY - minY) * 0.1;
            const xMin = minX - paddingX;
            const xMax = maxX + paddingX;
            const yMin = minY - paddingY;
            const yMax = maxY + paddingY;

            function toCanvasX(x) {
                return margin + ((x - xMin) / (xMax - xMin)) * plotWidth;
            }
            function toCanvasY(y) {
                return margin + ((yMax - y) / (yMax - yMin)) * plotHeight;
            }

            // Find nearest point
            let nearestIdx = null;
            let nearestDist = Infinity;
            data.points.forEach((point, idx) => {
                const cx = toCanvasX(point.x);
                const cy = toCanvasY(point.y);
                const dist = Math.sqrt((mouseX - cx) ** 2 + (mouseY - cy) ** 2);
                if (dist < 15 && dist < nearestDist) {
                    nearestIdx = idx;
                    nearestDist = dist;
                }
            });

            if (nearestIdx !== hoveredPoint) {
                hoveredPoint = nearestIdx;
                drawScatter();

                if (hoveredPoint !== null) {
                    const point = data.points[hoveredPoint];
                    document.getElementById('info').textContent =
                        'Token: "' + point.label + '" | Position: (' +
                        point.x.toFixed(3) + ', ' + point.y.toFixed(3) + ')';
                } else {
                    document.getElementById('info').textContent =
                        'Hover over a point to see the token';
                }
            }
        });

        // Initial draw
        drawScatter();

        // Redraw on window resize
        window.onresize = function() {
            drawScatter();
        };
    </script>
</body>
</html>`, method, method, n, method,
		map[string]string{"PCA": "global", "t-SNE": "local"}[method],
		jsData.String())

	// Write to file
	return os.WriteFile(filename, []byte(html), 0644)
}

// jsonString escapes a string for safe embedding in JavaScript/JSON
func jsonString(s string) string {
	// Simple escaping for common cases
	s = strings.ReplaceAll(s, "\\", "\\\\")
	s = strings.ReplaceAll(s, "\"", "\\\"")
	s = strings.ReplaceAll(s, "\n", "\\n")
	s = strings.ReplaceAll(s, "\r", "\\r")
	s = strings.ReplaceAll(s, "\t", "\\t")
	return "\"" + s + "\""
}

// formatJSArrayString formats a string slice as a JavaScript array
func formatJSArrayString(arr []string) string {
	if len(arr) == 0 {
		return "[]"
	}
	var sb strings.Builder
	sb.WriteString("[")
	for i, v := range arr {
		if i > 0 {
			sb.WriteString(",")
		}
		// Escape quotes and backslashes
		escaped := strings.ReplaceAll(v, "\\", "\\\\")
		escaped = strings.ReplaceAll(escaped, "\"", "\\\"")
		escaped = strings.ReplaceAll(escaped, "\n", "\\n")
		escaped = strings.ReplaceAll(escaped, "\r", "\\r")
		escaped = strings.ReplaceAll(escaped, "\t", "\\t")
		sb.WriteString("\"" + escaped + "\"")
	}
	sb.WriteString("]")
	return sb.String()
}
