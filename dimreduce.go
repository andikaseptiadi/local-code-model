package main

import (
	"fmt"
	"math"
	"math/rand"
)

// ===========================================================================
// DIMENSIONALITY REDUCTION - PCA and t-SNE
// ===========================================================================
//
// WHAT'S GOING ON HERE:
// This file implements algorithms to reduce high-dimensional embeddings (like
// 64-dimensional or 128-dimensional token embeddings) down to 2D for visualization.
//
// WHY THIS MATTERS:
// - Token embeddings exist in high-dimensional space (e.g., 64D, 128D, 512D)
// - Humans can't visualize more than 3 dimensions
// - These algorithms preserve relationships while projecting to 2D
// - Lets us see which tokens are "close" to each other semantically
//
// TWO ALGORITHMS IMPLEMENTED:
// 1. PCA (Principal Component Analysis): Fast, linear, preserves global structure
// 2. t-SNE (t-Distributed Stochastic Neighbor Embedding): Slower, non-linear, preserves local structure
//
// EDUCATIONAL PHILOSOPHY:
// Like the rest of this codebase, we implement from first principles with
// clarity over performance. Real implementations would use optimized libraries.
//
// ===========================================================================

// ===========================================================================
// PCA (Principal Component Analysis)
// ===========================================================================
//
// INTUITION:
// Imagine you have points in 3D space that roughly form a flat pancake.
// PCA finds the 2D plane that best fits those points and projects them onto it.
//
// ALGORITHM:
// 1. Center the data (subtract mean)
// 2. Compute covariance matrix (how dimensions vary together)
// 3. Find eigenvectors (principal directions of variation)
// 4. Project data onto top 2 eigenvectors
//
// COMPLEXITY: O(n*d^2) where n=points, d=dimensions
// For our use case: ~100-1000 points in 64-128D -> very fast
//
// ===========================================================================

// PCA reduces high-dimensional data to 2D using Principal Component Analysis.
//
// Input: embeddings is an (n, d) tensor where n = number of points, d = embedding dimension
// Output: (n, 2) tensor with 2D coordinates
func PCA(embeddings *Tensor) (*Tensor, error) {
	if len(embeddings.shape) != 2 {
		return nil, fmt.Errorf("PCA expects 2D tensor, got shape %v", embeddings.shape)
	}

	n := embeddings.shape[0] // number of points
	d := embeddings.shape[1] // embedding dimension

	if n < 2 {
		return nil, fmt.Errorf("PCA requires at least 2 points, got %d", n)
	}

	// Step 1: Center the data (subtract mean from each dimension)
	centered := NewTensor(n, d)
	for j := 0; j < d; j++ {
		// Compute mean of this dimension
		mean := 0.0
		for i := 0; i < n; i++ {
			mean += embeddings.At(i, j)
		}
		mean /= float64(n)

		// Subtract mean
		for i := 0; i < n; i++ {
			centered.Set(embeddings.At(i, j)-mean, i, j)
		}
	}

	// Step 2: Compute covariance matrix (d x d)
	// Cov = (1/n) * X^T * X where X is centered data
	//
	// OPTIMIZATION NOTE: For large d, this is the bottleneck.
	// Production code would use LAPACK or similar optimized library.
	cov := NewTensor(d, d)
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			sum := 0.0
			for k := 0; k < n; k++ {
				sum += centered.At(k, i) * centered.At(k, j)
			}
			cov.Set(sum/float64(n), i, j)
		}
	}

	// Step 3: Find top 2 eigenvectors using power iteration
	// (simplified eigenvalue decomposition)
	//
	// EDUCATIONAL NOTE: Full eigendecomposition is complex. We use
	// power iteration which is simple and finds dominant eigenvectors.
	// Production code would use QR algorithm or similar.
	pc1 := powerIteration(cov, 100) // First principal component
	pc2 := powerIteration(deflate(cov, pc1), 100) // Second principal component

	// Step 4: Project centered data onto principal components
	result := NewTensor(n, 2)
	for i := 0; i < n; i++ {
		// Project onto first principal component
		proj1 := 0.0
		for j := 0; j < d; j++ {
			proj1 += centered.At(i, j) * pc1[j]
		}
		result.Set(proj1, i, 0)

		// Project onto second principal component
		proj2 := 0.0
		for j := 0; j < d; j++ {
			proj2 += centered.At(i, j) * pc2[j]
		}
		result.Set(proj2, i, 1)
	}

	return result, nil
}

// powerIteration finds the dominant eigenvector of a symmetric matrix.
//
// ALGORITHM:
// 1. Start with random vector
// 2. Repeatedly multiply by matrix and normalize
// 3. Converges to eigenvector with largest eigenvalue
//
// WHY IT WORKS: Repeated matrix multiplication amplifies the dominant direction.
func powerIteration(matrix *Tensor, iterations int) []float64 {
	d := matrix.shape[0]

	// Initialize with random vector
	v := make([]float64, d)
	for i := 0; i < d; i++ {
		v[i] = rand.NormFloat64()
	}

	// Normalize
	v = normalize(v)

	// Power iteration
	for iter := 0; iter < iterations; iter++ {
		// Multiply: v_new = matrix * v
		vNew := make([]float64, d)
		for i := 0; i < d; i++ {
			for j := 0; j < d; j++ {
				vNew[i] += matrix.At(i, j) * v[j]
			}
		}

		// Normalize
		vNew = normalize(vNew)
		v = vNew
	}

	return v
}

// deflate removes the component of a matrix in the direction of an eigenvector.
// This allows finding the next eigenvector.
//
// Formula: A_deflated = A - λ * v * v^T
// where λ is eigenvalue, v is eigenvector
func deflate(matrix *Tensor, eigenvector []float64) *Tensor {
	d := matrix.shape[0]

	// Compute eigenvalue: λ = v^T * A * v
	Av := make([]float64, d)
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			Av[i] += matrix.At(i, j) * eigenvector[j]
		}
	}

	eigenvalue := 0.0
	for i := 0; i < d; i++ {
		eigenvalue += eigenvector[i] * Av[i]
	}

	// Deflate: A - λ * v * v^T
	result := NewTensor(d, d)
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			result.Set(matrix.At(i, j)-eigenvalue*eigenvector[i]*eigenvector[j], i, j)
		}
	}

	return result
}

// normalize normalizes a vector to unit length.
func normalize(v []float64) []float64 {
	norm := 0.0
	for _, x := range v {
		norm += x * x
	}
	norm = math.Sqrt(norm)

	if norm < 1e-10 {
		// Avoid division by zero
		return v
	}

	result := make([]float64, len(v))
	for i := range v {
		result[i] = v[i] / norm
	}
	return result
}

// ===========================================================================
// t-SNE (t-Distributed Stochastic Neighbor Embedding)
// ===========================================================================
//
// INTUITION:
// t-SNE tries to preserve "neighborhoods" when projecting to 2D. If two points
// are close in high-dimensional space, they should be close in 2D. If they're
// far apart, they should be far apart in 2D.
//
// KEY INSIGHT:
// Uses probability distributions to measure "closeness":
// - In high-D: Gaussian similarity (nearby points have high probability)
// - In 2D: t-distribution similarity (heavy tails prevent crowding)
//
// ALGORITHM:
// 1. Compute pairwise similarities in high-D space
// 2. Initialize random 2D positions
// 3. Iteratively adjust 2D positions to match high-D similarities
//
// COMPLEXITY: O(n^2 * d) for naive implementation
// Our vocab sizes are small (~100-500 tokens) so this is acceptable.
//
// WHY t-DISTRIBUTION IN 2D?
// Heavy tails allow moderate distances in high-D to become larger distances
// in 2D, preventing all points from crowding into a ball.
//
// ===========================================================================

// TSNE reduces high-dimensional data to 2D using t-SNE.
//
// Input: embeddings is an (n, d) tensor
// Output: (n, 2) tensor with 2D coordinates
//
// Parameters:
// - perplexity: controls neighborhood size (typically 5-50)
// - iterations: number of gradient descent steps (typically 1000)
// - learningRate: step size for gradient descent (typically 100-500)
func TSNE(embeddings *Tensor, perplexity float64, iterations int, learningRate float64) (*Tensor, error) {
	if len(embeddings.shape) != 2 {
		return nil, fmt.Errorf("t-SNE expects 2D tensor, got shape %v", embeddings.shape)
	}

	n := embeddings.shape[0]

	if n < 2 {
		return nil, fmt.Errorf("t-SNE requires at least 2 points, got %d", n)
	}

	// Step 1: Compute pairwise affinities in high-dimensional space
	// P[i,j] = probability that i picks j as neighbor (Gaussian kernel)
	//
	// EDUCATIONAL NOTE: We use symmetric SNE (simplification).
	// Full t-SNE uses asymmetric then symmetrizes: P_ij = (P_j|i + P_i|j) / 2n
	P := computeAffinities(embeddings, perplexity)

	// Step 2: Initialize 2D embedding with small random values
	Y := NewTensor(n, 2)
	for i := 0; i < n; i++ {
		Y.Set(rand.NormFloat64()*0.0001, i, 0)
		Y.Set(rand.NormFloat64()*0.0001, i, 1)
	}

	// Step 3: Gradient descent to minimize KL divergence
	// KL(P || Q) = sum_i sum_j P_ij * log(P_ij / Q_ij)
	//
	// Gradient: dC/dY_i = 4 * sum_j (P_ij - Q_ij) * (Y_i - Y_j) * (1 + ||Y_i - Y_j||^2)^-1
	momentum := NewTensor(n, 2) // For momentum-based gradient descent

	for iter := 0; iter < iterations; iter++ {
		// Compute Q matrix (affinities in 2D space using t-distribution)
		Q := compute2DAffinities(Y)

		// Compute gradient
		grad := NewTensor(n, 2)
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				if i == j {
					continue
				}

				// Difference in 2D
				dy0 := Y.At(i, 0) - Y.At(j, 0)
				dy1 := Y.At(i, 1) - Y.At(j, 1)
				distSq := dy0*dy0 + dy1*dy1

				// Gradient contribution from this pair
				// (P_ij - Q_ij) * (Y_i - Y_j) / (1 + dist^2)
				pq := P.At(i, j) - Q.At(i, j)
				factor := pq / (1 + distSq)

				grad.Set(grad.At(i, 0)+4*factor*dy0, i, 0)
				grad.Set(grad.At(i, 1)+4*factor*dy1, i, 1)
			}
		}

		// Update with momentum (helps escape local minima)
		alpha := 0.5 // momentum coefficient
		if iter > 250 {
			alpha = 0.8 // increase momentum after initial phase
		}

		for i := 0; i < n; i++ {
			for j := 0; j < 2; j++ {
				// Momentum update
				m := alpha*momentum.At(i, j) - learningRate*grad.At(i, j)
				momentum.Set(m, i, j)

				// Position update
				Y.Set(Y.At(i, j)+m, i, j)
			}
		}

		// Early exaggeration: multiply P by 4 for first 100 iterations
		// Helps form tight clusters early on
		if iter == 100 {
			for i := 0; i < n; i++ {
				for j := 0; j < n; j++ {
					P.Set(P.At(i, j)/4, i, j)
				}
			}
		}
	}

	return Y, nil
}

// computeAffinities computes pairwise affinities in high-dimensional space.
//
// Uses Gaussian kernel: P_j|i = exp(-||x_i - x_j||^2 / 2σ_i^2)
// where σ_i is chosen to match target perplexity.
//
// PERPLEXITY: Roughly, the effective number of neighbors. Higher perplexity
// means each point considers more neighbors as "close".
func computeAffinities(X *Tensor, perplexity float64) *Tensor {
	n := X.shape[0]
	d := X.shape[1]

	// Compute pairwise squared distances
	distSq := NewTensor(n, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			sum := 0.0
			for k := 0; k < d; k++ {
				diff := X.At(i, k) - X.At(j, k)
				sum += diff * diff
			}
			distSq.Set(sum, i, j)
		}
	}

	// For each point, find σ that gives target perplexity
	// Perplexity = 2^H where H is Shannon entropy
	targetEntropy := math.Log(perplexity)

	P := NewTensor(n, n)

	for i := 0; i < n; i++ {
		// Binary search for σ_i
		sigmaMin, sigmaMax := 1e-10, 1e10
		sigma := 1.0

		for attempt := 0; attempt < 50; attempt++ {
			// Compute P_j|i with current sigma
			sum := 0.0
			for j := 0; j < n; j++ {
				if i != j {
					sum += math.Exp(-distSq.At(i, j) / (2 * sigma * sigma))
				}
			}

			// Normalize and compute entropy
			entropy := 0.0
			for j := 0; j < n; j++ {
				if i == j {
					continue
				}
				p := math.Exp(-distSq.At(i, j)/(2*sigma*sigma)) / sum
				if p > 1e-10 {
					entropy -= p * math.Log(p)
				}
			}

			// Adjust sigma based on entropy
			if math.Abs(entropy-targetEntropy) < 1e-5 {
				break
			}

			if entropy > targetEntropy {
				sigmaMax = sigma
				sigma = (sigmaMin + sigma) / 2
			} else {
				sigmaMin = sigma
				sigma = (sigma + sigmaMax) / 2
			}
		}

		// Store probabilities
		sum := 0.0
		for j := 0; j < n; j++ {
			if i != j {
				sum += math.Exp(-distSq.At(i, j) / (2 * sigma * sigma))
			}
		}
		for j := 0; j < n; j++ {
			if i != j {
				P.Set(math.Exp(-distSq.At(i, j)/(2*sigma*sigma))/sum, i, j)
			}
		}
	}

	// Symmetrize: P_ij = (P_j|i + P_i|j) / 2n
	// Multiply by 4 for early exaggeration (will be divided by 4 later)
	Psym := NewTensor(n, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			Psym.Set(4*(P.At(i, j)+P.At(j, i))/(2*float64(n)), i, j)
		}
	}

	return Psym
}

// compute2DAffinities computes pairwise affinities in 2D space using t-distribution.
//
// Q_ij = (1 + ||y_i - y_j||^2)^-1 / Z
// where Z is normalization constant
//
// WHY T-DISTRIBUTION (STUDENT-T WITH 1 DOF)?
// Heavy tails allow moderate distances to spread out, preventing crowding.
func compute2DAffinities(Y *Tensor) *Tensor {
	n := Y.shape[0]

	Q := NewTensor(n, n)
	sum := 0.0

	// Compute unnormalized Q and sum for normalization
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				continue
			}

			dy0 := Y.At(i, 0) - Y.At(j, 0)
			dy1 := Y.At(i, 1) - Y.At(j, 1)
			distSq := dy0*dy0 + dy1*dy1

			q := 1.0 / (1.0 + distSq)
			Q.Set(q, i, j)
			sum += q
		}
	}

	// Normalize
	if sum < 1e-10 {
		sum = 1e-10 // avoid division by zero
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			Q.Set(Q.At(i, j)/sum, i, j)
		}
	}

	// Ensure minimum value to avoid numerical issues
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if Q.At(i, j) < 1e-12 {
				Q.Set(1e-12, i, j)
			}
		}
	}

	return Q
}
