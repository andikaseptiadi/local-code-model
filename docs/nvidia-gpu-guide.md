# NVIDIA GPU Selection Guide for ML Workloads

This guide helps you choose the right NVIDIA GPU for your specific machine learning workload, with current pricing (October 2025) and performance characteristics for transformer training and inference.

## Table of Contents

1. [GPU Lineup Overview](#gpu-lineup-overview)
2. [Training Workloads](#training-workloads)
3. [Inference Workloads](#inference-workloads)
4. [Cloud Pricing Comparison](#cloud-pricing-comparison)
5. [Decision Framework](#decision-framework)
6. [Hardware Specifications](#hardware-specifications)

---

## GPU Lineup Overview

NVIDIA's data center GPU lineup for AI/ML in 2025 spans from cost-effective inference accelerators to frontier-scale training GPUs:

| GPU Family | Generation | Target Use Case | Key Advantage |
|------------|-----------|-----------------|---------------|
| **L4** | Ada Lovelace | Cost-effective inference | Lowest cost, 72W TDP |
| **L40S** | Ada Lovelace | Inference + graphics | Balanced price/performance |
| **A10** | Ampere | Entry-level AI | Good for small models |
| **A100** | Ampere | Mainstream training | Mature ecosystem, wide availability |
| **H100** | Hopper | Large-scale training | Widely available, cost-effective |
| **H200** | Hopper | Extended context | 2× memory vs H100 |
| **B100** | Blackwell | Air-cooled training | Drop-in H100 replacement |
| **B200** | Blackwell | Frontier models | Next-gen performance |

**Key Insight:** Your choice should balance three factors:
1. **Model size** (parameters)
2. **Context length** (sequence length)
3. **Budget** (cost per hour or hardware purchase)

---

## Training Workloads

### Small Models (<100M parameters)

**Best Choice: L40S or A10**

- **L40S Pricing:** $1.95/hr (Modal), $0.80/hr (AWS 3-year reserved)
- **A10 Pricing:** $0.80-1.20/hr on major clouds
- **Hardware Cost:** L40S $7,500, A10 $3,000-4,000

**Why?**
- Small models don't benefit from Tensor Core performance of H100/A100
- Memory bandwidth is sufficient (L40S: 864 GB/s, A10: 600 GB/s)
- Lower TDP (L40S: 350W, A10: 150W) = lower operating costs
- Can train 30-50M parameter models faster than CPU

**Example Workloads:**
- 6-layer GPT (30M params): 2-3 hours on L40S
- BERT-Base (110M params): 4-6 hours on A10
- Fine-tuning pre-trained models <100M params

**Go Integration Pattern:**
```go
// Recommended: Use cuBLAS for basic GEMM operations
// No need for advanced Tensor Core optimizations
device := cuda.NewDevice(0)
result := device.MatMul(A, B)  // Uses cuBLAS SGEMM
```

---

### Medium Models (100M-10B parameters)

**Best Choice: A100 (40GB/80GB)**

- **A100 40GB Pricing:** ~$2.50-3.00/hr on specialized clouds
- **A100 80GB Pricing:** ~$3.50-4.00/hr on specialized clouds
- **AWS/GCP/Azure:** $4-6/hr after 2025 price cuts (44% reduction)

**Why?**
- HBM2e memory (40GB/80GB) fits most models up to 7B parameters
- 312 TFLOPS FP16 Tensor Core performance
- Mature software stack (CUDA 11+, well-optimized libraries)
- MIG (Multi-Instance GPU) support for multi-tenancy
- Best price/performance for mainstream models

**Performance Metrics:**
- **GPT-2 (1.5B params):** ~12 hours for 100K steps (batch size 32)
- **GPT-J (6B params):** ~3-4 days for full pre-training
- **LLaMA-7B fine-tuning:** ~6-12 hours depending on dataset

**Use Cases:**
- Pre-training models 1-7B parameters
- Fine-tuning up to 13B parameters (with mixed precision)
- Research and prototyping
- Small team production workloads

**Memory Guidance:**
- **40GB:** Models up to 3B params (training), 7B params (inference)
- **80GB:** Models up to 7B params (training), 13B params (inference)

---

### Large Models (10B-100B parameters)

**Best Choice: H100 (80GB)**

- **H100 80GB Pricing:**
  - Specialized clouds: $1.90-3.50/hr (after 2025 price reductions)
  - AWS P5 instances: $12.29/hr per GPU (8-GPU instances)
  - GCP: $11.06/hr per GPU
  - Azure: $6.98/hr per GPU

**Why?**
- 3× faster than A100 for transformer training
- 989 TFLOPS FP16 Tensor Core performance (3.1× A100)
- 80GB HBM3 memory with 3.35 TB/s bandwidth (2.2× A100)
- Transformer Engine for automatic FP8 mixed precision
- 4th-gen NVLink (900 GB/s for multi-GPU)

**Performance Metrics:**
- **LLaMA-13B:** ~2-3 days for full pre-training (8× H100)
- **LLaMA-65B:** ~1-2 weeks for full pre-training (64× H100)
- **GPT-3 scale (175B):** 3-4 weeks on 256-512× H100

**Use Cases:**
- Pre-training models 10-70B parameters
- Fine-tuning models 70-100B parameters
- Production training pipelines
- Multi-GPU distributed training

**When to Upgrade from A100:**
- Training time >48 hours (3× speedup pays for itself)
- Models >13B parameters (memory becomes constraint)
- Need for FP8 precision (2× speedup over FP16)

---

### Large Models with Extended Context (100B+ parameters, long sequences)

**Best Choice: H200 (141GB)**

- **H200 Pricing:** $3.72-10.60/hr (20-25% premium over H100)
- **AWS/GCP:** Early availability, pricing comparable to H100 8-GPU configs

**Why?**
- 141GB HBM3e memory (1.76× H100) for larger models or longer contexts
- 4.8 TB/s memory bandwidth (1.4× H100) = better throughput
- 1.6× faster than H100 on long-context workloads (128K+ tokens)
- Same Tensor Core performance as H100 (989 TFLOPS FP16)

**Use Cases:**
- Models 100B+ parameters (GPT-3 scale and beyond)
- Extended context windows (64K-128K tokens) like DeepSeek-R1
- Long-document understanding tasks
- Multi-modal models combining text, image, audio

**Context Length Guidance:**
| Model Size | H100 (80GB) | H200 (141GB) |
|-----------|-------------|--------------|
| 13B params | 32K tokens | 128K tokens |
| 70B params | 8K tokens | 32K tokens |
| 175B params | 2K tokens (requires multi-GPU) | 8K tokens |

**When to Upgrade from H100:**
- Context length >16K tokens (memory constraint)
- Model size >70B parameters
- Batch size limited by memory (H200 enables 1.5-2× larger batches)

---

### Frontier Models (400B+ parameters, trillion-scale)

**Best Choice: B200 (192GB) or GB200 NVL72**

- **B200 Pricing:** $6.25/hr (Modal serverless), ~$45,000-50,000 per GPU (OEM)
- **DGX B200:** $515,410 for 8-GPU system
- **GB200 NVL72:** Price on request (72× B200 + Grace CPUs)

**Why?**
- 192GB HBM3e memory (2.4× H100, 1.36× H200)
- 8 TB/s memory bandwidth (2.4× H100, 1.67× H200)
- 20 PFLOPS sparse FP4 compute (specialized for transformers)
- 2nd-gen Transformer Engine with FP4 precision
- Up to 4× faster training than H100, 30× faster inference

**Performance Metrics:**
- **DGX B200:** 3× training performance of DGX H100
- **Inference:** 15× faster than DGX H100 (with FP4 quantization)
- **Target:** Trillion-parameter models, extreme multi-modal workloads

**Use Cases:**
- Frontier research (models >400B parameters)
- Next-generation multimodal foundation models
- Extreme context windows (>128K tokens)
- Future-proofing infrastructure investments

**Important Caveats (October 2025):**
- Limited availability (early adopters only)
- Software ecosystem still maturing
- No large-scale training benchmarks published yet
- 25%+ cost premium over H200
- Reliability challenges being worked through

**When to Choose B200:**
- Planning for 2026-2027 model scale
- Building AI-first infrastructure today
- Budget allows for premium hardware
- Not time-constrained (can wait for availability)

---

## Inference Workloads

### Low-Latency Inference (batch size 1-4)

**Best Choice: L4**

- **L4 Pricing:** $0.30-0.80/hr depending on provider
- **Hardware Cost:** $1,500-2,500 per GPU

**Why?**
- Ultra-low TDP (72W) = high density (4-8 GPUs per server)
- 24GB memory sufficient for most inference workloads
- PCIe Gen4 interface (lower latency than SXM)
- Cost-effective for single-sample inference
- Great for serving 1-13B parameter models

**Performance Metrics:**
- **GPT-2 (1.5B):** 10-15ms latency (batch size 1)
- **LLaMA-7B:** 50-80ms latency (batch size 1, FP16)
- **BERT-Base:** 2-5ms latency (batch size 1)

**Use Cases:**
- Real-time chatbots
- API serving (REST/gRPC)
- Single-user applications
- Development and testing

**Cost Comparison:**
- **L4:** $0.50/hr × 1 GPU = $0.50/hr for 100 req/s
- **A100:** $3.00/hr × 1 GPU = $3.00/hr for 400 req/s
- **Winner:** L4 for <200 req/s, A100 for higher throughput

---

### High-Throughput Inference (batch size 8-32)

**Best Choice: L40S or A100**

- **L40S Pricing:** $1.95/hr (Modal), $0.80/hr (AWS reserved)
- **A100 Pricing:** $2.50-3.50/hr (specialized clouds)

**Why?**
- High memory bandwidth enables large batch processing
- Tensor Cores accelerate matrix operations in batched inference
- L40S: 48GB memory, 864 GB/s bandwidth, $1.95/hr
- A100: 80GB memory, 2 TB/s bandwidth, $3.00/hr

**When to Choose Which:**
- **L40S:** Batch sizes 8-16, models <13B params, cost-sensitive
- **A100:** Batch sizes 16-32, models 13-70B params, throughput-critical

**Performance Metrics (LLaMA-7B, batch size 16):**
- **L40S:** ~200 tokens/sec throughput
- **A100 40GB:** ~300 tokens/sec throughput
- **A100 80GB:** ~350 tokens/sec throughput

**Use Cases:**
- Batch API processing
- Scheduled inference jobs
- Multi-user serving with request batching
- Document processing pipelines

---

### Large Model Inference (13B-70B parameters)

**Best Choice: H100 or H200**

- **H100 Pricing:** $1.90-3.50/hr (specialized clouds after price cuts)
- **H200 Pricing:** $3.72-10.60/hr (20-25% premium)

**Why?**
- 80GB (H100) or 141GB (H200) memory for large models
- Extremely high memory bandwidth (3.35-4.8 TB/s)
- Tensor Cores with FP8 support (2× throughput over FP16)
- 4th/5th-gen NVLink for multi-GPU inference

**When to Choose Which:**
- **H100:** Models 13-70B params, standard context (<16K tokens)
- **H200:** Models 70B+ params, extended context (32K-128K tokens)

**Performance Metrics:**
- **LLaMA-70B (H100):** ~50 tokens/sec (batch size 1), ~150 tokens/sec (batch size 8)
- **LLaMA-70B (H200):** ~60 tokens/sec (batch size 1), ~200 tokens/sec (batch size 8)
- **DeepSeek-R1 (H200):** Supports 128K context vs 32K on H100

**Use Cases:**
- Serving large foundation models (GPT-3 scale)
- Long-context applications (summarization, document QA)
- Multi-modal inference
- High-value applications where latency matters

**Multi-GPU Inference:**
- Tensor parallelism across 2-8 GPUs for models >70B
- Pipeline parallelism for extremely large models
- NVLink communication critical (900 GB/s per GPU)

---

### Frontier Model Inference (100B+ parameters)

**Best Choice: B200**

- **B200 Pricing:** $6.25/hr (Modal serverless)
- **Target:** 100B-1T parameter models

**Why?**
- 192GB memory per GPU (2.4× H100)
- 8 TB/s memory bandwidth (2.4× H100)
- FP4 Tensor Cores for 30× faster inference vs H100
- 2nd-gen Transformer Engine optimized for inference

**Performance Claims (NVIDIA):**
- 30× faster inference than H100 (with FP4 quantization)
- Supports models up to 1T parameters with multi-GPU
- Optimized for extremely large context windows

**Use Cases:**
- Frontier model serving (GPT-4 scale and beyond)
- Extreme context applications (>128K tokens)
- Future-proofing inference infrastructure
- High-value enterprise deployments

**Caveats:**
- Limited availability (October 2025)
- Software optimizations still rolling out
- 25-40% cost premium over H200
- Best for forward-looking deployments

---

## Cloud Pricing Comparison

### Major Cloud Providers (October 2025)

After AWS announced 44% price reductions in mid-2025, pricing across major clouds has become more competitive:

| GPU | AWS (per GPU) | GCP (per GPU) | Azure (per GPU) | Notes |
|-----|--------------|--------------|----------------|-------|
| **A100 80GB** | $4-5/hr | $4.50-5.50/hr | $4-5/hr | 44% reduction from 2024 |
| **H100 80GB** | $12.29/hr (P5) | $11.06/hr | $6.98/hr | 8-GPU instances |
| **H200 141GB** | $13-15/hr | $12-14/hr | $8-10/hr | Limited availability |
| **L4 24GB** | $0.50-0.80/hr | $0.60-0.90/hr | $0.70-1.00/hr | Cost-effective inference |
| **L40S 48GB** | $0.80/hr (3yr) | $1.50-2.00/hr | $1.50-2.00/hr | AWS reserved best |

**Key Insight:** Azure appears cheapest but has lower availability quotas. AWS and GCP have better availability after 2025 price cuts.

---

### Specialized GPU Clouds (Often Cheaper)

Specialized providers often beat hyperscalers by 40-70% on cost:

| Provider | H100 SXM | H100 PCIe | A100 80GB | L4 | B200 |
|----------|---------|-----------|-----------|-----|------|
| **Modal** | - | - | - | - | $6.25/hr |
| **Lambda Labs** | $3.29/hr | $2.49/hr | $1.80-2.20/hr | $0.40-0.60/hr | - |
| **RunPod** | $2.25-2.50/hr | $1.90-2.20/hr | $1.40-1.80/hr | $0.30-0.50/hr | - |
| **Hyperstack** | $2.40/hr | $1.90/hr | $1.60-2.00/hr | $0.40-0.60/hr | - |
| **TensorDock** | $2.25/hr | $1.91/hr (spot) | $1.50-1.90/hr | $0.35-0.55/hr | - |

**Trade-offs:**
- ✅ **Pro:** 40-70% cheaper, flexible commitment
- ❌ **Con:** Lower availability, less reliability, fewer managed services
- ✅ **Pro:** Great for batch training, prototyping, burst workloads
- ❌ **Con:** Not suitable for production inference (uptime SLAs)

**Recommendation:**
- **Training:** Use specialized clouds (cost-effective, high burst capacity)
- **Inference (production):** Use major clouds (better uptime, SLAs, managed services)

---

### Hardware Purchase vs Cloud (TCO Analysis)

For long-running workloads, purchasing hardware can be more cost-effective:

#### H100 80GB (SXM5) Example:
- **Hardware Cost:** ~$30,000-40,000 per GPU
- **DGX H100 (8× H100):** ~$300,000-400,000
- **Cloud Cost (AWS):** $12.29/hr × 24hr × 30 days = $8,849/month per GPU

**Break-even Analysis:**
```
GPU Cost / Cloud Cost per Month = Break-even Months
$35,000 / $8,849 = ~4 months
```

**Additional Costs to Consider:**
- Power: ~700W per GPU × $0.10/kWh × 24hr × 30 days = $504/month
- Cooling: ~$100-200/month per GPU
- Network: ~$200-500/month (for multi-GPU setups)
- Maintenance: ~$500-1000/month (IT staff time)

**Total TCO (Own):** ~$35,000 + $1,300/month
**Total TCO (Cloud):** $8,849/month

**Break-even with All Costs:** ~5-6 months of continuous use

**When to Buy:**
- Running training jobs >6 months
- Building permanent ML infrastructure
- Team of 5+ researchers (shared resource)
- Compliance requires on-premises hardware

**When to Rent:**
- Short-term projects (<6 months)
- Burst workloads (training spikes)
- Testing different GPU types
- No capital budget for hardware

---

## Decision Framework

### Quick Decision Tree

```
START: What's your workload?

├─ TRAINING
│  ├─ Model size?
│  │  ├─ <100M params → L40S / A10
│  │  ├─ 100M-10B params → A100 (40GB or 80GB)
│  │  ├─ 10B-100B params → H100
│  │  ├─ 100B+ params with long context → H200
│  │  └─ 400B+ params (frontier) → B200
│  │
│  └─ Budget?
│     ├─ Cost-sensitive → Use A100 (best value)
│     ├─ Time-sensitive → Use H100/H200 (3× faster)
│     └─ Future-proofing → Use B200 (if available)
│
└─ INFERENCE
   ├─ Latency requirement?
   │  ├─ Low latency (batch size 1-4) → L4
   │  ├─ High throughput (batch size 8-32) → L40S / A100
   │  └─ Large model (13B-70B) → H100 / H200
   │
   └─ Model size?
      ├─ <13B params → L4 / L40S
      ├─ 13-70B params → A100 / H100
      └─ 70B+ params → H100 / H200 / B200
```

---

### Workload-Specific Recommendations

#### Research & Prototyping
- **Start:** A100 40GB ($2.50-3.00/hr)
- **Scale:** H100 when model >13B or training time >48hrs
- **Budget:** L40S for cost-sensitive projects

#### Production Training (Continuous)
- **Consider:** Purchasing DGX H100 or H200 systems
- **Break-even:** ~5-6 months of continuous use
- **Cloud:** Use specialized providers (40-70% cheaper)

#### Production Inference (Latency-Critical)
- **Best:** Major cloud providers (AWS/GCP/Azure) for SLAs
- **GPU:** L4 for <13B models, H100 for 13-70B models
- **Architecture:** Multi-region deployment for redundancy

#### Production Inference (Throughput-Critical)
- **Best:** A100 or L40S with batching
- **Alternative:** H100 with FP8 quantization (2× throughput)
- **Multi-GPU:** Tensor parallelism for models >70B

#### Startup / Small Team
- **Training:** A100 on specialized clouds (best value)
- **Inference:** L4 on major clouds (good SLAs, low cost)
- **Strategy:** Rent, don't buy (preserve capital)

#### Enterprise / Large Team
- **Training:** DGX H100/H200 on-premises (TCO advantage)
- **Inference:** Multi-cloud (AWS + GCP) for redundancy
- **Strategy:** Hybrid (own hardware + cloud burst capacity)

---

## Hardware Specifications

### Detailed Specs Table

| GPU | Architecture | Memory | Bandwidth | TDP | FP16 TFLOPS | FP8 TFLOPS | Price (OEM) |
|-----|-------------|--------|-----------|-----|-------------|------------|-------------|
| **L4** | Ada Lovelace | 24GB GDDR6 | 300 GB/s | 72W | 121 (Tensor) | 242 | $1,500-2,500 |
| **L40S** | Ada Lovelace | 48GB GDDR6 | 864 GB/s | 350W | 362 (Tensor) | 733 | $7,000-8,000 |
| **A10** | Ampere | 24GB GDDR6 | 600 GB/s | 150W | 125 (Tensor) | N/A | $3,000-4,000 |
| **A100 40GB** | Ampere | 40GB HBM2e | 1,555 GB/s | 400W | 312 (Tensor) | N/A | $10,000-12,000 |
| **A100 80GB** | Ampere | 80GB HBM2e | 2,039 GB/s | 400W | 312 (Tensor) | N/A | $15,000-18,000 |
| **H100 SXM** | Hopper | 80GB HBM3 | 3,350 GB/s | 700W | 989 (Tensor) | 1,979 | $30,000-40,000 |
| **H100 PCIe** | Hopper | 80GB HBM3 | 2,000 GB/s | 350W | 756 (Tensor) | 1,513 | $25,000-35,000 |
| **H200** | Hopper | 141GB HBM3e | 4,800 GB/s | 700W | 989 (Tensor) | 1,979 | $35,000-45,000 |
| **B100** | Blackwell | 192GB HBM3e | 8,000 GB/s | 700W | ~1,500 (est) | ~3,000 (est) | $30,000-35,000 |
| **B200** | Blackwell | 192GB HBM3e | 8,000 GB/s | 1000W | ~2,000 (est) | 20,000 (FP4) | $45,000-50,000 |

**Notes:**
- TDP = Thermal Design Power (affects cooling and power costs)
- SXM form factor has higher bandwidth and TDP than PCIe
- FP8 support only on Hopper (H100/H200) and Blackwell (B100/B200)
- B100/B200 specs are estimated based on NVIDIA announcements

---

### Form Factor Comparison (SXM vs PCIe)

| Feature | SXM (Server Module) | PCIe (Card) |
|---------|-------------------|-------------|
| **Memory Bandwidth** | Full speed (3.35-4.8 TB/s) | Reduced (~2 TB/s) |
| **TDP** | 700W (H100/H200) | 350W (H100 PCIe) |
| **NVLink** | 900 GB/s (4th gen) | 600 GB/s (3rd gen) |
| **Multi-GPU** | Optimized for 8-GPU configs | 2-4 GPUs typical |
| **Cooling** | Liquid cooling required | Air cooling possible |
| **Price Premium** | +15-20% over PCIe | Baseline |
| **Best For** | Large-scale training | Inference, small training |

**When to Use SXM:**
- Multi-GPU training (8+ GPUs)
- Large models requiring NVLink bandwidth
- Datacenter deployments with liquid cooling
- Maximum performance needed

**When to Use PCIe:**
- 1-4 GPU setups
- Inference workloads
- Air-cooled servers
- Cost-sensitive deployments

---

### Memory Hierarchy Impact on Transformers

Understanding memory hierarchy is critical for transformer performance:

```
Level 1: Registers (fastest, smallest)
  ├─ Tensor Cores use registers for 4×4 matrix ops
  └─ ~20 KB per SM (Streaming Multiprocessor)

Level 2: L1 Cache / Shared Memory
  ├─ Programmable cache (up to 128 KB per SM)
  └─ Critical for Flash Attention block sizes

Level 3: L2 Cache
  ├─ 40-60 MB total (varies by GPU)
  └─ Shared across all SMs

Level 4: HBM (GPU Memory)
  ├─ 24-192 GB (depending on GPU)
  ├─ 300-8,000 GB/s bandwidth
  └─ THIS IS USUALLY THE BOTTLENECK

Level 5: System Memory (CPU DRAM)
  ├─ Accessed via PCIe (25-64 GB/s)
  └─ Avoid at all costs for training
```

**Why This Matters for GPUs:**

1. **Attention is Memory-Bound:**
   - Standard attention: O(N²) memory accesses to HBM
   - Flash Attention: O(N) memory accesses by using L1/L2 cache
   - Result: 2-4× speedup just from better memory patterns

2. **Batch Size Limited by HBM:**
   - Larger batch → better GPU utilization
   - But batch size limited by HBM capacity
   - H200's 141GB vs H100's 80GB = 1.76× larger batches

3. **Context Length = Memory Intensive:**
   - Attention memory grows as O(N²) where N = sequence length
   - KV cache grows as O(N) per layer
   - Long context (128K tokens) requires 100+ GB just for KV cache

**Practical Optimization:**
```go
// Bad: Materializes full N² attention matrix in HBM
// Memory: O(N²) = 128K² × 4 bytes = 64 GB for single head!
attnScores := MatMul(Q, K_T)  // 128K × 128K matrix
attnProbs := Softmax(attnScores)
output := MatMul(attnProbs, V)

// Good: Flash Attention (tiled computation)
// Memory: O(N) = 128K × 4 bytes = 512 KB per head
output := FlashAttention(Q, K, V, blockSize=128)
```

**GPU Selection Impact:**
- **L4/L40S:** 24-48 GB → max ~16K context for 13B model
- **A100 80GB:** max ~32K context for 13B model
- **H100 80GB:** max ~32K context for 13B model (faster though)
- **H200 141GB:** max ~128K context for 13B model

---

## Summary: Quick Selection Guide

### For Training:

| Model Size | Best GPU | Price Range | Training Speed |
|-----------|---------|-------------|----------------|
| <100M | L40S / A10 | $0.80-2.00/hr | 2-3 hrs for 30M |
| 100M-10B | A100 | $2.50-4.00/hr | 12-48 hrs for 1.5B |
| 10B-70B | H100 | $1.90-12.00/hr | 2-7 days for 13B |
| 70B-175B | H200 | $3.72-15.00/hr | 1-4 weeks for 70B |
| 400B+ | B200 | $6.25+/hr | Frontier research |

### For Inference:

| Use Case | Best GPU | Price Range | Throughput |
|----------|---------|-------------|------------|
| Low latency (<100ms) | L4 | $0.30-0.80/hr | 100-200 req/s |
| High throughput (batched) | L40S / A100 | $1.00-3.50/hr | 200-400 req/s |
| Large models (13-70B) | H100 | $1.90-12.00/hr | 50-150 tok/s |
| Extended context (128K) | H200 | $3.72-15.00/hr | 60-200 tok/s |

### Cost Optimization:

1. **Use specialized clouds for training** (40-70% cheaper than AWS/GCP/Azure)
2. **Use major clouds for production inference** (better SLAs, uptime)
3. **Consider purchasing hardware** if continuous use >6 months
4. **Start with A100** for prototyping (best value)
5. **Upgrade to H100/H200** when speed matters more than cost

---

## Related Documentation

- [GPU Acceleration Guide](./gpu-acceleration-guide.md) - Technical implementation details
- [Optimization Guide](./optimization-guide.md) - Performance tuning strategies
- [Training Workflows](./training-workflows.md) - Complete training pipelines

