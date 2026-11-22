<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:0D3B6E,50:1A56A0,100:2E75B6&height=220&section=header&text=DeepSeek%20from%20Scratch&fontSize=48&fontColor=ffffff&fontAlignY=40&desc=MLA%20%C2%B7%20RoPE%20%C2%B7%20Mixture%20of%20Experts%20%C2%B7%20Multi-Token%20Prediction%20%C2%B7%20FP8%20%E2%80%94%20rebuilt%20from%20first%20principles&descAlignY=60&descSize=16"/>

<br/>

[![Author](https://img.shields.io/badge/Author-SHIVA%20KIRAN%20DADISHETTY-0D3B6E?style=for-the-badge)](.)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](.)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](.)
[![Status](https://img.shields.io/badge/Status-Active%20Development-22C55E?style=for-the-badge)](.)
[![License](https://img.shields.io/badge/License-MIT-F59E0B?style=for-the-badge)](./LICENSE)

<br/>

[![Topics Complete](https://img.shields.io/badge/Topics%20Complete-11%20%2F%2026-1A56A0?style=for-the-badge)](.)
[![Stage 1](https://img.shields.io/badge/Stage%201%20%7C%20Foundations-3%2F3-22C55E?style=for-the-badge)](.)
[![Stage 2](https://img.shields.io/badge/Stage%202%20%7C%20Attention%20%26%20KV%20Cache-6%2F6-22C55E?style=for-the-badge)](.)
[![Stage 3](https://img.shields.io/badge/Stage%203%20%7C%20MLA%20%26%20RoPE-2%2F6-2E75B6?style=for-the-badge)](.)
[![Stage 4](https://img.shields.io/badge/Stage%204%20%7C%20Mixture%20of%20Experts-0%2F4-6B7280?style=for-the-badge)](.)
[![Stage 5](https://img.shields.io/badge/Stage%205%20%7C%20Multi--Token%20Prediction-0%2F3-6B7280?style=for-the-badge)](.)
[![Stage 6](https://img.shields.io/badge/Stage%206%20%7C%20Quantization-0%2F3-6B7280?style=for-the-badge)](.)
[![Stage 7](https://img.shields.io/badge/Stage%207%20%7C%20Summary-0%2F1-6B7280?style=for-the-badge)](.)

</div>

---

## About This Repository

This repository rebuilds the architecture behind **DeepSeek-V2 and DeepSeek-V3** from first principles — every major innovation derived, drawn, and coded by hand.

It starts where [`llm-from-scratch`](https://github.com/shivakiran-ai/llm-from-scratch) ends. That series built a GPT-2 model. This one asks: **what did DeepSeek change, and why?** Multi-Head Latent Attention. Decoupled RoPE. Fine-grained Mixture of Experts with shared experts. Multi-Token Prediction. FP8 training.

Each idea is introduced as the **solution to a problem** created by the previous one. The KV cache makes inference fast — but huge. MQA and GQA shrink it — but cost quality. MLA compresses it — but clashes with RoPE. Decoupled RoPE fixes that. And so on.

This is part of my preparation for doctoral research in LLM pretraining and inference.

---

## How to Use This Repository

Every topic folder contains the same files:

| File | Purpose |
|------|---------|
| `README.md` | Sharp summary — key concept, figures, core insight, paper connection. **Start here.** |
| `TopicN_Title.docx` | Complete deep dive — every derivation, all mathematics, full code reference, every design decision explained. |
| `notebook.ipynb` | Fully annotated, runnable PyTorch implementation — built from scratch, extensively commented. |
| `images/` | The diagrams for that topic — one running example and colour scheme across the series. |

Read the `README.md` first. Run the notebook to see it in action. Open the `.docx` to go completely deep.

---

## Progress

```
Overall  ██████████████████░░░░░░░░░░░░░░░░░░░░░░░░  11 / 26 topics complete

Stage 1  Foundations             ████████████████████  3 / 3  ✅ COMPLETE
Stage 2  Attention & KV Cache    ████████████████████  6 / 6  ✅ COMPLETE
Stage 3  MLA & RoPE              ███████░░░░░░░░░░░░░  2 / 6  ← active
Stage 4  Mixture of Experts      ░░░░░░░░░░░░░░░░░░░░  0 / 4
Stage 5  Multi-Token Prediction  ░░░░░░░░░░░░░░░░░░░░  0 / 3
Stage 6  Quantization            ░░░░░░░░░░░░░░░░░░░░  0 / 3
Stage 7  Summary                 ░░░░░░░░░░░░░░░░░░░░  0 / 1
```

*Updated with every new topic. Star this repository ⭐ to follow the journey.*

---

## The Series Roadmap

### 🧱 Stage 1 — Foundations ✅

> *Where DeepSeek fits and the GPT-style architecture it starts from*

| # | Topic | Status | Core Concept |
|---|-------|:------:|--------------|
| 01 | [DeepSeek Basics](./01_deepseek_basics/README.md) | ✅ | What DeepSeek is · V2/V3/R1 · The four innovations this series builds |
| 02 | [LLM Architecture Overview](./02_llm_architecture/README.md) | ✅ | Tokens → embeddings → transformer blocks → logits · GPT-style decoder |
| 03 | [Introduction to Attention](./03_attention_intro/README.md) | ✅ | Why attention · context vectors · from RNNs to Transformers |

---

### 🧠 Stage 2 — Attention & the KV Cache ✅

> *From self-attention to the inference memory problem that motivates everything after*

| # | Topic | Status | Core Concept |
|---|-------|:------:|--------------|
| 04 | [Self-Attention Mechanism](./04_self_attention/README.md) | ✅ | W_Q · W_K · W_V · Q·Kᵀ · scale by √d_k · softmax · weights·V |
| 05 | [Causal Attention](./05_causal_attention/README.md) | ✅ | Causal mask · −∞ before softmax · renormalisation equivalence · dropout |
| 06 | [Multi-Head Attention](./06_multihead_attention/README.md) | ✅ | Split weights per head · per-head Q, K, V · concat · out_proj |
| 07 | [Key-Value Cache](./07_kv_cache/README.md) | ✅ | Prefill vs decode · cache K, V not Q · O(n²) → O(n) · cache size formula |
| 08 | [Multi-Query Attention](./08_multi_query_attention/README.md) | ✅ | One shared K and V for all heads · cache ÷ n_heads · quality trade-off |
| 09 | [Grouped-Query Attention](./09_grouped_query_attention/README.md) | ✅ | K, V shared per group of heads · the MHA ↔ MQA middle ground |

---

### 🗜️ Stage 3 — Multi-Head Latent Attention & Positional Encodings

> *DeepSeek's compressed KV cache, and how RoPE is made to fit with it*

| # | Topic | Status | Core Concept |
|---|-------|:------:|--------------|
| 10 | [Multi-Head Latent Attention (MLA)](./10_multihead_latent_attention/README.md) | ✅ | Low-rank latent KV · cache the latent, not K and V · weight absorption |
| 11 | [Multi-Head Latent Attention — Coded](./11_mla_code/README.md) | ✅ | One MLA decoding step, number by number · RopelessMLA in PyTorch · cache growth and memory tests |
| 12 | [Integer & Binary Positional Encodings](./12_integer_binary_positional_encoding/README.md) | 🔲 | Why position matters · integer and binary encodings and their limits |
| 13 | [Sinusoidal Positional Encodings](./13_sinusoidal_positional_encoding/README.md) | 🔲 | sin/cos at many frequencies · relative positions as rotations |
| 14 | [Rotary Positional Encodings (RoPE)](./14_rotary_positional_encoding/README.md) | 🔲 | Rotate Q and K by position · relative position in the dot product |
| 15 | [DeepSeek MLA + RoPE (Decoupled RoPE)](./15_deepseek_mla_rope/README.md) | 🔲 | Why RoPE breaks weight absorption · decoupled rotary keys and queries |

---

### 🔀 Stage 4 — Mixture of Experts

> *Sparse expert layers, load balancing, and DeepSeek's MoE redesign*

| # | Topic | Status | Core Concept |
|---|-------|:------:|--------------|
| 16 | [Mixture of Experts — Introduction](./16_moe_intro/README.md) | 🔲 | Router · top-k experts · sparse activation · total vs active params |
| 17 | [MoE Load-Balancing Techniques](./17_moe_load_balancing/README.md) | 🔲 | Auxiliary loss · load balancing loss · capacity factor · token dropping |
| 18 | [DeepSeek's Mixture of Experts](./18_deepseek_moe/README.md) | 🔲 | Fine-grained experts · shared experts · auxiliary-loss-free balancing |
| 19 | [Mixture of Experts — Coded](./19_moe_code/README.md) | 🔲 | DeepSeek-style MoE layer from scratch in PyTorch |

---

### 🎯 Stage 5 — Multi-Token Prediction

> *Predicting several future tokens per position, and DeepSeek's sequential MTP modules*

| # | Topic | Status | Core Concept |
|---|-------|:------:|--------------|
| 20 | [Multi-Token Prediction — Introduction](./20_mtp_intro/README.md) | 🔲 | Predict several future tokens · denser training signal · speculative decoding |
| 21 | [DeepSeek's Multi-Token Prediction](./21_deepseek_mtp/README.md) | 🔲 | Sequential MTP modules · shared embedding & head · causal chain kept |
| 22 | [Multi-Token Prediction — Coded](./22_mtp_code/README.md) | 🔲 | DeepSeek-style MTP modules from scratch in PyTorch |

---

### ⚡ Stage 6 — Quantization

> *Low-precision training and inference: mixed precision, fine-grained FP8, accumulation*

| # | Topic | Status | Core Concept |
|---|-------|:------:|--------------|
| 23 | [Introduction to LLM Quantization](./23_quantization_intro/README.md) | 🔲 | FP32 · BF16 · FP8 · INT8 · scale & zero point · memory vs accuracy |
| 24 | [DeepSeek Quantization Part 1](./24_deepseek_quantization_p1/README.md) | 🔲 | Mixed-precision framework · fine-grained (tile/block-wise) FP8 quantization |
| 25 | [DeepSeek Quantization Part 2](./25_deepseek_quantization_p2/README.md) | 🔲 | Higher-precision accumulation · online quantization |

---

### 🏁 Stage 7 — Summary

> *Putting every DeepSeek innovation together*

| # | Topic | Status | Core Concept |
|---|-------|:------:|--------------|
| 26 | [DeepSeek from Scratch — Summary](./26_summary/README.md) | 🔲 | MLA + RoPE · DeepSeekMoE · MTP · FP8 — the full picture |

---

## The Chain — How Every Topic Connects

```
GPT-style Transformer  (Topics 01 – 06)
        │  attention needs K and V for every past token
        ▼
KV Cache  (07)
        │  fast decoding — BUT memory grows with layers × heads × context
        ▼
Multi-Query → Grouped-Query Attention  (08 – 09)
        │  share K, V across heads — smaller cache, BUT lower quality
        ▼
Multi-Head Latent Attention  (10 – 11)
        │  cache one small latent vector — BUT standard RoPE breaks it
        ▼
Positional Encodings → RoPE → Decoupled RoPE  (12 – 15)
        │  attention is now efficient — the FFN is the next bottleneck
        ▼
Mixture of Experts → DeepSeekMoE  (16 – 19)
        │  huge capacity, few active params — training signal per token still sparse
        ▼
Multi-Token Prediction  (20 – 22)
        │  denser signal — BUT training at this scale is expensive
        ▼
FP8 Quantization & Mixed Precision  (23 – 25)
        ▼
DeepSeek-V3 — the full picture  (26)
```

> Each topic is the **solution** to the problem created by the previous topic.

---

## Design Philosophy

**1. No magic.**
Every layer is written by hand. No `from_pretrained()`, no hidden abstractions.

**2. The why before the what.**
Every innovation starts with the problem it solves — usually a memory or compute bottleneck visible in the previous topic.

**3. Every figure is checked.**
Diagrams are generated from real tensors. Shapes, row sums and equivalences (e.g. cached vs recomputed attention) are verified in code before they are drawn.

**4. Everything connects to the literature.**

| Paper | Relevance | Topics |
|-------|-----------|--------|
| Vaswani et al. (2017) — *Attention Is All You Need* | Transformer, scaled dot-product and multi-head attention | 02 – 06 |
| Shazeer (2019) — *Fast Transformer Decoding: One Write-Head is All You Need* | Multi-Query Attention | 08 |
| Ainslie et al. (2023) — *GQA* | Grouped-Query Attention | 09 |
| Su et al. (2021) — *RoFormer* | Rotary Positional Embeddings | 14 – 15 |
| DeepSeek-AI (2024) — *DeepSeek-V2* | Multi-Head Latent Attention, decoupled RoPE, DeepSeekMoE | 10 – 11, 15, 18 |
| Dai et al. (2024) — *DeepSeekMoE* | Fine-grained and shared experts | 18 – 19 |
| Shazeer et al. (2017) — *Outrageously Large Neural Networks* | Sparsely-gated Mixture of Experts | 16 – 17 |
| Fedus et al. (2021) — *Switch Transformers* | Load-balancing loss, capacity factor | 17 |
| Gloeckle et al. (2024) — *Better & Faster LLMs via Multi-token Prediction* | Multi-Token Prediction | 20 |
| DeepSeek-AI (2024) — *DeepSeek-V3 Technical Report* | Aux-loss-free balancing, MTP modules, FP8 training | 18, 21 – 25 |
| Micikevicius et al. (2018) — *Mixed Precision Training* | Mixed-precision foundations | 23 – 24 |

---

## Repository Structure

```
deepseek-from-scratch/
│
├── README.md                                ← You are here
├── LICENSE
├── .gitignore
│
├── 01_deepseek_basics/                      ✅ Complete
├── 02_llm_architecture/                     ✅ Complete
├── 03_attention_intro/                      ✅ Complete
├── 04_self_attention/                       ✅ Complete
├── 05_causal_attention/                     ✅ Complete
├── 06_multihead_attention/                  ✅ Complete
├── 07_kv_cache/                             ✅ Complete
├── 08_multi_query_attention/                ✅ Complete
├── 09_grouped_query_attention/              ✅ Complete
├── 10_multihead_latent_attention/           ✅ Complete
├── 11_mla_code/                             ✅ Complete
├── 12_integer_binary_positional_encoding/   🔲 Planned
├── 13_sinusoidal_positional_encoding/       🔲 Planned
├── 14_rotary_positional_encoding/           🔲 Planned
├── 15_deepseek_mla_rope/                    🔲 Planned
├── 16_moe_intro/                            🔲 Planned
├── 17_moe_load_balancing/                   🔲 Planned
├── 18_deepseek_moe/                         🔲 Planned
├── 19_moe_code/                             🔲 Planned
├── 20_mtp_intro/                            🔲 Planned
├── 21_deepseek_mtp/                         🔲 Planned
├── 22_mtp_code/                             🔲 Planned
├── 23_quantization_intro/                   🔲 Planned
├── 24_deepseek_quantization_p1/             🔲 Planned
├── 25_deepseek_quantization_p2/             🔲 Planned
├── 26_summary/                              🔲 Planned
│
└── (each topic: README.md · TopicN_Title.docx · notebook.ipynb · images/)
```

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/shivakiran-ai/deepseek-from-scratch.git
cd deepseek-from-scratch

# Install dependencies
pip install torch numpy jupyter matplotlib tiktoken
```

> **No GPU required** for the attention, KV cache and MLA topics. A GPU helps from the Mixture of Experts topics onwards.

---

## Prerequisites

| Requirement | Level |
|------------|-------|
| Python | Comfortable — functions, classes |
| PyTorch | Basic — tensors, `nn.Module`, autograd |
| Transformers | Recommended — this series follows `llm-from-scratch` |
| Linear Algebra | Helpful — matrix multiplication, low-rank factorisation |

> This series is a direct continuation of [`llm-from-scratch`](https://github.com/shivakiran-ai/llm-from-scratch). If you have not built a GPT-2 model before, start there.

---

## Related Repositories

| Repository | Description |
|-----------|-------------|
| [`neural-networks-from-scratch`](https://github.com/shivakiran-ai/neural-networks-from-scratch) | Neurons, backpropagation, optimizers and regularization in pure Python. |
| [`llm-from-scratch`](https://github.com/shivakiran-ai/llm-from-scratch) | A complete GPT-2 built from first principles — tokenizer to pretraining. The foundation for this series. |

---

## About

This series is part of my preparation for PhD research in machine learning, focused on LLM pretraining and inference efficiency.

DeepSeek-V3 reached frontier quality at a fraction of the usual training cost — not through one trick, but through a chain of engineering decisions, each fixing the bottleneck the last one exposed. This series rebuilds that chain, one component at a time.

Every topic is documented with:
- Handwritten notes capturing the thinking process
- Diagrams generated from real, verified tensors
- A complete PyTorch implementation
- Connections to the original research papers

---

## License

This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for details.

---

<div align="center">

<br/>

*If this repository helped you understand DeepSeek more deeply, consider giving it a ⭐*

<br/>

**SHIVA KIRAN DADISHETTY**

*Building one component at a time. No shortcuts.*

<br/>

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:2E75B6,50:1A56A0,100:0D3B6E&height=120&section=footer"/>

</div>
