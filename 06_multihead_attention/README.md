<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:0D3B6E,100:1A56A0&height=130&text=Topic%2006%20%E2%80%94%20Multi-Head%20Attention&fontSize=24&fontColor=ffffff&fontAlignY=52&desc=Technical%20Report%20%7C%20Stage%202%3A%20Attention%20%26%20the%20KV%20Cache%20%7C%20DeepSeek%20from%20Scratch&descSize=12&descAlignY=78"/>

<br/>

[![Topic](https://img.shields.io/badge/Topic-06%20of%2026-0D3B6E?style=for-the-badge)](.)
[![Stage](https://img.shields.io/badge/Stage%202-Attention%20%26%20the%20KV%20Cache-1A56A0?style=for-the-badge)](.)
[![Status](https://img.shields.io/badge/Status-Diagrams%20Ready-2E75B6?style=for-the-badge)](.)

**[← Back to the DeepSeek from Scratch series](../README.md)**

</div>

---

## Abstract

> 🔲 *To be written from the handwritten notes for this topic.*

---

## At a Glance

| | |
|---|---|
| **Topic** | 06 of 26 — Stage 2 |
| **Core concept** | Split weights per head · per-head Q, K, V · concat · out_proj |
| **Status** | 🔄 Diagrams ready — write-up in progress |
| **Builds on** | Topic 05 — Causal Attention |

---

## Figures

Every diagram uses one running example and colour scheme across the series. Shades come from a small computed example, not decoration.

**Figure 1 — Input embeddings**

<img src="./images/step1_input_embeddings.png" width="100%"/>

**Figure 2 — Qkv projection**

<img src="./images/step2_qkv_projection.png" width="100%"/>

**Figure 3 — Split heads**

<img src="./images/step3_split_heads.png" width="100%"/>

**Figure 4 — Per head qkv**

<img src="./images/step4_per_head_qkv.png" width="100%"/>

**Figure 5 — Attention scores**

<img src="./images/step5_attention_scores.png" width="100%"/>

**Figure 6 — Attention weights**

<img src="./images/step6_attention_weights.png" width="100%"/>

**Figure 7 — Context matrix**

<img src="./images/step7_context_matrix.png" width="100%"/>

**Figure 8 — Final context**

<img src="./images/step8_final_context.png" width="100%"/>

**Figure 9 — Multi head attention**

<img src="./images/multi_head_attention.png" width="100%"/>

---

## Files in This Directory

| File | Description |
|------|-------------|
| `README.md` | This document — conceptual summary and reader entry point |
| `Topic6_<Title>.docx` | 🔲 Deep-dive technical report (to be added) |
| `<Title>.ipynb` | 🔲 Annotated PyTorch implementation (to be added) |
| `images/` | Diagrams used in this README and the docx |

---

## Navigation

**Previous:** &nbsp; [Topic 05 — Causal Attention](../05_causal_attention/README.md)

**Next:** &nbsp; [Topic 07 — Key-Value Cache](../07_kv_cache/README.md)

---

<div align="center">

**DeepSeek from Scratch**  
A technical report series rebuilding the key ideas behind DeepSeek-V2 / V3 from first principles.

**Shiva Kiran Dadishetty** &nbsp;·&nbsp; Independent Research &nbsp;·&nbsp; Texas, USA  
[github.com/shivakiran-ai/deepseek-from-scratch](https://github.com/shivakiran-ai/deepseek-from-scratch)

*Document version 0.1 &nbsp;·&nbsp; Last updated November 2026*

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:1A56A0,100:0D3B6E&height=80&section=footer"/>

</div>
