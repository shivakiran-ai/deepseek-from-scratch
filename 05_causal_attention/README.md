<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:0D3B6E,100:1A56A0&height=130&text=Topic%2005%20%E2%80%94%20Causal%20Attention&fontSize=24&fontColor=ffffff&fontAlignY=52&desc=Technical%20Report%20%7C%20Stage%202%3A%20Attention%20%26%20the%20KV%20Cache%20%7C%20DeepSeek%20from%20Scratch&descSize=12&descAlignY=78"/>

<br/>

[![Topic](https://img.shields.io/badge/Topic-05%20of%2026-0D3B6E?style=for-the-badge)](.)
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
| **Topic** | 05 of 26 — Stage 2 |
| **Core concept** | Causal mask · −∞ before softmax · renormalisation equivalence · dropout |
| **Status** | 🔄 Diagrams ready — write-up in progress |
| **Builds on** | Topic 04 — Self-Attention Mechanism |

---

## Figures

Every diagram uses one running example and colour scheme across the series. Shades come from a small computed example, not decoration.

**Figure 1 — Why mask**

<img src="./images/ca1_why_mask.png" width="100%"/>

**Figure 2 — Causal mask**

<img src="./images/ca2_causal_mask.png" width="100%"/>

**Figure 3 — Mask before softmax**

<img src="./images/ca3_mask_before_softmax.png" width="100%"/>

**Figure 4 — Mask after softmax**

<img src="./images/ca4_mask_after_softmax.png" width="100%"/>

**Figure 5 — Dropout context**

<img src="./images/ca5_dropout_context.png" width="100%"/>

---

## Files in This Directory

| File | Description |
|------|-------------|
| `README.md` | This document — conceptual summary and reader entry point |
| `Topic5_<Title>.docx` | 🔲 Deep-dive technical report (to be added) |
| `<Title>.ipynb` | 🔲 Annotated PyTorch implementation (to be added) |
| `images/` | Diagrams used in this README and the docx |

---

## Navigation

**Previous:** &nbsp; [Topic 04 — Self-Attention Mechanism](../04_self_attention/README.md)

**Next:** &nbsp; [Topic 06 — Multi-Head Attention](../06_multihead_attention/README.md)

---

<div align="center">

**DeepSeek from Scratch**  
A technical report series rebuilding the key ideas behind DeepSeek-V2 / V3 from first principles.

**Shiva Kiran Dadishetty** &nbsp;·&nbsp; Independent Research &nbsp;·&nbsp; Texas, USA  
[github.com/shivakiran-ai/deepseek-from-scratch](https://github.com/shivakiran-ai/deepseek-from-scratch)

*Document version 0.1 &nbsp;·&nbsp; Last updated November 2026*

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:1A56A0,100:0D3B6E&height=80&section=footer"/>

</div>
