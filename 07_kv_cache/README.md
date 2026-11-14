<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:0D3B6E,100:1A56A0&height=130&text=Topic%2007%20%E2%80%94%20Key-Value%20Cache&fontSize=24&fontColor=ffffff&fontAlignY=52&desc=Technical%20Report%20%7C%20Stage%202%3A%20Attention%20%26%20the%20KV%20Cache%20%7C%20DeepSeek%20from%20Scratch&descSize=12&descAlignY=78"/>

<br/>

[![Topic](https://img.shields.io/badge/Topic-07%20of%2026-0D3B6E?style=for-the-badge)](.)
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
| **Topic** | 07 of 26 — Stage 2 |
| **Core concept** | Prefill vs decode · cache K, V not Q · O(n²) → O(n) · cache size formula |
| **Status** | 🔄 Diagrams ready — write-up in progress |
| **Builds on** | Topic 06 — Multi-Head Attention |

---

## Figures

Every diagram uses one running example and colour scheme across the series. Shades come from a small computed example, not decoration.

**Figure 1 — Inference loop**

<img src="./images/kv1_inference_loop.png" width="100%"/>

**Figure 2 — Only last row**

<img src="./images/kv2_only_last_row.png" width="100%"/>

**Figure 3 — Repeated work**

<img src="./images/kv3_repeated_work.png" width="100%"/>

**Figure 4 — Cached step**

<img src="./images/kv4_cached_step.png" width="100%"/>

**Figure 5 — Cache growth**

<img src="./images/kv5_cache_growth.png" width="100%"/>

**Figure 6 — Tradeoff**

<img src="./images/kv6_tradeoff.png" width="100%"/>

**Figure 7 — Cache size**

<img src="./images/kv7_cache_size.png" width="100%"/>

---

## Files in This Directory

| File | Description |
|------|-------------|
| `README.md` | This document — conceptual summary and reader entry point |
| `Topic7_<Title>.docx` | 🔲 Deep-dive technical report (to be added) |
| `<Title>.ipynb` | 🔲 Annotated PyTorch implementation (to be added) |
| `images/` | Diagrams used in this README and the docx |

---

## Navigation

**Previous:** &nbsp; [Topic 06 — Multi-Head Attention](../06_multihead_attention/README.md)

**Next:** &nbsp; [Topic 08 — Multi-Query Attention](../08_multi_query_attention/README.md)

---

<div align="center">

**DeepSeek from Scratch**  
A technical report series rebuilding the key ideas behind DeepSeek-V2 / V3 from first principles.

**Shiva Kiran Dadishetty** &nbsp;·&nbsp; Independent Research &nbsp;·&nbsp; Texas, USA  
[github.com/shivakiran-ai/deepseek-from-scratch](https://github.com/shivakiran-ai/deepseek-from-scratch)

*Document version 0.1 &nbsp;·&nbsp; Last updated November 2026*

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:1A56A0,100:0D3B6E&height=80&section=footer"/>

</div>
