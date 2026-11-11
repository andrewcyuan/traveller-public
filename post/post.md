---
title: "Traveller-delta: Agent memory system with format-specific search tools beats frontier models on accuracy and latency"
author: "Andrew Yuan"
---

# Introduction

Markdown repositories have become the standard for agent memory. However, in enterprise environments where log formats are consistent and retrieval must be precise, we find that markdown is lossy and inefficient. We introduce Traveller-delta, an agent which operates over full log history with domain-specific search tools and the help of Jev, a decision model, to rank initial candidates. We find that on LongMemEval-v2, Traveller-delta with Gemini achieves 89.14% accuracy (402/451), exceeding the published AgentRunbook-C V2 reference by 13.53 percentage points and our Codex CLI GPT-6.1 Sol control by 5.54 percentage points. Separately, the Luna state-search configuration reduces mean latency from the reference’s 130.54 to 35.32 seconds at 81.60% accuracy. Notably, Traveller-delta was discovered semi-autonomously, with a swarm of research agents that, in conjunction with a single human researcher, produced and experimented on nine distinct architectures before converging on delta.

| System (LME-v2 Small) | Accuracy | Mean query latency | Cost |
|---|---:|---:|---|
| **Traveller-delta, Gemini 3.8 Flash** | **89.14%** | 114.81 s | $70.80 |
| Traveller-delta, Claude Opus 5.5 medium | 88.69% | 18.98 s | $71.49 |
| Codex CLI, GPT-6 Astra medium | 86.25% | 25.24 s | $192.46 |
| Codex CLI, GPT-6.1 Sol high | 83.59% | 28.34 s | $41.71 |
| Traveller-delta, Luna state-search | 81.60% | 35.32 s | $8.41 |
| AgentRunbook-C V2 published reference | 75.61% | 130.54 s | Not recorded |

*Costs were estimated for Codex because the tokens were run through a non-usage-based subscription.*

# LongMemEval-v2

LongMemEval-v2 is an agent memory evaluation which feeds 100 variable-length sequences per domain (up to 25M tokens per Small haystack) of realistic enterprise action data, then asks 451 questions about that data, spanning static state recall, dynamic state tracking, workflow knowledge, environment gotchas, and premise awareness . Questions are divided into three answer formats: true/false, multiple choice, and short answer. In this way, LongMemEval-v2 creates a realistic benchmark for questions that would be asked of an internal chatbot.

# Existing work

Enterprise agent memory systems have been a topic of focus since the popularization of generative AI in 2022. During the past four years, two themes have dominated: **Retrieval augmented generation (RAG)** and **markdown repositories**.

**RAG** means splitting incoming information into chunks based on token length, generating embeddings for these chunks, and then dumping them into a vector database. At retrieval time, the query or a piece of text generated from the query, is embedded into the same dimensionality as the stored chunks. Similarity between the query vector `Q` and a given chunk vector `C` is determined by $$\frac{\mathbf{Q} \cdot \mathbf{C}}{|\mathbf{Q}| |\mathbf{C}|}$$. The most common issue with RAG systems, beyond the low intelligence of embedding models in general, is their failure to reason across chunks. See [MultiHop-RAG](https://arxiv.org/abs/2401.15391) and [GraphRAG](https://arxiv.org/abs/2404.16130). We also observed that in our run, the query-to-slice RAG baseline (42.8% overall on Small in the authors’ results; not a local rerun) performed poorly, especially on environment gotchas (20.7%) and workflow knowledge (41.5%) ([source](https://arxiv.org/html/2605.12493v1)).

**Markdown repositories** were an ad-hoc solution which grew into a replacement for RAG, starting with personal and coding agents. Markdown repositories are simply file systems of markdown documents, which an agent usually edits using unix library and simple file editing tools. Today, they support many of the most well-known enterprise agents. However, while markdown-based systems may save user preferences adequately, they are overwhelmed by volume and conflicting evidence. Letta Dream, a frontier markdown-based memory system, scored just 25.5% accuracy in our testing on LongMemEval-v2. The run answered UNKNOWN on 256/451 questions (56.76%); 253 of these answers were wrong. Of the 256 abstentions, 69 had no selected evidence and 187 had some evidence. In question `634973c3`, the original histories contain the Summary type choices Grow Business, Run Business, and Transform Business, but the frozen memory omits them. In `8c55d6bc`, the memory retains report navigation but omits the source rule that reports show the top five most-viewed products. These are documented omissions in this memory-only configuration, whose retriever could not access raw histories.

# Traveller-delta: a Format-Specific Frontier Agentic Memory System

In many enterprise use cases, like the ones simulated by LongMemEval-v2, log formats are known and have regular metadata, as they usually come from observability tools. Given some known metadata format, we seek to design a high-performance system for agentic memory. These are our design goals:

1. **High accuracy**: Get correct answers and abstain when the answer is unknown.
2. **Low-latency**: Complete search in a realistic time for enterprise use cases
3. **Format-optimized**: Reduce unnecessary translation by storing and retrieving using format-native tools.

**Accuracy**.
We chose to store all incoming logs losslessly as JSON. Specifically, the eval has 100 trajectories, which are arrays of states in a theoretical enterprise web environment.

```json
{
  "id": "illustrative-trajectory",
  "original_goal": "Inspect the report",
  "states": [
    {
      "text": "Original state observation",
      "screenshot": "screenshots/illustrative-trajectory/0.png"
    }
  ]
}
```


In this way, theoretically, every answer is findable given sufficient retrieval work. The most similar setup to this is [agent-runbook-c](https://xiaowu0162.github.io/longmemeval-v2/agentrunbook-c-v2/).

**Latency**.
Looking at past agents, we were dismayed at their latency (goal 2). Agent-runbook-c and ops-memory-v1 had latency of over 130 seconds per query, making them unrealistic options for enterprise use. To counter this, we ranked every trajectory with Jev, using a summary of the original job, using the output noul variable to form a ranking. We then used the ranking to guide the agent's search while allowing access to every trajectory. In this way, the agent became more likely to find the right answer sooner.

**Format optimization**.
Storing logs in their native format was a major improvement. We also equipped the agent with a strong native search tool, `state_search`. This tool indexes original state text with SQLite FTS5, supports phrases, Boolean operators, and prefixes, and returns BM25-ranked snippets with trajectory and state identifiers (12 by default, at most 40). We saw that equipping the agent with this tool immediately improved performance, with a 1.66% relative increase in accuracy (+1.33 percentage points) and 4.51% decrease in latency compared to the historical fast-handoff baseline.


# Experiments

Below summarized are several different versions of Traveller that we experimented with. 

| Configuration / control | Correct / total | Accuracy | Mean query latency | Cost (USD) |
|---|---:|---:|---:|---|
| Luna high, fast handoff | 362/451 | 80.27% | 36.99 s | $8.19 |
| Luna high, state-search | 368/451 | 81.60% | 35.32 s | $8.41 |
| Luna xhigh | 365/451 | 80.93% | 46.78 s | $8.94 |
| Luna high, source-check | 360/451 | 79.82% | 41.19 s | $8.21 |
| Gemini 3.8 Flash retrieval + Qwen | 402/451 | 89.14% | 114.81 s | $70.80 |
| Claude Opus 5.5 medium + Qwen | 400/451 | 88.69% | 18.98 s | $71.49 |
| DeepSeek V4.1 Flash 1017 + Qwen | 352/451 | 78.05% | — | $57.24 |
| Codex CLI, GPT-6.1 Sol high | 377/451 | 83.59% | 28.34 s | $41.71 |
| Codex CLI, GPT-6 Astra medium | 389/451 | 86.25% | 25.24 s | $192.46 |
| AgentRunbook-C (paper, Small) | Not supplied as counts | 74.9% | 108.3 s | Not recorded |
| AgentRunbook-C V2 reference | Not supplied as counts | 75.61% | 130.54 s | Not recorded |
| ops-memory-v1 | Not located | Not established | — | Not recorded |
| Letta Dream | 115/451 | 25.50% | — | Not recorded |



| Additional experiment | Accuracy | Interpretation |
|---|---:|---|
| State-search, three fresh repeats | 80.78% ± 0.90 pp | Mean query time 34.76 ± 0.15 s; sample SD across runs |
| Fast-handoff, three fresh repeats | 78.79% ± 1.05 pp | Mean query time 34.18 ± 0.98 s |
| Xhigh, three fresh repeats | 79.75% ± 0.68 pp | Mean query time 40.07 ± 0.79 s |
| Source-check, three fresh repeats | 77.97% ± 0.46 pp | Mean query time 36.04 ± 1.18 s |
| Gemini direct answer + online learning | 87.36% | Changed answering and learning protocol; not a fixed-reader ablation |

The 12-run replication cost $100.77 confirmed plus $1.68 uncertain in total. Experiment corrections added $2.49 reported charges plus $5.50 uncertainty; these are incremental batch costs, not full-run costs.



# Working with research agents

Traveller-delta was the fourth strategy we tried, and was just one of nine. We deployed dozens of long-running research agents using Codex to autonomously explore architecture ideas in parallel. After writing testing adapters for two common evaluations -- BEAM 1M and LongMemEval-v2 -- several dozen coding agents worked over the course of two weeks to explore, tune, and improve these architectures.

Working with agents made it possible to explore several designs in parallel. Instead of implementing every idea in sequence, we could give separate agents different approaches and compare what they produced. The shared evaluation harness gave that work a common target.

The harder part was deciding what the results actually meant. A change in the reader, search budget, or evidence format could change the score alongside the architecture itself. Saved outputs and experiment logs mattered as much as the implementations: they let us inspect failures and distinguish a promising idea from a better result caused by a different setup.

# Reflection, limitations, and conclusion

Popular approaches in agent memory are remarkably stratified. RAG is very fast and very inaccurate, markdown repositories are slow and questionably accurate, and continual-learning approaches, where model weights themselves are updated, are very computationally expensive and unproven. Traveller offers an approach that, in a domain, can be faster and more accurate than markdown repositories.

The most salient limitation of Traveller-delta is performance at massive scale. Because of lossless log storage, space consumption could be unwieldy. However, we speculate that due to the initial Jev-ranking step of Traveller-delta, the approach could simply be adapted to load only a subset of those chunks deemed the most relevant, allowing the engineer to easily tune the total resource consumption of Traveller-delta.

Traveller is not a general memory system, but serves as a model for how specialized memory systems can beat general ones. Our result supports the [mismanaged genius hypothesis](https://alexzhang13.github.io/blog/2026/mgh/), which states that unoptimized model harnesses obscure latent model capabilities. We believe that Traveller-delta has direct applicability to enterprise chatbots and agents, and overall we hope that it represents a useful contribution to improving agent memory as a whole.


# Links
[Github repository](https://github.com/andrewcyuan/traveller-public)
