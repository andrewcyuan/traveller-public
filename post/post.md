---
title: "Traveller-delta: creating a SOTA memory system with specialized agent search tools"
author: "Andrew Yuan"
---

# Introduction

The modern standard for high-fidelity agent memory is a markdown repository.[^agent-memory] Agents write and update markdown notes as new information comes in, with a periodic reorganization phase ("sleeping"). Markdown repositories are great for remembering user preferences. However, in terms of accurately retrieving facts, error rate climbs quickly as context ramps up.

Around a month ago, I decided to take a dive into the world of agent memory, and see if I could come up with a better alternative to markdown. I found an excellent eval for retrieval in information-rich environments: LongMemEval-v2. And, with the help of dozens of autonomous research agents, I ended up creating Traveller-delta, a system that achieved state-of-the-art results in accuracy, latency, and cost.


| System (LongMemEval-v2 Small) | Accuracy | Mean query latency | Cost |
|---|---:|---:|---|
| **Traveller-delta, Gemini 3.8 Flash** | **89.14%** | 114.81 s | $70.80 |
| Traveller-delta, Claude Opus 5.5 medium | 88.69% | 18.98 s | $71.49 |
| Traveller-delta, Luna high, state-search | 81.60% | 35.32 s | $8.41 |
| Codex CLI, GPT-6 Astra medium | 86.25% | 25.24 s | $192.46 |
| Codex CLI, GPT-6.1 Sol high | 83.59% | 28.34 s | $41.71 |
| AgentRunbook-C V2 published reference | 75.61% | 130.54 s | Not recorded |
| Letta Dream | 25.50% | 19.30 s (retrieval) | $93.48 estimated |


## About LongMemEval-v2

LongMemEval-v2 is an agent memory evaluation which feeds 100 variable-length sequences per domain (up to 25M tokens per sequence) of realistic enterprise action data, then asks 451 questions about that data, spanning static state recall, dynamic state tracking, workflow knowledge, environment gotchas, and premise awareness. Questions are divided into three answer formats: true/false, multiple choice, and short answer. In this way, LongMemEval-v2 creates a realistic benchmark for questions that would be asked of an internal chatbot.[^lme]

# Existing work

Enterprise agent memory systems have been a topic of focus since the popularization of generative AI in 2022. During the past four years, two themes have dominated: **Retrieval augmented generation (RAG)** and **markdown repositories**.

**RAG** means splitting incoming information into chunks based on token length, generating embeddings for these chunks,[^common-embedders] and then dumping them into a vector database. At retrieval time, the query or a piece of text generated from the query, is embedded into the same dimensionality as the stored chunks. Similarity between the query vector `Q` and a given chunk vector `C` is determined by $$\frac{\mathbf{Q} \cdot \mathbf{C}}{|\mathbf{Q}| |\mathbf{C}|}$$. The most common issue with RAG systems, beyond the low intelligence of embedding models in general, is their failure to reason across chunks.[^multihop][^graphrag] We also observed that in our run, the query-to-slice RAG baseline (42.8% overall on Small in the authors’ results; not a local rerun) performed poorly, especially on environment gotchas (20.7%) and workflow knowledge (41.5%)[^rag-results].

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


In this way, theoretically, every answer is findable given sufficient retrieval work. The most similar setup to this is agent-runbook-c[^agent-runbook-c].

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
| Gemini 3.8 Flash retrieval + Qwen | 402/451 | 89.14% | 114.81 s | $70.80[^gemini-cost] |
| Claude Opus 5.5 medium + Qwen | 400/451 | 88.69% | 18.98 s | $71.49 |
| DeepSeek V4.1 Flash 1017 + Qwen | 352/451 | 78.05% | — | $57.24 |
| Codex CLI, GPT-6.1 Sol high | 377/451 | 83.59% | 28.34 s | $41.71 |
| Codex CLI, GPT-6 Astra medium | 389/451 | 86.25% | 25.24 s | $192.46 |
| AgentRunbook-C (paper, Small) | Not supplied as counts | 74.9% | 108.3 s | Not recorded |
| AgentRunbook-C V2 reference | Not supplied as counts | 75.61% | 130.54 s | Not recorded |
| ops-memory-v1 | Not located | Not established | — | Not recorded |
| Letta Dream | 115/451 | 25.50% | 19.30 s (retrieval) | $93.48 estimated |




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

Traveller is not a general memory system, but serves as a model for how specialized memory systems can beat general ones. Our result supports the mismanaged genius hypothesis[^mismanaged-genius], which states that unoptimized model harnesses obscure latent model capabilities. We believe that Traveller-delta has direct applicability to enterprise chatbots and agents, and overall we hope that it represents a useful contribution to improving agent memory as a whole.


# Links
[Github repository](https://github.com/andrewcyuan/traveller-public)

[^agent-memory]: See Cognition's [Agent Memory Repo](https://cognition.com/agent-memory-repo), Letta's [Introducing Context Repositories: Git-based Memory for Coding Agents](https://www.letta.com/blog/context-repositories/), Anthropic's [How Claude remembers your project](https://code.claude.com/docs/en/memory), LangChain's [How we built Agent Builder's memory system](https://www.langchain.com/blog/how-we-built-agent-builders-memory), and Cline's [Memory Bank](https://docs.cline.bot/best-practices/memory-bank).

[^common-embedders]: Popular embedding models include OpenAI's `text-embedding-3-large`, Perplexity's [`pplx-embed-v1-4B`](https://arxiv.org/abs/2602.11151), and Qwen's `Qwen3-Embedding-8B` (used by the LME-v2 authors).

[^agent-runbook-c]: [AgentRunbook-C V2](https://xiaowu0162.github.io/longmemeval-v2/agentrunbook-c-v2/), the published reference used in our October 24 comparison.

[^mismanaged-genius]: [The Mismanaged Geniuses Hypothesis](https://alexzhang13.github.io/blog/2026/mgh/), Alex Zhang, Zhening Li, and Omar Khattab.

[^multihop]: Tang and Yang, [MultiHop-RAG](https://arxiv.org/abs/2401.15391): evaluates retrieval and reasoning failures on multi-hop questions.

[^graphrag]: Edge et al., [From Local to Global](https://arxiv.org/abs/2404.16130): evaluates conventional RAG limitations on corpus-wide synthesis.

[^lme]: Wu et al., [LongMemEval-V2](https://arxiv.org/html/2605.12493v1), sections 3.2 and 5.1. Small has 100 trajectories per domain; the 25M-token figure describes haystacks, not each trajectory.

[^rag-results]: [LongMemEval-V2 Table 2](https://arxiv.org/html/2605.12493v1): Small overall/static/dynamic/workflow/gotchas accuracy is 42.8/47.1/42.5/41.5/20.7% for slice RAG, 51.0/52.4/49.6/52.8/41.4% for slice+notes, and 58.6/66.1/58.3/52.8/31.0% for AgentRunbook-R. Mean query times are 0.1/0.2/26.9 seconds. These are authors' results, not our reruns; category scores alone do not identify a cross-chunk failure mechanism.

