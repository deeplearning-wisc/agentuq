# AgentUQ on $\tau^{2}$-Bench Harness
Official codebase of the paper "_Uncertainty Quantification in LLM Agents: Foundations, Emerging Challenges, and Opportunities_", ACL 2026, a position paper on agent uncertainty quantification (UQ).
> By [Changdae Oh](https://changdaeoh.github.io/)<sup>1</sup>, [Seongheon Park](https://seongheon-96.github.io/)<sup>1</sup>, [To Eun Kim](https://kimdanny.github.io/)<sup>2</sup>, [Jiatong Li](https://cslijt.github.io/)<sup>1</sup>, [Wendi Li](https://wendili.github.io/)<sup>1</sup>, [Samuel Yeh](https://mhyeh.github.io/)<sup>1</sup>, 
> <br/>
[Sean Du](https://d12306.github.io/)<sup>3</sup>, [Hamed Hassani](https://www.seas.upenn.edu/~hassani/)<sup>4</sup>, [Paul Bogdan](https://cps.usc.edu/)<sup>5</sup>, [Dawn Song](https://dawnsong.io/)<sup>6</sup>, and [Sharon Li](https://pages.cs.wisc.edu/~sharonli/)<sup>1</sup>.
> <br/>
> <br/>
> <sup>1</sup>University of Wisconsin--Madison, <sup>2</sup>Carnegie Mellon University, <sup>3</sup>Nanyang Technological University, 
> <br/>
<sup>4</sup>University of Pennsylvania, <sup>5</sup>University of Southern California, <sup>6</sup>University of California, Berkeley

[![Paper](https://img.shields.io/badge/ArXiv-2602.05073-orange)](https://arxiv.org/pdf/2602.05073)
[![Website](https://img.shields.io/badge/Website-agentuq-royalblue)](https://agentuq.github.io/)
[![Dataset](https://img.shields.io/badge/Artifacts-tau2UQ-teal)](https://huggingface.co/datasets/changdae/tau2-uq-artifacts)

## News
- [Jun 16, 2026] Full code release. Sorry about my laziness🥲
- [Apr 10, 2026] [$\tau^2$-bench UQ artifacts](https://huggingface.co/datasets/changdae/tau2-uq-artifacts) (actual trajectories and uncertainty measurements) used in our paper are now available on HuggingFace datasets🤗 
- [Apr 5, 2026] AgentUQ position paper got accepted to [ACL 2026](https://2026.aclweb.org/) (main conference)🎉
- [Feb 26, 2026] AgentUQ position paper got accepted to ICLR 2026 Workshop, [Agentic AI in the Wild: From Hallucinations to Reliable Autonomy](https://hallucination-reliable-agentic-ai.github.io/)🎉

## Overview
This repository extends [$\tau^2$-bench](https://github.com/sierra-research/tau2-bench) with an **uncertainty quantification (UQ) pipeline** that captures token-level log-probabilities during multi-turn agent-user-tool interactions, aggregates them into trajectory-level uncertainty estimates, and evaluates how well those estimates predict task failure.

This fork adds:

- **Runtime UQ tracking** -- automatic token-level logprob capture during simulation
- **Post-hoc UQ analysis** -- aggregation of token-level data into turn-level and trajectory-level summaries
- **UQ evaluation** -- AUROC, AUARC, Pearson/Spearman/Kendall correlations between uncertainty and task failure
- **Observation UQ scoring** -- auxiliary LLM-based estimation of observation surprise (user messages and tool results), which is introduced in Figure 8 of the paper.

## Installation

```bash
git clone https://github.com/deeplearning-wisc/agentuq.git
cd agentuq
pip install -e .
```

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

## Quick Start

### 1. Run a simulation with UQ tracking

```bash
tau2 run \
    --domain retail \
    --agent-llm gpt-4.1 \
    --user-llm gpt-4.1 \
    --num-trials 1 \
    --output-dir ./results \
    --agent-llm-args '{"logprobs": true, "top_logprobs": 20}' \
    --user-llm-args '{"logprobs": true, "top_logprobs": 20}'
```

The key arguments are `logprobs: true` and `top_logprobs: 20`, which enable token-level logprob capture. The UQ summary is automatically embedded in the result JSON.
In practice, top 5 logprobs almost captures >= 97.5% of probability mass; you can just track top 5 for light caching of artifacts or so.

### 2. Extract and analyze UQ data

```bash
# Extract token-level data into sidecar JSONL files
tau2 extract-uq-from-trajs \
    --results ./results/gpt-4.1_retail.json \
    --output-dir ./uq_logprobs

# Aggregate into per-turn and per-trajectory summaries
tau2 analyze-uq-logprobs \
    --input ./uq_logprobs \
    --output-dir ./uq_analysis
```

### 3. Evaluate UQ estimates

```bash
tau2 evaluate-uq \
    --mode embedded \
    --results ./results/gpt-4.1_retail.json \
    --output-dir ./uq_eval
```

This outputs AUROC, AUARC, and correlation metrics measuring how well uncertainty predicts task failure.

### 4. Score observation uncertainty

```bash
tau2 score-observation-uq \
    --results ./results/gpt-4.1_retail.json \
    --output-dir ./uq_obs \
    --scorer-mode agent_llm \
    --scorer-api-base http://127.0.0.1:8000/v1 \
    --rescore-backend chat_replay
```

This replays each conversation and scores how "surprising" each observation was from the agent's perspective.

--- 

## UQ Pipeline

### Runtime Tracking

During simulation, token-level logprobs are captured for every LLM generation:

| Field | Description |
|-------|-------------|
| `chosen_logprob` | Log-probability of the chosen token |
| `chosen_prob` | Probability of the chosen token |
| `topk_entropy` | Shannon entropy over top-k token distribution |
| `topk_mass` | Total probability mass of top-k tokens |
| `top_logprobs` | Full top-k candidate list |

These are stored as sidecar JSONL files partitioned by task.

### Trajectory-Level Summary

Each simulation run includes a `uq_summary` with per-role statistics:

- `avg_token_nll` -- average negative log-likelihood per token 
- `mean_topk_entropy` -- average Shannon entropy over top-k distributions
- `trajectory_nll` -- total cross-entropy across all tokens
- `min_chosen_prob` -- minimum token probability
- `total_tokens` -- count of generated tokens

### Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| AUROC | Can uncertainty rank failures above successes? |
| AUARC | Accuracy-rejection curve: does removing high-uncertainty tasks improve accuracy? |
| Pearson | Linear correlation between uncertainty and 1-reward |
| Spearman | Rank correlation |
| Kendall tau-b | Ordinal association |

### Observation UQ

Complements action-side UQ by scoring how surprising observations are:

- **`agent_llm` mode**: Uses the agent's own model with its domain system prompt
- **`auxiliary_llm` mode**: Uses a separate observer model

---

## How It Works (Pseudocode)

### Step 1: Token-Level Logprob Extraction

After every LLM generation call during simulation, the pipeline extracts
per-token uncertainty measurements from the response:

```python
# For each token in the LLM response:
for token in response.choices[0].logprobs:

    chosen_logprob = token.logprob 
    chosen_prob    = exp(chosen_logprob)

    # Top-k entropy: Shannon entropy over the top-k candidate distribution
    topk_probs = [exp(lp) for lp in token.top_logprobs]
    mass       = sum(topk_probs)
    normalized = [p / mass for p in topk_probs] 
    topk_entropy = -sum(p * log(p) for p in normalized)

    # Save to sidecar JSONL (one record per token)
    save({
        token, chosen_logprob, chosen_prob,
        topk_entropy, topk_mass=mass,
        role,        # "assistant" or "user"
        turn_idx,    # which turn in the conversation
        token_idx,   # position within the turn
        domain, task_id, trial, seed
    })
```

### Step 2: Turn-Level and Trajectory-Level Aggregation

Token-level records are grouped and aggregated at two levels:

```python
# ── Turn-level: group tokens by (task_id, trial, seed, role, turn_idx) ──

for each turn:
    tokens = all token records in this turn
    N      = len(tokens)

    turn_nll         = sum(-t.chosen_logprob for t in tokens)  
    avg_token_nll    = turn_nll / N                            
    mean_topk_entropy = sum(t.topk_entropy for t in tokens) / N 
    min_chosen_prob  = min(t.chosen_prob for t in tokens)       

# ── Trajectory-level: group tokens by (task_id, trial, seed, role) ──

for each trajectory:
    tokens = all token records across all turns
    N      = len(tokens)

    trajectory_nll    = sum(-t.chosen_logprob for t in tokens)
    avg_token_nll     = trajectory_nll / N
    mean_topk_entropy = sum(t.topk_entropy for t in tokens) / N
    min_chosen_prob   = min(t.chosen_prob for t in tokens)

# ── Combined role: merge assistant + user tokens for a joint estimate ──

combined_nll       = assistant.trajectory_nll + user.trajectory_nll
combined_N         = assistant.N + user.N
combined_avg_nll   = combined_nll / combined_N
```

### Step 3: Uncertainty Evaluation

Trajectory-level uncertainty is joined with task rewards to evaluate
whether uncertainty predicts task failure:

```python
# ── Setup: pair each trajectory's uncertainty with its reward ──

pairs = []
for each trajectory:
    uncertainty = trajectory.avg_token_nll       # or any UQ metric
    reward      = trajectory.reward              # 0.0 (fail) or 1.0 (success)
    pairs.append((uncertainty, reward))

# ── AUROC: can uncertainty rank failures above successes? ──

failures  = [u for u, r in pairs if r < threshold]
successes = [u for u, r in pairs if r >= threshold]
ranks     = rankdata([u for u, r in pairs])    
sum_ranks_pos = sum(ranks[i] for i where label[i] == failure)

auroc = (sum_ranks_pos - n_fail * (n_fail + 1) / 2) / (n_fail * n_success)
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `tau2 run` | Run benchmark simulation |
| `tau2 extract-uq-from-trajs` | Extract UQ data from result JSON to sidecar files |
| `tau2 analyze-uq-logprobs` | Aggregate token-level UQ into summaries |
| `tau2 evaluate-uq` | Evaluate UQ estimates (AUROC, correlations) |
| `tau2 score-observation-uq` | Score observation uncertainty |

See `examples/` for complete usage scripts.

## Example Scripts
bash scripts with verbose annotations
- `examples/run_with_uq.sh` -- Full pipeline: simulate, extract, analyze
- `examples/evaluate_uq.sh` -- Evaluate UQ prediction quality
- `examples/score_observation_uq.sh` -- Observation UQ scoring with agent/auxiliary LLM


## Citation
```
@article{oh2026uncertainty,
    title={Uncertainty Quantification in LLM Agents: Foundations, Emerging Challenges, and Opportunities},
    author={Oh, Changdae and Park, Seongheon and Kim, To Eun and Li, Jiatong and Li, Wendi and Yeh, Samuel and Du, Xuefeng and Hassani, Hamed and Bogdan, Paul and Song, Dawn and Li, Sharon},
    journal={arXiv preprint arXiv:2602.05073},
    year={2026}
}
```

## License
This work is released under the MIT License.

## Acknowledgement
We thank the authors of [$\tau^2$-bench](https://github.com/sierra-research/tau2-bench), allowing us to easily build the overall pipeline.
