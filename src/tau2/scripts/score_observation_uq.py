"""Score observation uncertainty: quantify UQ over user messages and tool
results from the agent's (or an auxiliary model's) perspective.

This script is the primary post-hoc tool for observation UQ.  It walks
saved trajectories, reconstructs the scorer's context at each observation
step, and obtains token-level logprobs via teacher-forced rescoring.

Two scorer modes are supported:

* **agent_llm** — reuse the agent's own LLM and system prompt so the
  logprobs reflect what the agent "expected" to observe.
* **auxiliary_llm** — use a separate observer model with a custom system
  prompt that acts as a world-model approximator.

The resulting token rows are written to sidecar JSONL files with a
``scorer`` field (``"agent_llm"`` or ``"auxiliary_llm"``) so downstream
analysis can distinguish them from the default ``"self"`` rows produced
during generation.
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Optional

from tau2.utils.uq_logprobs import (
    append_rows_jsonl,
    build_token_logprob_rows,
    reset_uq_logprobs_context,
    set_uq_logprobs_context,
)
from tau2.utils.uq_rescoring import (
    message_to_target_text,
    rescore_message_with_chat_replay,
    rescore_message_with_completions,
    to_chat_message,
)

# Default system prompt template for the auxiliary observer model.
# The {domain_policy} placeholder is filled at runtime when the domain
# is known; when no domain policy is available the placeholder is
# replaced with a short fallback note.
DEFAULT_AUXILIARY_SYSTEM_PROMPT_TEMPLATE = """\
You are an impartial observer monitoring a customer-service conversation \
between an AI agent and a human user.

Your task is to predict the next observation that the agent will receive. \
An observation is either:
  1. A user message — the customer's natural-language reply, or
  2. A tool result  — the deterministic output returned by a backend API \
after the agent issues a tool call.

Given the full conversation history up to this point, generate the most \
likely continuation as if you were the user or the tool backend. \
Aim to model the distribution of plausible observations as faithfully as \
possible: assign high probability to expected responses and low \
probability to surprising ones.

<domain_knowledge>
{domain_policy}
</domain_knowledge>"""

_AUXILIARY_FALLBACK_DOMAIN_NOTE = (
    "No domain-specific policy is available. "
    "Rely on the conversation context to infer the domain."
)

# Template mirroring ``tau2.agent.llm_agent.SYSTEM_PROMPT`` so we can
# reconstruct the agent's view without importing the agent module (which
# may pull heavy dependencies).
_AGENT_SYSTEM_PROMPT_TEMPLATE = """\
<instructions>
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
</instructions>
<policy>
{domain_policy}
</policy>"""


def _get_domain_policy(domain: str) -> str:
    """Retrieve the domain policy string via the registry."""
    from tau2.registry import registry

    env_constructor = registry.get_env_constructor(domain)
    environment = env_constructor()
    return environment.get_policy()


def _iter_result_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return sorted(path.rglob("*.json"))


def _load_results(
    path: Path,
) -> tuple[Optional[str], Optional[str], Optional[dict], list[dict[str, Any]]]:
    """Load a result JSON and return (domain, agent_llm, agent_args, simulations)."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return None, None, None, []
    info = payload.get("info", {})
    domain = info.get("environment_info", {}).get("domain_name")
    agent_llm = info.get("agent_info", {}).get("llm")
    agent_args = info.get("agent_info", {}).get("llm_args") or {}
    simulations = payload.get("simulations", [])
    if not isinstance(simulations, list):
        simulations = []
    return domain, agent_llm, agent_args, simulations


def _build_scorer_context_messages(
    messages: list[dict[str, Any]],
    obs_idx: int,
    system_prompt: str,
) -> list[dict[str, Any]]:
    """Build the chat-API message list that the scorer model sees before
    the observation at *obs_idx*.

    The context is: system prompt + all messages before *obs_idx* that are
    visible from the scorer's perspective.
    """
    chat_messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for m in messages[:obs_idx]:
        if not isinstance(m, dict):
            continue
        chat_m = to_chat_message(m)
        if chat_m is not None:
            chat_messages.append(chat_m)
    return chat_messages


def _load_scored_simulations(
    output_dir: Path,
    scorer_mode: str,
) -> set[tuple]:
    """Scan sidecar JSONL files for already-scored (task_id, trial, seed, scorer) tuples."""
    scored: set[tuple] = set()
    if not output_dir.exists():
        return scored
    for jsonl_path in sorted(output_dir.glob("logprobs_*.jsonl")):
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("scorer") != scorer_mode:
                        continue
                    key = (
                        str(row.get("task_id", "")),
                        row.get("trial"),
                        row.get("seed"),
                        row.get("scorer"),
                    )
                    scored.add(key)
        except OSError:
            continue
    return scored


def _compute_obs_uq_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate token rows into a single observation UQ summary dict."""
    n = len(rows)
    if n == 0:
        return {}
    sum_neg_logprob = 0.0
    sum_entropy = 0.0
    min_prob = None
    for row in rows:
        lp = row.get("chosen_logprob")
        if lp is not None:
            sum_neg_logprob += -lp
        ent = row.get("topk_entropy")
        if ent is not None:
            sum_entropy += ent
        prob = row.get("chosen_prob")
        if prob is not None:
            if min_prob is None or prob < min_prob:
                min_prob = prob
    return {
        "total_tokens": n,
        "trajectory_nll": sum_neg_logprob,
        "avg_token_nll": sum_neg_logprob / n if n > 0 else None,
        "mean_topk_entropy": sum_entropy / n if n > 0 else None,
        "min_chosen_prob": min_prob,
    }


def score_single_trajectory_observations(
    messages: list[dict[str, Any]],
    domain: str,
    scorer_mode: str,
    scorer_llm: str,
    scorer_api_base: Optional[str] = None,
    scorer_api_key: Optional[str] = None,
    scorer_api_version: Optional[str] = None,
    scorer_llm_args: Optional[dict] = None,
    rescore_backend: str = "completion_echo",
    top_k: int = 20,
    timeout_sec: int = 120,
    auxiliary_system_prompt: Optional[str] = None,
    score_user_messages: bool = True,
    score_tool_messages: bool = True,
) -> dict[str, Any]:
    """Score observations in a single trajectory and return UQ summary.

    This function is designed for on-the-spot use (called from
    ``run.py`` after orchestrator finishes).

    Returns a dict with keys ``"observation_user"``,
    ``"observation_tool"``, and ``"observation_combined"``.
    """
    # Auto-detect Azure settings when the scorer LLM contains "azure"
    effective_api_base = scorer_api_base
    effective_api_key = scorer_api_key
    effective_api_version = scorer_api_version
    if "azure" in scorer_llm.lower():
        if effective_api_base is None:
            effective_api_base = os.environ.get("AZURE_API_BASE")
        if effective_api_key is None:
            effective_api_key = os.environ.get("AZURE_API_KEY")
        if effective_api_version is None:
            effective_api_version = os.environ.get("AZURE_API_VERSION")

    if effective_api_base is None:
        raise ValueError("scorer_api_base is required for observation UQ scoring")

    domain_policy = _get_domain_policy(domain)
    if scorer_mode == "agent_llm":
        system_prompt = _AGENT_SYSTEM_PROMPT_TEMPLATE.format(
            domain_policy=domain_policy
        )
    elif scorer_mode == "auxiliary_llm":
        if auxiliary_system_prompt is not None:
            system_prompt = auxiliary_system_prompt
        else:
            system_prompt = DEFAULT_AUXILIARY_SYSTEM_PROMPT_TEMPLATE.format(
                domain_policy=domain_policy or _AUXILIARY_FALLBACK_DOMAIN_NOTE
            )
    else:
        raise ValueError(f"Unknown scorer_mode: {scorer_mode}")

    temperature = (scorer_llm_args or {}).get("temperature")
    all_rows_by_role: dict[str, list[dict[str, Any]]] = {
        "observation_user": [],
        "observation_tool": [],
    }

    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "user" and not score_user_messages:
            continue
        if role == "tool" and not score_tool_messages:
            continue
        if role not in {"user", "tool"}:
            continue

        target_text = message_to_target_text(msg)
        if target_text is None:
            continue

        # Build scorer context: system prompt + history before this observation
        scorer_context = _build_scorer_context_messages(
            messages, msg_idx, system_prompt
        )

        # Build a synthetic message list for rescoring: context + target at idx
        rescore_messages = scorer_context + [msg]
        rescore_idx = len(rescore_messages) - 1

        pseudo_response = {}
        used_backend = None
        if rescore_backend in {"completion_echo", "auto"}:
            pseudo_response = rescore_message_with_completions(
                messages=rescore_messages,
                idx=rescore_idx,
                model=scorer_llm,
                api_base=effective_api_base,
                api_key=effective_api_key,
                temperature=temperature,
                top_k=top_k,
                timeout_sec=timeout_sec,
                api_version=effective_api_version,
            )
            if isinstance(pseudo_response, dict) and len(pseudo_response) > 0:
                used_backend = "completion_echo"

        if (
            not isinstance(pseudo_response, dict) or len(pseudo_response) == 0
        ) and rescore_backend in {"chat_replay", "auto"}:
            chat_response, _exact = rescore_message_with_chat_replay(
                messages=rescore_messages,
                idx=rescore_idx,
                model=scorer_llm,
                api_base=effective_api_base,
                api_key=effective_api_key,
                temperature=temperature,
                top_k=top_k,
                timeout_sec=timeout_sec,
                require_exact_match=False,
                api_version=effective_api_version,
            )
            if isinstance(chat_response, dict) and len(chat_response) > 0:
                pseudo_response = chat_response
                used_backend = "chat_replay"

        if not isinstance(pseudo_response, dict) or len(pseudo_response) == 0:
            continue

        rows = build_token_logprob_rows(
            raw_response=pseudo_response,
            model=scorer_llm,
            role=role,
            message_timestamp=msg.get("timestamp"),
            turn_idx=msg.get("turn_idx"),
            scorer=scorer_mode,
            request_params={
                "temperature": temperature,
                "logprobs": True,
                "top_logprobs": top_k,
                "rescored": True,
                "rescore_backend": used_backend,
                "scorer_mode": scorer_mode,
            },
        )
        if rows:
            obs_key = "observation_user" if role == "user" else "observation_tool"
            all_rows_by_role[obs_key].extend(rows)
            append_rows_jsonl(rows)

    summary: dict[str, Any] = {}
    for key, rows in all_rows_by_role.items():
        if rows:
            summary[key] = _compute_obs_uq_stats(rows)

    # Combined observation summary
    all_obs_rows = (
        all_rows_by_role["observation_user"] + all_rows_by_role["observation_tool"]
    )
    if all_obs_rows:
        summary["observation_combined"] = _compute_obs_uq_stats(all_obs_rows)

    return summary


def score_observation_uq(
    results_path: Path,
    output_dir: Path,
    scorer_mode: str = "agent_llm",
    scorer_llm: Optional[str] = None,
    scorer_api_base: Optional[str] = None,
    scorer_api_key: Optional[str] = None,
    scorer_api_version: Optional[str] = None,
    scorer_llm_args: Optional[dict] = None,
    rescore_backend: str = "completion_echo",
    top_k: int = 20,
    timeout_sec: int = 120,
    partition_mode: str = "task",
    task_ids: Optional[list[str]] = None,
    num_tasks: Optional[int] = None,
    score_user_messages: bool = True,
    score_tool_messages: bool = True,
    auxiliary_system_prompt: Optional[str] = None,
    no_resume: bool = False,
) -> dict[str, Any]:
    """Batch-score observations from saved result files.

    Walks each simulation's messages, identifies observations (user and
    tool messages), reconstructs the scorer's context, and obtains
    logprobs via teacher-forced rescoring.

    Supports incremental resume: when *output_dir* already contains
    sidecar JSONL files, simulations that were already scored (matched
    by ``(task_id, trial, seed, scorer)``) are skipped.  Pass
    ``no_resume=True`` to force re-scoring of everything.
    """
    os.environ["TAU2_UQ_LOGPROBS_DIR"] = str(output_dir)
    os.environ["TAU2_UQ_LOGPROBS_PARTITION_MODE"] = partition_mode

    stats: dict[str, Any] = {
        "files": 0,
        "simulations": 0,
        "simulations_skipped": 0,
        "observations_total": 0,
        "observations_scored": 0,
        "observations_failed": 0,
        "rows_written": 0,
        "scorer_mode": scorer_mode,
    }

    # --- Resume: load already-scored simulations ---
    scored_sims: set[tuple] = set()
    if not no_resume:
        scored_sims = _load_scored_simulations(output_dir, scorer_mode)
        if scored_sims:
            print(
                f"[resume] Found {len(scored_sims)} already-scored simulation(s) "
                f"in {output_dir} — will skip these."
            )

    for result_file in _iter_result_files(results_path):
        stats["files"] += 1
        domain, agent_llm, agent_args, simulations = _load_results(result_file)

        if domain is None:
            continue

        # Resolve scorer LLM: default to agent's own LLM in agent_llm mode
        effective_scorer_llm = scorer_llm
        if effective_scorer_llm is None:
            if scorer_mode == "agent_llm":
                effective_scorer_llm = agent_llm
            else:
                raise ValueError("scorer_llm must be specified for auxiliary_llm mode")
        if effective_scorer_llm is None:
            continue

        # Auto-detect Azure settings when the scorer LLM contains "azure"
        effective_api_base = scorer_api_base
        effective_api_key = scorer_api_key
        effective_api_version = scorer_api_version
        if "azure" in effective_scorer_llm.lower():
            if effective_api_base is None:
                effective_api_base = os.environ.get("AZURE_API_BASE")
            if effective_api_key is None:
                effective_api_key = os.environ.get("AZURE_API_KEY")
            if effective_api_version is None:
                effective_api_version = os.environ.get("AZURE_API_VERSION")

        if effective_api_base is None:
            raise ValueError("scorer_api_base is required for observation UQ scoring")

        # Build the system prompt once per result file
        domain_policy = _get_domain_policy(domain)
        if scorer_mode == "agent_llm":
            system_prompt = _AGENT_SYSTEM_PROMPT_TEMPLATE.format(
                domain_policy=domain_policy
            )
        elif auxiliary_system_prompt is not None:
            system_prompt = auxiliary_system_prompt
        else:
            system_prompt = DEFAULT_AUXILIARY_SYSTEM_PROMPT_TEMPLATE.format(
                domain_policy=domain_policy or _AUXILIARY_FALLBACK_DOMAIN_NOTE
            )

        temperature = (scorer_llm_args or agent_args or {}).get("temperature")

        # Resolve the effective set of task IDs to score.
        effective_task_ids = None
        if task_ids is not None:
            effective_task_ids = set(task_ids)
        if num_tasks is not None:
            # Collect unique task IDs in the order they appear.
            seen: dict[str, None] = {}
            for s in simulations:
                tid = s.get("task_id")
                if tid is not None and tid not in seen:
                    if effective_task_ids is not None and tid not in effective_task_ids:
                        continue
                    seen[tid] = None
            first_n = set(list(seen.keys())[:num_tasks])
            effective_task_ids = first_n

        for sim in simulations:
            task_id = sim.get("task_id")
            if effective_task_ids is not None and task_id not in effective_task_ids:
                continue

            trial = sim.get("trial")
            seed = sim.get("seed")

            # Skip already-scored simulations when resuming
            sim_key = (str(task_id), trial, seed, scorer_mode)
            if sim_key in scored_sims:
                stats["simulations_skipped"] += 1
                continue

            stats["simulations"] += 1
            messages = sim.get("messages", [])
            if not isinstance(messages, list):
                continue

            token = set_uq_logprobs_context(
                domain=domain,
                task_id=task_id,
                trial=trial,
                seed=seed,
            )
            try:
                for msg_idx, msg in enumerate(messages):
                    if not isinstance(msg, dict):
                        continue
                    role = msg.get("role")
                    if role == "user" and not score_user_messages:
                        continue
                    if role == "tool" and not score_tool_messages:
                        continue
                    if role not in {"user", "tool"}:
                        continue

                    target_text = message_to_target_text(msg)
                    if target_text is None:
                        continue

                    stats["observations_total"] += 1

                    # Build scorer context
                    scorer_context = _build_scorer_context_messages(
                        messages, msg_idx, system_prompt
                    )
                    rescore_messages = scorer_context + [msg]
                    rescore_idx = len(rescore_messages) - 1

                    try:
                        pseudo_response = {}
                        used_backend = None

                        if rescore_backend in {"completion_echo", "auto"}:
                            pseudo_response = rescore_message_with_completions(
                                messages=rescore_messages,
                                idx=rescore_idx,
                                model=effective_scorer_llm,
                                api_base=effective_api_base,
                                api_key=effective_api_key,
                                temperature=temperature,
                                top_k=top_k,
                                timeout_sec=timeout_sec,
                                api_version=effective_api_version,
                            )
                            if (
                                isinstance(pseudo_response, dict)
                                and len(pseudo_response) > 0
                            ):
                                used_backend = "completion_echo"

                        if (
                            not isinstance(pseudo_response, dict)
                            or len(pseudo_response) == 0
                        ) and rescore_backend in {"chat_replay", "auto"}:
                            chat_response, _exact = rescore_message_with_chat_replay(
                                messages=rescore_messages,
                                idx=rescore_idx,
                                model=effective_scorer_llm,
                                api_base=effective_api_base,
                                api_key=effective_api_key,
                                temperature=temperature,
                                top_k=top_k,
                                timeout_sec=timeout_sec,
                                require_exact_match=False,
                                api_version=effective_api_version,
                            )
                            if (
                                isinstance(chat_response, dict)
                                and len(chat_response) > 0
                            ):
                                pseudo_response = chat_response
                                used_backend = "chat_replay"

                        if (
                            not isinstance(pseudo_response, dict)
                            or len(pseudo_response) == 0
                        ):
                            stats["observations_failed"] += 1
                            continue

                        rows = build_token_logprob_rows(
                            raw_response=pseudo_response,
                            model=effective_scorer_llm,
                            role=role,
                            message_timestamp=msg.get("timestamp"),
                            turn_idx=msg.get("turn_idx"),
                            scorer=scorer_mode,
                            request_params={
                                "temperature": temperature,
                                "logprobs": True,
                                "top_logprobs": top_k,
                                "rescored": True,
                                "rescore_backend": used_backend,
                                "scorer_mode": scorer_mode,
                            },
                        )
                        if rows:
                            append_rows_jsonl(rows)
                            stats["observations_scored"] += 1
                            stats["rows_written"] += len(rows)
                        else:
                            stats["observations_failed"] += 1
                    except Exception:
                        stats["observations_failed"] += 1
            finally:
                reset_uq_logprobs_context(token)

    stats["output_dir"] = str(output_dir)
    stats["partition_mode"] = partition_mode
    return stats


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Score observation uncertainty: quantify UQ over user messages "
            "and tool results from the agent's or an auxiliary model's perspective."
        )
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to one Tau2 result JSON file or a directory containing JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write observation UQ sidecar JSONL files.",
    )
    parser.add_argument(
        "--scorer-mode",
        default="agent_llm",
        choices=["agent_llm", "auxiliary_llm"],
        help=(
            "Scorer mode. 'agent_llm' uses the agent's own LLM and system prompt. "
            "'auxiliary_llm' uses a separate observer model. Default: agent_llm."
        ),
    )
    parser.add_argument(
        "--scorer-llm",
        default=None,
        help=(
            "Model name for the scorer LLM. Defaults to the agent's LLM in "
            "agent_llm mode. Required for auxiliary_llm mode."
        ),
    )
    parser.add_argument(
        "--scorer-api-base",
        default=None,
        help=(
            "API base URL for scoring. For Azure models (scorer-llm contains "
            "'azure'), defaults to AZURE_API_BASE env var."
        ),
    )
    parser.add_argument(
        "--scorer-api-key",
        default=None,
        help=(
            "API key for the scorer endpoint. For Azure models, defaults to "
            "AZURE_API_KEY env var."
        ),
    )
    parser.add_argument(
        "--scorer-api-version",
        default=None,
        help=(
            "API version for Azure OpenAI. Defaults to AZURE_API_VERSION env var "
            "when the scorer LLM contains 'azure'."
        ),
    )
    parser.add_argument(
        "--scorer-llm-args",
        type=json.loads,
        default=None,
        help="JSON dict of LLM arguments for the scorer (e.g. '{\"temperature\": 0.0}').",
    )
    parser.add_argument(
        "--rescore-backend",
        default="completion_echo",
        choices=["completion_echo", "chat_replay", "auto"],
        help="Rescoring backend. Default: completion_echo.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k logprobs to request. Default: 20.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=120,
        help="HTTP timeout per rescoring request in seconds. Default: 120.",
    )
    parser.add_argument(
        "--partition-mode",
        default="task",
        choices=["task", "single"],
        help="Output partition mode. Default: task.",
    )
    parser.add_argument(
        "--task-ids",
        nargs="+",
        default=None,
        help="Filter to specific task IDs.",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Score only the first N unique tasks. Can combine with --task-ids.",
    )
    parser.add_argument(
        "--no-user-messages",
        action="store_true",
        help="Skip scoring user messages (only score tool results).",
    )
    parser.add_argument(
        "--no-tool-messages",
        action="store_true",
        help="Skip scoring tool messages (only score user messages).",
    )
    parser.add_argument(
        "--auxiliary-system-prompt",
        default=None,
        help="Custom system prompt for auxiliary_llm mode.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing scored simulations and re-score everything.",
    )
    args = parser.parse_args()

    stats = score_observation_uq(
        results_path=Path(args.results),
        output_dir=Path(args.output_dir),
        scorer_mode=args.scorer_mode,
        scorer_llm=args.scorer_llm,
        scorer_api_base=args.scorer_api_base,
        scorer_api_key=args.scorer_api_key,
        scorer_api_version=args.scorer_api_version,
        scorer_llm_args=args.scorer_llm_args,
        rescore_backend=args.rescore_backend,
        top_k=args.top_k,
        timeout_sec=args.timeout_sec,
        partition_mode=args.partition_mode,
        task_ids=args.task_ids,
        num_tasks=args.num_tasks,
        score_user_messages=not args.no_user_messages,
        score_tool_messages=not args.no_tool_messages,
        auxiliary_system_prompt=args.auxiliary_system_prompt,
        no_resume=args.no_resume,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
