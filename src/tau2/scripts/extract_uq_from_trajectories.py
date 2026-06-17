import argparse
import json
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
    rescore_message_with_chat_replay,
    rescore_message_with_completions,
)


def _iter_result_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return sorted(path.rglob("*.json"))


def _load_simulations(
    path: Path,
) -> tuple[
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[dict],
    Optional[dict],
    list[dict[str, Any]],
]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return None, None, None, None, None, []
    info = payload.get("info", {})
    domain = info.get("environment_info", {}).get("domain_name")
    agent_llm = info.get("agent_info", {}).get("llm")
    user_llm = info.get("user_info", {}).get("llm")
    agent_args = info.get("agent_info", {}).get("llm_args") or {}
    user_args = info.get("user_info", {}).get("llm_args") or {}
    simulations = payload.get("simulations", [])
    if not isinstance(simulations, list):
        simulations = []
    return domain, agent_llm, user_llm, agent_args, user_args, simulations


def _model_for_role(
    role: str, agent_llm: Optional[str], user_llm: Optional[str]
) -> Optional[str]:
    if role == "assistant":
        return agent_llm
    if role == "user":
        return user_llm
    return None


def _request_params_for_role(role: str, agent_args: dict, user_args: dict) -> dict:
    src = agent_args if role == "assistant" else user_args
    return {
        "temperature": src.get("temperature"),
        "logprobs": src.get("logprobs"),
        "top_logprobs": src.get("top_logprobs"),
    }


def extract_uq_from_trajectories(
    results_path: Path,
    output_dir: Path,
    partition_mode: str = "task",
    rescore_missing: bool = False,
    rescore_api_base: Optional[str] = None,
    rescore_api_key: Optional[str] = None,
    rescore_top_k: int = 20,
    rescore_timeout_sec: int = 120,
    rescore_backend: str = "chat_replay",
    chat_replay_require_exact_match: bool = True,
) -> dict[str, Any]:
    os.environ["TAU2_UQ_LOGPROBS_DIR"] = str(output_dir)
    os.environ["TAU2_UQ_LOGPROBS_PARTITION_MODE"] = partition_mode

    stats = {
        "files": 0,
        "simulations": 0,
        "messages_total": 0,
        "messages_with_raw_data": 0,
        "messages_with_logprobs": 0,
        "messages_without_logprobs": 0,
        "messages_rescored": 0,
        "messages_rescore_failed": 0,
        "messages_rescored_chat_replay": 0,
        "messages_rescored_completion_echo": 0,
        "messages_chat_replay_mismatch": 0,
        "rows_written": 0,
    }

    for result_file in _iter_result_files(results_path):
        stats["files"] += 1
        domain, agent_llm, user_llm, agent_args, user_args, simulations = (
            _load_simulations(result_file)
        )
        for sim in simulations:
            stats["simulations"] += 1
            task_id = sim.get("task_id")
            trial = sim.get("trial")
            seed = sim.get("seed")
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
                    stats["messages_total"] += 1
                    if not isinstance(msg, dict):
                        continue
                    role = msg.get("role")
                    if role not in {"assistant", "user"}:
                        continue
                    raw_data = msg.get("raw_data")
                    if isinstance(raw_data, dict):
                        stats["messages_with_raw_data"] += 1
                    else:
                        raw_data = {}

                    model = _model_for_role(
                        role=role, agent_llm=agent_llm, user_llm=user_llm
                    )
                    if model is None:
                        continue
                    rows = build_token_logprob_rows(
                        raw_response=raw_data,
                        model=model,
                        role=role,
                        message_timestamp=msg.get("timestamp"),
                        turn_idx=msg.get("turn_idx"),
                        request_params=_request_params_for_role(
                            role=role,
                            agent_args=agent_args or {},
                            user_args=user_args or {},
                        ),
                    )
                    if len(rows) == 0:
                        if (
                            rescore_missing
                            and rescore_api_base is not None
                            and model.startswith("openai/")
                        ):
                            try:
                                req_temp = _request_params_for_role(
                                    role=role,
                                    agent_args=agent_args or {},
                                    user_args=user_args or {},
                                ).get("temperature")
                                pseudo_response = {}
                                used_backend = None
                                if rescore_backend in {"chat_replay", "auto"}:
                                    chat_response, exact_match = (
                                        rescore_message_with_chat_replay(
                                            messages=messages,
                                            idx=msg_idx,
                                            model=model,
                                            api_base=rescore_api_base,
                                            api_key=rescore_api_key,
                                            temperature=req_temp,
                                            top_k=rescore_top_k,
                                            timeout_sec=rescore_timeout_sec,
                                            require_exact_match=chat_replay_require_exact_match,
                                        )
                                    )
                                    if (
                                        isinstance(chat_response, dict)
                                        and len(chat_response) > 0
                                    ):
                                        pseudo_response = chat_response
                                        used_backend = "chat_replay"
                                    elif not exact_match:
                                        stats["messages_chat_replay_mismatch"] += 1
                                if (
                                    not isinstance(pseudo_response, dict)
                                    or len(pseudo_response) == 0
                                ) and rescore_backend in {
                                    "completion_echo",
                                    "auto",
                                    "chat_replay",
                                }:
                                    pseudo_response = rescore_message_with_completions(
                                        messages=messages,
                                        idx=msg_idx,
                                        model=model,
                                        api_base=rescore_api_base,
                                        api_key=rescore_api_key,
                                        temperature=req_temp,
                                        top_k=rescore_top_k,
                                        timeout_sec=rescore_timeout_sec,
                                    )
                                    if (
                                        isinstance(pseudo_response, dict)
                                        and len(pseudo_response) > 0
                                    ):
                                        used_backend = "completion_echo"
                                if isinstance(pseudo_response, dict):
                                    rows = build_token_logprob_rows(
                                        raw_response=pseudo_response,
                                        model=model,
                                        role=role,
                                        message_timestamp=msg.get("timestamp"),
                                        turn_idx=msg.get("turn_idx"),
                                        request_params={
                                            "temperature": _request_params_for_role(
                                                role=role,
                                                agent_args=agent_args or {},
                                                user_args=user_args or {},
                                            ).get("temperature"),
                                            "logprobs": True,
                                            "top_logprobs": rescore_top_k,
                                            "rescored": True,
                                            "rescore_backend": used_backend,
                                        },
                                    )
                                    if len(rows) > 0:
                                        stats["messages_rescored"] += 1
                                        if used_backend == "chat_replay":
                                            stats["messages_rescored_chat_replay"] += 1
                                        elif used_backend == "completion_echo":
                                            stats[
                                                "messages_rescored_completion_echo"
                                            ] += 1
                                    else:
                                        stats["messages_rescore_failed"] += 1
                            except Exception:
                                stats["messages_rescore_failed"] += 1
                        if len(rows) == 0:
                            stats["messages_without_logprobs"] += 1
                            continue
                    stats["messages_with_logprobs"] += 1
                    append_rows_jsonl(rows)
                    stats["rows_written"] += len(rows)
            finally:
                reset_uq_logprobs_context(token)

    stats["output_dir"] = str(output_dir)
    stats["partition_mode"] = partition_mode
    return stats


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Backfill UQ sidecar artifacts from pre-extracted trajectory files. "
            "Can either extract existing logprobs from raw_data or rescore missing ones."
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
        help="Directory to write extracted UQ JSONL files.",
    )
    parser.add_argument(
        "--partition-mode",
        default="task",
        choices=["task", "single"],
        help="Output partition mode. Default is task.",
    )
    parser.add_argument(
        "--rescore-missing",
        action="store_true",
        help=(
            "If set, messages without raw logprobs are rescored via /v1/completions "
            "using teacher-forced prompt echo."
        ),
    )
    parser.add_argument(
        "--rescore-api-base",
        default=None,
        help="OpenAI-compatible API base URL used for rescoring (e.g. http://127.0.0.1:8000/v1).",
    )
    parser.add_argument(
        "--rescore-api-key",
        default=None,
        help="Optional API key for rescoring endpoint.",
    )
    parser.add_argument(
        "--rescore-top-k",
        type=int,
        default=20,
        help="Top-k logprobs used in rescoring mode. Default is 20.",
    )
    parser.add_argument(
        "--rescore-timeout-sec",
        type=int,
        default=120,
        help="HTTP timeout per rescoring request in seconds. Default is 120.",
    )
    parser.add_argument(
        "--rescore-backend",
        default="chat_replay",
        choices=["chat_replay", "completion_echo", "auto"],
        help=(
            "Rescoring backend. chat_replay uses /chat/completions with original history; "
            "completion_echo uses /completions echo forcing; auto tries chat_replay then fallback."
        ),
    )
    parser.add_argument(
        "--chat-replay-require-exact-match",
        action="store_true",
        default=False,
        help="Require regenerated text to exactly match saved text for chat_replay rows.",
    )
    args = parser.parse_args()
    stats = extract_uq_from_trajectories(
        results_path=Path(args.results),
        output_dir=Path(args.output_dir),
        partition_mode=args.partition_mode,
        rescore_missing=args.rescore_missing,
        rescore_api_base=args.rescore_api_base,
        rescore_api_key=args.rescore_api_key,
        rescore_top_k=args.rescore_top_k,
        rescore_timeout_sec=args.rescore_timeout_sec,
        rescore_backend=args.rescore_backend,
        chat_replay_require_exact_match=args.chat_replay_require_exact_match,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
