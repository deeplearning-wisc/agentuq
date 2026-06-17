import hashlib
import json
import math
import os
import re
import threading
from contextvars import ContextVar, Token
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from tau2.utils.utils import DATA_DIR

_WRITE_LOCK = threading.Lock()
_OUTPUT_PATH: Optional[Path] = None
_OUTPUT_PATH_BY_KEY: dict[str, Path] = {}
_LOGPROBS_CONTEXT: ContextVar[dict[str, Any]] = ContextVar(
    "tau2_uq_logprobs_context", default={}
)
_UQ_ACCUMULATOR: ContextVar[dict[str, dict[str, Any]]] = ContextVar(
    "tau2_uq_accumulator", default={}
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sanitize_filename_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


# Most filesystems (ext4, btrfs, XFS, APFS, NTFS) limit filenames to 255
# bytes.  We use a conservative limit to leave room for encoding overhead.
_MAX_FILENAME_BYTES = 255


def _truncate_filename(file_name: str) -> str:
    """Truncate a filename to fit within the filesystem limit.

    When the filename (UTF-8 encoded) exceeds ``_MAX_FILENAME_BYTES``, the
    stem is shortened and a short hash of the *original* full stem is appended
    to preserve uniqueness.
    """
    if len(file_name.encode("utf-8")) <= _MAX_FILENAME_BYTES:
        return file_name
    # Separate extension (e.g. ".jsonl")
    dot_idx = file_name.rfind(".")
    if dot_idx > 0:
        stem, ext = file_name[:dot_idx], file_name[dot_idx:]
    else:
        stem, ext = file_name, ""
    # 8-char hex hash of the original stem for uniqueness
    digest = hashlib.sha256(stem.encode("utf-8")).hexdigest()[:8]
    suffix = f"_{digest}{ext}"
    max_stem_bytes = _MAX_FILENAME_BYTES - len(suffix.encode("utf-8"))
    # Truncate stem to fit (char-by-char to avoid breaking multi-byte chars)
    truncated = stem
    while len(truncated.encode("utf-8")) > max_stem_bytes:
        truncated = truncated[:-1]
    return truncated + suffix


def set_uq_logprobs_context(
    *,
    domain: Optional[str] = None,
    task_id: Optional[str] = None,
    trial: Optional[int] = None,
    seed: Optional[int] = None,
) -> tuple[Token, Token]:
    ctx = {
        "domain": domain,
        "task_id": task_id,
        "trial": trial,
        "seed": seed,
    }
    ctx_token = _LOGPROBS_CONTEXT.set(ctx)
    acc_token = _UQ_ACCUMULATOR.set({})
    return ctx_token, acc_token


def reset_uq_logprobs_context(token: tuple[Token, Token] | Token) -> None:
    if isinstance(token, tuple):
        ctx_token, acc_token = token
        _LOGPROBS_CONTEXT.reset(ctx_token)
        _UQ_ACCUMULATOR.reset(acc_token)
    else:
        # Backward compatibility: single token (context only)
        _LOGPROBS_CONTEXT.reset(token)


def _get_uq_logprobs_context() -> dict[str, Any]:
    return deepcopy(_LOGPROBS_CONTEXT.get())


def _update_accumulator(role: str, rows: list[dict[str, Any]]) -> None:
    """Update the per-role in-memory UQ accumulator with token rows."""
    accum = _UQ_ACCUMULATOR.get()
    if role not in accum:
        accum[role] = {
            "num_tokens": 0,
            "sum_neg_logprob": 0.0,
            "sum_entropy": 0.0,
            "min_chosen_prob": None,
        }
    stats = accum[role]
    for row in rows:
        stats["num_tokens"] += 1
        chosen_logprob = row.get("chosen_logprob")
        if chosen_logprob is not None:
            stats["sum_neg_logprob"] += -chosen_logprob
        entropy = row.get("topk_entropy")
        if entropy is not None:
            stats["sum_entropy"] += entropy
        chosen_prob = row.get("chosen_prob")
        if chosen_prob is not None:
            cur_min = stats["min_chosen_prob"]
            if cur_min is None or chosen_prob < cur_min:
                stats["min_chosen_prob"] = chosen_prob


def get_trajectory_uq_summary() -> Optional[dict[str, Any]]:
    """
    Return the trajectory-level UQ summary accumulated during the current
    simulation context.

    Returns a dict keyed by role (e.g. ``"assistant"``, ``"user"``), each
    containing aggregate uncertainty metrics.  When data exists for more
    than one role, a ``"combined"`` key is added with metrics aggregated
    across all roles.  Returns ``None`` when no token data has been
    accumulated.
    """
    accum = _UQ_ACCUMULATOR.get()
    if not accum:
        return None
    summary: dict[str, Any] = {}
    for role, stats in accum.items():
        n = stats["num_tokens"]
        summary[role] = {
            "total_tokens": n,
            "trajectory_nll": stats["sum_neg_logprob"],
            "avg_token_nll": stats["sum_neg_logprob"] / n if n > 0 else None,
            "mean_topk_entropy": stats["sum_entropy"] / n if n > 0 else None,
            "min_chosen_prob": stats["min_chosen_prob"],
        }

    # Build a combined (agent + user) summary when multiple roles are present.
    if len(accum) > 1:
        total_n = 0
        total_neg_logprob = 0.0
        total_entropy = 0.0
        combined_min_prob = None
        for stats in accum.values():
            total_n += stats["num_tokens"]
            total_neg_logprob += stats["sum_neg_logprob"]
            total_entropy += stats["sum_entropy"]
            p = stats["min_chosen_prob"]
            if p is not None and (combined_min_prob is None or p < combined_min_prob):
                combined_min_prob = p
        summary["combined"] = {
            "total_tokens": total_n,
            "trajectory_nll": total_neg_logprob,
            "avg_token_nll": total_neg_logprob / total_n if total_n > 0 else None,
            "mean_topk_entropy": (
                total_entropy / total_n if total_n > 0 else None
            ),
            "min_chosen_prob": combined_min_prob,
        }

    return summary


def _get_partition_key(mode: str, context: dict[str, Any]) -> str:
    domain = context.get("domain")
    task_id = context.get("task_id")
    trial = context.get("trial")
    seed = context.get("seed")
    if mode == "task" and domain and task_id and trial is not None:
        return f"task::{domain}::{task_id}::trial{trial}::seed{seed}"
    return "single"


def _get_output_path(context: dict[str, Any]) -> Path:
    global _OUTPUT_PATH
    global _OUTPUT_PATH_BY_KEY

    explicit_path = os.getenv("TAU2_UQ_LOGPROBS_PATH")
    if explicit_path:
        path = Path(explicit_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    mode = os.getenv("TAU2_UQ_LOGPROBS_PARTITION_MODE", "task").strip().lower()
    if mode not in {"task", "single"}:
        mode = "task"

    output_dir = Path(
        os.getenv("TAU2_UQ_LOGPROBS_DIR", str(DATA_DIR / "uq_logprobs"))
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    partition_key = _get_partition_key(mode=mode, context=context)

    if partition_key == "single":
        if _OUTPUT_PATH is not None:
            return _OUTPUT_PATH
        file_name = f"logprobs_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.jsonl"
        _OUTPUT_PATH = output_dir / file_name
        return _OUTPUT_PATH

    if partition_key in _OUTPUT_PATH_BY_KEY:
        return _OUTPUT_PATH_BY_KEY[partition_key]

    domain = _sanitize_filename_component(str(context.get("domain", "unknown_domain")))
    task_id = _sanitize_filename_component(str(context.get("task_id", "unknown_task")))
    trial = context.get("trial")
    seed = context.get("seed")
    trial_str = f"trial{trial}" if trial is not None else "trialNA"
    seed_str = f"seed{seed}" if seed is not None else "seedNA"
    file_name = _truncate_filename(
        f"logprobs_{domain}_{task_id}_{trial_str}_{seed_str}.jsonl"
    )
    path = output_dir / file_name
    _OUTPUT_PATH_BY_KEY[partition_key] = path
    return path


def strip_logprobs_from_response(raw_response: dict[str, Any]) -> dict[str, Any]:
    """
    Return a copy of the response with bulky token logprobs removed.
    """
    cleaned = deepcopy(raw_response)
    cleaned.pop("logprobs", None)
    message = cleaned.get("message")
    if isinstance(message, dict):
        message.pop("logprobs", None)
    choices = cleaned.get("choices", [])
    if not isinstance(choices, list):
        return cleaned
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        choice.pop("logprobs", None)
        message = choice.get("message")
        if isinstance(message, dict):
            message.pop("logprobs", None)
    return cleaned


def _extract_token_rows(raw_response: dict[str, Any]) -> list[dict[str, Any]]:
    # Full OpenAI-style response: {"choices":[{"logprobs":{"content":[...]}}]}
    choices = raw_response.get("choices", [])
    if isinstance(choices, list) and choices:
        choice0 = choices[0] if isinstance(choices[0], dict) else {}
        logprobs = choice0.get("logprobs", {})
    else:
        # Choice-level payload saved in message.raw_data
        # {"logprobs":{"content":[...]}, "message": ...}
        logprobs = raw_response.get("logprobs", {})
    if not isinstance(logprobs, dict):
        return []
    content = logprobs.get("content", [])
    if not isinstance(content, list):
        return []
    return [item for item in content if isinstance(item, dict)]


def _compute_topk_entropy(
    topk_logprobs: list[float],
) -> tuple[Optional[float], Optional[float]]:
    if not topk_logprobs:
        return None, None
    probs = [math.exp(lp) for lp in topk_logprobs]
    mass = sum(probs)
    if mass <= 0:
        return None, None
    normalized = [p / mass for p in probs]
    entropy = -sum(p * math.log(max(p, 1e-12)) for p in normalized)
    return entropy, mass


def build_token_logprob_rows(
    raw_response: dict[str, Any],
    *,
    model: str,
    role: str,
    message_timestamp: Optional[str] = None,
    turn_idx: Optional[int] = None,
    request_params: Optional[dict[str, Any]] = None,
    scorer: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Convert OpenAI-compatible chat logprobs payload to row-wise records.

    Each row is one generated token with chosen token logprob and top-k candidates.

    Parameters
    ----------
    scorer : str, optional
        Identifies who produced the logprobs.  ``"self"`` (default when
        ``None``) means the generating model itself; ``"agent_llm"`` or
        ``"auxiliary_llm"`` indicate observation-UQ scoring.
    """
    context = _get_uq_logprobs_context()
    token_rows = _extract_token_rows(raw_response)
    if not token_rows:
        return []

    records: list[dict[str, Any]] = []
    for token_idx, item in enumerate(token_rows):
        token = item.get("token")
        chosen_logprob = _safe_float(item.get("logprob"))

        topk_entries = item.get("top_logprobs", [])
        if not isinstance(topk_entries, list):
            topk_entries = []

        topk_tokens: list[str] = []
        topk_logprobs: list[float] = []
        for candidate in topk_entries:
            if not isinstance(candidate, dict):
                continue
            candidate_token = candidate.get("token")
            candidate_logprob = _safe_float(candidate.get("logprob"))
            if candidate_token is None or candidate_logprob is None:
                continue
            topk_tokens.append(str(candidate_token))
            topk_logprobs.append(candidate_logprob)

        topk_entropy, topk_mass = _compute_topk_entropy(topk_logprobs)
        record = {
            "recorded_at": _utc_now(),
            "message_timestamp": message_timestamp,
            "model": model,
            "role": role,
            "turn_idx": turn_idx,
            "token_idx": token_idx,
            "token": token,
            "chosen_logprob": chosen_logprob,
            "chosen_prob": math.exp(chosen_logprob)
            if chosen_logprob is not None
            else None,
            "topk_tokens": topk_tokens,
            "topk_logprobs": topk_logprobs,
            "topk_entropy": topk_entropy,
            "topk_mass": topk_mass,
            "domain": context.get("domain"),
            "task_id": context.get("task_id"),
            "trial": context.get("trial"),
            "seed": context.get("seed"),
        }
        if scorer is not None:
            record["scorer"] = scorer
        if request_params:
            record["request_params"] = request_params
        records.append(record)
    return records


def append_rows_jsonl(rows: list[dict[str, Any]]) -> Optional[Path]:
    if not rows:
        return None
    context = _get_uq_logprobs_context()
    output_path = _get_output_path(context=context)
    with _WRITE_LOCK:
        with open(output_path, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_path


def save_response_logprobs(
    *,
    raw_response: dict[str, Any],
    model: str,
    role: str,
    message_timestamp: Optional[str],
    turn_idx: Optional[int],
    request_params: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """
    Save token-level top-k logprobs to a sidecar JSONL file and update the
    in-memory trajectory-level UQ accumulator.
    """
    try:
        rows = build_token_logprob_rows(
            raw_response=raw_response,
            model=model,
            role=role,
            message_timestamp=message_timestamp,
            turn_idx=turn_idx,
            request_params=request_params,
        )
        if not rows:
            return None
        # Update in-memory accumulator for trajectory-level UQ
        _update_accumulator(role=role, rows=rows)
        output_path = append_rows_jsonl(rows)
        if output_path is None:
            return None
        return {
            "enabled": True,
            "format": "jsonl",
            "path": str(output_path),
            "num_tokens": len(rows),
            "schema_version": "uq_logprobs_v1",
        }
    except Exception as e:
        logger.warning(f"Failed to save token logprobs sidecar: {e}")
        return {
            "enabled": False,
            "error": str(e),
            "schema_version": "uq_logprobs_v1",
        }
