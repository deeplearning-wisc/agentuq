import argparse
import csv
import json
from pathlib import Path
from typing import Optional


def _as_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iter_jsonl_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    return sorted(input_path.rglob("*.jsonl"))


def _init_agg() -> dict:
    return {
        "num_tokens": 0,
        "sum_neg_logprob": 0.0,
        "sum_entropy": 0.0,
        "min_chosen_prob": None,
    }


def _update_agg(agg: dict, row: dict) -> None:
    agg["num_tokens"] += 1
    chosen_logprob = _as_float(row.get("chosen_logprob"))
    if chosen_logprob is not None:
        agg["sum_neg_logprob"] += -chosen_logprob

    entropy = _as_float(row.get("topk_entropy"))
    if entropy is not None:
        agg["sum_entropy"] += entropy

    chosen_prob = _as_float(row.get("chosen_prob"))
    if chosen_prob is not None:
        current_min = agg["min_chosen_prob"]
        if current_min is None or chosen_prob < current_min:
            agg["min_chosen_prob"] = chosen_prob


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_uq_logprobs(input_path: Path, output_dir: Path) -> None:
    files = _iter_jsonl_files(input_path)
    if len(files) == 0:
        raise ValueError(f"No JSONL files found under {input_path}")

    # Keys now include scorer as a grouping dimension.
    # turn_key = (domain, task_id, trial, seed, scorer, role, turn_discriminator)
    # traj_key = (domain, task_id, trial, seed, scorer, role)
    turn_aggs: dict[tuple, dict] = {}
    traj_aggs: dict[tuple, dict] = {}
    file_stats: list[dict] = []

    for file_path in files:
        num_rows = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                row = json.loads(line)
                num_rows += 1

                domain = row.get("domain")
                task_id = row.get("task_id")
                trial = row.get("trial")
                seed = row.get("seed")
                role = row.get("role")
                scorer = row.get("scorer", "self")
                turn_idx = row.get("turn_idx")
                message_timestamp = row.get("message_timestamp")

                # When turn_idx is null (logprobs are saved before the
                # orchestrator assigns indices), fall back to
                # message_timestamp which is unique per generate() call.
                turn_discriminator = (
                    turn_idx if turn_idx is not None else message_timestamp
                )
                turn_key = (
                    domain,
                    task_id,
                    trial,
                    seed,
                    scorer,
                    role,
                    turn_discriminator,
                )
                if turn_key not in turn_aggs:
                    turn_aggs[turn_key] = _init_agg()
                _update_agg(turn_aggs[turn_key], row)

                traj_key = (domain, task_id, trial, seed, scorer, role)
                if traj_key not in traj_aggs:
                    traj_aggs[traj_key] = _init_agg()
                _update_agg(traj_aggs[traj_key], row)

        file_stats.append({"file_path": str(file_path), "num_rows": num_rows})

    # Derive sequential turn indices per trajectory.
    # traj_key for turns grouping = turn_key[:6] = (domain, task_id, trial, seed, scorer, role)
    traj_turns: dict[tuple, list] = {}
    for turn_key in turn_aggs:
        traj_key = turn_key[:6]
        traj_turns.setdefault(traj_key, []).append(turn_key)

    turn_idx_map: dict[tuple, int] = {}
    for turns in traj_turns.values():
        for idx, turn_key in enumerate(
            sorted(turns, key=lambda k: (k[6] is None, str(k[6])))
        ):
            turn_idx_map[turn_key] = idx

    turn_rows = []
    for turn_key, agg in sorted(
        turn_aggs.items(), key=lambda kv: turn_idx_map.get(kv[0], 0)
    ):
        domain, task_id, trial, seed, scorer, role, _disc = turn_key
        resolved_turn_idx = turn_idx_map[turn_key]
        num_tokens = agg["num_tokens"]
        mean_entropy = (
            agg["sum_entropy"] / num_tokens if num_tokens > 0 else None
        )
        avg_token_nll = agg["sum_neg_logprob"] / num_tokens if num_tokens > 0 else None
        turn_rows.append(
            {
                "domain": domain,
                "task_id": task_id,
                "trial": trial,
                "seed": seed,
                "scorer": scorer,
                "role": role,
                "turn_idx": resolved_turn_idx,
                "num_tokens": num_tokens,
                "turn_nll": agg["sum_neg_logprob"],
                "avg_token_nll": avg_token_nll,
                "mean_topk_entropy": mean_entropy,
                "min_chosen_prob": agg["min_chosen_prob"],
            }
        )

    trajectory_rows = []
    for (domain, task_id, trial, seed, scorer, role), agg in sorted(traj_aggs.items()):
        num_tokens = agg["num_tokens"]
        mean_entropy = (
            agg["sum_entropy"] / num_tokens if num_tokens > 0 else None
        )
        avg_token_nll = agg["sum_neg_logprob"] / num_tokens if num_tokens > 0 else None
        trajectory_rows.append(
            {
                "domain": domain,
                "task_id": task_id,
                "trial": trial,
                "seed": seed,
                "scorer": scorer,
                "role": role,
                "total_tokens": num_tokens,
                "trajectory_nll": agg["sum_neg_logprob"],
                "avg_token_nll": avg_token_nll,
                "mean_topk_entropy": mean_entropy,
                "min_chosen_prob": agg["min_chosen_prob"],
            }
        )

    # Build combined trajectory rows.  Group per-role aggs by
    # (domain, task_id, trial, seed, scorer) and merge when multiple
    # roles are present within the same scorer.
    task_roles: dict[tuple, list[tuple]] = {}
    for traj_key, agg in traj_aggs.items():
        base_key = traj_key[:5]  # (domain, task_id, trial, seed, scorer)
        task_roles.setdefault(base_key, []).append((traj_key, agg))

    for base_key, role_entries in sorted(task_roles.items()):
        if len(role_entries) < 2:
            continue
        domain, task_id, trial, seed, scorer = base_key
        total_n = 0
        total_neg_logprob = 0.0
        total_entropy = 0.0
        combined_min_prob = None
        for _key, agg in role_entries:
            total_n += agg["num_tokens"]
            total_neg_logprob += agg["sum_neg_logprob"]
            total_entropy += agg["sum_entropy"]
            p = agg["min_chosen_prob"]
            if p is not None and (combined_min_prob is None or p < combined_min_prob):
                combined_min_prob = p
        trajectory_rows.append(
            {
                "domain": domain,
                "task_id": task_id,
                "trial": trial,
                "seed": seed,
                "scorer": scorer,
                "role": "combined",
                "total_tokens": total_n,
                "trajectory_nll": total_neg_logprob,
                "avg_token_nll": (total_neg_logprob / total_n if total_n > 0 else None),
                "mean_topk_entropy": (
                    total_entropy / total_n if total_n > 0 else None
                ),
                "min_chosen_prob": combined_min_prob,
            }
        )

    _write_csv(
        output_dir / "uq_logprobs_file_stats.csv",
        ["file_path", "num_rows"],
        file_stats,
    )
    _write_csv(
        output_dir / "uq_turn_summary.csv",
        [
            "domain",
            "task_id",
            "trial",
            "seed",
            "scorer",
            "role",
            "turn_idx",
            "num_tokens",
            "turn_nll",
            "avg_token_nll",
            "mean_topk_entropy",
            "min_chosen_prob",
        ],
        turn_rows,
    )
    _write_csv(
        output_dir / "uq_trajectory_summary.csv",
        [
            "domain",
            "task_id",
            "trial",
            "seed",
            "scorer",
            "role",
            "total_tokens",
            "trajectory_nll",
            "avg_token_nll",
            "mean_topk_entropy",
            "min_chosen_prob",
        ],
        trajectory_rows,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze token-level top-k logprob sidecar files for uncertainty metrics."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to one JSONL file or a directory containing JSONL logprob files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save summary CSV files.",
    )
    args = parser.parse_args()
    analyze_uq_logprobs(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
