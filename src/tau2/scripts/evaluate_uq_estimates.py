import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Optional


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iter_result_files(result_path: Path) -> list[Path]:
    if result_path.is_file():
        return [result_path]
    if not result_path.exists():
        raise FileNotFoundError(f"Result path does not exist: {result_path}")
    files = sorted(result_path.rglob("*.json")) + sorted(result_path.rglob("*.jsonl"))
    return files


def _extract_reward_rows_from_simulation(
    sim: dict[str, Any],
    *,
    default_domain: Optional[str],
) -> list[dict[str, Any]]:
    reward_info = sim.get("reward_info", {})
    reward = None
    if isinstance(reward_info, dict):
        reward = _as_float(reward_info.get("reward"))
    return [
        {
            "domain": default_domain,
            "task_id": sim.get("task_id"),
            "trial": sim.get("trial"),
            "seed": sim.get("seed"),
            "reward": reward,
        }
    ]


def _extract_uq_reward_rows_from_simulation(
    sim: dict[str, Any],
    *,
    default_domain: Optional[str],
    role: str,
    uncertainty_column: str,
) -> list[dict[str, Any]]:
    """Extract joined (uncertainty, reward) rows from a simulation that has uq_summary."""
    uq_summary = sim.get("uq_summary")
    if not isinstance(uq_summary, dict):
        return []
    role_summary = uq_summary.get(role)
    if not isinstance(role_summary, dict):
        return []
    uncertainty = _as_float(role_summary.get(uncertainty_column))
    if uncertainty is None:
        return []
    reward_info = sim.get("reward_info", {})
    reward = None
    if isinstance(reward_info, dict):
        reward = _as_float(reward_info.get("reward"))
    return [
        {
            "domain": default_domain,
            "task_id": sim.get("task_id"),
            "trial": sim.get("trial"),
            "seed": sim.get("seed"),
            "role": role,
            "uncertainty": uncertainty,
            "reward": reward,
            # Also carry the full UQ stats for the output CSV
            "total_tokens": role_summary.get("total_tokens"),
            "trajectory_nll": _as_float(role_summary.get("trajectory_nll")),
            "avg_token_nll": _as_float(role_summary.get("avg_token_nll")),
            "mean_topk_entropy": _as_float(role_summary.get("mean_topk_entropy")),
            "min_chosen_prob": _as_float(role_summary.get("min_chosen_prob")),
        }
    ]


def load_reward_table(result_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file_path in _iter_result_files(result_path):
        if file_path.suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                continue
            simulations = payload.get("simulations", [])
            if not isinstance(simulations, list):
                continue
            domain = (
                payload.get("info", {})
                .get("environment_info", {})
                .get("domain_name", None)
            )
            for sim in simulations:
                if not isinstance(sim, dict):
                    continue
                rows.extend(
                    _extract_reward_rows_from_simulation(sim=sim, default_domain=domain)
                )
        elif file_path.suffix == ".jsonl":
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line == "":
                        continue
                    sim = json.loads(line)
                    if not isinstance(sim, dict):
                        continue
                    rows.extend(
                        _extract_reward_rows_from_simulation(
                            sim=sim,
                            default_domain=None,
                        )
                    )
    return rows


def load_uq_reward_table_from_results(
    result_path: Path,
    role: str,
    uncertainty_column: str,
) -> list[dict[str, Any]]:
    """
    Load joined (uncertainty, reward) rows directly from Tau2 results files
    that contain embedded ``uq_summary`` per simulation.
    """
    rows: list[dict[str, Any]] = []
    for file_path in _iter_result_files(result_path):
        if file_path.suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                continue
            simulations = payload.get("simulations", [])
            if not isinstance(simulations, list):
                continue
            domain = (
                payload.get("info", {})
                .get("environment_info", {})
                .get("domain_name", None)
            )
            for sim in simulations:
                if not isinstance(sim, dict):
                    continue
                rows.extend(
                    _extract_uq_reward_rows_from_simulation(
                        sim=sim,
                        default_domain=domain,
                        role=role,
                        uncertainty_column=uncertainty_column,
                    )
                )
        elif file_path.suffix == ".jsonl":
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line == "":
                        continue
                    sim = json.loads(line)
                    if not isinstance(sim, dict):
                        continue
                    rows.extend(
                        _extract_uq_reward_rows_from_simulation(
                            sim=sim,
                            default_domain=None,
                            role=role,
                            uncertainty_column=uncertainty_column,
                        )
                    )
    return rows


def load_uq_trajectory_table(
    path: Path,
    role: str,
    scorer: Optional[str] = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if role and row.get("role") != role:
                continue
            # Filter by scorer when provided.  Rows without a scorer
            # column are treated as "self" for backward compatibility.
            if scorer is not None:
                row_scorer = row.get("scorer", "self")
                if row_scorer != scorer:
                    continue
            rows.append(row)
    return rows


def _key4(row: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    return (
        row.get("domain"),
        row.get("task_id"),
        row.get("trial"),
        row.get("seed"),
    )


def _key3(row: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (
        row.get("task_id"),
        row.get("trial"),
        row.get("seed"),
    )


def join_uq_with_rewards(
    uq_rows: list[dict[str, Any]],
    reward_rows: list[dict[str, Any]],
    uncertainty_column: str,
    invert_uncertainty: bool,
) -> list[dict[str, Any]]:
    reward_by_key4 = {_key4(r): r for r in reward_rows}
    reward_by_key3 = {_key3(r): r for r in reward_rows}

    joined: list[dict[str, Any]] = []
    for row in uq_rows:
        reward_row = reward_by_key4.get(_key4(row))
        if reward_row is None:
            reward_row = reward_by_key3.get(_key3(row))
        if reward_row is None:
            continue
        uncertainty = _as_float(row.get(uncertainty_column))
        if uncertainty is None:
            continue
        if invert_uncertainty:
            uncertainty = -uncertainty
        reward = _as_float(reward_row.get("reward"))
        joined.append(
            {
                "domain": row.get("domain"),
                "task_id": row.get("task_id"),
                "trial": row.get("trial"),
                "seed": row.get("seed"),
                "role": row.get("role"),
                "uncertainty": uncertainty,
                "reward": reward,
            }
        )
    return joined


def _rankdata(values: list[float]) -> list[float]:
    sorted_pairs = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(sorted_pairs):
        j = i
        while (
            j + 1 < len(sorted_pairs) and sorted_pairs[j + 1][1] == sorted_pairs[i][1]
        ):
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[sorted_pairs[k][0]] = avg_rank
        i = j + 1
    return ranks


def compute_auroc(scores: list[float], labels: list[int]) -> Optional[float]:
    n = len(scores)
    if n == 0:
        return None
    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _rankdata(scores)
    sum_ranks_pos = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    auroc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auroc


def compute_pearson(x: list[float], y: list[float]) -> Optional[float]:
    n = len(x)
    if n == 0 or len(y) != n:
        return None
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)
    if var_x <= 0 or var_y <= 0:
        return None
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    return cov / math.sqrt(var_x * var_y)


def compute_spearman(x: list[float], y: list[float]) -> Optional[float]:
    """Spearman rank correlation coefficient.

    Converts *x* and *y* to ranks (with average-rank tie handling via
    ``_rankdata``) and then returns the Pearson correlation of the ranks.
    """
    if len(x) == 0 or len(x) != len(y):
        return None
    return compute_pearson(_rankdata(x), _rankdata(y))


def compute_kendall_tau(x: list[float], y: list[float]) -> Optional[float]:
    """Kendall's tau-b rank correlation coefficient.

    Counts concordant vs. discordant pairs over all (i, j) with i < j,
    adjusting for ties in *x* and *y* (tau-b formulation).
    """
    n = len(x)
    if n < 2 or len(y) != n:
        return None
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    ties_xy = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                ties_xy += 1
                ties_x += 1
                ties_y += 1
            elif dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif (dx > 0 and dy > 0) or (dx < 0 and dy < 0):
                concordant += 1
            else:
                discordant += 1
    n_pairs = n * (n - 1) // 2
    denom = math.sqrt((n_pairs - ties_x) * (n_pairs - ties_y))
    if denom <= 0:
        return None
    return (concordant - discordant) / denom


def compute_auarc(
    uncertainties: list[float],
    rewards: list[float],
    failure_threshold: float,
) -> tuple[Optional[float], list[dict[str, Any]]]:
    """Compute Area Under the Accuracy-Rejection Curve.

    Sort by uncertainty descending (most uncertain first), then progressively
    reject the most uncertain examples.  At each rejection rate the accuracy
    is computed over the *remaining* (non-rejected) examples.
    """
    if len(uncertainties) == 0:
        return None, []
    # Sort descending by uncertainty — reject the most uncertain first.
    data = sorted(zip(uncertainties, rewards), key=lambda x: x[0], reverse=True)
    n = len(data)
    successes = [1 if reward >= failure_threshold else 0 for _, reward in data]
    total_successes = sum(successes)

    curve: list[dict[str, Any]] = []
    # rejection_rate=0  →  all examples kept  →  overall accuracy
    prev_accuracy = total_successes / n if n > 0 else 0.0
    curve.append(
        {
            "coverage": 1.0,
            "rejection_rate": 0.0,
            "accuracy": prev_accuracy,
        }
    )

    # Progressively reject one example at a time (most uncertain first).
    remaining_successes = total_successes
    for k in range(n):
        remaining_successes -= successes[k]
        remaining = n - (k + 1)
        rejection = (k + 1) / n
        accuracy = remaining_successes / remaining if remaining > 0 else 1.0
        curve.append(
            {
                "coverage": remaining / n,
                "rejection_rate": rejection,
                "accuracy": accuracy,
            }
        )

    # Trapezoidal integration (rejection_rate on x-axis, accuracy on y-axis)
    area = 0.0
    for i in range(1, len(curve)):
        dx = curve[i]["rejection_rate"] - curve[i - 1]["rejection_rate"]
        area += dx * (curve[i]["accuracy"] + curve[i - 1]["accuracy"]) / 2.0

    return area, curve


def _write_evaluation_outputs(
    valid_rows: list[dict[str, Any]],
    uncertainties: list[float],
    rewards: list[float],
    output_dir: Path,
    uncertainty_column: str,
    invert_uncertainty: bool,
    role: str,
    failure_threshold: float,
    *,
    extra_joined_fields: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Shared helper that computes UQ metrics and writes output CSVs + JSON.
    """
    failures = [1 if reward < failure_threshold else 0 for reward in rewards]
    negative_rewards = [-reward for reward in rewards]

    auroc = compute_auroc(scores=uncertainties, labels=failures)
    auarc, auarc_curve = compute_auarc(
        uncertainties=uncertainties,
        rewards=rewards,
        failure_threshold=failure_threshold,
    )
    pearson_uncertainty_negative_reward = compute_pearson(
        x=uncertainties, y=negative_rewards
    )
    spearman_uncertainty_negative_reward = compute_spearman(
        x=uncertainties, y=negative_rewards
    )
    kendall_uncertainty_negative_reward = compute_kendall_tau(
        x=uncertainties, y=negative_rewards
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    joined_fieldnames = [
        "domain",
        "task_id",
        "trial",
        "seed",
        "role",
        "uncertainty",
        "reward",
    ]
    if extra_joined_fields:
        joined_fieldnames.extend(extra_joined_fields)

    joined_csv = output_dir / "uq_eval_joined.csv"
    with open(joined_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=joined_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(valid_rows)

    curve_csv = output_dir / "uq_eval_auarc_curve.csv"
    with open(curve_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["coverage", "rejection_rate", "accuracy"]
        )
        writer.writeheader()
        writer.writerows(auarc_curve)

    metrics = {
        "num_examples_total": len(valid_rows),
        "num_examples_with_reward": len(
            [r for r in valid_rows if r.get("reward") is not None]
        ),
        "uncertainty_column": uncertainty_column,
        "invert_uncertainty": invert_uncertainty,
        "role": role,
        "failure_threshold": failure_threshold,
        "metrics": {
            "auroc_failure_vs_uncertainty": auroc,
            "auarc": auarc,
            "pearson_uncertainty_negative_reward": pearson_uncertainty_negative_reward,
            "spearman_uncertainty_negative_reward": spearman_uncertainty_negative_reward,
            "kendall_uncertainty_negative_reward": kendall_uncertainty_negative_reward,
        },
        "outputs": {
            "joined_csv": str(joined_csv),
            "auarc_curve_csv": str(curve_csv),
        },
    }

    metrics_path = output_dir / "uq_eval_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def evaluate_uq_estimates(
    uq_trajectory_csv: Path,
    result_path: Path,
    output_dir: Path,
    uncertainty_column: str = "avg_token_nll",
    role: str = "assistant",
    failure_threshold: float = 0.0,
    invert_uncertainty: bool = False,
    scorer: Optional[str] = None,
) -> dict[str, Any]:
    """
    Original CSV-based evaluation: joins a separate UQ trajectory CSV with
    reward data from the results JSON.
    """
    uq_rows = load_uq_trajectory_table(path=uq_trajectory_csv, role=role, scorer=scorer)
    reward_rows = load_reward_table(result_path=result_path)
    joined_rows = join_uq_with_rewards(
        uq_rows=uq_rows,
        reward_rows=reward_rows,
        uncertainty_column=uncertainty_column,
        invert_uncertainty=invert_uncertainty,
    )

    valid_rows = [r for r in joined_rows if r["reward"] is not None]
    uncertainties = [r["uncertainty"] for r in valid_rows]
    rewards = [r["reward"] for r in valid_rows]

    return _write_evaluation_outputs(
        valid_rows=valid_rows,
        uncertainties=uncertainties,
        rewards=rewards,
        output_dir=output_dir,
        uncertainty_column=uncertainty_column,
        invert_uncertainty=invert_uncertainty,
        role=role,
        failure_threshold=failure_threshold,
    )


def evaluate_uq_from_results(
    result_path: Path,
    output_dir: Path,
    uncertainty_column: str = "avg_token_nll",
    role: str = "assistant",
    failure_threshold: float = 0.0,
    invert_uncertainty: bool = False,
) -> dict[str, Any]:
    """
    Evaluate UQ using embedded ``uq_summary`` from the results JSON directly.

    This mode does not require a separate UQ trajectory CSV — it reads both
    uncertainty estimates and rewards from the main Tau2 results file(s),
    enabling domain-wide evaluation in a single command.
    """
    joined_rows = load_uq_reward_table_from_results(
        result_path=result_path,
        role=role,
        uncertainty_column=uncertainty_column,
    )
    if invert_uncertainty:
        for row in joined_rows:
            row["uncertainty"] = -row["uncertainty"]

    valid_rows = [r for r in joined_rows if r.get("reward") is not None]
    uncertainties = [r["uncertainty"] for r in valid_rows]
    rewards = [r["reward"] for r in valid_rows]

    # Also write a standalone UQ trajectory CSV for easy reuse
    output_dir.mkdir(parents=True, exist_ok=True)
    uq_traj_csv = output_dir / "uq_trajectory_summary.csv"
    uq_traj_fields = [
        "domain",
        "task_id",
        "trial",
        "seed",
        "role",
        "total_tokens",
        "trajectory_nll",
        "avg_token_nll",
        "mean_topk_entropy",
        "min_chosen_prob",
        "reward",
    ]
    with open(uq_traj_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=uq_traj_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(valid_rows)

    metrics = _write_evaluation_outputs(
        valid_rows=valid_rows,
        uncertainties=uncertainties,
        rewards=rewards,
        output_dir=output_dir,
        uncertainty_column=uncertainty_column,
        invert_uncertainty=invert_uncertainty,
        role=role,
        failure_threshold=failure_threshold,
        extra_joined_fields=[
            "total_tokens",
            "trajectory_nll",
            "avg_token_nll",
            "mean_topk_entropy",
            "min_chosen_prob",
        ],
    )
    metrics["outputs"]["uq_trajectory_csv"] = str(uq_traj_csv)
    # Re-write metrics with updated outputs
    metrics_path = output_dir / "uq_eval_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate uncertainty estimates against rewards using AUROC, AUARC, Pearson, Spearman, and Kendall's tau."
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["csv", "embedded", "auto"],
        help=(
            "Evaluation mode. 'csv' requires --uq-trajectory-csv and --results. "
            "'embedded' reads uq_summary directly from --results (no separate CSV needed). "
            "'auto' uses 'embedded' if --uq-trajectory-csv is not provided, else 'csv'. "
            "Default: auto."
        ),
    )
    parser.add_argument(
        "--uq-trajectory-csv",
        default=None,
        help="Path to uq_trajectory_summary.csv generated by analyze_uq_logprobs (csv mode).",
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to Tau2 results JSON file or directory containing result files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save evaluation outputs.",
    )
    parser.add_argument(
        "--uncertainty-column",
        default="avg_token_nll",
        help="Uncertainty metric to evaluate. Default: avg_token_nll.",
    )
    parser.add_argument(
        "--role",
        default="assistant",
        help="Role to evaluate: 'assistant', 'user', or 'combined' (agent + user). Default: assistant.",
    )
    parser.add_argument(
        "--failure-threshold",
        type=float,
        default=0.0,
        help="Failure label threshold: failure if reward < threshold. Default is 0.0.",
    )
    parser.add_argument(
        "--invert-uncertainty",
        action="store_true",
        help="Invert uncertainty score sign before evaluation.",
    )
    parser.add_argument(
        "--scorer",
        default=None,
        help=(
            "Filter UQ rows by scorer value (e.g. 'self', 'agent_llm', "
            "'auxiliary_llm'). Only used in csv mode. Default: no filter."
        ),
    )
    args = parser.parse_args()

    mode = args.mode
    if mode == "auto":
        mode = "csv" if args.uq_trajectory_csv else "embedded"

    if mode == "csv":
        if args.uq_trajectory_csv is None:
            parser.error("--uq-trajectory-csv is required in 'csv' mode.")
        metrics = evaluate_uq_estimates(
            uq_trajectory_csv=Path(args.uq_trajectory_csv),
            result_path=Path(args.results),
            output_dir=Path(args.output_dir),
            uncertainty_column=args.uncertainty_column,
            role=args.role,
            failure_threshold=args.failure_threshold,
            invert_uncertainty=args.invert_uncertainty,
            scorer=args.scorer,
        )
    else:
        metrics = evaluate_uq_from_results(
            result_path=Path(args.results),
            output_dir=Path(args.output_dir),
            uncertainty_column=args.uncertainty_column,
            role=args.role,
            failure_threshold=args.failure_threshold,
            invert_uncertainty=args.invert_uncertainty,
        )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
