import argparse
import json

from tau2.config import (
    DEFAULT_AGENT_IMPLEMENTATION,
    DEFAULT_LLM_AGENT,
    DEFAULT_LLM_TEMPERATURE_AGENT,
    DEFAULT_LLM_TEMPERATURE_USER,
    DEFAULT_LLM_USER,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_ERRORS,
    DEFAULT_MAX_STEPS,
    DEFAULT_NUM_TRIALS,
    DEFAULT_SEED,
    DEFAULT_USER_IMPLEMENTATION,
)
from tau2.data_model.simulation import RunConfig
from tau2.run import get_options, run_domain


def add_run_args(parser):
    """Add run arguments to a parser."""
    domains = get_options().domains
    parser.add_argument(
        "--domain",
        "-d",
        type=str,
        choices=domains,
        help="The domain to run the simulation on",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=DEFAULT_NUM_TRIALS,
        help="The number of times each task is run. Default is 1.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=DEFAULT_AGENT_IMPLEMENTATION,
        choices=get_options().agents,
        help=f"The agent implementation to use. Default is {DEFAULT_AGENT_IMPLEMENTATION}.",
    )
    parser.add_argument(
        "--agent-llm",
        type=str,
        default=DEFAULT_LLM_AGENT,
        help=f"The LLM to use for the agent. Default is {DEFAULT_LLM_AGENT}.",
    )
    parser.add_argument(
        "--agent-llm-args",
        type=json.loads,
        default={"temperature": DEFAULT_LLM_TEMPERATURE_AGENT},
        help=f"The arguments to pass to the LLM for the agent. Default is '{{\"temperature\": {DEFAULT_LLM_TEMPERATURE_AGENT}}}'.",
    )
    parser.add_argument(
        "--user",
        type=str,
        choices=get_options().users,
        default=DEFAULT_USER_IMPLEMENTATION,
        help=f"The user implementation to use. Default is {DEFAULT_USER_IMPLEMENTATION}.",
    )
    parser.add_argument(
        "--user-llm",
        type=str,
        default=DEFAULT_LLM_USER,
        help=f"The LLM to use for the user. Default is {DEFAULT_LLM_USER}.",
    )
    parser.add_argument(
        "--user-llm-args",
        type=json.loads,
        default={"temperature": DEFAULT_LLM_TEMPERATURE_USER},
        help=f"The arguments to pass to the LLM for the user. Default is '{{\"temperature\": {DEFAULT_LLM_TEMPERATURE_USER}}}'.",
    )
    parser.add_argument(
        "--task-set-name",
        type=str,
        default=None,
        choices=get_options().task_sets,
        help="The task set to run the simulation on. If not provided, will load default task set for the domain.",
    )
    parser.add_argument(
        "--task-split-name",
        type=str,
        default="base",
        help="The task split to run the simulation on. If not provided, will load 'base' split.",
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        nargs="+",
        help="(Optional) run only the tasks with the given IDs. If not provided, will run all tasks.",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="The number of tasks to run.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"The maximum number of steps to run the simulation. Default is {DEFAULT_MAX_STEPS}.",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=DEFAULT_MAX_ERRORS,
        help=f"The maximum number of tool errors allowed in a row in the simulation. Default is {DEFAULT_MAX_ERRORS}.",
    )
    parser.add_argument(
        "--save-to",
        type=str,
        required=False,
        help="The path to save the simulation results. Will be saved to data/simulations/<save_to>.json. If not provided, will save to <domain>_<agent>_<user>_<llm_agent>_<llm_user>_<timestamp>.json. If the file already exists, it will try to resume the run.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help=f"The maximum number of concurrent simulations to run. Default is {DEFAULT_MAX_CONCURRENCY}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"The seed to use for the simulation. Default is {DEFAULT_SEED}.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=DEFAULT_LOG_LEVEL,
        help=f"The log level to use for the simulation. Default is {DEFAULT_LOG_LEVEL}.",
    )
    parser.add_argument(
        "--enforce-communication-protocol",
        action="store_true",
        default=False,
        help="Enforce communication protocol rules (e.g., no mixed messages with text and tool calls). Default is False.",
    )


def main():
    parser = argparse.ArgumentParser(description="Tau2 command line interface")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a benchmark")
    add_run_args(run_parser)
    run_parser.set_defaults(
        func=lambda args: run_domain(
            RunConfig(
                domain=args.domain,
                task_set_name=args.task_set_name,
                task_split_name=args.task_split_name,
                task_ids=args.task_ids,
                num_tasks=args.num_tasks,
                agent=args.agent,
                llm_agent=args.agent_llm,
                llm_args_agent=args.agent_llm_args,
                user=args.user,
                llm_user=args.user_llm,
                llm_args_user=args.user_llm_args,
                num_trials=args.num_trials,
                max_steps=args.max_steps,
                max_errors=args.max_errors,
                save_to=args.save_to,
                max_concurrency=args.max_concurrency,
                seed=args.seed,
                log_level=args.log_level,
                enforce_communication_protocol=args.enforce_communication_protocol,
            )
        )
    )

    # Play command
    play_parser = subparsers.add_parser(
        "play", help="Play manual mode - interact with a domain as the agent"
    )
    play_parser.set_defaults(func=lambda args: run_manual_mode())

    # View command
    view_parser = subparsers.add_parser("view", help="View simulation results")
    view_parser.add_argument(
        "--dir",
        type=str,
        help="Directory containing simulation files. Defaults to data/simulations if not specified.",
    )
    view_parser.add_argument(
        "--file",
        type=str,
        help="Path to the simulation results file to view",
    )
    view_parser.add_argument(
        "--only-show-failed",
        action="store_true",
        help="Only show failed tasks.",
    )
    view_parser.add_argument(
        "--only-show-all-failed",
        action="store_true",
        help="Only show tasks that failed in all trials.",
    )
    view_parser.set_defaults(func=lambda args: run_view_simulations(args))

    # Domain command
    domain_parser = subparsers.add_parser("domain", help="Show domain documentation")
    domain_parser.add_argument(
        "domain",
        type=str,
        help="Name of the domain to show documentation for (e.g., 'airline', 'mock')",
    )
    domain_parser.set_defaults(func=lambda args: run_show_domain(args))

    # Start command
    start_parser = subparsers.add_parser("start", help="Start all servers")
    start_parser.set_defaults(func=lambda args: run_start_servers())

    # Check data command
    check_data_parser = subparsers.add_parser(
        "check-data", help="Check if data directory is properly configured"
    )
    check_data_parser.set_defaults(func=lambda args: run_check_data())

    # Evaluate trajectories command
    evaluate_parser = subparsers.add_parser(
        "evaluate-trajs", help="Evaluate trajectories and update rewards"
    )
    evaluate_parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to trajectory files, directories, or glob patterns",
    )
    evaluate_parser.add_argument(
        "-o",
        "--output-dir",
        help="Directory to save updated trajectory files with recomputed rewards. If not provided, only displays metrics.",
    )
    evaluate_parser.set_defaults(func=lambda args: run_evaluate_trajectories(args))

    # Analyze UQ logprobs command
    analyze_uq_parser = subparsers.add_parser(
        "analyze-uq-logprobs",
        help="Analyze token-level UQ logprob sidecar files",
    )
    analyze_uq_parser.add_argument(
        "--input",
        required=True,
        help="Path to one JSONL file or a directory containing JSONL logprob files.",
    )
    analyze_uq_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save UQ summary CSV files.",
    )
    analyze_uq_parser.set_defaults(func=lambda args: run_analyze_uq_logprobs(args))

    # Evaluate UQ estimates command
    eval_uq_parser = subparsers.add_parser(
        "evaluate-uq",
        help="Evaluate UQ scores with AUROC, AUARC, and reward correlation",
    )
    eval_uq_parser.add_argument(
        "--mode",
        default="auto",
        choices=["csv", "embedded", "auto"],
        help=(
            "'csv': join separate UQ CSV with rewards from --results. "
            "'embedded': read uq_summary directly from --results (no CSV needed). "
            "'auto': use 'embedded' if --uq-trajectory-csv is omitted. Default: auto."
        ),
    )
    eval_uq_parser.add_argument(
        "--uq-trajectory-csv",
        default=None,
        help="Path to uq_trajectory_summary.csv from analyze-uq-logprobs (csv mode).",
    )
    eval_uq_parser.add_argument(
        "--results",
        required=True,
        help="Path to Tau2 result JSON or a directory with result files.",
    )
    eval_uq_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save UQ evaluation outputs.",
    )
    eval_uq_parser.add_argument(
        "--uncertainty-column",
        default="avg_token_nll",
        help="Uncertainty metric to evaluate. Default: avg_token_nll.",
    )
    eval_uq_parser.add_argument(
        "--role",
        default="assistant",
        help="Role to evaluate. Default: assistant.",
    )
    eval_uq_parser.add_argument(
        "--failure-threshold",
        type=float,
        default=0.0,
        help="Failure if reward < threshold. Default 0.0.",
    )
    eval_uq_parser.add_argument(
        "--invert-uncertainty",
        action="store_true",
        help="Invert uncertainty score sign before evaluation.",
    )
    eval_uq_parser.add_argument(
        "--scorer",
        default=None,
        help=(
            "Filter UQ rows by scorer value (e.g. 'self', 'agent_llm', "
            "'auxiliary_llm'). Only used in csv mode. Default: no filter."
        ),
    )
    eval_uq_parser.set_defaults(func=lambda args: run_evaluate_uq(args))

    # Extract UQ from existing trajectories command
    extract_uq_parser = subparsers.add_parser(
        "extract-uq-from-trajs",
        help="Extract/backfill UQ sidecar artifacts from existing trajectory files",
    )
    extract_uq_parser.add_argument(
        "--results",
        required=True,
        help="Path to one Tau2 result JSON file or a directory containing JSON files.",
    )
    extract_uq_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write extracted UQ JSONL files.",
    )
    extract_uq_parser.add_argument(
        "--partition-mode",
        default="task",
        choices=["task", "single"],
        help="Output partition mode. Default is task.",
    )
    extract_uq_parser.add_argument(
        "--rescore-missing",
        action="store_true",
        help="Rescore messages without saved logprobs via additional inference.",
    )
    extract_uq_parser.add_argument(
        "--rescore-api-base",
        default=None,
        help="OpenAI-compatible API base for rescoring (e.g. http://127.0.0.1:8000/v1).",
    )
    extract_uq_parser.add_argument(
        "--rescore-api-key",
        default=None,
        help="Optional API key for the rescoring endpoint.",
    )
    extract_uq_parser.add_argument(
        "--rescore-top-k",
        type=int,
        default=20,
        help="Top-k for rescoring logprobs. Default 20.",
    )
    extract_uq_parser.add_argument(
        "--rescore-timeout-sec",
        type=int,
        default=120,
        help="Timeout per rescoring request in seconds. Default 120.",
    )
    extract_uq_parser.add_argument(
        "--rescore-backend",
        default="chat_replay",
        choices=["chat_replay", "completion_echo", "auto"],
        help="Rescoring backend. Default chat_replay.",
    )
    extract_uq_parser.add_argument(
        "--chat-replay-require-exact-match",
        action="store_true",
        default=False,
        help="Only accept chat-replay rows when regenerated text exactly matches target.",
    )
    extract_uq_parser.set_defaults(func=lambda args: run_extract_uq_from_trajs(args))

    # Score observation UQ command
    obs_uq_parser = subparsers.add_parser(
        "score-observation-uq",
        help="Score observation uncertainty over user messages and tool results",
    )
    obs_uq_parser.add_argument(
        "--results",
        required=True,
        help="Path to one Tau2 result JSON file or a directory containing JSON files.",
    )
    obs_uq_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write observation UQ sidecar JSONL files.",
    )
    obs_uq_parser.add_argument(
        "--scorer-mode",
        default="agent_llm",
        choices=["agent_llm", "auxiliary_llm"],
        help=(
            "Scorer mode. 'agent_llm' uses the agent's own LLM and system prompt. "
            "'auxiliary_llm' uses a separate observer model. Default: agent_llm."
        ),
    )
    obs_uq_parser.add_argument(
        "--scorer-llm",
        default=None,
        help=(
            "Model name for the scorer LLM. Defaults to the agent's LLM in "
            "agent_llm mode. Required for auxiliary_llm mode."
        ),
    )
    obs_uq_parser.add_argument(
        "--scorer-api-base",
        default=None,
        help=(
            "API base URL for scoring. For Azure models (scorer-llm contains "
            "'azure'), defaults to AZURE_API_BASE env var."
        ),
    )
    obs_uq_parser.add_argument(
        "--scorer-api-key",
        default=None,
        help=(
            "API key for the scorer endpoint. For Azure models, defaults to "
            "AZURE_API_KEY env var."
        ),
    )
    obs_uq_parser.add_argument(
        "--scorer-api-version",
        default=None,
        help=(
            "API version for Azure OpenAI. Defaults to AZURE_API_VERSION env var "
            "when the scorer LLM contains 'azure'."
        ),
    )
    obs_uq_parser.add_argument(
        "--scorer-llm-args",
        type=json.loads,
        default=None,
        help="JSON dict of LLM arguments for the scorer (e.g. '{\"temperature\": 0.0}').",
    )
    obs_uq_parser.add_argument(
        "--rescore-backend",
        default="completion_echo",
        choices=["completion_echo", "chat_replay", "auto"],
        help="Rescoring backend. Default: completion_echo.",
    )
    obs_uq_parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k logprobs to request. Default: 20.",
    )
    obs_uq_parser.add_argument(
        "--timeout-sec",
        type=int,
        default=120,
        help="HTTP timeout per rescoring request in seconds. Default: 120.",
    )
    obs_uq_parser.add_argument(
        "--partition-mode",
        default="task",
        choices=["task", "single"],
        help="Output partition mode. Default: task.",
    )
    obs_uq_parser.add_argument(
        "--task-ids",
        nargs="+",
        default=None,
        help="Filter to specific task IDs.",
    )
    obs_uq_parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Score only the first N unique tasks. Can combine with --task-ids.",
    )
    obs_uq_parser.add_argument(
        "--no-user-messages",
        action="store_true",
        help="Skip scoring user messages (only score tool results).",
    )
    obs_uq_parser.add_argument(
        "--no-tool-messages",
        action="store_true",
        help="Skip scoring tool messages (only score user messages).",
    )
    obs_uq_parser.add_argument(
        "--auxiliary-system-prompt",
        default=None,
        help="Custom system prompt for auxiliary_llm mode.",
    )
    obs_uq_parser.set_defaults(func=lambda args: run_score_observation_uq(args))

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return

    args.func(args)


def run_view_simulations(args):
    from tau2.scripts.view_simulations import main as view_main

    view_main(
        sim_file=args.file,
        only_show_failed=args.only_show_failed,
        only_show_all_failed=args.only_show_all_failed,
        sim_dir=args.dir,
    )


def run_show_domain(args):
    from tau2.scripts.show_domain_doc import main as domain_main

    domain_main(args.domain)


def run_start_servers():
    from tau2.scripts.start_servers import main as start_main

    start_main()


def run_check_data():
    from tau2.scripts.check_data import main as check_data_main

    check_data_main()


def run_evaluate_trajectories(args):
    import sys

    from loguru import logger

    from tau2.scripts.evaluate_trajectories import evaluate_trajectories

    logger.configure(handlers=[{"sink": sys.stderr, "level": "ERROR"}])

    evaluate_trajectories(args.paths, args.output_dir)


def run_analyze_uq_logprobs(args):
    from pathlib import Path

    from tau2.scripts.analyze_uq_logprobs import analyze_uq_logprobs

    analyze_uq_logprobs(input_path=Path(args.input), output_dir=Path(args.output_dir))


def run_evaluate_uq(args):
    from pathlib import Path

    from tau2.scripts.evaluate_uq_estimates import (
        evaluate_uq_estimates,
        evaluate_uq_from_results,
    )

    mode = args.mode
    if mode == "auto":
        mode = "csv" if args.uq_trajectory_csv else "embedded"

    if mode == "csv":
        if args.uq_trajectory_csv is None:
            raise ValueError("--uq-trajectory-csv is required in 'csv' mode.")
        metrics = evaluate_uq_estimates(
            uq_trajectory_csv=Path(args.uq_trajectory_csv),
            result_path=Path(args.results),
            output_dir=Path(args.output_dir),
            uncertainty_column=args.uncertainty_column,
            role=args.role,
            failure_threshold=args.failure_threshold,
            invert_uncertainty=args.invert_uncertainty,
            scorer=getattr(args, "scorer", None),
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


def run_score_observation_uq(args):
    from pathlib import Path

    from tau2.scripts.score_observation_uq import score_observation_uq

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
    )
    print(json.dumps(stats, indent=2))


def run_extract_uq_from_trajs(args):
    from pathlib import Path

    from tau2.scripts.extract_uq_from_trajectories import extract_uq_from_trajectories

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


def run_manual_mode():
    from tau2.scripts.manual_mode import main as manual_main

    manual_main()


if __name__ == "__main__":
    main()
