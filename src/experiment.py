"""
Experiment runner: compares model performance on ETHICS benchmark
with and without psalm injection in the system prompt.

Usage:
    python -m src.experiment                    # run full experiment
    python -m src.experiment --quick            # quick smoke test (50 samples)
    python -m src.experiment --subset justice    # single subset
    python -m src.experiment --psalm 23          # test with specific psalm
    python -m src.experiment --psalm-mode random_n --psalm-count 5  # 5 random psalms
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog

from .psalms import PsalmLoader, PsalmMode, PsalmInjection
from .ethics_tasks import make_ethics_task, SUBSETS


MODELS = [
    "anthropic/claude-sonnet-4-20250514",
    "openai/gpt-4o",
]

RESULTS_DIR = Path(__file__).parent.parent / "results"


def extract_score(log: EvalLog) -> dict:
    """Pull accuracy and metadata from an EvalLog."""
    results = log.results
    metrics = {}
    if results and results.scores:
        for score in results.scores:
            metrics.update({k: v.value for k, v in score.metrics.items()})
    return {
        "model": str(log.eval.model),
        "accuracy": metrics.get("accuracy", None),
        "stderr": metrics.get("stderr", None),
        "samples": log.eval.dataset.samples if log.eval.dataset else None,
        "status": str(log.status),
    }


def run_condition(
    subset: str,
    model: str,
    injection: PsalmInjection,
    limit: int | None,
    log_dir: str,
) -> dict:
    """Run a single experimental condition and return the result dict."""
    task = make_ethics_task(subset, injection=injection, limit=limit)

    logs = inspect_eval(
        task,
        model=model,
        log_dir=log_dir,
        cache_prompt=True,
    )

    log = logs[0]
    result = extract_score(log)
    result["subset"] = subset
    result["condition"] = injection.description
    result["psalm_numbers"] = injection.psalm_numbers
    return result


def run_experiment(
    subsets: list[str],
    models: list[str],
    injection: PsalmInjection,
    limit: int | None = None,
) -> list[dict]:
    """Run full A/B experiment: vanilla vs psalm-injected, across subsets and models."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_dir = str(RESULTS_DIR / "logs" / timestamp)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "logs").mkdir(exist_ok=True)

    vanilla = PsalmInjection(mode=PsalmMode.NONE, psalm_numbers=[], text="")
    all_results = []

    for model in models:
        for subset in subsets:
            print(f"\n{'='*60}")
            print(f"Model: {model} | Subset: {subset}")
            print(f"{'='*60}")

            # Condition A: Vanilla
            print(f"\n--- Condition A: Vanilla ---")
            result_a = run_condition(subset, model, vanilla, limit, log_dir)
            all_results.append(result_a)
            print(f"  Accuracy: {result_a['accuracy']}")

            # Condition B: Psalm-injected
            print(f"\n--- Condition B: {injection.description} ---")
            result_b = run_condition(subset, model, injection, limit, log_dir)
            all_results.append(result_b)
            print(f"  Accuracy: {result_b['accuracy']}")

            # Delta
            if result_a["accuracy"] is not None and result_b["accuracy"] is not None:
                delta = result_b["accuracy"] - result_a["accuracy"]
                direction = "+" if delta >= 0 else ""
                print(f"\n  Delta: {direction}{delta:.4f}")

    # Save raw results
    results_file = RESULTS_DIR / f"results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")

    return all_results


def build_injection(args) -> PsalmInjection:
    """Build a PsalmInjection from CLI args."""
    loader = PsalmLoader()
    mode = PsalmMode(args.psalm_mode)
    psalm_list = None
    if args.psalms:
        psalm_list = [int(x.strip()) for x in args.psalms.split(",")]
    return loader.inject(
        mode=mode,
        psalm_number=args.psalm,
        psalm_list=psalm_list,
        n=args.psalm_count,
        seed=args.seed,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run ETHICS benchmark with psalm injection experiment"
    )
    parser.add_argument(
        "--subset",
        choices=list(SUBSETS.keys()) + ["all"],
        default="all",
        help="ETHICS subset to evaluate (default: all)",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=MODELS,
        help="Model(s) to evaluate",
    )
    parser.add_argument(
        "--psalm-mode",
        choices=[m.value for m in PsalmMode if m != PsalmMode.NONE],
        default="specific",
        help="How to select psalms for injection",
    )
    parser.add_argument(
        "--psalm",
        type=int,
        default=23,
        help="Specific psalm number (for --psalm-mode specific)",
    )
    parser.add_argument(
        "--psalms",
        type=str,
        default=None,
        help="Comma-separated list of psalm numbers (for --psalm-mode specific_list), e.g. '1,23,42,51'",
    )
    parser.add_argument(
        "--psalm-count",
        type=int,
        default=3,
        help="Number of psalms (for --psalm-mode random_n)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for psalm selection reproducibility",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of eval samples per subset",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke test: 50 samples, commonsense only",
    )

    args = parser.parse_args()

    if args.quick:
        args.subset = "commonsense"
        args.limit = 50

    subsets = list(SUBSETS.keys()) if args.subset == "all" else [args.subset]
    injection = build_injection(args)

    print(f"Psalm injection: {injection.description}")
    print(f"Models: {args.model}")
    print(f"Subsets: {subsets}")
    print(f"Limit: {args.limit or 'all'}")

    results = run_experiment(
        subsets=subsets,
        models=args.model,
        injection=injection,
        limit=args.limit,
    )

    # Print summary table
    from .analysis import print_comparison_table
    print_comparison_table(results)


if __name__ == "__main__":
    main()
