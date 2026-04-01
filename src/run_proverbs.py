"""
Proverbs experiment runner.

Runs the ETHICS benchmark with Proverbs injected into the system prompt.
Uses the same A/B design as the psalm experiments.

Usage:
    python -m src.run_proverbs --mode random_n --count 10 --seed 42
    python -m src.run_proverbs --mode specific_list --chapters "1,2,8"
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from inspect_ai import eval as inspect_eval

from .scripture import ScriptureLoader, ScriptureInjection, PsalmMode
from .psalms import PsalmInjection
from .ethics_tasks import make_ethics_task, SUBSETS
from .experiment import extract_score


MODELS = [
    "anthropic/claude-sonnet-4-20250514",
    "openai/gpt-4o",
]

RESULTS_DIR = Path(__file__).parent.parent / "results"


def run_condition(subset, model, injection, limit, log_dir):
    task = make_ethics_task(subset, injection=injection, limit=limit)
    logs = inspect_eval(task, model=model, log_dir=log_dir, cache_prompt=True)
    log = logs[0]
    result = extract_score(log)
    result["subset"] = subset
    result["condition"] = injection.description
    result["chapter_numbers"] = injection.chapter_numbers if isinstance(injection, ScriptureInjection) else []
    return result


def run_proverbs_experiment(subsets, models, injection, limit=None):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_dir = str(RESULTS_DIR / "logs" / f"proverbs_{timestamp}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    vanilla_psalm = PsalmInjection(mode=PsalmMode.NONE, psalm_numbers=[], text="")
    all_results = []

    for model in models:
        for subset in subsets:
            print(f"\n{'='*60}")
            print(f"Model: {model} | Subset: {subset}")
            print(f"{'='*60}")

            print(f"\n--- Condition A: Vanilla ---")
            result_a = run_condition(subset, model, vanilla_psalm, limit, log_dir)
            all_results.append(result_a)
            print(f"  Accuracy: {result_a['accuracy']}")

            print(f"\n--- Condition B: {injection.description} ---")
            result_b = run_condition(subset, model, injection, limit, log_dir)
            all_results.append(result_b)
            print(f"  Accuracy: {result_b['accuracy']}")

            if result_a["accuracy"] is not None and result_b["accuracy"] is not None:
                delta = result_b["accuracy"] - result_a["accuracy"]
                sign = "+" if delta >= 0 else ""
                print(f"\n  Delta: {sign}{delta:.4f}")

    results_file = RESULTS_DIR / f"proverbs_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run ETHICS benchmark with Proverbs injection experiment"
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
        "--mode",
        choices=["specific", "specific_list", "random", "random_n", "all"],
        default="random_n",
        help="How to select Proverbs chapters",
    )
    parser.add_argument(
        "--chapter",
        type=int,
        default=1,
        help="Specific chapter number (for --mode specific)",
    )
    parser.add_argument(
        "--chapters",
        type=str,
        default=None,
        help="Comma-separated list of chapter numbers (for --mode specific_list)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of chapters (for --mode random_n)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of eval samples per subset",
    )

    args = parser.parse_args()
    subsets = list(SUBSETS.keys()) if args.subset == "all" else [args.subset]

    loader = ScriptureLoader("Proverbs")
    mode = PsalmMode(args.mode)
    chapter_list = None
    if args.chapters:
        chapter_list = [int(x.strip()) for x in args.chapters.split(",")]

    injection = loader.inject(
        mode=mode,
        chapter_number=args.chapter,
        chapter_list=chapter_list,
        n=args.count,
        seed=args.seed,
    )

    print(f"Proverbs injection: {injection.description}")
    print(f"Models: {args.model}")
    print(f"Subsets: {subsets}")
    print(f"Limit: {args.limit or 'all'}")

    results = run_proverbs_experiment(
        subsets=subsets,
        models=args.model,
        injection=injection,
        limit=args.limit,
    )

    from .analysis import print_comparison_table
    print_comparison_table(results)


if __name__ == "__main__":
    main()
