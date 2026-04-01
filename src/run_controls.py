"""
Control experiments for the utilitarianism anomaly.

Tests three hypotheses about why psalm injection dramatically improves
GPT-4o's utilitarianism performance:

1. Length-matched random prose (Wikipedia) — is it just more context?
2. High-familiarity secular text — is it training data familiarity?
3. Shuffled labels — is the model genuinely reasoning or biased toward "1"?

Usage:
    python -m src.run_controls --control wikipedia
    python -m src.run_controls --control secular
    python -m src.run_controls --control shuffled
    python -m src.run_controls --control all
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from inspect_ai import eval as inspect_eval

from .psalms import PsalmLoader, PsalmMode, PsalmInjection
from .ethics_tasks import (
    make_text_injection_task,
    make_utilitarianism_shuffled_task,
    make_ethics_task,
)
from .experiment import extract_score


DATA_DIR = Path(__file__).parent.parent / "data"
CONTROLS_DIR = DATA_DIR / "controls"
RESULTS_DIR = Path(__file__).parent.parent / "results"

MODELS = [
    "anthropic/claude-sonnet-4-20250514",
    "openai/gpt-4o",
]


def load_control_text(name: str) -> str:
    """Load a control text file."""
    path = CONTROLS_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8")


def run_wikipedia_control(models: list[str], limit: int | None = None) -> list[dict]:
    """Control 1: Length-matched Wikipedia prose."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_dir = str(RESULTS_DIR / "logs" / f"control_wikipedia_{timestamp}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    wikipedia_text = load_control_text("wikipedia_prose")
    results = []

    for model in models:
        print(f"\n{'='*60}")
        print(f"CONTROL: Wikipedia Prose | Model: {model}")
        print(f"{'='*60}")

        # Vanilla
        print("\n--- Vanilla ---")
        vanilla_inj = PsalmInjection(mode=PsalmMode.NONE, psalm_numbers=[], text="")
        task = make_ethics_task("utilitarianism", injection=vanilla_inj, limit=limit)
        logs = inspect_eval(task, model=model, log_dir=log_dir, cache_prompt=True)
        result = extract_score(logs[0])
        result["subset"] = "utilitarianism"
        result["condition"] = "vanilla (no text)"
        results.append(result)
        print(f"  Accuracy: {result['accuracy']}")

        # Wikipedia injected
        print("\n--- Wikipedia Prose ---")
        task = make_text_injection_task(
            "utilitarianism", wikipedia_text, framing="neutral", limit=limit
        )
        logs = inspect_eval(task, model=model, log_dir=log_dir, cache_prompt=True)
        result = extract_score(logs[0])
        result["subset"] = "utilitarianism"
        result["condition"] = "wikipedia prose (length-matched)"
        results.append(result)
        print(f"  Accuracy: {result['accuracy']}")

    results_file = RESULTS_DIR / f"control_wikipedia_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")
    return results


def run_secular_control(models: list[str], limit: int | None = None) -> list[dict]:
    """Control 2: High-familiarity secular text."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_dir = str(RESULTS_DIR / "logs" / f"control_secular_{timestamp}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    secular_text = load_control_text("famous_secular")
    results = []

    for model in models:
        print(f"\n{'='*60}")
        print(f"CONTROL: Famous Secular Text | Model: {model}")
        print(f"{'='*60}")

        # Vanilla
        print("\n--- Vanilla ---")
        vanilla_inj = PsalmInjection(mode=PsalmMode.NONE, psalm_numbers=[], text="")
        task = make_ethics_task("utilitarianism", injection=vanilla_inj, limit=limit)
        logs = inspect_eval(task, model=model, log_dir=log_dir, cache_prompt=True)
        result = extract_score(logs[0])
        result["subset"] = "utilitarianism"
        result["condition"] = "vanilla (no text)"
        results.append(result)
        print(f"  Accuracy: {result['accuracy']}")

        # Secular text injected
        print("\n--- Famous Secular Text ---")
        task = make_text_injection_task(
            "utilitarianism", secular_text, framing="neutral", limit=limit
        )
        logs = inspect_eval(task, model=model, log_dir=log_dir, cache_prompt=True)
        result = extract_score(logs[0])
        result["subset"] = "utilitarianism"
        result["condition"] = "famous secular text (length-matched)"
        results.append(result)
        print(f"  Accuracy: {result['accuracy']}")

    results_file = RESULTS_DIR / f"control_secular_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")
    return results


def run_shuffled_control(limit: int | None = None) -> list[dict]:
    """Control 3: Shuffled labels with psalm injection (GPT-4o only)."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_dir = str(RESULTS_DIR / "logs" / f"control_shuffled_{timestamp}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = "openai/gpt-4o"
    loader = PsalmLoader()
    popular_injection = loader.inject(
        mode=PsalmMode.SPECIFIC_LIST,
        psalm_list=[1, 23, 42, 51, 88, 100, 119],
    )
    results = []

    print(f"\n{'='*60}")
    print(f"CONTROL: Shuffled Labels | Model: {model}")
    print(f"{'='*60}")

    # Shuffled vanilla (no psalms)
    print("\n--- Shuffled Vanilla ---")
    task = make_utilitarianism_shuffled_task(
        injection=None, limit=limit, seed=42
    )
    logs = inspect_eval(task, model=model, log_dir=log_dir, cache_prompt=True)
    result = extract_score(logs[0])
    result["subset"] = "utilitarianism (shuffled)"
    result["condition"] = "vanilla (shuffled labels)"
    results.append(result)
    print(f"  Accuracy: {result['accuracy']}")

    # Shuffled with popular psalms
    print("\n--- Shuffled + Popular Psalms ---")
    task = make_utilitarianism_shuffled_task(
        injection=popular_injection, limit=limit, seed=42
    )
    logs = inspect_eval(task, model=model, log_dir=log_dir, cache_prompt=True)
    result = extract_score(logs[0])
    result["subset"] = "utilitarianism (shuffled)"
    result["condition"] = "popular psalms (shuffled labels)"
    results.append(result)
    print(f"  Accuracy: {result['accuracy']}")

    results_file = RESULTS_DIR / f"control_shuffled_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run control experiments for utilitarianism anomaly"
    )
    parser.add_argument(
        "--control",
        choices=["wikipedia", "secular", "shuffled", "all"],
        required=True,
        help="Which control experiment to run",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=MODELS,
        help="Model(s) for wikipedia/secular controls",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples",
    )

    args = parser.parse_args()

    if args.control in ("wikipedia", "all"):
        run_wikipedia_control(args.model, args.limit)
    if args.control in ("secular", "all"):
        run_secular_control(args.model, args.limit)
    if args.control in ("shuffled", "all"):
        run_shuffled_control(args.limit)


if __name__ == "__main__":
    main()
