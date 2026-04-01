"""
Statistical analysis and reporting for experiment results.

Compares vanilla vs psalm-injected conditions and computes
significance metrics.
"""

import json
from pathlib import Path

from scipy import stats
from tabulate import tabulate


RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_results(results_file: str | Path) -> list[dict]:
    with open(results_file) as f:
        return json.load(f)


def pair_results(results: list[dict]) -> list[dict]:
    """Pair vanilla and psalm-injected results by (model, subset)."""
    vanilla = {}
    psalm = {}
    for r in results:
        key = (r["model"], r["subset"])
        if r["condition"] == "vanilla (no psalm)":
            vanilla[key] = r
        else:
            psalm[key] = r

    pairs = []
    for key in vanilla:
        if key in psalm:
            v = vanilla[key]
            p = psalm[key]
            delta = None
            if v["accuracy"] is not None and p["accuracy"] is not None:
                delta = p["accuracy"] - v["accuracy"]
            pairs.append({
                "model": key[0],
                "subset": key[1],
                "vanilla_acc": v["accuracy"],
                "psalm_acc": p["accuracy"],
                "vanilla_stderr": v.get("stderr"),
                "psalm_stderr": p.get("stderr"),
                "delta": delta,
                "psalm_condition": p["condition"],
                "psalm_numbers": p["psalm_numbers"],
            })
    return pairs


def compute_significance(pairs: list[dict]) -> list[dict]:
    """Add statistical significance info to paired results.

    Uses a two-proportion z-test on accuracy scores.
    """
    for pair in pairs:
        v_acc = pair["vanilla_acc"]
        p_acc = pair["psalm_acc"]
        v_se = pair.get("vanilla_stderr")
        p_se = pair.get("psalm_stderr")

        if v_acc is None or p_acc is None:
            pair["z_stat"] = None
            pair["p_value"] = None
            pair["significant"] = None
            continue

        if v_se and p_se and (v_se > 0 or p_se > 0):
            se_diff = (v_se**2 + p_se**2) ** 0.5
            if se_diff > 0:
                z = (p_acc - v_acc) / se_diff
                p_val = 2 * (1 - stats.norm.cdf(abs(z)))
                pair["z_stat"] = z
                pair["p_value"] = p_val
                pair["significant"] = p_val < 0.05
            else:
                pair["z_stat"] = None
                pair["p_value"] = None
                pair["significant"] = None
        else:
            pair["z_stat"] = None
            pair["p_value"] = None
            pair["significant"] = None

    return pairs


def print_comparison_table(results: list[dict]):
    """Print a formatted comparison table from raw results."""
    pairs = pair_results(results)
    pairs = compute_significance(pairs)

    headers = [
        "Model", "Subset", "Vanilla Acc", "Psalm Acc",
        "Delta", "p-value", "Sig?"
    ]
    rows = []
    for p in pairs:
        model_short = p["model"].split("/")[-1]
        v_acc = f"{p['vanilla_acc']:.4f}" if p["vanilla_acc"] is not None else "N/A"
        p_acc = f"{p['psalm_acc']:.4f}" if p["psalm_acc"] is not None else "N/A"

        if p["delta"] is not None:
            sign = "+" if p["delta"] >= 0 else ""
            delta = f"{sign}{p['delta']:.4f}"
        else:
            delta = "N/A"

        p_val = f"{p['p_value']:.4f}" if p["p_value"] is not None else "N/A"
        sig = "Yes" if p.get("significant") else ("No" if p.get("significant") is not None else "N/A")

        rows.append([model_short, p["subset"], v_acc, p_acc, delta, p_val, sig])

    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS: Vanilla vs Psalm-Injected")
    if pairs and pairs[0].get("psalm_condition"):
        print(f"Psalm condition: {pairs[0]['psalm_condition']}")
    print("=" * 80)
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Summary across subsets per model
    if len(pairs) > 1:
        print("\n--- Summary by Model ---")
        models = sorted(set(p["model"] for p in pairs))
        for model in models:
            model_pairs = [p for p in pairs if p["model"] == model]
            deltas = [p["delta"] for p in model_pairs if p["delta"] is not None]
            if deltas:
                avg_delta = sum(deltas) / len(deltas)
                sign = "+" if avg_delta >= 0 else ""
                print(f"  {model.split('/')[-1]}: avg delta = {sign}{avg_delta:.4f} "
                      f"across {len(deltas)} subsets")


def analyze_results_file(results_file: str | Path):
    """Load and analyze a results JSON file."""
    results = load_results(results_file)
    print_comparison_table(results)


def find_latest_results() -> Path | None:
    """Find the most recent results file."""
    if not RESULTS_DIR.exists():
        return None
    files = sorted(RESULTS_DIR.glob("results_*.json"))
    return files[-1] if files else None


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        analyze_results_file(sys.argv[1])
    else:
        latest = find_latest_results()
        if latest:
            print(f"Analyzing: {latest}")
            analyze_results_file(latest)
        else:
            print("No results files found. Run the experiment first:")
            print("  python -m src.experiment --quick")
