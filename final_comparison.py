
import json
from pathlib import Path

import pandas as pd

ROOT = Path(".")
OUTPUT_DIR = ROOT / "final_comparison"
OUTPUT_DIR.mkdir(exist_ok=True)

EXPERIMENTS = {
    "Gemma original-order": {
        "folder": "gemma_evaluation",
        "summary": "summary.json",
        "csv": "per_recording_metrics.csv",
    },
    "Gemma shuffled-input": {
        "folder": "gemma_planning_evaluation",
        "summary": "summary.json",
        "csv": "per_recording_metrics.csv",
    },
    "Random baseline": {
        "folder": "baseline_evaluation",
        "summary": "summary.json",
        "csv": "per_recording_metrics.csv",
    },
    "Rule-based baseline": {
        "folder": "rule_baseline_evaluation",
        "summary": "summary.json",
        "csv": "per_recording_metrics.csv",
    },
}


def read_summary(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    summary_rows = []
    recording_frames = []

    for experiment, config in EXPERIMENTS.items():
        folder = ROOT / config["folder"]
        summary_path = folder / config["summary"]
        csv_path = folder / config["csv"]

        if not summary_path.exists():
            print(f"WARNING: Missing summary: {summary_path}")
            continue

        summary = read_summary(summary_path)

        summary_rows.append({
            "experiment": experiment,
            "model_or_baseline": summary.get(
                "model", summary.get("baseline", "not specified")
            ),
            "recordings": summary.get(
                "recordings", summary.get("phones", None)
            ),
            "valid_count": summary.get(
                "valid_permutation_count",
                summary.get("first_attempt_valid_count", None),
            ),
            "valid_rate": summary.get(
                "valid_permutation_rate",
                summary.get("first_attempt_valid_rate", None),
            ),
            "mean_edit_similarity": summary.get(
                "mean_edit_similarity",
                summary.get(
                    "mean_edit_similarity_all_returned",
                    summary.get(
                        "first_attempt_mean_edit_similarity_all_returned",
                        summary.get(
                            "first_attempt_mean_edit_similarity_valid_only",
                            None,
                        ),
                    ),
                ),
            ),
            "exact_match_count": summary.get(
                "exact_match_count",
                summary.get(
                    "exact_match_rate_all_recordings",
                    summary.get("first_attempt_exact_match_count", None),
                ),
            ),
            "source_summary": str(summary_path),
            "interpretation_note": summary.get("note", ""),
        })

        if csv_path.exists():
            frame = pd.read_csv(csv_path)
            frame.insert(0, "experiment", experiment)
            recording_frames.append(frame)
        else:
            print(f"WARNING: Missing per-recording CSV: {csv_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "summary_comparison.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    if recording_frames:
        detail_df = pd.concat(recording_frames, ignore_index=True, sort=False)
        detail_path = OUTPUT_DIR / "per_recording_comparison.csv"
        detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
        print(f"Per-recording comparison saved: {detail_path.resolve()}")

    (OUTPUT_DIR / "summary_comparison.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== EXPERIMENT SUMMARY COMPARISON ===")
    print(summary_df.to_string(index=False))
    print(f"\nSummary CSV saved: {summary_path.resolve()}")
    print(
        "\nNote: different input conditions and metric subsets mean these "
        "scores should not automatically be treated as a fair model ranking."
    )


if __name__ == "__main__":
    main()
