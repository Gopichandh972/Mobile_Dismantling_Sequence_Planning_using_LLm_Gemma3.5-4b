
import json
from pathlib import Path

import pandas as pd

GEMMA_DIR = Path("gemma_evaluation")
BASELINE_DIR = Path("baseline_evaluation")
OUTPUT_DIR = Path("comparison_results")


def main():
    gemma_path = GEMMA_DIR / "per_recording_metrics.csv"
    baseline_path = BASELINE_DIR / "per_recording_metrics.csv"

    if not gemma_path.exists() or not baseline_path.exists():
        raise FileNotFoundError(
            "Missing Gemma or baseline per-recording metrics CSV."
        )

    gemma = pd.read_csv(gemma_path)
    baseline = pd.read_csv(baseline_path)

    # Restrict the main comparison to recordings where Gemma produced
    # a valid permutation on its first attempt.
    valid = gemma[gemma["first_attempt_valid"] == True].copy()

    if valid.empty:
        raise ValueError("No valid first-attempt Gemma predictions found.")

    # The baseline CSV now contains the average over 100 random trials.
    merged = valid.merge(
        baseline[["recording_id", "mean_edit_similarity", "std_edit_similarity"]],
        on="recording_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(valid):
        raise ValueError("Some valid Gemma recordings are missing baseline results.")

    merged = merged.rename(columns={
        "first_attempt_edit_similarity": "gemma_edit_similarity",
        "mean_edit_similarity": "random_mean_edit_similarity",
        "std_edit_similarity": "random_std_edit_similarity",
    })

    merged["similarity_difference"] = (
        merged["gemma_edit_similarity"]
        - merged["random_mean_edit_similarity"]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_DIR / "per_recording_comparison.csv", index=False)

    gemma_summary = json.loads(
        (GEMMA_DIR / "summary.json").read_text(encoding="utf-8")
    )
    baseline_summary = json.loads(
        (BASELINE_DIR / "summary.json").read_text(encoding="utf-8")
    )

    summary = {
        "gemma_model": gemma_summary["model"],
        "all_recordings": int(len(gemma)),
        "gemma_first_attempt_valid_count": int(
            gemma["first_attempt_valid"].sum()
        ),
        "gemma_first_attempt_valid_rate": float(
            gemma["first_attempt_valid"].mean()
        ),
        "gemma_exact_matches_all_recordings": int(
            (gemma["first_attempt_exact_match"] == True).sum()
        ),
        "gemma_exact_match_rate_all_recordings": float(
            (gemma["first_attempt_exact_match"] == True).sum() / len(gemma)
        ),
        "common_valid_recordings": int(len(merged)),
        "gemma_mean_edit_similarity_common_valid": float(
            merged["gemma_edit_similarity"].mean()
        ),
        "random_mean_edit_similarity_same_recordings": float(
            merged["random_mean_edit_similarity"].mean()
        ),
        "mean_similarity_difference": float(
            merged["similarity_difference"].mean()
        ),
        "random_baseline_overall_mean_all_recordings": float(
            baseline_summary["mean_edit_similarity"]
        ),
        "interpretation_note": (
            "The paired subset includes recordings where Gemma's first "
            "prediction was a valid permutation. Random scores are averaged "
            "over 100 shuffles per recording. This measures order agreement "
            "with annotations, not physical safety or optimality."
        ),
    }

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("=== GEMMA VS RANDOM BASELINE ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved comparison to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
