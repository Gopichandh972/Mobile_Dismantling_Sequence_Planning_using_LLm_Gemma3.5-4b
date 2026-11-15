
import json
from pathlib import Path
from collections import Counter

import pandas as pd

ROOT = Path(".")
OUTPUT_DIR = ROOT / "final_comparable_evaluation"
OUTPUT_DIR.mkdir(exist_ok=True)

EXPERIMENTS = {
    "Gemma shuffled-input": (
        "gemma_planning_evaluation/predictions.json",
        "predicted_operations",
        "reference_operations",
    ),
    "Random baseline": (
        "baseline_evaluation/predictions.json",
        "random_prediction",
        "reference_operations",
    ),
    "Rule-based baseline": (
        "rule_baseline_evaluation/predictions.json",
        "predicted_operations",
        "reference_operations",
    ),
}
def edit_similarity(reference, prediction):
    previous = list(range(len(prediction) + 1))

    for i, ref_item in enumerate(reference, start=1):
        current = [i]

        for j, pred_item in enumerate(prediction, start=1):
            cost = 0 if ref_item == pred_item else 1
            current.append(min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + cost,
            ))

        previous = current

    distance = previous[-1]
    return 1.0 - distance / max(len(reference), len(prediction), 1)


def main():
    all_rows = []

    for method, (relative_path, pred_key, ref_key) in EXPERIMENTS.items():
        path = ROOT / relative_path

        if not path.exists():
            print(f"ERROR: Missing file: {path}")
            print("Check the prediction JSON filenames before continuing.")
            return

        records = json.loads(path.read_text(encoding="utf-8"))

        for item in records:
            prediction = item.get(pred_key)
        if prediction is None:
            prediction = item.get("prediction")
        if prediction is None:
            prediction = item.get("random_prediction")
        if prediction is None:
            prediction = item.get("predicted_operations")
            reference = item.get(ref_key)

            # Accommodate the different schemas used by saved experiments.
            if reference is None:
                reference = item.get("reference")

            if prediction is None or reference is None:
                raise ValueError(
                    f"Missing prediction/reference for {method}, "
                    f"recording {item.get('recording_id')}"
                )

            prediction = list(prediction)
            reference = list(reference)

            all_rows.append({
                "method": method,
                "recording_id": item["recording_id"],
                "phone": item["phone"],
                "reference_length": len(reference),
                "prediction_length": len(prediction),
                "valid_permutation": Counter(prediction) == Counter(reference),
                "edit_similarity": edit_similarity(reference, prediction),
                "exact_match": prediction == reference,
            })

    details = pd.DataFrame(all_rows)
    if details.empty:
       raise ValueError(
        "No records were loaded. Check the prediction JSON schemas and field names."
    )

    print("Loaded records:", len(details))
    print("Columns:", details.columns.tolist())
    print(details.head(2).to_string(index=False))

    # Ensure all methods are evaluated on exactly the same recording IDs.
    id_sets = {
        method: set(group["recording_id"])
        for method, group in details.groupby("method")
    }
    common_ids = set.intersection(*id_sets.values())

    if not common_ids:
        raise ValueError("No common recording IDs across methods.")

    details = details[details["recording_id"].isin(common_ids)].copy()

    summary = (
        details.groupby("method")
        .agg(
            recordings=("recording_id", "nunique"),
            valid_count=("valid_permutation", "sum"),
            valid_rate=("valid_permutation", "mean"),
            mean_edit_similarity=("edit_similarity", "mean"),
            exact_match_count=("exact_match", "sum"),
            exact_match_rate=("exact_match", "mean"),
        )
        .reset_index()
    )

    details.to_csv(
        OUTPUT_DIR / "per_recording_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        OUTPUT_DIR / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Common recordings:", len(common_ids))
    print("\n=== COMPARABLE EVALUATION ===")
    print(summary.to_string(index=False))
    print(f"\nSaved results to: {OUTPUT_DIR.resolve()}")
    print(
        "\nNote: these metrics measure agreement with observed annotations. "
        "They do not establish physical feasibility or safety."
    )


if __name__ == "__main__":
    main()
         