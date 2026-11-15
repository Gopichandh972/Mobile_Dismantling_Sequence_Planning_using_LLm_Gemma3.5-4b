
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

INPUT_FILE = Path("sequences.json")
OUTPUT_DIR = Path("rule_baseline_evaluation")


def edit_similarity(reference, prediction):
    """Normalized Levenshtein similarity."""
    if not reference and not prediction:
        return 1.0

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


def get_positions(operations):
    """First position of each distinct operation in one sequence."""
    positions = {}

    for index, operation in enumerate(operations):
        positions.setdefault(operation, index)

    return positions


def learn_order_preferences(training_sequences):
    """
    Learn pairwise ordering preferences from training recordings only.
    Each recording contributes at most one vote per distinct operation pair.
    """
    votes = defaultdict(lambda: [0, 0])

    for operations in training_sequences:
        positions = get_positions(operations)
        unique_ops = list(positions)

        for i, a in enumerate(unique_ops):
            for b in unique_ops[i + 1:]:
                if positions[a] < positions[b]:
                    votes[(a, b)][0] += 1
                else:
                    votes[(a, b)][1] += 1

    preferences = {}

    for (a, b), (a_before_b, b_before_a) in votes.items():
        preferences[(a, b)] = (
            1 if a_before_b > b_before_a
            else -1 if b_before_a > a_before_b
            else 0
        )

    return preferences


def rank_operations(input_operations, preferences):
    """
    Rank operation labels using pairwise training preferences.
    Ties use a deterministic alphabetical tie-break.
    Repeated instances are retained in the output.
    """
    unique_ops = list(dict.fromkeys(input_operations))
    scores = {operation: 0.0 for operation in unique_ops}

    for i, a in enumerate(unique_ops):
        for b in unique_ops[i + 1:]:
            relation = preferences.get((a, b))

            if relation is None:
                reverse = preferences.get((b, a))
                relation = -reverse if reverse is not None else 0

            scores[a] += relation
            scores[b] -= relation

    ranked_labels = sorted(
        unique_ops,
        key=lambda operation: (-scores[operation], operation.lower()),
    )

    # Preserve each operation's original multiplicity.
    remaining = Counter(input_operations)
    prediction = []

    for operation in ranked_labels:
        prediction.extend([operation] * remaining[operation])

    return prediction


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    data = json.loads(INPUT_FILE.read_text(encoding="utf-8"))

    if len(data) < 2:
        raise ValueError("At least two recordings are required.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    predictions = []

    for index, held_out in enumerate(data, start=1):
        training = [
            item for item in data
            if item["recording_id"] != held_out["recording_id"]
        ]

        preferences = learn_order_preferences(
            [item["operations"] for item in training]
        )

        reference = list(held_out["operations"])
        # The input is treated as an unordered multiset.
        prediction = rank_operations(reference, preferences)

        valid = Counter(reference) == Counter(prediction)
        similarity = edit_similarity(reference, prediction)
        exact = prediction == reference

        rows.append({
            "recording_id": held_out["recording_id"],
            "phone": held_out["phone"],
            "operation_count": len(reference),
            "valid_permutation": valid,
            "edit_similarity": similarity,
            "exact_match": exact,
        })

        predictions.append({
            "recording_id": held_out["recording_id"],
            "phone": held_out["phone"],
            "prediction": prediction,
            "reference": reference,
            "valid_permutation": valid,
        })

        print(
            f"[{index}/{len(data)}] {held_out['recording_id']} | "
            f"{held_out['phone']} | similarity={similarity:.4f} | "
            f"exact={exact}",
            flush=True,
        )

    results = pd.DataFrame(rows)
    results.to_csv(
        OUTPUT_DIR / "per_recording_metrics.csv",
        index=False,
    )

    (OUTPUT_DIR / "predictions.json").write_text(
        json.dumps(predictions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {
        "baseline": "leave_one_recording_out_pairwise_order_preferences",
        "recordings": len(results),
        "valid_permutation_count": int(
            results["valid_permutation"].sum()
        ),
        "valid_permutation_rate": float(
            results["valid_permutation"].mean()
        ),
        "mean_edit_similarity": float(
            results["edit_similarity"].mean()
        ),
        "exact_match_count": int(results["exact_match"].sum()),
        "exact_match_rate": float(results["exact_match"].mean()),
        "note": (
            "Pairwise order preferences are learned without the held-out "
            "recording. This is an observational baseline, not a physically "
            "verified or safety-aware planner. Ties use alphabetical order."
        ),
    }

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== RULE-BASED BASELINE SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nResults saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
