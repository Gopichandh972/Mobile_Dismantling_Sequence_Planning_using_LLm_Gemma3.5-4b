
import json
import random
from pathlib import Path

import pandas as pd

INPUT_FILE = Path("sequences.json")
OUTPUT_DIR = Path("baseline_evaluation")
N_TRIALS = 100
BASE_SEED = 42


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

    return 1 - previous[-1] / max(len(reference), len(prediction), 1)


def main():
    sequences = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trial_rows = []
    recording_rows = []

    for sequence in sequences:
        reference = sequence["operations"]
        scores = []
        exact_matches = 0

        for trial in range(N_TRIALS):
            rng = random.Random(BASE_SEED + trial * 1000 + len(trial_rows))
            shuffled = reference.copy()
            rng.shuffle(shuffled)

            score = edit_similarity(reference, shuffled)
            scores.append(score)
            exact_matches += int(shuffled == reference)

            trial_rows.append({
                "recording_id": sequence["recording_id"],
                "phone": sequence["phone"],
                "trial": trial + 1,
                "edit_similarity": score,
                "exact_match": shuffled == reference,
            })

        recording_rows.append({
            "recording_id": sequence["recording_id"],
            "phone": sequence["phone"],
            "operation_count": len(reference),
            "mean_edit_similarity": sum(scores) / N_TRIALS,
            "std_edit_similarity": pd.Series(scores).std(ddof=1),
            "exact_matches": exact_matches,
            "exact_match_rate": exact_matches / N_TRIALS,
        })

    trial_df = pd.DataFrame(trial_rows)
    recording_df = pd.DataFrame(recording_rows)

    trial_df.to_csv(OUTPUT_DIR / "all_random_trials.csv", index=False)
    recording_df.to_csv(OUTPUT_DIR / "per_recording_metrics.csv", index=False)

    summary = {
        "baseline": "random_shuffle",
        "trials_per_recording": N_TRIALS,
        "recordings": len(sequences),
        "mean_edit_similarity": float(trial_df["edit_similarity"].mean()),
        "std_edit_similarity_across_all_trials": float(
            trial_df["edit_similarity"].std(ddof=1)
        ),
        "exact_match_rate": float(trial_df["exact_match"].mean()),
        "note": (
            "Random order is a baseline, not a physically informed "
            "disassembly planner. Compare models on the same recordings."
        ),
    }

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"\nSaved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
