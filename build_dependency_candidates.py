
import json
from pathlib import Path
from collections import Counter

INPUT_FILE = Path("sequences.json")
OUTPUT_FILE = Path("dependency_candidates.json")


def main():
    sequences = json.loads(INPUT_FILE.read_text(encoding="utf-8"))

    # Count how often operation A occurs before operation B
    # in the same annotated recording.
    pair_counts = Counter()
    pair_recordings = {}

    for sequence in sequences:
        operations = sequence["operations"]
        unique_ops = list(dict.fromkeys(operations))

        for i, operation_a in enumerate(unique_ops):
            first_a = operations.index(operation_a)

            for operation_b in unique_ops[i + 1:]:
                first_b = operations.index(operation_b)

                if first_a < first_b:
                    pair = (operation_a, operation_b)
                else:
                    pair = (operation_b, operation_a)

                pair_counts[pair] += 1
                pair_recordings.setdefault(pair, set()).add(
                    sequence["recording_id"]
                )

    candidates = []

    for (before, after), count in pair_counts.items():
        recording_ids = sorted(pair_recordings[(before, after)])

        candidates.append({
            "before": before,
            "after": after,
            "recordings_observed_together": count,
            "recording_ids": recording_ids,
            "status": "UNVERIFIED_CANDIDATE",
            "evidence_note": (
                "Co-occurrence and relative order must be checked against "
                "the original annotations. This is not proof of a mandatory "
                "physical prerequisite."
            ),
        })

    candidates.sort(
        key=lambda item: (
            -item["recordings_observed_together"],
            item["before"],
            item["after"],
        )
    )

    OUTPUT_FILE.write_text(
        json.dumps(candidates, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Recordings analyzed:", len(sequences))
    print("Candidate ordered pairs:", len(candidates))
    print("Output saved to:", OUTPUT_FILE.resolve())
    print("\nFirst 15 candidates:")

    for item in candidates[:15]:
        print(
            f"{item['before']} -> {item['after']} "
            f"| co-occurring recordings: "
            f"{item['recordings_observed_together']}"
        )


if __name__ == "__main__":
    main()
