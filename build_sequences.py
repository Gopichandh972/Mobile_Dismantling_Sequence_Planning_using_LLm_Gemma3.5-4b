
import json
from pathlib import Path

import pandas as pd

INPUT_FILE = Path("annotations.csv")
OUTPUT_FILE = Path("sequences.json")


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required = {"recording_id", "phone", "start_seconds", "operation"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    sequences = []

    for recording_id, group in df.groupby("recording_id", sort=False):
        group = group.sort_values(["start_seconds", "end_seconds"])

        phone_names = group["phone"].dropna().unique()
        if len(phone_names) != 1:
            raise ValueError(
                f"{recording_id} contains multiple phone names: {phone_names}"
            )

        operations = [
            str(operation).strip()
            for operation in group["operation"]
            if pd.notna(operation)
            and str(operation).strip().casefold() != "noise"
        ]

        sequences.append({
            "recording_id": str(recording_id),
            "phone": str(phone_names[0]),
            "operations": operations,
        })

    OUTPUT_FILE.write_text(
        json.dumps(sequences, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Recordings: {len(sequences)}")
    print(f"Sequences with no operations: {sum(not s['operations'] for s in sequences)}")
    print(f"Total operation instances: {sum(len(s['operations']) for s in sequences)}")
    print(f"Unique operation labels: {len(set(op for s in sequences for op in s['operations']))}")
    print(f"Saved to: {OUTPUT_FILE.resolve()}")

    print("\nSequence lengths:")
    for sequence in sequences:
        print(
            f"{sequence['recording_id']} | {sequence['phone']} | "
            f"{len(sequence['operations'])} operations"
        )

    print("\nFirst sequence preview:")
    print(json.dumps(sequences[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
