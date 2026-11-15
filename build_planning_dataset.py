
import json
import random
from pathlib import Path

INPUT_FILE = Path("sequences.json")
OUTPUT_FILE = Path("planning_dataset.json")
SEED = 42


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    sequences = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    rng = random.Random(SEED)
    dataset = []

    for sequence in sequences:
        reference = list(sequence["operations"])
        shuffled = reference.copy()
        rng.shuffle(shuffled)

        # Avoid leaving the sequence in its original order when possible.
        if len(shuffled) > 1 and shuffled == reference:
            shuffled = shuffled[1:] + shuffled[:1]

        dataset.append({
            "recording_id": sequence["recording_id"],
            "phone": sequence["phone"],
            "input_operations": shuffled,
            "reference_operations": reference,
        })

    OUTPUT_FILE.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Recordings: {len(dataset)}")
    print(f"Output saved to: {OUTPUT_FILE.resolve()}")
    print("\nExample:")
    print(json.dumps(dataset[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

