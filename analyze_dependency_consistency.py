
import json
from collections import defaultdict
from pathlib import Path

INPUT_FILE = Path("sequences.json")
OUTPUT_FILE = Path("dependency_consistency.json")


def main():
    sequences = json.loads(INPUT_FILE.read_text(encoding="utf-8"))

    # Store each pair's observed ordering across recordings
    pair_directions = defaultdict(lambda: {"a_before_b": 0, "b_before_a": 0})

    for sequence in sequences:
        operations = sequence["operations"]
        positions = {}

        for index, operation in enumerate(operations):
            positions.setdefault(operation, index)

        unique_ops = list(positions)

        for i, a in enumerate(unique_ops):
            for b in unique_ops[i + 1:]:
                if positions[a] < positions[b]:
                    pair_directions[(a, b)]["a_before_b"] += 1
                else:
                    pair_directions[(a, b)]["b_before_a"] += 1

    results = []

    for (a, b), counts in pair_directions.items():
        total = counts["a_before_b"] + counts["b_before_a"]
        if total == 0:
            continue

        majority = max(counts["a_before_b"], counts["b_before_a"])
        consistency = majority / total

        results.append({
            "operation_a": a,
            "operation_b": b,
            "a_before_b_count": counts["a_before_b"],
            "b_before_a_count": counts["b_before_a"],
            "recordings_compared": total,
            "majority_order_consistency": round(consistency, 4),
            "status": (
                "CONSISTENT_OBSERVATION"
                if consistency == 1.0
                else "VARIABLE_OBSERVATION"
            ),
            "warning": (
                "Observed order is not proof of a mandatory physical dependency."
            ),
        })

    results.sort(
        key=lambda item: (
            -item["recordings_compared"],
            -item["majority_order_consistency"],
        )
    )

    OUTPUT_FILE.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    consistent = sum(
        item["status"] == "CONSISTENT_OBSERVATION" for item in results
    )

    print("Operation pairs analyzed:", len(results))
    print("Pairs with one observed direction:", consistent)
    print("Pairs with variable directions:", len(results) - consistent)
    print("Saved to:", OUTPUT_FILE.resolve())

    print("\nTop 15 pairs:")
    for item in results[:15]:
        print(
            f"{item['operation_a']} vs {item['operation_b']} | "
            f"A before B: {item['a_before_b_count']} | "
            f"B before A: {item['b_before_a_count']} | "
            f"Consistency: {item['majority_order_consistency']}"
        )


if __name__ == "__main__":
    main()
