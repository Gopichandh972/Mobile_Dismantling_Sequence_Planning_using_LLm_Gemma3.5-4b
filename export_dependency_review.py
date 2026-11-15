
import json
from pathlib import Path

import pandas as pd

INPUT_FILE = Path("dependency_consistency.json")
OUTPUT_FILE = Path("dependency_review.csv")


def main():
    data = json.loads(INPUT_FILE.read_text(encoding="utf-8"))

    rows = []

    for item in data:
        if item["a_before_b_count"] >= item["b_before_a_count"]:
            before = item["operation_a"]
            after = item["operation_b"]
            observed_count = item["a_before_b_count"]
        else:
            before = item["operation_b"]
            after = item["operation_a"]
            observed_count = item["b_before_a_count"]

        rows.append({
            "before_operation": before,
            "after_operation": after,
            "recordings_supporting_order": observed_count,
            "recordings_compared": item["recordings_compared"],
            "observed_order_consistency": item["majority_order_consistency"],
            "evidence_recordings": "",
            "review_status": "UNREVIEWED",
            "review_notes": "",
        })

    df = pd.DataFrame(rows)
    df.sort_values(
        ["recordings_supporting_order", "observed_order_consistency"],
        ascending=[False, False],
        inplace=True,
    )
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"Candidate rows: {len(df)}")
    print(f"Saved to: {OUTPUT_FILE.resolve()}")
    print("Open the CSV in Excel to review candidate relationships.")


if __name__ == "__main__":
    main()
