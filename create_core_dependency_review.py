
import pandas as pd
from pathlib import Path

INPUT_FILE = Path("dependency_review_top30.csv")
OUTPUT_FILE = Path("core_dependency_review.csv")


def main():
    df = pd.read_csv(INPUT_FILE)

    # Keep candidates with at least four supporting recordings.
    core = df[df["recordings_supporting_order"] >= 4].copy()

    core["constraint_type"] = "UNCLASSIFIED"
    core["verification_status"] = "NEEDS_EVIDENCE"
    core["evidence_source"] = ""
    core["review_notes"] = ""

    core.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("Core candidate relationships:", len(core))
    print("Saved to:", Path(OUTPUT_FILE).resolve())
    print("\nCandidate relationships:")
    print(
        core[
            [
                "before_operation",
                "after_operation",
                "recordings_supporting_order",
                "constraint_type",
                "verification_status",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
