
import pandas as pd
from pathlib import Path

INPUT_FILE = Path("core_dependency_review.csv")
OUTPUT_FILE = Path("core_dependency_review_classified.csv")


def classify(row):
    before = row["before_operation"].strip().lower()
    after = row["after_operation"].strip().lower()

    # These are candidates for device-specific verification,
    # not automatically confirmed physical constraints.
    if (
        "battery connector disconnection" in before
        and after in {"battery removal", "motherboard removal"}
    ):
        return (
            "CANDIDATE_PREREQUISITE",
            "NEEDS_DEVICE_SPECIFIC_EVIDENCE",
            "Verify the exact connector, component, and device model."
        )

    if (
        before in {"screw removal", "bottom screw removal"}
        and after in {
            "battery removal",
            "motherboard removal",
            "speaker removal",
            "cable disconnection",
        }
    ):
        return (
            "CANDIDATE_PREREQUISITE",
            "NEEDS_DEVICE_SPECIFIC_EVIDENCE",
            "Generic screw labels do not identify which component is secured."
        )

    return (
        "OBSERVED_ORDER_ONLY",
        "UNVERIFIED",
        "Recurring annotation order alone does not establish a mandatory dependency."
    )


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    classifications = df.apply(classify, axis=1, result_type="expand")
    classifications.columns = [
        "constraint_type",
        "verification_status",
        "review_notes",
    ]

    df[["constraint_type", "verification_status", "review_notes"]] = (
        classifications
    )

    df["evidence_source"] = ""
    df["review_status"] = "UNREVIEWED"

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("Relationships reviewed:", len(df))
    print("\nClassification counts:")
    print(df["constraint_type"].value_counts().to_string())
    print("\nVerification status counts:")
    print(df["verification_status"].value_counts().to_string())
    print(f"\nSaved to: {Path(OUTPUT_FILE).resolve()}")


if __name__ == "__main__":
    main()
