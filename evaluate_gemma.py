
import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

INPUT_FILE = Path("sequences.json")
OUTPUT_DIR = Path("gemma_evaluation")
MODEL = "gemma3:4b"
API_URL = "http://localhost:11434/api/chat"
TIMEOUT = 240

SYSTEM_PROMPT = """
You are a phone-disassembly sequence organizer.
Reorder the supplied operations into a plausible disassembly order.
Return only a JSON array of strings.
Use every supplied operation exactly as many times as it appears.
Do not add, remove, rename, merge, or split operations.
Do not include explanations or markdown.
The input is an observed operation list, not a verified safety procedure.
"""


def edit_similarity(reference, prediction):
    """Normalized Levenshtein similarity, from 0 to 1."""
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


def is_valid_permutation(reference, prediction):
    return (
        isinstance(prediction, list)
        and all(isinstance(item, str) for item in prediction)
        and Counter(reference) == Counter(prediction)
    )


def call_model(phone, operations):
    payload = {
        "model": MODEL,
        "stream": False,
        "format": {
            "type": "array",
            "items": {"type": "string"},
        },
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Phone model: {phone}\n"
                    "Reorder these operations. Preserve every operation "
                    "and its count exactly:\n"
                    + json.dumps(operations, ensure_ascii=False)
                ),
            },
        ],
        "options": {
            "temperature": 0,
            "num_predict": max(256, len(operations) * 32),
        },
    }

    response = requests.post(API_URL, json=payload, timeout=TIMEOUT)
    response.raise_for_status()
    data = response.json()
    content = data["message"]["content"]
    result = json.loads(content)

    if not isinstance(result, list) or not all(
        isinstance(item, str) for item in result
    ):
        raise ValueError("Model did not return a JSON array of strings")

    return result


def safe_call(phone, operations):
    try:
        return call_model(phone, operations), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    sequences = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    predictions_output = []

    for index, sequence in enumerate(sequences, start=1):
        recording_id = sequence["recording_id"]
        phone = sequence["phone"]
        reference = sequence["operations"]

        print(
            f"[{index}/{len(sequences)}] {recording_id} | {phone} "
            f"| {len(reference)} operations",
            flush=True,
        )

        first_prediction, first_error = safe_call(phone, reference)
        first_valid = (
            first_prediction is not None
            and is_valid_permutation(reference, first_prediction)
        )

        retry_prediction = None
        retry_error = None
        retry_valid = False

        if not first_valid:
            print("  First attempt invalid/failed; retrying once.", flush=True)
            time.sleep(1)
            retry_prediction, retry_error = safe_call(phone, reference)
            retry_valid = (
                retry_prediction is not None
                and is_valid_permutation(reference, retry_prediction)
            )

        first_similarity = (
            edit_similarity(reference, first_prediction)
            if first_prediction is not None else None
        )
        retry_similarity = (
            edit_similarity(reference, retry_prediction)
            if retry_prediction is not None else None
        )

        # Order metrics are meaningful only if every input operation
        # appears exactly once per its original count.
        order_similarity = (
            edit_similarity(reference, first_prediction)
            if first_valid else None
        )

        row = {
            "recording_id": recording_id,
            "phone": phone,
            "reference_length": len(reference),
            "first_attempt_returned": first_prediction is not None,
            "first_attempt_valid": first_valid,
            "first_attempt_error": first_error,
            "first_attempt_edit_similarity": first_similarity,
            "first_attempt_exact_match": (
                first_prediction == reference if first_valid else None
            ),
            "first_attempt_order_similarity": order_similarity,
            "retry_attempted": not first_valid,
            "retry_returned": retry_prediction is not None,
            "retry_valid": retry_valid if not first_valid else None,
            "retry_error": retry_error,
            "retry_edit_similarity": retry_similarity,
            "retry_exact_match": (
                retry_prediction == reference if retry_valid else None
            ),
        }
        rows.append(row)

        predictions_output.append({
            "recording_id": recording_id,
            "phone": phone,
            "reference_operations": reference,
            "first_prediction": first_prediction,
            "first_error": first_error,
            "first_valid_permutation": first_valid,
            "retry_prediction": retry_prediction,
            "retry_error": retry_error,
            "retry_valid_permutation": retry_valid,
        })

        print(
            f"  first_valid={first_valid}; "
            f"edit_similarity={first_similarity}; "
            f"exact={row['first_attempt_exact_match']}",
            flush=True,
        )

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "per_recording_metrics.csv", index=False)

    (OUTPUT_DIR / "predictions.json").write_text(
        json.dumps(predictions_output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    first_valid_rows = results[results["first_attempt_valid"]]
    retry_rows = results[results["retry_attempted"]]
    retry_valid_rows = retry_rows[retry_rows["retry_valid"] == True]

    summary = {
        "model": MODEL,
        "recordings": len(results),
        "unique_phone_models": int(results["phone"].nunique()),
        "first_attempt_valid_count": int(results["first_attempt_valid"].sum()),
        "first_attempt_valid_rate": float(results["first_attempt_valid"].mean()),
        "first_attempt_mean_edit_similarity_all_returned": (
            float(results["first_attempt_edit_similarity"].dropna().mean())
            if results["first_attempt_edit_similarity"].notna().any()
            else None
        ),
        "first_attempt_mean_edit_similarity_valid_only": (
            float(first_valid_rows["first_attempt_edit_similarity"].mean())
            if len(first_valid_rows) else None
        ),
        "first_attempt_exact_match_count": int(
            (results["first_attempt_exact_match"] == True).sum()
        ),
        "retry_attempted_count": len(retry_rows),
        "retry_valid_count": len(retry_valid_rows),
        "retry_mean_edit_similarity_valid_only": (
            float(retry_valid_rows["retry_edit_similarity"].mean())
            if len(retry_valid_rows) else None
        ),
        "retry_exact_match_count": int(
            (retry_valid_rows["retry_exact_match"] == True).sum()
        ),
        "note": (
            "Agreement with observed annotations only; not proof of optimal "
            "or physically safe disassembly. Invalid permutations are excluded "
            "from order-agreement metrics. Repeated labels are scored by count."
        ),
    }

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== GEMMA EVALUATION SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nResults saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
