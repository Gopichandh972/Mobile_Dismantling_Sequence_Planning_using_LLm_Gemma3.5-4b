
import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

INPUT_FILE = Path("planning_dataset.json")
OUTPUT_DIR = Path("gemma_planning_evaluation")
MODEL = "gemma3:4b"
API_URL = "http://localhost:11434/api/chat"
TIMEOUT = 240

SYSTEM_PROMPT = """
You are reconstructing an observed phone-disassembly operation sequence.

The input is a shuffled list of operations from one recording.
Return the operations in the most plausible order you can infer.

Rules:
- Return only a JSON array of strings.
- Use every supplied operation exactly as many times as it appears.
- Do not add, remove, rename, merge, or split operations.
- Do not include explanations or markdown.
- The original annotated order is not provided.
- Do not claim that your proposed order is physically safe or optimal.
"""


def edit_similarity(reference, prediction):
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

    return 1.0 - previous[-1] / max(len(reference), len(prediction), 1)


def valid_permutation(reference, prediction):
    return (
        isinstance(prediction, list)
        and all(isinstance(item, str) for item in prediction)
        and Counter(reference) == Counter(prediction)
    )


def call_gemma(phone, input_operations):
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
                    "Reconstruct the operation order from this shuffled list. "
                    "Preserve every operation and its count exactly:\n"
                    + json.dumps(input_operations, ensure_ascii=False)
                ),
            },
        ],
        "options": {
            "temperature": 0,
            "num_predict": max(256, len(input_operations) * 32),
        },
    }

    response = requests.post(API_URL, json=payload, timeout=TIMEOUT)
    response.raise_for_status()

    content = response.json()["message"]["content"]
    prediction = json.loads(content)

    if not isinstance(prediction, list) or not all(
        isinstance(item, str) for item in prediction
    ):
        raise ValueError("Model did not return a JSON array of strings")

    return prediction


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing {INPUT_FILE}")

    dataset = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    saved_predictions = []

    for index, item in enumerate(dataset, start=1):
        recording_id = item["recording_id"]
        phone = item["phone"]
        input_operations = item["input_operations"]
        reference = item["reference_operations"]

        print(
            f"[{index}/{len(dataset)}] {recording_id} | {phone} "
            f"| {len(input_operations)} operations",
            flush=True,
        )

        prediction = None
        error = None

        try:
            prediction = call_gemma(phone, input_operations)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        is_valid = (
            prediction is not None
            and valid_permutation(reference, prediction)
        )

        similarity = (
            edit_similarity(reference, prediction)
            if prediction is not None
            else None
        )

        exact_match = (
            prediction == reference
            if is_valid
            else False
        )

        rows.append({
            "recording_id": recording_id,
            "phone": phone,
            "reference_length": len(reference),
            "returned_prediction": prediction is not None,
            "valid_permutation": is_valid,
            "error": error,
            "edit_similarity": similarity,
            "exact_match": exact_match,
        })

        saved_predictions.append({
            "recording_id": recording_id,
            "phone": phone,
            "input_operations": input_operations,
            "reference_operations": reference,
            "prediction": prediction,
            "valid_permutation": is_valid,
            "error": error,
        })

        print(
            f"  valid={is_valid}; similarity={similarity}; "
            f"exact={exact_match}",
            flush=True,
        )

        time.sleep(0.5)

    results = pd.DataFrame(rows)
    results.to_csv(
        OUTPUT_DIR / "per_recording_metrics.csv",
        index=False,
    )

    (OUTPUT_DIR / "predictions.json").write_text(
        json.dumps(saved_predictions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    returned = results[results["returned_prediction"]]
    valid = results[results["valid_permutation"]]

    summary = {
        "model": MODEL,
        "task": "reconstruct_annotated_order_from_shuffled_operations",
        "recordings": len(results),
        "returned_prediction_count": int(results["returned_prediction"].sum()),
        "valid_permutation_count": int(results["valid_permutation"].sum()),
        "valid_permutation_rate": float(results["valid_permutation"].mean()),
        "mean_edit_similarity_all_returned": (
            float(returned["edit_similarity"].mean())
            if len(returned) else None
        ),
        "mean_edit_similarity_valid_only": (
            float(valid["edit_similarity"].mean())
            if len(valid) else None
        ),
        "exact_match_count": int(results["exact_match"].sum()),
        "exact_match_rate_all_recordings": float(results["exact_match"].mean()),
        "note": (
            "Measures agreement with the annotated order after shuffling "
            "the operation list. Does not establish physical feasibility, "
            "safety, or optimality. This dataset contains 20 recordings."
        ),
    }

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== GEMMA PLANNING EVALUATION ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
