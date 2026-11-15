
import re
from pathlib import Path

import pandas as pd

INPUT_FILE = Path("timestampss.txt")
OUTPUT_FILE = Path("annotations.csv")


def to_seconds(timestamp):
    minutes, seconds = map(int, timestamp.strip().split(":"))
    if not 0 <= seconds < 60:
        raise ValueError(f"Invalid timestamp: {timestamp}")
    return minutes * 60 + seconds


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing {INPUT_FILE}")

    lines = INPUT_FILE.read_text(encoding="utf-8-sig").splitlines()

    rows = []
    recording_id = None
    phone = None

    heading_pattern = re.compile(
        r"^\s*(\d+)\.\s+(.+?)\s+[–—-]\s+https?://",
        re.IGNORECASE,
    )
    header_pattern = re.compile(r"^\s*Start\s+End\s+Operation\s*$", re.I)
    interval_pattern = re.compile(
        r"^\s*(\d+:\d{2})\s+(\d+:\d{2})\s+(.+?)\s*$"
    )

    for line_number, line in enumerate(lines, start=1):
        heading = heading_pattern.match(line)
        if heading:
            recording_number = int(heading.group(1))
            phone = heading.group(2).strip()
            recording_id = f"V{recording_number:02d}"
            continue

        if header_pattern.match(line):
            continue

        match = interval_pattern.match(line)
        if not match:
            continue

        if recording_id is None or phone is None:
            raise ValueError(f"Interval before recording heading at line {line_number}")

        start_text, end_text, operation = match.groups()
        start_seconds = to_seconds(start_text)
        end_seconds = to_seconds(end_text)

        if end_seconds <= start_seconds:
            raise ValueError(f"Invalid interval at line {line_number}")

        rows.append({
            "recording_id": recording_id,
            "phone": phone,
            "start": start_text,
            "end": end_text,
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "duration_seconds": end_seconds - start_seconds,
            "operation": operation.strip(),
            "source_line": line_number,
        })

    if not rows:
        raise ValueError("No intervals found. Check timestampss.txt.")

    df = pd.DataFrame(rows)

    # Validate intervals within each recording, not across phone models.
    overlaps = []
    gaps = []

    for rec_id, group in df.groupby("recording_id", sort=False):
        group = group.sort_values("start_seconds")
        previous_end = None

        for row in group.itertuples():
            if previous_end is not None:
                if row.start_seconds < previous_end:
                    overlaps.append((rec_id, row.start_seconds, previous_end))
                elif row.start_seconds > previous_end:
                    gaps.append((rec_id, previous_end, row.start_seconds))
            previous_end = row.end_seconds if previous_end is None else max(
                previous_end, row.end_seconds
            )

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"Parsed intervals: {len(df)}")
    print(f"Recordings found: {df['recording_id'].nunique()}")
    print(f"Phone models found: {df['phone'].nunique()}")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    print(f"Invalid intervals: {int((df['duration_seconds'] <= 0).sum())}")
    print(f"Overlaps within recordings: {len(overlaps)}")
    print(f"Gaps within recordings: {len(gaps)}")
    print(f"Output: {OUTPUT_FILE.resolve()}")

    print("\nIntervals per recording:")
    print(
        df.groupby(["recording_id", "phone"], sort=False)
        .size()
        .to_string()
    )

    if overlaps:
        print("\nFirst overlaps to inspect:")
        print(overlaps[:10])

    if gaps:
        print("\nFirst gaps to inspect:")
        print(gaps[:10])


if __name__ == "__main__":
    main()
