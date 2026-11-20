"""Independently reconstruct the primary counts from released decisions using SQLite."""

import csv
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    systems = json.loads((ROOT / "results/heldout.json").read_text(encoding="utf-8"))["systems"]
    query = Path(__file__).with_suffix(".sql").read_text(encoding="utf-8")
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    try:
        db.execute("CREATE TABLE audit_rows(record TEXT NOT NULL)")
        with (ROOT / "results/heldout_decisions.jsonl").open(encoding="utf-8") as stream:
            db.executemany("INSERT INTO audit_rows VALUES (?)", ((line,) for line in stream if line.strip()))
        rows = [dict(row) for row in db.execute(query)]
    finally:
        db.close()
    if {row["method"] for row in rows} != set(systems):
        raise ValueError("Methods in the decision log differ from the score report.")
    for row in rows:
        for key, value in row.items():
            if key != "method" and systems[row["method"]][key] != value:
                raise ValueError(f"{row['method']}: SQLite and the published report disagree for {key}.")
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
