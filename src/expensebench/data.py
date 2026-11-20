"""Load the frozen benchmark and expose one expense's observable records."""

import json
import sqlite3
import sysconfig
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
if not (DEFAULT_DATA_DIR / "observations.jsonl").is_file():
    DEFAULT_DATA_DIR = Path(sysconfig.get_path("data")) / "share" / "expense-policy-benchmark"


def _rows(data_dir, name):
    path = (Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR) / name
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def load_cases(data_dir=None):
    """Return the experiment manifest. Split and pairing metadata are evaluator-only."""
    return _rows(data_dir, "cases.jsonl")


def load_gold(data_dir=None):
    """Return evaluator-only labels and minimal supporting record IDs by case ID."""
    return {row["case_id"]: row for row in _rows(data_dir, "gold.jsonl")}


def build_database(path=":memory:", data_dir=None):
    """Materialize observable facts from public JSONL in a new SQLite database.

    The returned connection is query-only. IDs identify records within a case;
    use case_id for every lookup. No labels, split, family or pair metadata are
    loaded. Reopen an existing database with sqlite3.connect instead.
    """
    if str(path) != ":memory:" and Path(path).exists() and Path(path).stat().st_size:
        raise FileExistsError(f"Database already exists: {path}")
    policies = _rows(data_dir, "policies.jsonl")
    observations = _rows(data_dir, "observations.jsonl")
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA foreign_keys = ON")
        db.executescript("""
            CREATE TABLE policies (
                id TEXT PRIMARY KEY, effective_from TEXT NOT NULL, text TEXT NOT NULL
            );
            CREATE TABLE expenses (
                case_id TEXT PRIMARY KEY, id TEXT NOT NULL, policy_id TEXT NOT NULL REFERENCES policies(id),
                category TEXT NOT NULL, amount_cents INTEGER NOT NULL CHECK(amount_cents > 0),
                spent_at TEXT NOT NULL, merchant TEXT NOT NULL, currency TEXT NOT NULL
            );
            CREATE TABLE evidence (
                case_id TEXT NOT NULL REFERENCES expenses(case_id), id TEXT NOT NULL,
                kind TEXT NOT NULL, description TEXT NOT NULL, value_json TEXT NOT NULL,
                PRIMARY KEY(case_id, id)
            );
        """)
        with db:
            db.executemany("INSERT INTO policies VALUES (:id, :effective_from, :text)", policies)
            for row in observations:
                db.execute(
                    "INSERT INTO expenses VALUES (:case_id,:id,:policy_id,:category,:amount_cents,"
                    ":spent_at,:merchant,:currency)",
                    {"case_id": row["case_id"], "policy_id": row["policy_id"], **row["expense"]},
                )
                db.executemany(
                    "INSERT INTO evidence VALUES (?, ?, ?, ?, ?)",
                    [
                        (
                            row["case_id"],
                            entry["id"],
                            entry["kind"],
                            entry["description"],
                            json.dumps(entry["value"]),
                        )
                        for entry in row["evidence"]
                    ],
                )
        db.execute("PRAGMA query_only = ON")
        return db
    except Exception:
        db.close()
        raise


def case_packet(connection, case_id):
    """Return exactly the expense, effective clause and evidence visible for one case."""
    record = connection.execute(
        "SELECT id,policy_id,category,amount_cents,spent_at,merchant,currency "
        "FROM expenses WHERE case_id = ?",
        (case_id,),
    ).fetchone()
    if record is None:
        raise KeyError(case_id)
    expense = dict(record)
    policy_id = expense.pop("policy_id")
    policy = dict(
        connection.execute(
            "SELECT id,effective_from,text FROM policies WHERE id = ?",
            (policy_id,),
        ).fetchone()
    )
    evidence = []
    for row in connection.execute(
        "SELECT id,kind,description,value_json FROM evidence WHERE case_id = ? ORDER BY id",
        (case_id,),
    ):
        item = dict(row)
        item["value"] = json.loads(item.pop("value_json"))
        evidence.append(item)
    return {"case_id": case_id, "expense": expense, "policy": policy, "evidence": evidence}
