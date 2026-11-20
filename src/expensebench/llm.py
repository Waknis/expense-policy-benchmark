"""Run the paired, local Qwen policy-information ablation via llama.cpp."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib import request


METHODS = ("llm_policy", "llm_evidence", "llm_finding")
MODEL = "qwen2.5-7b-instruct-q4_k_m"
SEED = 42
MAX_TOKENS = 240
SYSTEM = (
    "You review corporate expense claims under the supplied written policy. "
    "Decide whether to approve the claim, reject it, or send it for human review. "
    "Use only the information shown. Apply the policy's instructions for exceptions "
    "and missing or conflicting information. Do not assume that undisclosed records "
    "exist or that they establish a violation. Amounts named amount_cents are in "
    "cents. Return a JSON object with action and evidence_ids. Select the applicable "
    "policy ID and the IDs of the specific records that support your decision. "
    "Use each ID exactly as supplied."
)
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["approve", "review", "reject"]},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["action", "evidence_ids"],
    "additionalProperties": False,
}
FINDING_INSTRUCTION = (
    " Add a finding as the first JSON field. State one concise, checkable fact that "
    "matters to this claim, such as a computed amount, date relationship, or "
    "missing record. Then provide the action and evidence_ids."
)
FINDING_SCHEMA = {
    **OUTPUT_SCHEMA,
    "properties": {"finding": {"type": "string"}, **OUTPUT_SCHEMA["properties"]},
    "required": ["finding", *OUTPUT_SCHEMA["required"]],
}


def messages_for(packet, method):
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    visible = {"policy": packet["policy"], "expense": packet["expense"]}
    if method != "llm_policy":
        visible["evidence"] = packet["evidence"]
    instructions = SYSTEM + FINDING_INSTRUCTION if method == "llm_finding" else SYSTEM
    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": json.dumps(visible, ensure_ascii=False, sort_keys=True)},
    ]


def parse_response(raw, finish_reason, method="llm_evidence"):
    """Retain malformed, truncated or structurally invalid outputs as invalid actions."""
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return "invalid", []
    if finish_reason != "stop" or not isinstance(parsed, dict):
        return "invalid", []
    expected = {"action", "evidence_ids"}
    if method == "llm_finding":
        expected.add("finding")
        if not isinstance(parsed.get("finding"), str) or not parsed["finding"].strip():
            return "invalid", []
    if set(parsed) != expected:
        return "invalid", []
    if parsed["action"] not in ("approve", "reject", "review"):
        return "invalid", []
    ids = parsed["evidence_ids"]
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        return "invalid", []
    return parsed["action"], ids


def request_body(packet, method, model=MODEL):
    """Reconstruct the exact UTF-8 bytes hashed and sent for one prediction."""
    messages = messages_for(packet, method)
    schema = FINDING_SCHEMA if method == "llm_finding" else OUTPUT_SCHEMA
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "seed": SEED,
        "max_tokens": MAX_TOKENS,
        "stream": False,
        "response_format": {"type": "json_schema", "json_schema": {"name": "decision", "schema": schema}},
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def predict(endpoint, model, method, split, packet):
    body = request_body(packet, method, model)
    req = request.Request(endpoint.rstrip("/") + "/chat/completions", data=body,
                          headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=180) as response:
        completion = json.load(response)
    choice = completion["choices"][0]
    raw = choice["message"]["content"]
    finish = choice["finish_reason"]
    action, citations = parse_response(raw, finish, method)
    return {
        "case_id": packet["case_id"],
        "split": split,
        "method": method,
        "model": model,
        "action": action,
        "evidence_ids": citations,
        "raw": raw,
        "finish_reason": finish,
        "prompt_tokens": completion["usage"]["prompt_tokens"],
        "completion_tokens": completion["usage"]["completion_tokens"],
        "prompt_sha256": hashlib.sha256(body).hexdigest(),
    }


def write_split(path, predictions, split):
    """Publish a completed split atomically, preserving the other split for this method."""
    rows = []
    if split != "all" and path.exists():
        with path.open(encoding="utf-8") as old:
            for line in old:
                previous = json.loads(line)
                if previous["split"] != split:
                    rows.append(previous)
    rows.extend(predictions)
    if len({row["case_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate case IDs in predictions")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent,
                                         delete=False, prefix="." + path.name + ".")
    tmp_path = Path(handle.name)
    try:
        with handle:
            for row in sorted(rows, key=lambda value: (value["split"], value["case_id"])):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def main(argv=None):
    from .data import build_database, case_packet, load_cases

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18081/v1")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    parser.add_argument("--method", choices=("all", *METHODS), default="all")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("results/predictions"))
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    cases = [c for c in load_cases() if args.split == "all" or c["split"] == args.split]
    if not cases:
        parser.error("selected split has no cases")
    db = build_database()
    try:
        packets = [(case["split"], case_packet(db, case["case_id"])) for case in cases]
    finally:
        db.close()

    for method in METHODS if args.method == "all" else (args.method,):
        def score(item):
            split, packet = item
            return predict(args.endpoint, args.model, method, split, packet)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            predictions = list(pool.map(score, packets))
        path = args.output_dir / f"{method}.jsonl"
        write_split(path, predictions, args.split)
        print(f"Wrote {len(predictions)} {args.split} predictions for {method} to {path}", flush=True)


if __name__ == "__main__":
    main()
