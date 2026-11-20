"""Generate v2's synthetic, two-stage expense decisions. No external data or dependencies."""

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import random


TOOLS = ("registry", "ledger", "document", "authorization")
PAIRS = {"train": 96, "dev": 24, "test": 24}


@dataclass(frozen=True)
class Check:
    tool: str
    field: str
    operator: str
    targets: tuple
    alternatives: tuple = ()


def equal(tool, field, wanted, *alternatives):
    return Check(tool, field, "equals", (wanted,), alternatives)


def limit(tool, field, operator, *targets):
    return Check(tool, field, operator, targets)


@dataclass(frozen=True)
class Family:
    name: str
    merchant: str
    clause: str
    first: Check
    second: Check


# Each family has its own authored policy wording and records. A numeric limit
# has three editions in the same split, analogous to different employer policies.
FAMILIES = {
    "train": (
        Family("fleet_bicycle_service", "Spoke and Hub",
               "A bicycle service claim must first be tied to a company fleet bicycle. Then check the "
               "documented service subtotal against the maximum of {b} cents.",
               equal("registry", "ownership_class", "company_fleet", "personal_bicycle", "rental_bicycle"),
               limit("document", "service_subtotal_cents", "at_most", 6500, 8000, 9500)),
        Family("translation_delivery", "Clear Text Cooperative",
               "Check the delivered translation first: at least {a} translated words must have been delivered. "
               "The rights authorization must then cover internal distribution.",
               limit("document", "translated_words", "at_least", 350, 500, 750),
               equal("authorization", "distribution_scope", "internal_distribution", "proof_only", "personal_use")),
        Family("field_office_day_pass", "Common Desk Hall",
               "The authorized worksite must be a field office before checking usage. Including this visit, "
               "the monthly day-pass count must be no greater than {b}.",
               equal("authorization", "worksite_class", "field_office", "home_office", "main_campus"),
               limit("ledger", "month_visits_including_current", "at_most", 4, 6, 8)),
        Family("shipment_insurance", "Parcel Cover Services",
               "Only shipments with a posted declared value of at least {a} cents qualify for insurance. "
               "Next check that the carrier is registered in the insured network.",
               limit("ledger", "declared_value_cents", "at_least", 10000, 15000, 25000),
               equal("registry", "carrier_network", "insured_network", "standard_network", "unlisted")),
        Family("badge_replacement", "Keycard Services",
               "Only a staff badge qualifies; check the credential registry first. Then verify at most "
               "{b} replacements, this request included, during the current badge year.",
               equal("registry", "credential_type", "staff_badge", "visitor_pass", "guest_qr"),
               limit("ledger", "badge_year_reissues", "at_most", 1, 2, 3)),
        Family("projector_delivery", "Screen Cart",
               "Start with the distance on the delivery slip: up to {a} kilometres qualifies. Then check "
               "the vehicle registry for a cargo vehicle.",
               limit("document", "delivery_km", "at_most", 50, 65, 80),
               equal("registry", "vehicle_class", "cargo_vehicle", "passenger_sedan", "motorcycle")),
        Family("archive_collection", "Boxroom Transfer",
               "First require at least {a} boxes on the posted collection order. If this qualifies, the "
               "shipping manifest must describe tamper-evident seals.",
               limit("ledger", "posted_box_count", "at_least", 8, 12, 15),
               equal("document", "seal_type", "tamper_evident", "plain_tape", "tie_wrap")),
        Family("media_destruction", "Media Shred Works",
               "The disposal request must authorize certified destruction. Once that condition holds, "
               "the certificate must cover at least {b} percent of the submitted media.",
               equal("authorization", "handling_class", "certified_destruction", "ordinary_disposal", "donation"),
               limit("document", "certificate_coverage_percent", "at_least", 85, 90, 95)),
    ),
    "dev": (
        Family("instrument_calibration", "Metrology Desk",
               "Begin with the lab's registration: a traceable standard is mandatory for this program. "
               "If confirmed, the proposed calibration interval may span up to {b} days.",
               equal("registry", "standard_class", "traceable_standard", "internal_reference", "training_reference"),
               limit("document", "interval_days", "at_most", 180, 270, 365)),
        Family("specimen_storage", "Field Cold Rooms",
               "Examine the peak temperature on the storage report first. It may not exceed {a} tenths "
               "of a degree Celsius. Next require a storage authorization specifically for research stock.",
               limit("document", "peak_tenths_celsius", "at_most", 40, 50, 60),
               equal("authorization", "stock_class", "research_stock", "personal_stock", "retail_inventory")),
        Family("visitor_interpretation", "Spoken Bridge",
               "Start by checking the timed session ledger. Sessions qualify at {a} interpreted minutes "
               "or longer. The visitor must then have granted written release for the session.",
               limit("ledger", "interpreted_minutes", "at_least", 20, 30, 45),
               equal("authorization", "consent_form", "written_release", "verbal_only", "release_declined")),
    ),
    "test": (
        Family("outage_generator_hire", "Grid Restore",
               "Determine the dispatch basis before counting hire hours. It must be an emergency outage. "
               "For an emergency, metered hire time qualifies through {b} hours inclusive.",
               equal("authorization", "dispatch_basis", "emergency_outage", "planned_test", "private_event"),
               limit("ledger", "metered_hire_hours", "at_most", 18, 24, 36)),
        Family("solvent_return", "Solvent Recovery Yard",
               "A carrier must hold an industrial collection permit before a load is assessed. The "
               "weighbridge docket must then show a return load of {b} kilograms or less.",
               equal("registry", "collection_permit", "industrial_collection", "municipal_only", "permit_lapsed"),
               limit("document", "net_return_kg", "at_most", 70, 100, 140)),
        Family("dependent_care_travel", "Worktrip Care",
               "Consult the posted itinerary first: reimbursement starts at {a} work-trip nights. For "
               "qualifying itineraries the provider invoice must specify dependent care.",
               limit("ledger", "posted_trip_nights", "at_least", 2, 3, 4),
               equal("document", "invoice_scope", "dependent_care", "housekeeping", "pet_care")),
        Family("warranty_repair_intake", "Renew Repair Network",
               "Read the warranty certificate before checking the vendor. The remaining term at intake "
               "must be at least {a} days. Next verify that the shop holds authorized repair status.",
               limit("document", "warranty_days_at_intake", "at_least", 5, 10, 14),
               equal("registry", "service_designation", "authorized_repair", "retail_only", "training_partner")),
        Family("accessible_sign_installation", "Readable Spaces",
               "First consult the site authorization; it must designate public access. Then the installer's "
               "test certificate must give a contrast ratio of at least {b} hundredths (450 means 4.50).",
               equal("authorization", "site_designation", "public_access", "private_residence", "staff_storage"),
               limit("document", "contrast_hundredths", "at_least", 450, 500, 550)),
        Family("independent_machine_inspection", "Inspection Exchange",
               "Read the examiner's certificate first. Independent accreditation is mandatory. Then "
               "check the asset ledger: this visit must cover at least {b} active machines.",
               equal("document", "examiner_status", "independent_accreditation", "internal_examiner", "trainee"),
               limit("ledger", "active_machines_inspected", "at_least", 5, 8, 10)),
    ),
}

PROCESS = (
    " Apply these requirements in the stated order. At the first unavailable value, refer the claim "
    "for review. Reject at the first known violation. Approve when both requirements are met. "
    "Cite this policy and the record for every check through the decisive check."
)


def digest(*parts):
    return sha256("|".join(map(str, ("expense-interactive-v2", *parts))).encode()).hexdigest()


def make_id(prefix, *parts):
    return prefix + "-" + digest(*parts)[:12]


def draw_value(check, target, state, rng):
    if state == "missing":
        return None
    if check.operator == "equals":
        return target if state == "pass" else rng.choice(check.alternatives)
    # Keep small discrete and percentage measurements in their physical range.
    span = min(1000, max(1, target // 4))
    if check.field.endswith("_percent"):
        span = min(span, 100 - target)
    if state == "pass":
        offset = rng.choice([0, 1, rng.randint(1, span)])
        return target + (offset if check.operator == "at_least" else -offset)
    offset = rng.choice([1, rng.randint(1, span)])
    return target + (-offset if check.operator == "at_least" else offset)


def state_pair(index, rng):
    """Choose two decision-changing states, differing at only one check."""
    focus = index % 2
    transition = (index // 2) % 3
    if transition == 0:
        arms = [["pass", "pass"], ["pass", "pass"]]
        arms[1][focus] = "fail"
    elif transition == 1:
        arms = [["pass", "pass"], ["pass", "pass"]]
        arms[1][focus] = "missing"
    else:
        # The second check can conflict with the first; the order is material.
        arms = [["pass", "pass"], ["pass", "pass"]]
        arms[0][focus], arms[1][focus] = "fail", "missing"
        if focus == 0:
            other = rng.choice(["pass", "missing", "fail"])
            arms[0][1] = arms[1][1] = other
    rng.shuffle(arms)
    return arms


def decoy(tool, rng):
    options = {
        "registry": ("office_region", ["east", "west", "central"]),
        "ledger": ("prior_month_line_count", [3, 8, 19]),
        "document": ("invoice_format", ["digital", "paper", "scanned_pdf"]),
        "authorization": ("filing_channel", ["mail", "portal", "service_desk"]),
    }
    field, values = options[tool]
    return field, rng.choice(values)


def build_pair(split, family, index):
    rng = random.Random(int(digest(family.name, index), 16))
    pair_id = make_id("M", family.name, index, "pair")
    # Edition is independent of the type of intervention.
    edition = int(digest(family.name, index, "edition"), 16) % 3
    checks = (family.first, family.second)
    targets = [c.targets[edition % len(c.targets)] for c in checks]
    policy = {
        "id": make_id("P", family.name, edition),
        "text": family.clause.format(a=targets[0], b=targets[1]) + PROCESS,
        "checks": [
            {"tool": c.tool, "field": c.field, "operator": c.operator, "target": t}
            for c, t in zip(checks, targets, strict=True)
        ],
    }
    expense = {"id": make_id("X", pair_id), "currency": "USD",
               "amount_cents": rng.randint(1600, 98000), "merchant": family.merchant}
    records = {}
    for tool in TOOLS:
        field, value = decoy(tool, rng)
        records[tool] = {"id": make_id("D", pair_id, tool), "field": field,
                         "value": value, "description": field.replace("_", " ")}
    arms = state_pair(index, rng)
    values = []
    for i, (check, target) in enumerate(zip(checks, targets, strict=True)):
        # Shared states get exactly the same observable value in both arms.
        values.append({
            state: draw_value(check, target, state, rng)
            for state in sorted({arm[i] for arm in arms})
        })
    cases, labels = [], []
    for arm_index, states in enumerate(arms):
        cid = make_id("C", pair_id, arm_index)
        tools = {name: dict(record) for name, record in records.items()}
        for i, (check, state) in enumerate(zip(checks, states, strict=True)):
            tools[check.tool].update(field=check.field, description=check.field.replace("_", " "),
                                     value=values[i][state])
        cases.append({"case_id": cid, "pair_id": pair_id, "family": family.name, "split": split,
                      "expense": expense, "policy": policy, "tools": tools})
        required = [policy["id"]]
        action, why = "approve", "Both ordered requirements are satisfied."
        for check, state in zip(checks, states, strict=True):
            required.append(tools[check.tool]["id"])
            if state == "missing":
                action, why = "review", f"The required {check.field} value is unavailable."
                break
            if state == "fail":
                action, why = "reject", f"The {check.field} requirement is violated."
                break
        labels.append({"case_id": cid, "action": action,
                       "required_evidence": required, "rationale": why})
    return cases, labels


def generate(destination, splits):
    for split in splits:
        folder = Path(destination) / split
        folder.mkdir(parents=True, exist_ok=True)
        cases, labels = [], []
        for family in FAMILIES[split]:
            for index in range(PAIRS[split]):
                pair, outcomes = build_pair(split, family, index)
                cases.extend(pair)
                labels.extend(outcomes)
        for name, rows in (("cases", cases), ("gold", labels)):
            with (folder / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as out:
                for row in sorted(rows, key=lambda r: r["case_id"]):
                    out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=FAMILIES, action="append")
    args = parser.parse_args()
    generate(args.output_dir, args.split or list(FAMILIES))
