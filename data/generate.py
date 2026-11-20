"""Regenerate the published synthetic fixtures from the v1 authored case designs."""

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


def write_fixtures(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    meta = None
    cases, policies, observations, gold = [], [], [], []
    directions = [
        ("approve", "reject"),
        ("reject", "review"),
        ("review", "approve"),
        ("reject", "approve"),
        ("approve", "review"),
        ("review", "reject"),
    ]
    merchants = {
        "meals": ["Fieldstone Kitchen", "Market Lunchroom", "Green Table"],
        "ground_transport": ["Cab North", "Rivertown Taxi", "Cedar Cabs"],
        "rail": ["Atlas Regional", "Valley Passenger Rail", "Civic Rail"],
        "training": ["Center for Applied Learning", "Workshop House", "Practical Academy"],
        "lodging": ["Quarry Hotel", "The Orchard Inn", "Larch House"],
        "air_travel": ["North Coast Air", "Transit Air", "Lake Airlines"],
        "equipment": ["Studio Supply", "Park Office Supply", "Keystone Goods"],
    }

    def hashed(*parts):
        """Stable public IDs without a family, split or action prefix."""
        return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()

    def rec(kind, description, value):
        return (kind, description, value)

    def family(name, split, category, text):
        """Start a policy family for the ensuing six matched pairs."""
        nonlocal meta
        policy_id = "P-" + hashed("policy", name)[:5].upper()
        policies.append({"id": policy_id, "effective_from": "2025-01-01", "text": text})
        meta = (name, split, category, policy_id)

    def pair(
        i, amount, extras, changed, outcomes, required, *, stamp=None, currency="USD", receipt=None
    ):
        """Instantiate a fixed pair by modifying one case record and its expected outcome."""
        name, split, category, policy_id = meta
        pair_key = f"{name}:{i}"
        date = stamp or f"2025-{3 + i:02d}-{9 + 2 * i:02d}T15:00:00Z"
        eid = "D-" + hashed(pair_key, "expense")[:5].upper()
        pair_id = "M-" + hashed("pair", pair_key)[:8]
        expense = {
            "id": eid,
            "category": category,
            "amount_cents": amount,
            "spent_at": date,
            "merchant": merchants[category][i % 3],
            "currency": currency,
        }
        if receipt is None:
            receipt = ["itemized", "itemized", "card slip", "itemized", "unreadable", "itemized"][i]
        facts = [
            rec("receipt", "Format recorded by document intake", receipt),
            rec(
                "business_purpose",
                "Employee's stated purpose",
                {
                    "meals": "Project meeting",
                    "ground_transport": "Travel during field assignment",
                    "rail": "Travel to regional office",
                    "training": "Employee skills workshop",
                    "lodging": "Overnight for assignment",
                    "air_travel": "Client site visit",
                    "equipment": "Workstation equipment",
                }[category],
            ),
            *extras,
        ]
        identifiers = {kind: "D-" + hashed(pair_key, kind)[:5].upper() for kind, _, _ in facts}
        assert len(identifiers) == len(facts) and eid not in identifiers.values()
        for arm, outcome in enumerate(directions[i]):
            val, explanation = outcomes[outcome]
            cid = "C-" + hashed("frozen-observation-v1", pair_key, arm)[:9]
            evidence = sorted(
                (
                    {
                        "id": identifiers[kind],
                        "kind": kind,
                        "description": desc,
                        "value": val if kind == changed else value,
                    }
                    for kind, desc, value in facts
                ),
                key=lambda r: r["id"],
            )
            observations.append(
                {"case_id": cid, "policy_id": policy_id, "expense": expense, "evidence": evidence}
            )
            cases.append(
                {
                    "case_id": cid,
                    "split": split,
                    "family": name,
                    "pair_id": pair_id,
                    "policy_id": policy_id,
                    "expense_id": eid,
                }
            )
            refs = required[outcome] if isinstance(required, dict) else required
            gold.append(
                {
                    "case_id": cid,
                    "action": outcome,
                    "required_evidence": sorted(
                        [policy_id] + [eid if x == "EXP" else identifiers[x] for x in refs]
                    ),
                    "rationale": explanation,
                }
            )

    family(
        "hosted_meal_allowance",
        "dev",
        "meals",
        "Hosted meal allowance. Approve up to and including USD 45.00 per confirmed attendee, including the host. "
        "Use the card total including tax and tip and the confirmed event roster. Reject a documented overage. "
        "If the roster count is unavailable, send for review. The receipt format and employee memo do not establish attendance.",
    )
    for i, (pax, delta) in enumerate([(2, 0), (4, -1), (3, -25), (2, -1), (4, 0), (3, 0)]):
        total = 4500 * pax + delta
        pair(
            i,
            total,
            [
                rec("confirmed_attendees", "Count on the signed event roster", None),
                rec("invited_attendees", "Calendar invitation count", pax + 2),
            ],
            "confirmed_attendees",
            {
                "approve": (pax, f"The total fits the limit for {pax} confirmed attendees."),
                "reject": (pax - 1, f"The total exceeds the limit for {pax - 1} confirmed attendees."),
                "review": (None, "The confirmed count is unavailable."),
            },
            {
                "approve": ["EXP", "confirmed_attendees"],
                "reject": ["EXP", "confirmed_attendees"],
                "review": ["confirmed_attendees"],
            },
        )

    family(
        "local_transport",
        "dev",
        "ground_transport",
        "Local transport. The dispatched trip classification is authoritative. Approve a journey between a company worksite "
        "and a customer site in either direction. Reject a home-to-regular-office commute in either direction. "
        "Route unresolved destinations or any other classification to review. A memo is insufficient to determine the route.",
    )
    for i in range(6):
        pair(
            i,
            [2400, 3560, 2700, 4900, 3100, 1800][i],
            [rec("dispatch_route", "Dispatcher-verified origin and destination", None)],
            "dispatch_route",
            {
                "approve": (
                    ["worksite to customer", "customer to worksite"][i % 2],
                    "Dispatch confirms an eligible work trip.",
                ),
                "reject": (
                    ["home to regular office", "regular office to home"][i % 2],
                    "Dispatch confirms an excluded commute.",
                ),
                "review": (
                    ["destination unresolved", "origin unresolved", "home to customer"][i % 3],
                    "The recorded route requires manual review.",
                ),
            },
            ["dispatch_route"],
        )

    family(
        "rail_contract",
        "dev",
        "rail",
        "Rail vendor contract. For a booking made through a reseller, look up that reseller in the current corporate "
        "travel registry. Approve active partners and reject suspended or terminated partners. A missing or provisional "
        "registry designation needs review. Direct operator bookings can be approved without a reseller lookup.",
    )
    for i in range(6):
        pair(
            i,
            [4800, 6300, 12900, 8600, 4500, 14000][i],
            [
                rec("booking_channel", "Supplier listed on purchase order", "reseller"),
                rec("registry_status", "Current registry designation of invoicing reseller", None),
                rec(
                    "historical_registry_status",
                    "Registry value from the previous year",
                    "active partner",
                ),
            ],
            "registry_status",
            {
                "approve": ("active partner", "The reseller is a current active partner."),
                "reject": (
                    ["suspended", "terminated"][i % 2],
                    "The reseller's current designation excludes reimbursement.",
                ),
                "review": ([None, "provisional"][i % 2], "Current reseller eligibility is unresolved."),
            },
            ["booking_channel", "registry_status"],
        )

    family(
        "course_enrollment",
        "dev",
        "training",
        "External courses. The published course start date controls enrollment deadlines. Approve a final enrollment "
        "confirmed on or before the course start date; reject a final enrollment confirmed later. Route missing "
        "confirmation dates to review. Calendar dates are UTC. The date on the course invoice does not establish enrollment.",
    )
    for i in range(6):
        start = dt.date(2025, 4 + i, 16)
        pair(
            i,
            [17500, 8500, 6500, 15000, 24000, 9500][i],
            [
                rec("course_start", "Published course start date (UTC)", str(start)),
                rec("enrollment_date", "Registrar's final enrollment confirmation date (UTC)", None),
                rec("invoice_date", "Date invoice was prepared", str(start - dt.timedelta(days=2))),
            ],
            "enrollment_date",
            {
                "approve": (
                    str(start - dt.timedelta(days=i % 2)),
                    "Enrollment was confirmed by the course start.",
                ),
                "reject": (
                    str(start + dt.timedelta(days=1 + i % 2)),
                    "Enrollment was confirmed after the course started.",
                ),
                "review": (None, "The registrar's confirmation date is unavailable."),
            },
            {
                "approve": ["course_start", "enrollment_date"],
                "reject": ["course_start", "enrollment_date"],
                "review": ["enrollment_date"],
            },
        )

    family(
        "lodging_room_rate",
        "test",
        "lodging",
        "Lodging cap. Use the room-only subtotal in USD cents before taxes, divided by the billed number of nights. "
        "In a metro rate zone the nightly cap is USD 220; in a standard zone it is USD 160. Approve rates at or below "
        "the cap and reject higher rates. If a required folio or zone field is unavailable, route to review. Ignore taxes, deposits and card totals.",
    )
    for i, (zone, nights) in enumerate(
        [("metro", 2), ("standard", 3), ("metro", 1), ("standard", 2), ("metro", 3), ("standard", 1)]
    ):
        limit = (22000 if zone == "metro" else 16000) * nights
        if i in {3, 5}:
            pair(
                i,
                39750,
                [
                    rec("rate_zone", "Destination band in the travel directory", None),
                    rec("nights", "Number of billed nights on booking", 2),
                    rec("room_subtotal_cents", "Room-only subtotal from folio, USD cents", 37500),
                ],
                "rate_zone",
                {
                    "approve": ("metro", "USD 187.50 per night fits the metro cap."),
                    "reject": ("standard", "USD 187.50 per night exceeds the standard cap."),
                    "review": (None, "The applicable zone cap is unknown."),
                },
                {
                    "approve": ["rate_zone", "nights", "room_subtotal_cents"],
                    "reject": ["rate_zone", "nights", "room_subtotal_cents"],
                    "review": ["rate_zone"],
                },
            )
            continue
        pair(
            i,
            limit + 3500,
            [
                rec("rate_zone", "Destination band in the travel directory", zone),
                rec("nights", "Number of billed nights on booking", nights),
                rec("room_subtotal_cents", "Room-only subtotal from folio, USD cents", None),
            ],
            "room_subtotal_cents",
            {
                "approve": (
                    limit - (i % 2) * 1750,
                    "The documented room subtotal is inside the zone allowance.",
                ),
                "reject": (
                    limit + [1, 599, 1700][i % 3],
                    "The room subtotal exceeds nights multiplied by the zone cap.",
                ),
                "review": (
                    None,
                    "There is no reliable room-only subtotal; the card total includes other charges.",
                ),
            },
            {
                "approve": ["rate_zone", "nights", "room_subtotal_cents"],
                "reject": ["rate_zone", "nights", "room_subtotal_cents"],
                "review": ["room_subtotal_cents"],
            },
        )

    family(
        "purchase_preapproval",
        "test",
        "equipment",
        "Equipment approvals. A purchase above USD 150.00 requires a department manager's recorded authorization "
        "strictly before the card charge, compared in UTC. Reject an authorization recorded at or after the charge. "
        "If the approval time cannot be verified, route to review. Approve amounts up to USD 150 without preapproval. "
        "The employee's request timestamp is never an authorization.",
    )
    for i in range(6):
        date = dt.datetime(2025, 4 + i, 13, 16, 30, tzinfo=dt.UTC)

        def fmt(stamp):
            return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")

        pair(
            i,
            [15001, 21900, 18999, 15100, 26900, 16500][i],
            [
                rec(
                    "manager_approved_at", "Timestamp on department manager's authorization (UTC)", None
                ),
                rec(
                    "requested_at",
                    "Employee request created in the portal (UTC)",
                    fmt(date - dt.timedelta(days=2)),
                ),
            ],
            "manager_approved_at",
            {
                "approve": (
                    fmt(date - dt.timedelta(minutes=1 + i)),
                    "The manager authorized this above-threshold charge beforehand.",
                ),
                "reject": (
                    fmt(date + dt.timedelta(minutes=[0, 1, 7][i % 3])),
                    "The manager authorization was at or after the charge.",
                ),
                "review": (
                    None,
                    "The charge exceeds the exemption and the manager's authorization time is unavailable.",
                ),
            },
            ["EXP", "manager_approved_at"],
            stamp=fmt(date),
        )

    family(
        "prior_reimbursement",
        "test",
        "ground_transport",
        "Duplicate reimbursements. Check the payment reconciliation row for the receipt reference. If the recorded "
        "reference matches the current scanned receipt and a previous payment is settled, reject the duplicate. "
        "If the matching previous payment is fully reversed or voided, approve. A pending or unreconciled matching "
        "payment goes to review. A settled payment for a different receipt does not block approval. If the prior "
        "receipt reference is missing, route to review.",
    )
    for i in range(6):
        ref = f"R-{480 + i * 37}"
        if i in {0, 2, 5}:
            pair(
                i,
                [4820, 7900, 2700, 8900, 5600, 6200][i],
                [
                    rec("receipt_reference", "Identifier transcribed by document intake", ref),
                    rec("prior_receipt_reference", "Receipt identifier on reconciliation row", None),
                    rec(
                        "prior_payment_status",
                        "Current bank reconciliation status for that row",
                        "settled",
                    ),
                ],
                "prior_receipt_reference",
                {
                    "approve": (ref + "-B", "The settled payment concerns a different receipt."),
                    "reject": (ref, "This receipt has already been paid."),
                    "review": (
                        None,
                        "The prior reference is unavailable, so the match cannot be resolved.",
                    ),
                },
                {
                    "approve": ["receipt_reference", "prior_receipt_reference", "prior_payment_status"],
                    "reject": ["receipt_reference", "prior_receipt_reference", "prior_payment_status"],
                    "review": ["prior_receipt_reference"],
                },
            )
            continue
        pair(
            i,
            [4820, 7900, 2700, 8900, 5600, 6200][i],
            [
                rec("receipt_reference", "Identifier transcribed by document intake", ref),
                rec("prior_receipt_reference", "Receipt identifier on reconciliation row", ref),
                rec("prior_payment_status", "Current bank reconciliation status for that row", None),
            ],
            "prior_payment_status",
            {
                "approve": (
                    ["fully reversed", "voided"][i % 2],
                    "The earlier payment for this receipt no longer disburses funds.",
                ),
                "reject": ("settled", "The bank shows a settled payment for this same receipt."),
                "review": (
                    ["pending", "unreconciled"][i % 2],
                    "The matching bank payment has not reached a conclusive state.",
                ),
            },
            ["receipt_reference", "prior_receipt_reference", "prior_payment_status"],
        )

    family(
        "foreign_meal_rate",
        "test",
        "meals",
        "Foreign travel meal. The cap is USD 60.00 inclusive. Multiply the original expense amount by the corporate "
        "purchase-date rate quoted as USD per unit of local currency. Round the result to the nearest USD cent, "
        "half cents up. Approve at or below the cap; reject above it. If that date's rate is unavailable, route to "
        "review. The month-end reference rate is for accounting only and cannot be substituted.",
    )
    for i, (amount, currency, ok, high) in enumerate(
        [
            (5000, "EUR", "1.2000", "1.2010"),
            (7500, "CAD", "0.8000", "0.8100"),
            (5000, "EUR", "1.1990", "1.2100"),
            (7500, "CAD", "0.7990", "0.8010"),
            (4000, "GBP", "1.5000", "1.5100"),
            (4000, "GBP", "1.4900", "1.5010"),
        ]
    ):
        pair(
            i,
            amount,
            [
                rec("purchase_date_fx", "Corporate rate on charge date, USD per local unit", None),
                rec(
                    "month_end_fx",
                    "Accounting rate at month end, USD per local unit",
                    high if i % 2 else ok,
                ),
            ],
            "purchase_date_fx",
            {
                "approve": (ok, "At the required rate, the rounded USD amount is within USD 60."),
                "reject": (high, "At the required rate, the rounded USD amount exceeds USD 60."),
                "review": (None, "The mandatory purchase-date rate is unavailable."),
            },
            {
                "approve": ["EXP", "purchase_date_fx"],
                "reject": ["EXP", "purchase_date_fx"],
                "review": ["purchase_date_fx"],
            },
            currency=currency,
        )

    family(
        "equipment_allowance_revision",
        "test",
        "equipment",
        "Equipment allowance. Add the current USD charge to equipment reimbursements already paid in its calendar "
        "half-year. For charges before 2025-07-01 the inclusive cap was USD 300.00; for charges on or after "
        "2025-07-01 it is USD 400.00. Approve if the total fits, reject an overage, and route unavailable "
        "half-year payment history to review. An annual total or an estimate from the employee cannot replace that history.",
    )
    for i, (month, day, amount) in enumerate(
        [(6, 30, 12000), (7, 1, 17000), (8, 11, 5000), (5, 22, 15500), (7, 18, 9900), (4, 6, 24000)]
    ):
        remaining = (30000 if month < 7 else 40000) - amount
        pair(
            i,
            amount,
            [
                rec(
                    "half_year_paid_cents",
                    "Paid equipment reimbursements in charge's calendar half-year, USD cents",
                    None,
                ),
                rec(
                    "annual_paid_cents",
                    "Employee annual screen, including other reimbursements, USD cents",
                    60000 + i * 5000,
                ),
            ],
            "half_year_paid_cents",
            {
                "approve": (
                    remaining - i % 2,
                    "The current and prior payments fit the effective allowance.",
                ),
                "reject": (
                    remaining + [1, 150, 500][i % 3],
                    "The combined amount exceeds the allowance in force on the charge date.",
                ),
                "review": (
                    None,
                    "The half-year history is unavailable; the annual screen cannot replace it.",
                ),
            },
            {
                "approve": ["EXP", "half_year_paid_cents"],
                "reject": ["EXP", "half_year_paid_cents"],
                "review": ["half_year_paid_cents"],
            },
            stamp=f"2025-{month:02d}-{day:02d}T12:05:00Z",
        )

    family(
        "airfare_cabin_exception",
        "test",
        "air_travel",
        "Airfare. Standard economy is allowed. Premium economy is reimbursable only if the longest single scheduled "
        "airborne segment lasts at least 360 minutes; connection and total itinerary times do not qualify. "
        "Reject a confirmed premium economy itinerary below that segment threshold. Route missing segment evidence "
        "for a premium fare to review. Reject business or first class irrespective of duration.",
    )
    for i in range(6):
        if i in {0, 2, 5}:
            pair(
                i,
                [48000, 65900, 71500, 36745, 49500, 63400][i],
                [
                    rec("cabin", "Fare cabin on ticket", None),
                    rec(
                        "longest_segment_minutes",
                        "Longest single flight in the operator's timetable, minutes",
                        None,
                    ),
                    rec(
                        "itinerary_minutes",
                        "Total itinerary including connections, minutes",
                        490 + i * 15,
                    ),
                ],
                "cabin",
                {
                    "approve": (
                        "standard economy",
                        "Standard economy is eligible regardless of segment length.",
                    ),
                    "reject": (
                        ["business", "first class"][i % 2],
                        "The booked cabin is prohibited regardless of duration.",
                    ),
                    "review": ("premium economy", "The premium cabin needs verified segment evidence."),
                },
                {
                    "approve": ["cabin"],
                    "reject": ["cabin"],
                    "review": ["cabin", "longest_segment_minutes"],
                },
            )
            continue
        pair(
            i,
            [48000, 65900, 71500, 36745, 49500, 63400][i],
            [
                rec("cabin", "Fare cabin on ticket", "premium economy"),
                rec(
                    "longest_segment_minutes",
                    "Longest single flight in the operator's timetable, minutes",
                    None,
                ),
                rec("itinerary_minutes", "Total itinerary including connections, minutes", 490 + i * 15),
            ],
            "longest_segment_minutes",
            {
                "approve": (
                    [360, 391, 405][i % 3],
                    "At least one segment meets the premium economy threshold.",
                ),
                "reject": ([359, 341, 330][i % 3], "No single segment reaches 360 minutes."),
                "review": (
                    None,
                    "The longest segment is unknown despite an available total itinerary time.",
                ),
            },
            ["cabin", "longest_segment_minutes"],
        )

    for fname, rows, key in [
        ("policies", policies, "id"),
        ("cases", cases, "case_id"),
        ("observations", observations, "case_id"),
        ("gold", gold, "case_id"),
    ]:
        (out / (fname + ".jsonl")).write_text(
            "".join(
                json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
                for r in sorted(rows, key=lambda r: r[key])
            ),
            encoding="utf-8",
            newline="\n",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination for the four frozen JSONL files.",
    )
    write_fixtures(parser.parse_args().output_dir)
