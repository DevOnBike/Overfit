"""Finds plans that stopped between gates.

WHY THIS EXISTS. On 2026-08-08 Task 3 of the metric-source seam plan was implemented, committed and put
into the product without a verifier or a reviewer ever running. Not because anyone decided to skip them —
the gate was deferred while the working tree was moving, and then simply not resumed. It surfaced hours
later, by accident, while answering an unrelated question.

The plan's own `STATUS:` line is the only tracker and it is maintained by hand, so nothing notices when a
task advances and then stops. This does.

    python Scripts/plan_gate_check.py
"""
import pathlib
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Ordered. Reaching one without the previous is not the concern; STOPPING at one is.
STAGES = ["ANALYSIS_READY", "APPROVED", "IMPLEMENTED", "VERIFIED", "REVIEWED", "PR_READY", "MERGED"]

# A stage that is dangerous to rest at, and what is missing when you do.
STALLS = {
    "IMPLEMENTED": ("VERIFIED", "code exists and nothing has judged whether its tests prove anything"),
    "VERIFIED": ("REVIEWED", "verified but never reviewed against the repository's own rules"),
    "REVIEWED": ("PR_READY", "reviewed but never taken through a PR gate"),
}

problems = 0

# A plan is a file whose name says so. docs/specs also holds a README, which has no status and
# should not be reported as a stalled plan — the first run of this script did exactly that.
for plan in sorted((ROOT / "docs" / "specs").glob("*-plan.md")):
    status = next((l for l in plan.read_text(encoding="utf-8", errors="replace").splitlines()
                   if l.startswith("STATUS:")), None)

    if status is None:
        print(f"[no status] {plan.name}")
        problems += 1

        continue

    reached = [s for s in STAGES if re.search(rf"\b{s}\b", status)]
    committed = "COMMITTED" in status.upper()

    if not reached:
        print(f"[unreadable] {plan.name}: {status[:110]}")
        problems += 1

        continue

    furthest = max(reached, key=STAGES.index)
    missing = STALLS.get(furthest)

    # UNGATED-and-committed is the shape that actually bit; call it out loudest.
    if committed and "VERIFIED" not in reached:
        print(f"[COMMITTED WITHOUT A GATE] {plan.name}")
        print(f"    reached {', '.join(reached)} — code is in the product and nothing verified it")
        problems += 1
    elif missing and not committed:
        print(f"[stalled at {furthest}] {plan.name}")
        print(f"    next is {missing[0]}: {missing[1]}")
        problems += 1

print(f"\n{problems} plan(s) need attention" if problems else "\nno plan is stalled between gates")
sys.exit(1 if problems else 0)
