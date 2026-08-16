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
STAGES = ["ANALYSIS_READY", "APPROVED", "IMPLEMENTED", "VERIFIED", "REVIEWED", "GATES_PASSED",
          "PR_READY", "MERGED", "OUTCOME_MEASURED"]

# The gate manifest, added 2026-08-13. The STATUS line is a scalar: it says how far the work got and
# cannot say WHICH check ran. Before the manifest, "the CISO gate was not required" and "the CISO gate
# was never dispatched" were the same absence in the plan file.
GATES = ["verifier", "reviewer", "mutation-proof", "performance", "security", "leak-scan", "AOT",
         "API-compatibility", "release-readiness"]

VERDICTS = ("PASS", "FAIL", "INCONCLUSIVE", "NOT_REQUIRED")

# `overfit-architect` closes with SIGNED, which is the same state the pipeline calls APPROVED. Accepted
# rather than "corrected" in seven plan files: a plan is not broken because two documents chose different
# words for one state, and a checker that reports a synonym as unreadable teaches people to ignore it.
ALIASES = {"SIGNED": "APPROVED"}


def check_manifest(text):
    """A missing line is NOT `NOT_REQUIRED` — it means nobody asked.

    Returns a list of messages rather than printing, so the caller can print the plan's name once and
    only when there is something to say.
    """
    found = []
    lines = {}

    for line in text.splitlines():
        m = re.match(r"\s{2,}([A-Za-z-]+):\s*(\S+)(.*)$", line)
        if m and m.group(1) in GATES:
            lines[m.group(1)] = (m.group(2).rstrip(","), m.group(3).strip())

    if not lines:
        return ["no GATES: block — nothing records what was checked"]

    for gate in GATES:
        if gate not in lines:
            found.append(f"gate NOT ASKED: {gate} — a missing line is not NOT_REQUIRED")

            continue

        verdict, tail = lines[gate]

        if verdict not in VERDICTS:
            found.append(f"gate {gate}: unreadable verdict {verdict!r} — expected {', '.join(VERDICTS)}")
        elif verdict == "NOT_REQUIRED" and not tail:
            # The reason is the whole point: it is what separates a gate nobody needed from one
            # nobody ran. Without it the manifest re-creates the ambiguity it exists to remove.
            found.append(f"gate {gate}: NOT_REQUIRED with no reason on the line")
        elif verdict == "FAIL":
            found.append(f"gate {gate}: FAIL — the plan must not advance past it")

    return found

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
    text = plan.read_text(encoding="utf-8", errors="replace")
    status = next((l for l in text.splitlines() if l.startswith("STATUS:")), None)

    if status is None:
        print(f"[no status] {plan.name}")
        problems += 1

        continue

    # QUARANTINED is terminal and deliberate: the plan is not in the pipeline and is not going to be.
    # Recognising it is the difference between "somebody forgot this" and "somebody decided this".
    if "QUARANTINED" in status.upper():
        print(f"[quarantined, not in the pipeline] {plan.name}")

        continue

    reached = [s for s in STAGES if re.search(rf"\b{s}\b", status)]
    reached += [canonical for word, canonical in ALIASES.items()
                if re.search(rf"\b{word}\b", status, re.I) and canonical not in reached]
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

    # The manifest is only demanded once there is work to have gated. Asking a plan that is still
    # ANALYSIS_READY for a verifier verdict would train people to write NOT_REQUIRED everywhere,
    # which is how a gate list becomes wallpaper.
    if STAGES.index(furthest) >= STAGES.index("IMPLEMENTED"):
        found = check_manifest(text)

        if found:
            print(f"[gate manifest] {plan.name}")

            for message in found:
                print(f"    {message}")

            problems += len(found)

print(f"\n{problems} plan(s) need attention" if problems else "\nno plan is stalled between gates")
sys.exit(1 if problems else 0)
