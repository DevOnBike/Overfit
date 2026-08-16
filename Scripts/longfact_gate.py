"""Runs the [LongFact] suite as a gate — all of it before a release, the touched areas before a merge.

WHY THIS EXISTS
    Counted from the runner's own results 2026-08-07: 1715 `[Fact]` against **256 `[LongFact]`**, 195 of
    them under `LanguageModels` — the model loaders and the runtime, which is the highest-value end-to-end
    surface in the repository. Nothing ran them on any schedule, and until 2026-08-06 there was no way to
    run them at all except by editing the attribute in source. A test that never runs is worse than no
    test: it looks like coverage.

    (The first version of this note said 358, from counting occurrences of the text `[LongFact` — which
    also counts every mention in a comment or doc block, 103 of them. 256 skipped plus 5 deliberate
    `[Fact(Skip=...)]` is exactly the 261 the runner reports. Count what executes.)

WHAT IT REFUSES TO DO
    It will not start while this box is an instrument. A 24-hour anomaly-guard measurement or a
    BenchmarkDotNet run owns the machine, and 358 model loads inside somebody's sampling window produces two
    wrong answers rather than one result. `MeasurementExclusion` enforces the benchmark half with a
    machine-wide mutex and exits 2; this script checks the guard half itself, because a marker file is not
    something the mutex knows about.

WHAT IT RECORDS
    The elapsed time, to `Tests/bin/longfact-timings.log`. "Run this before every release" means something
    very different at eight minutes than at six hours, and nobody had the number.

    Lab-dependent tests are reported SEPARATELY when the local Kubernetes lab is not up. They skip rather
    than fail — a machine without a cluster must not go red — but a skip folded into a total is a decision
    nobody made, so they are named and the gate exits 4 until somebody accepts the gap with
    `--allow-no-lab`. Exit codes: 0 complete, 1 something failed, 3 refused to start, 4 green but the lab
    part did not run.

USAGE
    python Scripts/longfact_gate.py                 # RELEASE gate — everything
    python Scripts/longfact_gate.py --changed       # PR gate — only areas the diff touches
    python Scripts/longfact_gate.py --area LanguageModels Anomalies
    python Scripts/longfact_gate.py --allow-no-lab  # accept that the lab tests did not run
"""
import argparse
import datetime
import pathlib
import re
import subprocess
import sys
import time

# Sibling script, always next to this one. It owns the classification, so it also owns the answer to
# "which attributes are LongFacts" — duplicating that list here is how the two drift apart.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import longfact_split  # noqa: E402  (path must be set first)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Repo root from this file's own location, NOT a hardcoded path: a developer's absolute
# path in a public repository is a leak, and it also means the script only runs on one box.
ROOT = pathlib.Path(__file__).resolve().parents[1]
TESTS = ROOT / "Tests"
NAMESPACE = "DevOnBike.Overfit.Tests"
TIMINGS = ROOT / "Tests" / "bin" / "longfact-timings.log"

# Source area -> test area. Only entries that differ from the identity mapping need to be here; anything
# else falls back to the same name, which is how the test tree is laid out (domain first, purpose second).
EXPLICIT = {
    "Intrinsics": "Core",
    "Tensors": "Core",
    "Ops": "Core",
    "Autograd": "DeepLearning",
    "Onnx": "Integrations",
    "Trees": "Trees",
    "Statistics": "Statistics",
}


def refuse_if_the_box_is_an_instrument():
    """A measurement in progress outranks a test run. Both would be wrong, and only one is recoverable."""
    marker = TESTS / "bin" / "fp-run-clean-start.txt"

    if not marker.exists():
        return

    stamp = marker.read_text(encoding="utf-8").split()[-1]
    started = datetime.datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=datetime.timezone.utc)
    age = (datetime.datetime.now(datetime.timezone.utc) - started).total_seconds() / 3600.0

    if age < 24:
        ends = started + datetime.timedelta(hours=24)
        print(f"REFUSING: an anomaly-guard measurement started {started:%Y-%m-%d %H:%M}Z and runs until "
              f"{ends:%Y-%m-%d %H:%M}Z ({24 - age:.1f} h left).")
        print("          358 model loads would land inside its samples. Wait, or delete the marker if the "
              "run is genuinely over.")
        sys.exit(3)


def changed_areas():
    """Test areas touched by the diff against the upstream branch."""
    r = subprocess.run(["git", "diff", "--name-only", "origin/HEAD...HEAD"],
                       cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")

    files = r.stdout.split() if r.returncode == 0 else []

    if not files:
        r = subprocess.run(["git", "diff", "--name-only", "HEAD"],
                           cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        files = r.stdout.split()

    areas = set()

    for f in files:
        parts = pathlib.PurePosixPath(f).parts

        if len(parts) < 3:
            continue

        if parts[0] == "Tests":
            areas.add(parts[1])
            continue

        if parts[0] == "Sources":
            area = parts[2] if len(parts) > 3 else parts[1]
            areas.add(EXPLICIT.get(area, area))

    # Only areas that exist as a test directory; anything else would filter to nothing and read as a pass.
    return sorted(a for a in areas if (TESTS / a).is_dir())


# An attribute followed by a method signature, NOT a bare text match. `[LongFact` also appears in
# comments and doc blocks — 103 times on 2026-08-07 — and counting those inflated the scope figure this
# script prints from 256 to 359. A gate that misreports its own scope is a gate nobody can size.
#
# It comes from longfact_split rather than being written again here. The copy that used to live at this
# line had drifted: it knew nothing of `ModelFact`/`FixtureFact`/`LabFact` and could not cross a nested
# bracket, so it counted 229 where the classifier counted 256 — two numbers for one question.
LONGFACT_METHOD = longfact_split.longfact_method_pattern()


def count_longfacts(areas):
    total = 0

    for p in TESTS.rglob("*.cs"):
        if any(x in p.parts for x in ("bin", "obj")):
            continue

        rel = p.relative_to(TESTS).parts

        if areas and (not rel or rel[0] not in areas):
            continue

        total += len(LONGFACT_METHOD.findall(p.read_text(encoding="utf-8", errors="replace")))

    return total


# Must match LabFact.Marker in Tests/LabFact.cs. The string lives in one place on each side and the
# check below fails loudly if they drift, because a marker nobody matches degrades this report to
# silence — which is the exact failure it exists to prevent.
LAB_MARKER = "LAB NOT AVAILABLE"


def verify_marker_still_matches():
    """Fails the run if `LabFact.Marker` and `LAB_MARKER` have drifted apart.

    A marker that matches nothing does not produce an error, it produces SILENCE — the lab report simply
    comes back empty and the gate reads as fully covered. That is the one failure mode this whole feature
    exists to prevent, so it is checked rather than assumed.
    """
    source = ROOT / "Tests" / "LabFact.cs"

    if not source.exists():
        print(f"REFUSING: {source.relative_to(ROOT)} is missing — lab skips could not be reported.")
        sys.exit(3)

    declared = re.search(r'Marker\s*=\s*"([^"]+)"', source.read_text(encoding="utf-8"))

    if not declared:
        print("REFUSING: LabFact.Marker not found in Tests/LabFact.cs.")
        sys.exit(3)

    if declared.group(1) != LAB_MARKER:
        print(f"REFUSING: LabFact.Marker is {declared.group(1)!r} but this script greps for "
              f"{LAB_MARKER!r}. A mismatch reports zero lab skips instead of erroring, so it is fatal "
              f"here. Fix both to the same string.")
        sys.exit(3)


def classify(trx_text):
    """Splits non-passing results into REAL FAILURES and LAB SKIPS.

    Two things are being separated that the previous version merged. `outcome != "Passed"` was treated
    as a failure, so every deliberate skip would have been printed under "FAILING TESTS" — harmless
    while nothing skipped on purpose, and actively misleading now that six tests do. And a lab skip is
    not an ordinary skip: a missing model fixture is a fact about somebody's disk, while a missing lab
    means a slice of the release gate did not run and somebody has to decide whether that is acceptable.
    """
    import xml.etree.ElementTree as ET

    failures, lab = [], []

    try:
        root = ET.fromstring(trx_text)
    except ET.ParseError:
        # Better a degraded report than none: fall back to the old scrape, but say so.
        print("         !! TRX did not parse — falling back to text scraping")

        return sorted(set(re.findall(r'testName="([^"]+)"[^>]*outcome="Failed"', trx_text))), []

    for element in root.iter():
        if not element.tag.endswith("UnitTestResult"):
            continue

        outcome = element.get("outcome", "")
        name = element.get("testName", "?")

        if outcome == "Passed":
            continue

        # The skip reason travels in ErrorInfo/Message for a NotExecuted result, same as an assertion
        # message does for a failure.
        message = " ".join((m.text or "") for m in element.iter() if m.tag.endswith("Message"))

        if outcome == "NotExecuted":
            if LAB_MARKER in message:
                lab.append(name)

            continue

        # Carry the MESSAGE, not just the name. A name says which test broke; only the message says
        # whether it is a missing fixture, an unreachable lab, a real assertion or a timeout — and those
        # four have four different owners. Added 2026-08-15 after a 41-area campaign reported 19 failures
        # across 5 areas and identified the cause of none of them, because every per-area TRX had been
        # overwritten by the next area before anybody read it.
        failures.append((name, " ".join(message.split())))

    # De-duplicate on the name while keeping the first message seen for it.
    seen, unique = set(), []

    for name, message in sorted(failures):
        if name in seen:
            continue

        seen.add(name)
        unique.append((name, message))

    return unique, sorted(set(lab))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--changed", action="store_true", help="PR gate: only areas the diff touches")
    ap.add_argument("--area", nargs="*", default=None, help="explicit test areas")
    ap.add_argument("--allow-no-lab", action="store_true",
                    help="accept a run where the lab-dependent tests did not run (see exit code 4)")
    args = ap.parse_args()

    refuse_if_the_box_is_an_instrument()
    verify_marker_still_matches()

    areas = args.area if args.area else (changed_areas() if args.changed else [])
    mode = "PR_GATE" if (args.changed or args.area) else "RELEASE_GATE"

    print(f"MODE   {mode}")
    print(f"AREAS  {', '.join(areas) if areas else 'all'}")
    print(f"TESTS  ~{count_longfacts(areas)} [LongFact] in scope")

    if (args.changed or args.area) and not areas:
        print("\nNo test area matches the diff. That is not a pass — say so, and check the mapping in this "
              "script before concluding the change is untested.")
        sys.exit(0)

    # ONE TRX PER AREA. A single shared name is overwritten by the next area, so on a multi-area campaign
    # every failure message is destroyed by the run that follows it — which is exactly what happened on
    # 2026-08-15, leaving 19 failures with names and no causes. The file is the only place the assertion
    # text survives, so it must not be a shared name.
    trx_stem = "-".join(a.replace(".", "_") for a in areas) if areas else "all"
    trx = ROOT / "Tests" / "bin" / f"longfact-{trx_stem[:80]}.trx"
    cmd = ["dotnet", "test", "./Tests/Tests.csproj", "-c", "Release",
           "--logger", f"trx;LogFileName={trx}"]

    if areas:
        # One filter, OR-ed. `~` is "contains" in VSTest's expression language.
        cmd += ["--filter", "|".join(f"FullyQualifiedName~{NAMESPACE}.{a}." for a in areas)]

    print(f"\n$ OVERFIT_RUN_LONG=1 {' '.join(cmd)}\n", flush=True)

    started = time.time()
    # DOTNET_CLI_UI_LANGUAGE is not cosmetic here. Measured 2026-08-07 on this box: the SDK is
    # Polish-localised, so a passing run prints `Powodzenie! — niepowodzenie: 0, powodzenie: 8` and a
    # failing one prints `Niepowodzenie`, neither of which the English patterns below match. The effect
    # is not a missing pretty summary — it is that the FAILING TEST NAMES come back empty from a run
    # that failed, which is the exact way this repository has already lost a real failure twice. Pin the
    # language for machine reading, and take the counts from the TRX the runner writes rather than from
    # console text in any language.
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env={**__import__("os").environ, "OVERFIT_RUN_LONG": "1",
                                              "DOTNET_CLI_UI_LANGUAGE": "en"})
    elapsed = time.time() - started

    out = r.stdout + "\n" + r.stderr

    # Exit 2 is MeasurementExclusion refusing because a benchmark holds the machine mutex — a different
    # thing from a test failure, and reporting it as one would send somebody hunting a bug that is not there.
    if r.returncode == 2 and "measurement" in out.lower():
        print("REFUSED by MeasurementExclusion: a benchmark holds the machine mutex. Not a test failure.")
        sys.exit(3)

    # The TRX is the authority: it is written by the runner as structured data, so it does not move when
    # the console language does, and it distinguishes "0 failed" from "nothing ran" — which an exit code
    # of 0 does not. A filter that matches no test also exits 0.
    counters, failed, lab_skips = {}, [], []

    if trx.exists():
        text = trx.read_text(encoding="utf-8", errors="replace")
        found = re.search(r"<Counters\b([^>]*)/>", text)

        if found:
            counters = dict(re.findall(r'(\w+)="(\d+)"', found.group(1)))

        failed, lab_skips = classify(text)

    if not failed:
        failed = sorted(set(re.findall(r"(?:Failed|Nie powiodło się)\s+([\w\.]+\.\w+)", out)))

    print(f"\nELAPSED  {elapsed/60:.1f} min   exit {r.returncode}")

    if counters:
        # Skips are total-minus-executed, NOT the `notExecuted` attribute. Measured on this runner
        # 2026-08-07: a run whose single result carries outcome="NotExecuted" still writes
        # notExecuted="0" — only `total` and `executed` are filled in. Printing the attribute reported
        # "not executed 0" on a run where everything skipped, which is precisely the reassuring-and-false
        # summary this gate exists to avoid.
        total = int(counters.get("total", 0))
        executed = int(counters.get("executed", 0))
        print(f"         TRX: total {total}   passed {counters.get('passed', '?')}"
              f"   failed {counters.get('failed', '?')}   skipped {total - executed}"
              f"   (of which lab: {len(lab_skips)})")

        if total == 0:
            print("         !! NOTHING MATCHED. Exit 0 on an empty run is not a pass — check the filter.")

        if total > 0 and executed == 0:
            print("         !! EVERY test in scope skipped. Nothing was verified; see the reasons below.")
    else:
        print("         !! no TRX produced — counts below are scraped from console text and may be "
              "language-dependent")

    if failed:
        print(f"\nFAILING TESTS ({len(failed)}) — name AND message, because a name is not a diagnosis:")

        for entry in failed[:40]:
            name, message = entry if isinstance(entry, tuple) else (entry, "")
            print("   ", name)

            if message:
                print(f"        {message[:400]}")

    # THE LAB IS REPORTED, NOT COUNTED. These tests skipped rather than failed, which is correct on a
    # machine with no cluster — but a skip that disappears into a total is a decision nobody made, and the
    # things they cover (the anomaly guard end to end, the calibration floors, the shadow run) are exactly
    # the ones whose absence is invisible from the code. So they are named, and the run does not claim to
    # be a clean pass until somebody says the omission is acceptable for this release.
    if lab_skips:
        print(f"\nLAB-DEPENDENT TESTS DID NOT RUN ({len(lab_skips)}) — the local Kubernetes lab was not "
              f"reachable:")

        for t in lab_skips:
            print("   ", t)

        print("\n   These checked NOTHING. To run them, bring the lab up and re-run this gate:")
        print(r"       k8s\monitoring\install.cmd  /  k8s\overfit\deploy.cmd     (once)")
        print(r"       k8s\monitoring\forward.cmd        Prometheus on 9090, 9098, 9099")
        print(r"       k8s\overfit\forward-replicas.cmd  one local port per replica")
        print("   To accept the gap for this release instead, re-run with --allow-no-lab, which records "
              "the decision in the timing log.")

    TIMINGS.parent.mkdir(parents=True, exist_ok=True)

    lab_note = ("no-lab-accepted" if lab_skips and args.allow_no_lab
                else f"no-lab({len(lab_skips)})" if lab_skips else "lab-covered")

    with TIMINGS.open("a", encoding="utf-8") as fh:
        fh.write(f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ}\t{mode}\t"
                 f"{','.join(areas) if areas else 'all'}\t{elapsed/60:.1f} min\texit {r.returncode}\t"
                 f"{lab_note}\n")

    print(f"\ntiming appended to {TIMINGS.relative_to(ROOT).as_posix()}")

    if r.returncode != 0:
        sys.exit(1)

    # Exit 4 is deliberately NOT 0 and NOT 1: nothing failed, but the gate is incomplete, and the two
    # cases need different reactions from whoever reads the exit code.
    if lab_skips and not args.allow_no_lab:
        print(f"\nEXIT 4 — everything that ran passed, but {len(lab_skips)} lab test(s) did not run. "
              "This is a decision, not a failure.")
        sys.exit(4)

    sys.exit(0)


# Guarded so the classification above can be imported and tested against a recorded TRX without
# launching a multi-hour run — the report is the part most likely to be wrong, and it was.
if __name__ == "__main__":
    main()
