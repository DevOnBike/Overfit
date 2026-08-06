"""Runs the [LongFact] suite as a gate — all of it before a release, the touched areas before a merge.

WHY THIS EXISTS
    Measured 2026-08-06: 1733 `[Fact]` against **358 `[LongFact]`**, 277 of them under `LanguageModels` —
    the model loaders and the runtime, which is the highest-value end-to-end surface in the repository.
    Nothing ran them on any schedule, and until the same day there was no way to run them at all except by
    editing the attribute in source. A test that never runs is worse than no test: it looks like coverage.

WHAT IT REFUSES TO DO
    It will not start while this box is an instrument. A 24-hour anomaly-guard measurement or a
    BenchmarkDotNet run owns the machine, and 358 model loads inside somebody's sampling window produces two
    wrong answers rather than one result. `MeasurementExclusion` enforces the benchmark half with a
    machine-wide mutex and exits 2; this script checks the guard half itself, because a marker file is not
    something the mutex knows about.

WHAT IT RECORDS
    The elapsed time, to `Tests/bin/longfact-timings.log`. "Run this before every release" means something
    very different at eight minutes than at six hours, and nobody had the number.

USAGE
    python .claude/longfact.py                 # RELEASE gate — everything
    python .claude/longfact.py --changed       # PR gate — only areas the diff touches
    python .claude/longfact.py --area LanguageModels Anomalies
"""
import argparse
import datetime
import pathlib
import re
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(r"D:\Overfit")
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


def count_longfacts(areas):
    total = 0

    for p in TESTS.rglob("*.cs"):
        if any(x in p.parts for x in ("bin", "obj")):
            continue

        rel = p.relative_to(TESTS).parts

        if areas and (not rel or rel[0] not in areas):
            continue

        total += len(re.findall(r"\[LongFact", p.read_text(encoding="utf-8", errors="replace")))

    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--changed", action="store_true", help="PR gate: only areas the diff touches")
    ap.add_argument("--area", nargs="*", default=None, help="explicit test areas")
    args = ap.parse_args()

    refuse_if_the_box_is_an_instrument()

    areas = args.area if args.area else (changed_areas() if args.changed else [])
    mode = "PR_GATE" if (args.changed or args.area) else "RELEASE_GATE"

    print(f"MODE   {mode}")
    print(f"AREAS  {', '.join(areas) if areas else 'all'}")
    print(f"TESTS  ~{count_longfacts(areas)} [LongFact] in scope")

    if (args.changed or args.area) and not areas:
        print("\nNo test area matches the diff. That is not a pass — say so, and check the mapping in this "
              "script before concluding the change is untested.")
        sys.exit(0)

    cmd = ["dotnet", "test", "./Tests/Tests.csproj", "-c", "Release"]

    if areas:
        # One filter, OR-ed. `~` is "contains" in VSTest's expression language.
        cmd += ["--filter", "|".join(f"FullyQualifiedName~{NAMESPACE}.{a}." for a in areas)]

    print(f"\n$ OVERFIT_RUN_LONG=1 {' '.join(cmd)}\n", flush=True)

    started = time.time()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env={**__import__("os").environ, "OVERFIT_RUN_LONG": "1"})
    elapsed = time.time() - started

    out = r.stdout + "\n" + r.stderr

    # Exit 2 is MeasurementExclusion refusing because a benchmark holds the machine mutex — a different
    # thing from a test failure, and reporting it as one would send somebody hunting a bug that is not there.
    if r.returncode == 2 and "measurement" in out.lower():
        print("REFUSED by MeasurementExclusion: a benchmark holds the machine mutex. Not a test failure.")
        sys.exit(3)

    failed = sorted(set(re.findall(r"(?:Failed|Nie powiodło się)\s+([\w\.]+\.\w+)", out)))
    summary = [l.strip() for l in out.splitlines()
               if re.search(r"(Passed!|Failed!|Przeszły|Nie powiodły)", l)]

    print(f"\nELAPSED  {elapsed/60:.1f} min   exit {r.returncode}")

    for l in summary[:4]:
        print("        ", l[:160])

    if failed:
        print(f"\nFAILING TESTS ({len(failed)}) — names, because a count is not actionable:")
        for f in failed[:40]:
            print("   ", f)

    TIMINGS.parent.mkdir(parents=True, exist_ok=True)

    with TIMINGS.open("a", encoding="utf-8") as fh:
        fh.write(f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ}\t{mode}\t"
                 f"{','.join(areas) if areas else 'all'}\t{elapsed/60:.1f} min\texit {r.returncode}\n")

    print(f"\ntiming appended to {TIMINGS.relative_to(ROOT).as_posix()}")
    sys.exit(0 if r.returncode == 0 else 1)


main()
