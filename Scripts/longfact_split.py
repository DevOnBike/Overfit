"""Splits [LongFact] into a HEAVY group and a LIGHT one, then runs the LIGHT remainder.

WHY THE SPLIT EXISTS — it is a measurement, not a preference
    The undivided run reached 106 of 175 chunks in 54.7 minutes and then spent 25 minutes inside a single
    chunk (`QwenGgufKnowledgeInjectionDemoTests`, a QLoRA fine-tune on a real Qwen-3B) without finishing it.
    The remaining chunks are dominated by the same class of work: QLoRA end-to-end, training RAM and
    step-time diagnostics, speculative-decode benches, sidecar builds. A gate in which one test can take a
    quarter of an hour cannot run on every merge, and pretending otherwise is how a gate gets disabled.

    This is also the answer to T7 in docs/test-gate-backlog.md, arrived at by measuring rather than by
    deciding in advance.

HOW A TEST IS CLASSIFIED — measurement first, names only where there is no measurement
    HEAVY if EITHER:
      (a) it was measured above HEAVY_SECONDS in the 2026-08-07 run, or
      (b) its name says it trains or benchmarks, and it has no measurement yet.
    (a) is evidence. (b) is a guess, and it is marked as one in the output so nobody later reads the list
    as if all of it had been timed. Every (b) entry becomes an (a) entry the first time the heavy group is
    run to completion.

WHAT THIS SCRIPT DOES
    Writes the classification to Scripts/longfact_heavy.txt, then runs every LIGHT test that has no result
    yet — one process per chunk, same as before, appending to the same progress log.
"""
import argparse
import datetime
import os
import pathlib
import re
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Repo root from this file's own location, NOT a hardcoded path: a developer's absolute
# path in a public repository is a leak, and it also means the script only runs on one box.
ROOT = pathlib.Path(__file__).resolve().parents[1]
TESTS = ROOT / "Tests"
BIN = TESTS / "bin"
CLASSIFICATION = ROOT / "Scripts" / "longfact_heavy.txt"
PROGRESS = BIN / "longfact-progress.log"
CHUNK_MAX = 12
HEAVY_SECONDS = 120.0

# Names that say the test trains or benchmarks. Used ONLY where no measurement exists.
HEAVY_TOKENS = ("QLora", "FineTune", "Training", "TrainStep", "TrainingRam", "KnowledgeInjection",
                "Bench", "Sweep", "Speculative", "DataParallelTraining", "Checkpoint")


def measured():
    """full test name -> (seconds, outcome), from every chunk TRX written so far."""
    found = {}

    for trx in sorted(BIN.glob("longfact-chunk-*.trx")):
        text = trx.read_text(encoding="utf-8", errors="replace")

        for m in re.finditer(
                r'<UnitTestResult[^>]*testName="([^"]+)"[^>]*duration="([^"]+)"[^>]*outcome="(\w+)"',
                text):
            parts = m.group(2).split(":")

            try:
                found[m.group(1)] = (
                    int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2]), m.group(3))
            except (ValueError, IndexError):
                continue

    return found


def all_longfacts():
    """Every [LongFact] full name, from the full-suite TRX crossed with the source attributes."""
    text = (BIN / "full-suite.trx").read_text(encoding="utf-8", errors="replace")
    skipped = re.findall(r'<UnitTestResult[^>]*testName="([^"]+)"[^>]*outcome="NotExecuted"', text)

    pattern = re.compile(
        r"\[(?:LongFact|ProductionAnomalyBaseFact)[^\]]*\](?:[ \t]*//[^\n]*)?(?:\s*\[[^\]]*\])*\s*public\s+(?:async\s+)?"
        r"[\w<>\[\]\.,\s]+?\s+(\w+)\s*\(")
    names = set()

    for p in TESTS.rglob("*.cs"):
        if any(x in p.parts for x in ("bin", "obj")):
            continue

        names.update(pattern.findall(p.read_text(encoding="utf-8", errors="replace")))

    return [n for n in skipped if n.rsplit(".", 1)[-1] in names]


def classify(names, timings):
    heavy, light = {}, []

    for n in names:
        seen = timings.get(n)

        if seen and seen[0] >= HEAVY_SECONDS:
            heavy[n] = f"measured {seen[0]:.0f}s"

            continue

        if seen:
            light.append(n)

            continue

        if any(t.lower() in n.lower() for t in HEAVY_TOKENS):
            heavy[n] = "GUESSED from the name — never measured"

            continue

        light.append(n)

    return heavy, light


def run_chunk(label, names, index, total, kind="light"):
    trx = BIN / f"longfact-{kind}-{index:03d}.trx"

    if trx.exists():
        trx.unlink()

    started = time.time()
    r = subprocess.run(
        ["dotnet", "test", "./Tests/Tests.csproj", "-c", "Release", "--no-build",
         "--logger", f"trx;LogFileName={trx}",
         "--filter", "|".join(f"FullyQualifiedName={n}" for n in names)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600,
        env={**os.environ, "OVERFIT_RUN_LONG": "1", "DOTNET_CLI_UI_LANGUAGE": "en"})
    elapsed = time.time() - started

    passed = failed = executed = 0
    bad = []

    if trx.exists():
        text = trx.read_text(encoding="utf-8", errors="replace")
        found = re.search(r"<Counters\b([^>]*)/>", text)

        if found:
            c = dict(re.findall(r'(\w+)="(\d+)"', found.group(1)))
            passed, failed = int(c.get("passed", 0)), int(c.get("failed", 0))
            executed = int(c.get("executed", 0))

        bad = sorted(set(m.group(1) for m in re.finditer(
            r'<UnitTestResult[^>]*testName="([^"]+)"[^>]*outcome="(?!Passed)(\w+)"', text)))

    state = "ok  " if (failed == 0 and executed == len(names)) else "FAIL"
    print(f"[{index:2d}/{total}] {state} {elapsed:6.0f}s  {len(names):2d} tests  pass {passed:2d} "
          f"fail {failed:2d}  {label}", flush=True)

    for b in bad[:8]:
        print(f"          !! {b.replace('DevOnBike.Overfit.Tests.', '')}", flush=True)
        msg = re.search(r'<UnitTestResult[^>]*testName="' + re.escape(b) + r'".*?<Message>(.*?)</Message>',
                        trx.read_text(encoding="utf-8", errors="replace"), re.S)

        if msg:
            print(f"             {msg.group(1).strip().splitlines()[0][:170]}", flush=True)

    with PROGRESS.open("a", encoding="utf-8") as fh:
        fh.write(f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ}\tLIGHT\t{label}\t"
                 f"{len(names)}\t{elapsed:.0f}s\tpass {passed}\tfail {failed}\t"
                 f"{';'.join(bad) if bad else '-'}\n")

    return passed, failed, len(names), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classify-only", action="store_true")
    ap.add_argument("--heavy", action="store_true",
                    help="run the HEAVY group instead of the light remainder, to give it real timings")
    args = ap.parse_args()

    timings = measured()
    names = all_longfacts()
    heavy, light = classify(names, timings)

    done = {n for n in names if n in timings}
    todo = [n for n in light if n not in done]

    guessed = sum(1 for v in heavy.values() if v.startswith("GUESSED"))
    print(f"TOTAL   {len(names)} [LongFact]")
    print(f"HEAVY   {len(heavy)}   ({len(heavy) - guessed} measured >= {HEAVY_SECONDS:.0f}s, "
          f"{guessed} GUESSED from the name and never measured)")
    print(f"LIGHT   {len(light)}   of which {len(done & set(light))} already have a result, "
          f"{len(todo)} still to run")

    CLASSIFICATION.write_text(
        "# [LongFact] HEAVY group — excluded from the per-merge gate, run before a release.\n"
        f"# Written {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ} by "
        "Scripts/longfact_split.py\n"
        f"# HEAVY means: measured at or above {HEAVY_SECONDS:.0f}s, OR its name says it trains or\n"
        "# benchmarks and it has never been measured. The second kind is marked GUESSED and is a\n"
        "# hypothesis, not a measurement — it becomes a measurement the first time the heavy group\n"
        "# runs to completion. Do not quote a GUESSED entry as a timing.\n#\n"
        + "".join(f"{n}\t{why}\n" for n, why in sorted(heavy.items())),
        encoding="utf-8")
    print(f"\nclassification -> {CLASSIFICATION.relative_to(ROOT).as_posix()}")

    if args.classify_only:
        return

    kind = "heavy" if args.heavy else "light"

    # One test per process for the heavy group. These are QLoRA fine-tunes, training-RAM diagnostics and
    # speculative-decode benches; grouping them would stack multi-gigabyte models in one process, which is
    # exactly the failure that forced the chunking in the first place (21.7 GB, 1.0 GB free, the monitoring
    # stack evicted). Their start-up cost is noise next to their runtime, so there is nothing to amortise.
    size = 1 if args.heavy else CHUNK_MAX
    todo = sorted(heavy) if args.heavy else todo

    if not todo:
        return

    groups = {}

    for n in todo:
        groups.setdefault(n.rsplit(".", 1)[0], []).append(n)

    plan = []

    for key in sorted(groups, key=lambda k: (len(groups[k]), k)):
        members = groups[key]

        for i in range(0, len(members), size):
            plan.append((key.replace("DevOnBike.Overfit.Tests.", ""), members[i:i + size]))

    print(f"\n=== running the {kind.upper()} group: {len(plan)} chunks, {len(todo)} tests ===")

    if args.heavy:
        print("    Expect hours. 5 of these were measured above 120 s and one reached 25 minutes without")
        print("    finishing; the other 29 are guesses from the name and get their first real number here.\n",
              flush=True)

    totals = [0, 0, 0]
    all_bad = []
    started = time.time()

    for i, (label, members) in enumerate(plan, 1):
        p, f, n, bad = run_chunk(label, members, i, len(plan), kind)
        totals = [totals[0] + p, totals[1] + f, totals[2] + n]
        all_bad += bad

    print(f"\n{kind.upper()} GROUP  {totals[2]} tests   passed {totals[0]}   failed {totals[1]}   "
          f"{(time.time() - started) / 60:.1f} min")

    if all_bad:
        print(f"\nFAILING ({len(all_bad)}):")

        for b in all_bad:
            print("   ", b.replace("DevOnBike.Overfit.Tests.", ""))


main()
