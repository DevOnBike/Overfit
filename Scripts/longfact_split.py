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


def longfact_attributes():
    """Every attribute that IS a LongFact, resolved from the source rather than listed by hand.

    Hardcoding the list was fine when there were two. Four more arrived on 2026-08-07 — `ModelFact`,
    `FixtureFact`, `LabFact`, plus subclasses of those — and a hardcoded list does not fail when it goes
    stale, it silently drops those tests out of the heavy group. So walk the inheritance to a fixpoint:
    anything deriving from a known LongFact is one.
    """
    declarations = []

    for p in TESTS.rglob("*.cs"):
        if any(x in p.parts for x in ("bin", "obj")):
            continue

        declarations += re.findall(r"class\s+(\w+)\s*:\s*(\w+)",
                                   p.read_text(encoding="utf-8", errors="replace"))

    known = {"LongFact"}

    for _ in range(8):  # BOUND: inheritance depth here is 2; 8 is slack, not a real limit.
        grown = {d for d, base in declarations if base in known}

        if grown <= known:
            break

        known |= grown

    return sorted(known)


def longfact_method_pattern():
    """One pattern, used by this script AND by longfact_gate — group 1 is the method name.

    Two copies of it gave two different answers to "how many [LongFact] are there", 256 against 229, which
    is precisely the mis-sized gate the gate's own docstring warns about.

    Two details are load-bearing. `(?:[^\\[\\]]|\\[[^\\]]*\\])*` rather than `[^\\]]*`, because the argument
    list can itself contain a bracket, as in `[ModelFact([Dir, RefJson], "3ms")]` — the simple form stopped
    at the array's closing bracket and five tests fell out of BOTH groups, running nowhere. And the
    optional `//` after the attribute, because several sites carry a trailing note there.
    """
    return re.compile(
        r"\[(?:" + "|".join(longfact_attributes()) + r")(?:[^\[\]]|\[[^\]]*\])*\](?:[ \t]*//[^\n]*)?"
        r"(?:\s*\[[^\]]*\](?:[ \t]*//[^\n]*)?)*\s*public\s+(?:async\s+)?"
        r"[\w<>\[\]\.,\s]+?\s+(\w+)\s*\(")


def all_longfacts():
    """Every [LongFact] full name, read from the SOURCE.

    It used to take the names from `full-suite.trx` and merely cross-check them against the source. That
    made a stale artefact the authority on what exists: the TRX in `Tests/bin` was written at 10:20 on
    2026-08-07 and five diagnostics added that afternoon were absent from it, so they landed in neither
    the heavy group nor the light one — invisible, and in exactly the way this gate exists to prevent.

    The source cannot go stale. A name assembled wrongly here is also not silent: the filter matches
    nothing, the chunk executes zero of one, and `run_chunk` reports FAIL.
    """
    pattern = longfact_method_pattern()
    found = []

    for p in TESTS.rglob("*.cs"):
        if any(x in p.parts for x in ("bin", "obj")):
            continue

        text = p.read_text(encoding="utf-8", errors="replace")
        namespace = re.search(r"^\s*namespace\s+([\w.]+)", text, re.M)

        if not namespace:
            continue

        for m in pattern.finditer(text):
            # The declaring class: the last TOP-LEVEL declaration above the attribute.
            #
            # Two narrower versions were wrong. `\bclass\s+(\w+)` also matches the word in prose, and
            # did — doc comments produced owners called `of` and `10`. Anchoring to a line start fixed
            # that but still took the nearest declaration, which is a NESTED helper wherever one sits
            # above the test: that gave `Data.Mnist.Replica.…` for a method on
            # `MnistDataParallelBenchTests`. Both errors produce a name that resolves to no test.
            #
            # Top-level is the shallowest indent in the file (block-scoped namespaces, .editorconfig).
            declarations = [
                (len(c.group(1)), c.group(2)) for c in re.finditer(
                    r"^([ \t]*)(?:(?:public|internal|private|protected|sealed|abstract|static|partial)"
                    r"[ \t]+)*class[ \t]+(\w+)", text[:m.start()], re.M)]

            if not declarations:
                continue

            outermost = min(d[0] for d in declarations)
            owner = next(name for indent, name in reversed(declarations) if indent == outermost)

            found.append(f"{namespace.group(1)}.{owner}.{m.group(1)}")

    return sorted(set(found))


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


# Must match LabFact.Marker in Tests/LabFact.cs, same as in longfact_gate.py.
LAB_MARKER = "LAB NOT AVAILABLE"


def split_results(trx_text):
    """(failures, skips, lab_skips) from one chunk's TRX — three states, because they mean three things."""
    import xml.etree.ElementTree as ET

    failures, skips, lab = [], [], []

    try:
        root = ET.fromstring(trx_text)
    except ET.ParseError:
        return sorted(set(re.findall(r'testName="([^"]+)"[^>]*outcome="Failed"', trx_text))), [], []

    for element in root.iter():
        if not element.tag.endswith("UnitTestResult"):
            continue

        outcome, name = element.get("outcome", ""), element.get("testName", "?")

        if outcome == "Passed":
            continue

        if outcome != "NotExecuted":
            failures.append(name)

            continue

        skips.append(name)
        message = " ".join((m.text or "") for m in element.iter() if m.tag.endswith("Message"))

        if LAB_MARKER in message:
            lab.append(name)

    return sorted(set(failures)), sorted(set(skips)), sorted(set(lab))


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
    bad, skipped, lab = [], [], []

    if trx.exists():
        text = trx.read_text(encoding="utf-8", errors="replace")
        found = re.search(r"<Counters\b([^>]*)/>", text)

        if found:
            c = dict(re.findall(r'(\w+)="(\d+)"', found.group(1)))
            passed, failed = int(c.get("passed", 0)), int(c.get("failed", 0))
            executed = int(c.get("executed", 0))

        bad, skipped, lab = split_results(text)

    # A skip is NOT a failure. Conflating them would report the whole heavy group red on a box whose lab
    # is down or whose model fixtures live elsewhere — and a group that is red for a reason nobody can
    # fix is a group people stop reading.
    state = "ok  " if (failed == 0 and executed + len(skipped) == len(names)) else "FAIL"
    note = (f"  skip {len(skipped)}" + (f" (lab {len(lab)})" if lab else "")) if skipped else ""
    print(f"[{index:2d}/{total}] {state} {elapsed:6.0f}s  {len(names):2d} tests  pass {passed:2d} "
          f"fail {failed:2d}{note}  {label}", flush=True)

    for b in bad[:8]:
        print(f"          !! {b.replace('DevOnBike.Overfit.Tests.', '')}", flush=True)
        msg = re.search(r'<UnitTestResult[^>]*testName="' + re.escape(b) + r'".*?<Message>(.*?)</Message>',
                        trx.read_text(encoding="utf-8", errors="replace"), re.S)

        if msg:
            print(f"             {msg.group(1).strip().splitlines()[0][:170]}", flush=True)

    # `kind`, not a hardcoded "LIGHT": every heavy chunk was about to be logged as light, which would have
    # made the one artefact that answers "how long does the release gate take" answer it wrongly.
    with PROGRESS.open("a", encoding="utf-8") as fh:
        fh.write(f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ}\t{kind.upper()}\t"
                 f"{label}\t{len(names)}\t{elapsed:.0f}s\tpass {passed}\tfail {failed}\t"
                 f"skip {len(skipped)}\tlab {len(lab)}\t{';'.join(bad) if bad else '-'}\n")

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


# Guarded, same as longfact_gate.py. Unguarded, `import longfact_split` to check one of its regexes
# STARTS A TEST RUN — which is what happened at 20:41 on 2026-08-07 and had to be killed.
if __name__ == "__main__":
    main()
