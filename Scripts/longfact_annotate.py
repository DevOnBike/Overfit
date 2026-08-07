"""Writes each [LongFact]'s measured runtime next to the attribute.

WHY
    "Run these before a release" means something very different at 3 seconds than at 10 minutes, and
    until 2026-08-07 nobody had a single number: none of these tests had ever been executed. The one that
    turned out to take 579 s sits in a file that gave no hint of it. Putting the number at the call site
    is what lets somebody choose a subset without running the whole thing first.

NOT "AVERAGE" — and the wording matters
    This is ONE run. An average needs repetitions, and writing "average" from a single sample is the kind
    of claim this repository keeps having to retract. The comment therefore says "measured once", with the
    date, so the next reader knows exactly how much weight it carries. Durations also include a cold
    process start and, for the model tests, reading multi-GB weights off disk; a second run on a warm file
    cache would be faster and that is not captured here either.

SOURCE
    The per-test <UnitTestResult duration=...> in the chunk TRX files, not the wall-clock of the chunk —
    the latter includes ~1-2 s of `dotnet test` start-up that is not the test's cost.

FAILED TESTS GET NO NUMBER
    A test that died on `connection refused` after 3 s did not take 3 s; it took 3 s to fail. Annotating
    that would be worse than saying nothing, so failures are marked as unmeasured with the reason.

IDEMPOTENT
    Re-running replaces an existing annotation rather than appending a second one.
"""
import argparse
import datetime
import pathlib
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Repo root from this file's own location, NOT a hardcoded path: a developer's absolute
# path in a public repository is a leak, and it also means the script only runs on one box.
ROOT = pathlib.Path(__file__).resolve().parents[1]
TESTS = ROOT / "Tests"
BIN = TESTS / "bin"
STAMP = "2026-08-07"

# `[LongFact]` optionally already annotated, then any further attributes, then the method name.
SITE = re.compile(
    r"(?P<indent>[ \t]*)\[(?P<attr>LongFact|ProductionAnomalyBaseFact)(?P<args>\([^)]*\))?\](?P<note>[ \t]*//[^\n]*)?"
    r"(?P<between>(?:\s*\[[^\]]*\])*\s*public\s+(?:async\s+)?[\w<>\[\]\.,\s]+?\s+)(?P<method>\w+)\s*\(")

# EXACTLY four spaces of indent — the namespace-level class, not a nested one. A "last class declared
# before this offset" rule looks right and is wrong: MnistDataParallelBenchTests declares a private
# nested `Replica` at line 31 and its test at line 88, so the naive rule attributed the test to `Replica`
# and silently failed to match it against the runner's results. It was one test out of 57 and it would
# have read as "runtime not measured yet" rather than as a bug. Four spaces is the same assumption the
# repository's own BanMultipleTopLevelTypes guard makes, and .editorconfig enforces block-scoped
# namespaces, so it holds here.
CLASS = re.compile(r"^ {4}(?:public|internal|private)?\s*(?:sealed\s+|abstract\s+|static\s+|partial\s+)*"
                   r"class\s+(\w+)", re.M)

# Tests whose recorded duration is real but MEANINGLESS, so the number must not be written down. A timing
# is only a timing if the test did the work; a run that returned early has a duration and no meaning, and
# annotating it would preserve, in source, a number produced by the very defect that was just fixed.
INVALIDATED = {
    ("GptAnomalyLoRATargetComparisonProductionTests",
     "LoRATargetStages_OnProductionBase_FlattenBenign_AndKeepDetection"):
        "runtime unknown — the 2026-08-07 run recorded <1 s, but that was the vacuous pass "
        "(missing fixture returned early); it now skips instead",
}


def durations():
    """full test name -> (seconds, outcome) from every chunk TRX."""
    found = {}

    for trx in (sorted(BIN.glob("longfact-chunk-*.trx"))
                + sorted(BIN.glob("longfact-light-*.trx"))
                + sorted(BIN.glob("longfact-heavy-*.trx"))
                + sorted(BIN.glob("longfact.trx"))):
        text = trx.read_text(encoding="utf-8", errors="replace")

        for m in re.finditer(
                r'<UnitTestResult[^>]*testName="([^"]+)"[^>]*duration="([^"]+)"[^>]*outcome="(\w+)"',
                text):
            name, dur, outcome = m.group(1), m.group(2), m.group(3)
            parts = dur.split(":")

            try:
                seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            except (ValueError, IndexError):
                continue

            found[name] = (seconds, outcome)

    return found


def human(seconds):
    """The notation `LongFact.TryParseRuntime` reads back: 115ms / 32s / 15min51s / 2h15min.

    Human first, because this string is read far more often than it is parsed — somebody deciding whether
    to run a subset wants "15min51s", not 951. Composite forms carry the smaller unit only when it is
    non-zero, so a clean value stays short ("4min", not "4min00s").
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"

    if seconds < 60:
        return f"{seconds:.0f}s"

    if seconds < 3600:
        minutes, rest = divmod(int(round(seconds)), 60)

        return f"{minutes}min{rest}s" if rest else f"{minutes}min"

    hours, rest = divmod(int(round(seconds)), 3600)
    minutes = rest // 60

    return f"{hours}h{minutes}min" if minutes else f"{hours}h"


def class_at(text, offset):
    """Name of the last class declared before this offset."""
    last = None

    for m in CLASS.finditer(text, 0, offset):
        last = m.group(1)

    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write the files; otherwise dry run")
    args = ap.parse_args()

    timings = durations()
    print(f"TRX      {len(timings)} test results across "
          f"{len(list(BIN.glob('longfact-chunk-*.trx')))} chunk file(s)")

    by_short = {}

    for full, value in timings.items():
        cls, method = full.rsplit(".", 2)[-2:]
        by_short[(cls, method)] = value

    heavy_short = set()
    heavy_file = ROOT / "Scripts" / "longfact_heavy.txt"

    if heavy_file.exists():
        for line in heavy_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or "\t" not in line:
                continue

            full = line.split("\t", 1)[0]
            heavy_short.add(tuple(full.rsplit(".", 2)[-2:]))

    print(f"HEAVY    {len(heavy_short)} tests in the heavy group (no per-merge measurement expected)")

    annotated = unmeasured = failed = untouched = 0
    changed_files = 0
    samples = []

    for p in sorted(TESTS.rglob("*.cs")):
        if any(x in p.parts for x in ("bin", "obj")):
            continue

        text = p.read_text(encoding="utf-8")

        if "[LongFact" not in text and "[ProductionAnomalyBaseFact" not in text:
            continue

        out = []
        last = 0
        touched = False

        for m in SITE.finditer(text):
            cls = class_at(text, m.start())
            key = (cls, m.group("method"))
            value = by_short.get(key)

            # The measured value goes in the ATTRIBUTE, everything else in a comment. A number that a tool
            # can read back is data — it can be summed to say what the gate costs, or sorted to name the
            # worst offenders. "No number, and here is why" is prose and belongs in prose; putting a
            # placeholder string in the attribute would make every consumer parse excuses.
            attribute_args = ""

            if key in INVALIDATED:
                # A number exists for these, and it is a lie. See INVALIDATED for why each one is there.
                note = f"  // {INVALIDATED[key]}"
                unmeasured += 1
            elif value is None and key in heavy_short:
                # The heavy group is deliberately not run per merge, so "no number" is its final state
                # until somebody runs that group — saying so is the useful thing.
                note = "  // heavy group, never measured — see Scripts/longfact_heavy.txt"
                unmeasured += 1
            elif value is None:
                # No result AND not heavy: it is probably still running. Leave the site exactly as it is
                # rather than stamping a placeholder that gets rewritten minutes later — an annotation
                # that churns is one nobody trusts, and the file history should show the number arriving
                # once.
                untouched += 1

                continue
            elif value[1] != "Passed":
                note = f"  // runtime unmeasured — the test failed after {human(value[0])} ({STAMP})"
                failed += 1
            else:
                attribute_args = f'("{human(value[0])}")'
                note = ""
                annotated += 1

                if len(samples) < 12:
                    samples.append(f"[LongFact{attribute_args}]  {cls}.{m.group('method')}")

            head = text[last:m.start()]
            # Rebuild with the attribute name that was THERE, not a hardcoded "LongFact". Writing the base
            # name back silently downgraded ProductionAnomalyBaseFact to LongFact and took the fixture
            # guard with it — a change no compiler can object to, because both names are valid attributes,
            # and the build went green on a test that had just lost its skip. Caught by reading the diff,
            # which was the only thing that could have caught it.
            rebuilt = (f"{m.group('indent')}[{m.group('attr')}{attribute_args}]{note}"
                       f"{m.group('between')}{m.group('method')}(")
            out.append(head)
            out.append(rebuilt)
            last = m.end()
            touched = True

        if not touched:
            continue

        out.append(text[last:])
        new = "".join(out)

        if new == text:
            continue

        changed_files += 1

        if args.apply:
            p.write_text(new, encoding="utf-8")

    print(f"\nannotated with a runtime : {annotated}")
    print(f"failed, so no runtime    : {failed}")
    print(f"heavy group, no number   : {unmeasured}")
    print(f"left alone (still running or not run yet) : {untouched}")
    print(f"files that would change  : {changed_files}")
    print(f"\nmode: {'APPLIED' if args.apply else 'DRY RUN — nothing written'}")
    print("\nsample:")

    for s in samples:
        print("   ", s)


main()
