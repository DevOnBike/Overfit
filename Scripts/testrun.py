"""Run a slice of the test suite with LIVE progress, serially when models are involved, and prove it.

    python Scripts/testrun.py --area LanguageModels          # chunked, serial, long tests on
    python Scripts/testrun.py --filter "FullyQualifiedName~Gpt2"
    python Scripts/testrun.py --area Anomalies --no-long     # fast tests only, stays parallel

WHY THIS FILE EXISTS — four separate incidents, all on 2026-08-20:

1. **A run with no visible progress is a run nobody can judge.** `dotnet test` output was captured into a
   buffer and printed at the end, so a 45-minute run showed nothing until it finished. "Is it stuck or
   working?" had no answer, and "I cannot see inside the chunk" described a choice I had made, not a limit
   of the tool. Progress here is streamed as it arrives.

2. **The whole `LanguageModels` area in one parallel run exhausted the machine.** xunit parallelises
   collections, several tests load multi-gigabyte checkpoints, and 61.6 GB does not hold them at once.
   Measured: 0.3 GB free, 89 GB of pagefile, 567 MB/s of disk READS with almost no writes — paging, not
   progress. After serialising: 17 GB resident, 29 GB free, 1.2 GB of pagefile. Chunking alone does NOT fix
   this; it shrinks the group that runs concurrently without stopping the concurrency.

3. **The parallelism switch is silent in both directions.** `xunit.runner.json` is not copied to output by
   xunit.v3.core.mtp-v2 4.0.0 — its props and targets carry no rule, so the file needs an explicit
   `<Content Include>` — and a RunSettings key on the command line is ignored without a word. Worse, when
   the switch DOES work the runner announces it only at `-v n`. `TG-T14`'s first serial arm was recorded
   INCONCLUSIVE for exactly this reason and may well have been working. So the mode is read back out of the
   run's own output, and a slice that cannot show it is reported UNKNOWN rather than assumed.

4. **`.claude/do.py` is scratch and was overwritten mid-task**, taking the first version of this runner with
   it. What must survive lives under `Scripts/`, like `Scripts/lab.py`.

**The serial setting is scoped to THIS RUN, not to the repository.** Plain `dotnet test` runs 2,764 tests in
22 seconds and loads no model at all — every model test is skipped without OVERFIT_RUN_LONG. Serialising the
whole suite would punish 1,078 tests for the sins of about 260. The toggle is applied to the source
`xunit.runner.json` (not the copy in bin, which a rebuild would overwrite mid-run) and restored in `finally`.
"""

import argparse
import io
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMESPACE = "DevOnBike.Overfit.Tests"
RUNNER_JSON = os.path.join(REPO, "Tests", "xunit.runner.json")

#: One line per completed test at `-v n`, e.g. "  Passed DevOnBike.Overfit.Tests.X.Y [12 ms]".
#:
#: The name is anchored on the ROOT NAMESPACE rather than on `\S+`. The loose version matched MSBuild's own
#: prose — "Failed to resolve...", "Skipped to..." — and reported ten RED lines for a run in which nothing
#: had failed. A false red is worse than no progress line at all: it is the one signal that must never lie.
RESULT = re.compile(r"^\s*(Passed|Failed|Skipped)\s+(DevOnBike\.\S+)")

#: The runner announces its own parallelism here. Absent means the run cannot vouch for its mode.
MODE = re.compile(r"parallel mode = (\w+)")

#: Heaviest first, so an exhausted box shows up in the first slice rather than the last.
LANGUAGE_MODEL_CHUNKS = [
    "Runtime", "Loading", "Diagnostics", "Constraints", "Tokenization", "Retrieval",
    "Skills", "Chat", "Tokenizers", "Whisper", "LoRA", "Tools", "Sampling",
    "Embeddings", "Agents", "GPT1", "Demo", "Memory", "Experimental",
]


def available_gb():
    try:
        import psutil

        return psutil.virtual_memory().available / 1073741824
    except ImportError:
        return float("nan")


def swap_gb():
    try:
        import psutil

        return psutil.swap_memory().used / 1073741824
    except ImportError:
        return float("nan")


def wait_for_memory(minimum_gb, seconds=300):
    """Starting a slice while the previous one's pages are still being reclaimed reproduces the thrash."""
    deadline = time.time() + seconds

    while time.time() < deadline:
        if not available_gb() < minimum_gb:
            return True

        time.sleep(5)

    return False


def restore_default():
    """Puts the runner config back to the repository's default, whatever a previous run left behind.

    **A killed process does not run `finally`.** On 2026-08-20 this runner was stopped mid-slice and left
    `parallelizeTestCollections: false` committed to the source file — the whole suite silently serialised
    for everyone, with nothing to point at. Cleaning up on the way OUT is not enough when the way out can be
    skipped; the reliable place is on the way IN, where a residue from any earlier crash is repaired before
    it can be inherited.
    """
    text = io.open(RUNNER_JSON, encoding="utf-8").read()
    fixed = re.sub(r'"parallelizeTestCollections":\s*false', '"parallelizeTestCollections": true', text)

    if fixed != text:
        io.open(RUNNER_JSON, "w", encoding="utf-8", newline="").write(fixed)
        print("  NOTE: a previous run left parallelism disabled in Tests/xunit.runner.json — restored.",
              flush=True)


def set_parallel(enabled):
    """Rewrites the source runner config and returns what it held before."""
    original = io.open(RUNNER_JSON, encoding="utf-8").read()
    want = "true" if enabled else "false"
    io.open(RUNNER_JSON, "w", encoding="utf-8", newline="").write(
        re.sub(r'"parallelizeTestCollections":\s*(true|false)',
               f'"parallelizeTestCollections": {want}', original))

    return original


def run_slice(label, test_filter, trx, env, every, quiet_pass):
    """Runs one filter with live progress. Returns (passed, failed, skipped, minutes, mode, failures)."""
    command = ["dotnet", "test", "./Tests/Tests.csproj", "-c", "Release", "-v", "n",
               "--logger", f"trx;LogFileName={trx}", "--filter", test_filter]

    started = time.time()
    seen = 0
    mode = "UNKNOWN"
    live_red = []
    build_errors = []

    process = subprocess.Popen(command, cwd=REPO, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                               errors="replace", bufsize=1)

    for line in process.stdout:
        found = MODE.search(line)

        if found:
            mode = found.group(1)
            print(f"    [{label}] runner says: parallel mode = {mode}", flush=True)

            continue

        # A build that refuses to start produces ZERO tests, and zero tests is indistinguishable from zero
        # failures in every summary that only counts reds. On 2026-08-20 an orphaned test host held
        # `Global\DevOnBike.Overfit.MachineMeasurement`, the build guard refused, and this runner reported
        # "OK ... passed 0 failed 0 skipped 0" for eleven consecutive areas. The reason is captured here so
        # the summary can say WHY nothing ran rather than merely that nothing failed.
        if ": error " in line or "Build FAILED" in line:
            stripped = line.strip()

            if stripped not in build_errors:
                build_errors.append(stripped)

            continue

        match = RESULT.match(line)

        if not match:
            continue

        outcome, name = match.group(1), match.group(2)
        seen += 1
        parts = name.rsplit(".", 2)
        short = ".".join(parts[-2:]) if len(parts) > 1 else name

        if outcome == "Failed":
            live_red.append(short)
            print(f"    [{label}] {seen:>5}  RED   {short}", flush=True)

            continue

        if outcome == "Passed" and not quiet_pass and seen % every == 0:
            print(f"    [{label}] {seen:>5}  ok    {short}   "
                  f"({(time.time() - started) / 60:.1f} min, RAM {available_gb():.0f} GB)", flush=True)

    process.wait()
    minutes = (time.time() - started) / 60

    passed = failed = skipped = 0
    failures = []

    try:
        for element in ET.parse(trx).getroot().iter():
            if not element.tag.endswith("UnitTestResult"):
                continue

            outcome = element.get("outcome", "")
            name = element.get("testName", "?")

            if outcome == "Passed":
                passed += 1
            elif outcome == "NotExecuted":
                skipped += 1
            else:
                message = " ".join(" ".join(
                    (m.text or "") for m in element.iter() if m.tag.endswith("Message")).split())
                failed += 1
                failures.append((name.split(f"{NAMESPACE}.")[-1], message[:240]))
    except (ET.ParseError, FileNotFoundError):
        # The TRX is the authority on counts. Without it the live tally is all there is, and saying so beats
        # reporting a number whose provenance is unclear.
        print(f"    [{label}] TRX unreadable — live tally {seen} tests, {len(live_red)} red", flush=True)

        return (seen - len(live_red), len(live_red), 0, minutes, mode,
                [(f, "(from live output; TRX missing)") for f in live_red], build_errors)

    return passed, failed, skipped, minutes, mode, failures, build_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--area", help="test area, e.g. LanguageModels or Anomalies")
    parser.add_argument("--filter", dest="explicit", help="a raw --filter expression, used as-is")
    parser.add_argument("--no-chunks", action="store_true", help="one run, no splitting")
    parser.add_argument("--from", dest="start_at",
                        help="skip chunks before this one — for resuming after a stopped run")
    parser.add_argument("--no-long", action="store_true",
                        help="do not set OVERFIT_RUN_LONG=1 (implies parallel: nothing loads a model)")
    parser.add_argument("--parallel", action="store_true",
                        help="keep xunit parallelism on even for a long run — it will exhaust the box")
    parser.add_argument("--every", type=int, default=25, help="print one passing test every N (default 25)")
    parser.add_argument("--quiet-pass", action="store_true", help="print only failures as they happen")
    parser.add_argument("--min-free-gb", type=float, default=20.0,
                        help="wait for this much free RAM before each slice (default 20)")
    args = parser.parse_args()

    if not args.area and not args.explicit:
        parser.error("give --area or --filter")

    env = dict(os.environ)
    env["DOTNET_CLI_UI_LANGUAGE"] = "en"

    if not args.no_long:
        env["OVERFIT_RUN_LONG"] = "1"

    # OVERFIT_GPT2_DIR only. `OVERFIT_MODEL_DIR` is NOT set here, and that is a correction, not an omission:
    # this runner used to default it to C:\qwen3b, and `SafetensorsGpt2LoaderTests` reads that variable to
    # find a GPT-2 safetensors file. Both C:\qwen3b and C:\gpt2 contain a `model.safetensors`, so the test
    # loaded QWEN's weights as GPT-2 Small and failed its bit-parity check — a red produced entirely by the
    # harness, on a loader that touches none of the code under test. Only three places in the repository read
    # the variable and all three want the anomaly production model, not Qwen. Left unset, each falls back to
    # its own correct default.
    env.setdefault("OVERFIT_GPT2_DIR", r"C:\gpt2")

    if args.explicit:
        slices = [("filter", args.explicit)]
    elif args.area == "LanguageModels" and not args.no_chunks:
        chunks = LANGUAGE_MODEL_CHUNKS

        if args.start_at:
            # Resuming is a real need — a stopped run should not force a repeat of the slices that already
            # passed — but a silent skip would be worse than the repeat. The names dropped are printed.
            if args.start_at not in chunks:
                parser.error(f"--from {args.start_at} is not a chunk; known: {', '.join(chunks)}")

            cut = chunks.index(args.start_at)

            if cut:
                print(f"  resuming at {args.start_at}, SKIPPING {cut} earlier chunk(s): "
                      f"{', '.join(chunks[:cut])}", flush=True)

            chunks = chunks[cut:]

        slices = [(c, f"FullyQualifiedName~{NAMESPACE}.LanguageModels.{c}.") for c in chunks]
    else:
        slices = [(args.area, f"FullyQualifiedName~{NAMESPACE}.{args.area}.")]

    restore_default()

    serial = not args.parallel and not args.no_long

    print(f"  {len(slices)} slice(s), OVERFIT_RUN_LONG={env.get('OVERFIT_RUN_LONG', 'unset')}, "
          f"{'SERIAL' if serial else 'parallel'}, free RAM {available_gb():.0f} GB, "
          f"swap {swap_gb():.0f} GB\n", flush=True)

    totals = {"passed": 0, "failed": 0, "skipped": 0}
    all_failures = []
    unknown_mode = []
    nothing_ran = []
    original = set_parallel(not serial) if serial else None

    try:
        for label, test_filter in slices:
            if not wait_for_memory(args.min_free_gb):
                print(f"  SKIPPED {label}: free RAM stayed under {args.min_free_gb} GB", flush=True)

                continue

            trx = os.path.join(REPO, "Tests", "bin", f"testrun-{label}.trx")
            before = available_gb()

            passed, failed, skipped, minutes, mode, failures, build_errors = run_slice(
                label, test_filter, trx, env, args.every, args.quiet_pass)

            totals["passed"] += passed
            totals["failed"] += failed
            totals["skipped"] += skipped
            all_failures.extend((label, t, m) for t, m in failures)

            if serial and mode != "none":
                unknown_mode.append(f"{label} (mode={mode})")

            # ZERO EXECUTED IS NOT OK. `failed == 0` was the whole verdict here until 2026-08-20, when a
            # build guard refused eleven slices in a row and every one of them printed "OK ... passed 0
            # failed 0 skipped 0". Nothing had failed because nothing had run, and the summary could not
            # tell those apart — the exact shape this repository keeps finding elsewhere and the reason
            # `TG-T14` sat unnoticed for six days.
            executed = passed + failed + skipped

            if executed == 0:
                nothing_ran.append(label)
                reason = build_errors[0][:150] if build_errors else "no tests matched and no build error seen"
                print(f"  !!  {label:<16}{minutes:>6.1f} min   NOTHING RAN — not a pass.\n"
                      f"      {reason}\n", flush=True)

                continue

            flag = "OK " if failed == 0 else "RED"
            print(f"  {flag} {label:<16}{minutes:>6.1f} min   passed {passed:>4}  failed {failed:>3}  "
                  f"skipped {skipped:>4}   RAM {before:.0f}->{available_gb():.0f} GB   parallel={mode}\n",
                  flush=True)
    finally:
        if original is not None:
            io.open(RUNNER_JSON, "w", encoding="utf-8", newline="").write(original)

    print(f"  TOTAL  passed {totals['passed']}   failed {totals['failed']}   "
          f"skipped {totals['skipped']}")

    if nothing_ran:
        print(f"\n  !! {len(nothing_ran)} slice(s) EXECUTED NOTHING and are not passes: "
              f"{', '.join(nothing_ran)}")

    if unknown_mode:
        print("\n  !! these slices did not confirm serial execution, so their memory behaviour proves "
              f"nothing: {', '.join(unknown_mode)}")

    if all_failures:
        print("\n  FAILURES:")

        for label, test, message in all_failures:
            print(f"    [{label}] {test}")
            print(f"        {message}")

    # A slice that executed nothing is a failure of the run, not a silent zero. Exiting 0 here would let a
    # refused build read as a green suite in any script that only checks the code.
    return 1 if totals["failed"] or nothing_ran else 0


if __name__ == "__main__":
    raise SystemExit(main())
