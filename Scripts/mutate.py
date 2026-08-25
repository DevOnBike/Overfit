"""Apply a source mutation, build, test, and restore — without leaving a stale binary behind.

    import sys
    sys.path.insert(0, r"D:\\Overfit\\Scripts")
    from mutate import mutation

    with mutation(path, old, new, "what this breaks"):
        run_the_suite()

**Why this file exists, and it is not a style preference.** The obvious harness is
``shutil.copy2(f, f + ".bak")``, edit, test, ``shutil.move(f + ".bak", f)``. **`copy2` preserves the
modification time**, so the restored file is *older* than the object built from the mutated source. MSBuild
compares those timestamps, decides the output is up to date, and **does not rebuild** — every run after the
restore executes the mutated library against correct source.

Measured on 2026-08-19: after one such restore the timestamp went **backwards by 113.5 seconds**, and the
next full suite reported **100 failures** with source that was byte-identical to a green commit. It took a
file-by-file bisection against `HEAD` to find that the source was never the problem. Worse than the wasted
hour: a measurement taken after a restore had been recorded as a result, and it may have been measuring the
mutated binary rather than the change it was attributing.

**So the restore stamps the file with the current time**, which is the whole point of this module. Two other
guards come with it, both from failures on record here:

- **A mutation that does not build is not a tested mutation.** Reporting "caught" or "escaped" for source the
  compiler rejected is worse than not running it.
- **The mutated file keeps its own line endings.** Until 2026-08-25 the read normalised them and the write
  did not put them back, so mutating a **CRLF** file rewrote every line of it to LF for the duration of the
  block. `shutil.move` of the `copy2` backup restored the bytes, so it was invisible in a completed run —
  but a harness killed mid-run left the file converted, and **a whole-file line-ending change is the most
  content-free and most review-destroying diff this repository can produce**, invisible in an editor and in
  most diffs unless somebody goes looking. Same shape as the `[FAIL]` defect above: the tool reports nothing
  wrong while the state is wrong. It never bit, because the file being mutated happened to be LF —
  `Demo/GpuProbe` is uniformly LF and most of the rest of the tree is CRLF.
- **A mutation that kills the test host prints no failure line**, and an absent failure marker reads exactly
  like a pass. Callers must check the test count, not only the presence of failures — `expected_tests` below
  is there to make that hard to skip.
- **The FAILURE MARKER IS NOT THE AUTHORITY. The summary line's count is.** Until 2026-08-25 `verdict` read
  victims only from `[FAIL]`, and **this repository's runner does not print `[FAIL]`** — it prints
  `Failed <fully-qualified-name> [41 ms]`, one line per failure. So a mutation that reddened **eleven** tests
  was reported as `*** ESCAPED *** (2795 tests ran and none noticed)`, and every guard above it passed,
  because the tests really had run and really had been built. **A false ESCAPED is worse than a crash**: it
  reads as "the test is vacuous", and the reader's next action is to weaken a test that was working.
  `verdict` now reads both markers **and cross-checks the victim list against `Failed: N`**. A disagreement
  is an `EXTRACTION FAILED`, which is a third outcome and must never be collapsed into either other one —
  "nothing noticed" and "I could not read what noticed" lead to opposite work.
"""

import io
import os
import shutil
import time
from contextlib import contextmanager


def touch(path):
    """Stamps a file with the current time so the build system sees it as changed."""
    now = time.time()
    os.utime(path, (now, now))


def as_written(fragment, text):
    """Re-line-ends a fragment written with ``\n`` so it can match ``text`` as that file actually is.

    **The endings move this way round, and the other way is a defect.** Anchors are typed by hand into
    Python strings, so they carry ``\n`` whatever the target file uses. Normalising the FILE would make
    them match — and would rewrite every line of a CRLF file, which is what this module used to do.
    Translating the FRAGMENT matches just as well and touches nothing.

    A fragment already written with ``\r\n`` is flattened first, so the result cannot come out ``\r\r\n``.

    On a file with mixed endings the majority wins and the anchor may then match zero times — which is
    loud, because ``mutation`` refuses to proceed on a count it did not expect.
    """
    fragment = fragment.replace("\r\n", "\n")
    crlf = text.count("\r\n")

    if crlf > text.count("\n") - crlf:
        return fragment.replace("\n", "\r\n")

    return fragment


@contextmanager
def mutation(path, old, new, label, occurrences=1):
    """Replaces ``old`` with ``new`` for the duration of the block, then restores and re-stamps.

    Raises before touching anything if ``old`` does not appear exactly ``occurrences`` times: an anchor that
    matches zero times silently tests nothing, and one that matches twice changes more than intended.

    **Byte-preserving.** The file is read and written as bytes, so a line the mutation does not touch comes
    out exactly as it went in, whatever the file's endings are — see the fourth bullet in this module's
    docstring for the run that established it. ``old`` and ``new`` are translated to the file's endings by
    :func:`as_written`, so callers keep writing anchors with ``\n``.
    """
    text = io.open(path, "rb").read().decode("utf-8")
    old = as_written(old, text)
    new = as_written(new, text)
    found = text.count(old)

    if found != occurrences:
        raise AssertionError(
            f"mutation '{label}': anchor matched {found} time(s), expected {occurrences} — nothing changed")

    backup = path + ".mutation-backup"
    shutil.copy2(path, backup)

    try:
        with io.open(path, "wb") as handle:
            handle.write(text.replace(old, new, occurrences).encode("utf-8"))

        touch(path)

        yield
    finally:
        shutil.move(backup, path)

        # The line this module exists for. copy2 carried the original mtime through the backup, so without
        # this the restored source looks older than the object built from the mutated text and the build is
        # skipped.
        touch(path)


def victims(output):
    """Every failing test's method name, from EITHER marker this repository's runners have printed.

    Two forms, and the second is the one that was missing until 2026-08-25:

        SomeClass.SomeTest(param: 1) [FAIL]
        Failed DevOnBike.Overfit.Tests.Core.Kernels.SomeTests.SomeTest(param: 1) [41 ms]

    A `[Theory]` victim carries its arguments, so the name is captured without them and the set collapses
    a parameterised test's cases into one name — which is why `verdict` reports the SUMMARY's count beside
    the names rather than `len` of this list.

    The summary line cannot be mistaken for a victim: it begins `Failed!` and `Failed:`, neither of which
    is `Failed` followed by whitespace.
    """
    import re

    marked = re.findall(r"\.(\w+)(?:\([^)]*\))?\s*\[FAIL\]", output)
    named = re.findall(r"^\s*Failed\s+[\w.]*\.(\w+)", output, re.MULTILINE)

    return sorted(set(marked) | set(named))


def verdict(output, expected_tests, label):
    """Turns a test run's output into CAUGHT / ESCAPED / EXTRACTION FAILED, refusing to guess when the run
    was not a run.

    **The count in the summary line is the authority, not the presence of a marker** — see the third bullet
    in this module's docstring for the run that established it.
    """
    import re

    if any(": error" in line for line in output.splitlines()):
        return f"{label}: *** DID NOT BUILD — the mutation was never tested ***"

    summary = re.search(r"(Passed!|Failed!)\s+-\s+Failed:\s+(\d+), Passed:\s+(\d+)", output)

    if summary is None:
        return f"{label}: *** NO SUMMARY — the host died, which is not a pass ***"

    reported = int(summary.group(2))
    ran = reported + int(summary.group(3))

    if ran < expected_tests:
        return f"{label}: *** PARTIAL: {ran} of {expected_tests} ran — not a pass ***"

    failures = victims(output)

    if reported > 0 and not failures:
        return (f"{label}: *** EXTRACTION FAILED — the summary reports {reported} failing test(s) and no "
                f"victim name could be read from this output. NOT escaped, NOT caught: fix the reader ***")

    if reported == 0 and failures:
        return (f"{label}: *** EXTRACTION FAILED — the summary reports 0 failing tests yet "
                f"{len(failures)} victim name(s) were read out ({failures[:3]}). NOT caught ***")

    if not failures:
        return f"{label}: *** ESCAPED *** ({ran} tests ran and none noticed)"

    return f"{label}: CAUGHT — {reported} failing test(s), e.g. {failures[:3]}"


def kill_tree(pid):
    r"""Kills a process and every descendant.

    **`subprocess.run(timeout=...)` kills only the direct child.** A `dotnet test` that times out leaves
    `testhost.exe` and `DevOnBike.Overfit.Tests.exe` alive, and this repository's build guard refuses to
    build while one of them holds `Global\DevOnBike.Overfit.MachineMeasurement`. On 2026-08-19 that turned
    one hung run into an afternoon: every subsequent run blocked on a mutex held by an orphan from the
    previous timeout, which read as "the code hangs" rather than "the harness leaks processes".
    """
    import subprocess

    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                   capture_output=True, text=True, timeout=60)


def kill_test_hosts():
    """Kills every test host on the machine, orphan or not, so the measurement mutex is free."""
    import subprocess

    script = ("Get-CimInstance Win32_Process | "
              "Where-Object { $_.Name -match 'Overfit.Tests|testhost|vstest' } | "
              "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }")

    subprocess.run(["powershell", "-NoProfile", "-Command", script],
                   capture_output=True, text=True, timeout=120)
