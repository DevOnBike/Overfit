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
- **A mutation that kills the test host prints no failure line**, and an absent `[FAIL]` reads exactly like a
  pass. Callers must check the test count, not only the presence of failures — `expected_tests` below is
  there to make that hard to skip.
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


@contextmanager
def mutation(path, old, new, label, occurrences=1):
    """Replaces ``old`` with ``new`` for the duration of the block, then restores and re-stamps.

    Raises before touching anything if ``old`` does not appear exactly ``occurrences`` times: an anchor that
    matches zero times silently tests nothing, and one that matches twice changes more than intended.
    """
    text = io.open(path, encoding="utf-8").read()
    found = text.count(old)

    if found != occurrences:
        raise AssertionError(
            f"mutation '{label}': anchor matched {found} time(s), expected {occurrences} — nothing changed")

    backup = path + ".mutation-backup"
    shutil.copy2(path, backup)

    try:
        io.open(path, "w", encoding="utf-8", newline="").write(text.replace(old, new, occurrences))
        touch(path)

        yield
    finally:
        shutil.move(backup, path)

        # The line this module exists for. copy2 carried the original mtime through the backup, so without
        # this the restored source looks older than the object built from the mutated text and the build is
        # skipped.
        touch(path)


def verdict(output, expected_tests, label):
    """Turns a test run's output into CAUGHT / ESCAPED, refusing to guess when the run was not a run."""
    import re

    if any(": error" in line for line in output.splitlines()):
        return f"{label}: *** DID NOT BUILD — the mutation was never tested ***"

    summary = re.search(r"(Passed!|Failed!)\s+-\s+Failed:\s+(\d+), Passed:\s+(\d+)", output)

    if summary is None:
        return f"{label}: *** NO SUMMARY — the host died, which is not a pass ***"

    ran = int(summary.group(2)) + int(summary.group(3))

    if ran < expected_tests:
        return f"{label}: *** PARTIAL: {ran} of {expected_tests} ran — not a pass ***"

    failures = sorted(set(re.findall(r"\.(\w+)(?:\([^)]*\))?\s*\[FAIL\]", output)))

    if not failures:
        return f"{label}: *** ESCAPED *** ({ran} tests ran and none noticed)"

    return f"{label}: CAUGHT by {failures[:3]}"
