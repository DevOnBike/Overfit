---
name: overfit-mutate
description: Break the behaviour a test claims to protect and prove the test notices, with the five guards that stop a mutation lying — refuse a dirty target, require the anchor to match exactly once, check the baseline is green, separate "did not compile" from "not caught", and verify the restore byte-for-byte. Use after writing any test whose failure matters, and whenever somebody says a change is "pinned by a test". Not anomaly-specific; it applies to any C# test in this repository.
model: opus
color: green
---

# Prove the test can fail

**A green suite says the tests pass. It does not say they could fail.** This repository has shipped an
assertion satisfied by channels that existed before its subject did, a fixture whose value coincided with the
fallback so both sides read 24, and a test whose fixture did not contain the pod it was making a claim about.
All three were green. All three were found by mutation.

## The procedure

Write this into the scratch file (your own `do-<agent>.py`) with the four constants
substituted, and execute it as a single invocation.

```python
"""One mutation, aimed at one named test."""
import pathlib
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

REPO = pathlib.Path(r"D:\Overfit")
TARGET = REPO / "REPLACE_ME.cs"
FILTER = "REPLACE_ME"            # dotnet test --filter
EXPECTED = "REPLACE_ME"          # the test that MUST go red
ANCHOR = """REPLACE_ME"""        # exact source, including indentation
MUTATED = """REPLACE_ME"""


def run():
    proc = subprocess.run(
        ["dotnet", "test", str(REPO / "Tests" / "Tests.csproj"), "-c", "Release", "--nologo",
         "--filter", FILTER],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    text = (proc.stdout or "") + (proc.stderr or "")

    return (proc.returncode,
            sorted({n.split(".")[-1] for n in
                    re.findall(r"(?:Niepowodzenie|Failed) (DevOnBike\S+)", text)}),
            [l.strip() for l in text.splitlines() if ": error " in l])


dirty = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", str(TARGET)])
print(f"--- target clean against HEAD: {dirty.returncode == 0}")

original = TARGET.read_text(encoding="utf-8")
count = original.count(ANCHOR)
print(f"--- anchor matches {count} time(s)" + ("" if count == 1 else "   <-- MUST be exactly 1"))

if count != 1:
    sys.exit("STOPPED: an ambiguous anchor mutates whichever site the replace happens to reach")

code, failed, _ = run()
print(f"--- baseline: rc={code} failed={failed or 'none'}")

if code != 0:
    sys.exit("STOPPED: baseline is red, so the mutation would say nothing")

try:
    TARGET.write_text(original.replace(ANCHOR, MUTATED), encoding="utf-8")
    code, failed, errors = run()

    if errors:
        print(f"--- DID NOT COMPILE — the mutation is invalid, not the test: {errors[0][:150]}")
    else:
        print(f"--- under mutation: rc={code} failed={failed or 'NONE'}")
        print(f"    -> {'CAUGHT by ' + EXPECTED if EXPECTED in failed else 'NOT CAUGHT'}")
finally:
    TARGET.write_text(original, encoding="utf-8")
    print(f"--- restored byte-for-byte: {TARGET.read_text(encoding='utf-8') == original}")
```

## The five guards, each of which has fired here

1. **Refuse to start if the target already differs from `HEAD`.** A harness killed mid-run leaves the source
   mutated; the next run then treats the mutated file as its baseline and cheerfully reports *restore
   verified*.
2. **Assert the anchor matches exactly once, and print the count.** On 2026-08-10 an anchor matched twice
   because `RunCustomTrend` carries a line byte-identical to `RunTrend`'s. A separate run matched **zero**
   times because a multi-line anchor did not account for CRLF — which would have read as "not caught" if the
   count had not been printed.
3. **Check the baseline is green first.** A mutation against a red baseline cannot distinguish "the test
   caught it" from "it was already failing".
4. **Separate "did not compile" from "not caught".** An invalid mutation is a mistake in the harness, not
   evidence about the test, and by exit code alone they are identical. Also: if another agent is editing the
   tree, a compile error may not be yours — re-check before diagnosing.
5. **Restore in a `finally` and compare byte-for-byte.** Not "restored" — *verified* restored.

## Reading the result

- **Red, on the test you named** — the test can fail, for the reason you think. This is the only outcome
  that licenses the phrase *pinned by a test*.
- **Red, on a different test** — say which. It may be a better guard than the one you were checking, or the
  mutation changed more than you intended.
- **GREEN — the important case, and it is a finding rather than a setback.** The behaviour is not covered.
  Three times on 2026-08-10 a green mutation exposed something real: a test whose fixture did not contain the
  pod it made a claim about; an `AN-F1` fix that **nothing in the entire `Anomalies` suite** protected,
  because every existing test had a history matching the climb so the distinguishing case was never
  exercised; and a signed plan whose mechanism could not fire at all. **Report it. Do not touch the harness
  until you understand why nothing noticed.**

## What to mutate

One behaviour that matters, not one line per file. Aim each mutation at a **named** test and say which. The
best targets are the ones where a wrong answer is silent: a gate that fails open instead of closed, a
constant-time comparison replaced by `StartsWith`, a counter not cleared on recovery, a missing-data path
returning `Healthy` instead of `InsufficientData`.

**Do not mutate to see what happens.** A mutation with no predicted victim is a fishing trip, and a green
result from one tells you nothing you can act on.
