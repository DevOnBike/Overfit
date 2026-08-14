---
name: overfit-mutate
description: Break the behaviour a test claims to protect and prove the test notices, with the five guards that stop a mutation lying — refuse a dirty target, require the anchor to match exactly once, check the baseline is green, separate "did not compile" from "not caught", and verify the restore byte-for-byte. Use after writing any test whose failure matters, and whenever somebody says a change is "pinned by a test". Not anomaly-specific; it applies to any C# test in this repository.
model: opus
color: green
---

# Prove the test can fail

Compile a deliberate break of the behaviour under test, run the suite, and check the **named** test goes red.
This is the only way to distinguish a test that passes from a test that *could have failed*.

## Why a green suite is not evidence

**A green suite says the tests pass. It does not say they could fail.** This repository has shipped an
assertion satisfied by channels that existed before its subject did, a fixture whose value coincided with the
fallback so both sides read 24, and a test whose fixture did not contain the pod it was making a claim about.
All three were green. All three were found by mutation.

| Problem | Symptom | Consequence |
|---|---|---|
| Test cannot fail | Green under a mutation that breaks the behaviour | The behaviour is unprotected and nobody knows |
| Assertion satisfied by something else | Asserts "any finding" where it means "this finding" | Passes on a neighbour's output, silent when the subject breaks |
| Fixture coincides with the fallback | Expected value equals what the code returns when it does nothing | Both sides read the same for two different reasons |
| Guard in the wrong place | Validation runs after the thing it guards | The check is visible in review and does nothing |

## When to Use

- After writing any test whose failure matters — in the same task, not later
- Before writing "pinned by a test" in a report, a commit message or a task row
- On any anomaly (`AN-*`, `RS-*`, `PS-*`) task — step 7 of `docs/aiops/aiops-task-protocol.md` requires it
- When a suite is green after a change you expected to be risky
- When a test has never been observed failing and its subject has changed since

## When Not to Use

- To explore what a mutation does. **A mutation with no predicted victim is a fishing trip**, and a green
  result from one tells you nothing you can act on
- On a tree another agent or process is editing — a compile error may not be yours
- For coverage questions (use `overfit-coverage-analysis`) or assertion shape (`overfit-assertion-quality`)
- As a substitute for reading the code path. Mutation shows a test notices, not that the behaviour is right
- When the question is "where should I look" rather than "does this hold" — `overfit-test-gap-analysis`
  reasons about mutation points without paying for a build. Use it to rank targets, then mutate the survivors

## Inputs

| Input | Required | Description |
|---|---|---|
| Target file | Yes | The production `.cs` file whose behaviour will be broken |
| Anchor | Yes | Exact source text to replace, including indentation. Must occur **exactly once** |
| Mutated text | Yes | The replacement. Must compile — an invalid mutation is a harness bug, not a finding |
| Expected victim | Yes | The **named** test that must go red. "Some test" is not a prediction |
| Filter | Yes | `dotnet test --filter`, narrow enough to be fast and wide enough to contain the victim |

## Workflow

### Step 1: Predict the victim before writing anything

Name the test that must fail, and why. This is what makes a green result a finding rather than a shrug.

### Step 2: Choose what to break

One behaviour that matters, not one line per file. The best targets are the ones where a wrong answer is
silent: a gate that fails open instead of closed, a constant-time comparison replaced by `StartsWith`, a
counter not cleared on recovery, a missing-data path returning `Healthy` instead of `InsufficientData`, a
validator moved to after the allocation it guards.

### Step 3: Write the harness into the scratch file

`.claude/do.py`, or your own `do-<agent>.py`, with the five constants substituted. Execute it as the single
invocation `python D:/Overfit/.claude/do.py`.

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
    # Output goes to a FILE, never a pipe, and the run carries its own hang timeout.
    #
    # A mutation can HANG the suite rather than redden it — this repository's decode dispatcher waits on
    # an untimed spin inside a lock, so a protocol mutated to never exhaust spins for ever. With
    # `capture_output=True` that is unrecoverable: the child exits, a GRANDCHILD keeps the pipe open, and
    # `communicate(timeout=...)` blocks straight past its own timeout. Measured 2026-08-14 — a harness sat
    # 30+ minutes, the `finally` that restores the mutated file was never reached, and an orphaned
    # DevOnBike.Overfit.Tests.exe burned 2268 CPU-seconds and then failed every later build with
    # OVERFITMEASURING, a message about a concurrent measurement that did not exist. The same shape cost
    # the main session a whole release-gate run the same day.
    #
    # A file handle cannot be held open by a grandchild in a way that blocks the parent, and
    # `--blame-hang` turns a hang into a REPORTED RESULT instead of a stuck process.
    log = REPO / "Tests" / "bin" / "mutate-arm.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    with log.open("wb") as sink:
        proc = subprocess.run(
            ["dotnet", "test", str(REPO / "Tests" / "Tests.csproj"), "-c", "Release", "--nologo",
             "--filter", FILTER,
             "--blame-hang", "--blame-hang-timeout", "5m"],
            stdout=sink, stderr=subprocess.STDOUT)

    text = log.read_bytes().decode("utf-8", errors="replace")

    return (proc.returncode,
            sorted({n.split(".")[-1] for n in
                    re.findall(r"(?:Niepowodzenie|Failed) (DevOnBike\S+)", text)}),
            [l.strip() for l in text.splitlines() if ": error " in l])


dirty = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", str(TARGET)])
print(f"--- target clean against HEAD: {dirty.returncode == 0}")

# Read BYTES and match against LF-normalised text. This tree is CRLF; an anchor pasted
# with "\n" matches 0 times and the run stops — six arms in a row cost a full harness
# cycle on 2026-08-12 to exactly that. Writes restore the file's original ending.
RAW = TARGET.read_bytes()
CRLF = b"\r\n" in RAW
original = RAW.decode("utf-8").replace("\r\n", "\n")


def write(text):
    TARGET.write_bytes((text.replace("\n", "\r\n") if CRLF else text).encode("utf-8"))


count = original.count(ANCHOR)
print(f"--- anchor matches {count} time(s)" + ("" if count == 1 else "   <-- MUST be exactly 1"))

if count != 1:
    sys.exit("STOPPED: an ambiguous anchor mutates whichever site the replace happens to reach")

code, failed, _ = run()
print(f"--- baseline: rc={code} failed={failed or 'none'}")

if code != 0:
    sys.exit("STOPPED: baseline is red, so the mutation would say nothing")

try:
    write(original.replace(ANCHOR, MUTATED))
    code, failed, errors = run()

    if errors:
        print(f"--- DID NOT COMPILE — the mutation is invalid, not the test: {errors[0][:150]}")
    else:
        print(f"--- under mutation: rc={code} failed={failed or 'NONE'}")
        # startswith, NOT `EXPECTED in failed`. A [Theory] comes back as
        # `MyTest(arg: 3)` and the runner truncates at the first space, so exact
        # membership reports a CAUGHT mutation as survived — an error whose
        # direction MANUFACTURES findings. Measured 2026-08-12: two false
        # "survived" lines against tests that had gone red.
        caught = any(f.startswith(EXPECTED) for f in failed)
        print(f"    -> {'CAUGHT by ' + EXPECTED if caught else 'NOT CAUGHT'}")
finally:
    write(original)
    # Compared as BYTES against what was read, so a silent line-ending conversion
    # cannot pass as a restore.
    print(f"--- restored byte-for-byte: {TARGET.read_bytes() == RAW}")
```

### Step 4: Read the result

| Outcome | Means |
|---|---|
| **Red, on the test you named** | The test can fail, for the reason you think. The only outcome that licenses *pinned by a test* |
| **Red, on a different test** | Say which. It may be a better guard than the one you were checking, or the mutation changed more than you intended |
| **Red, on several** | Often informative — a traversal mutation on `NR-4` reddened five, including both cycle tests, which showed those produce clean failures rather than hangs |
| **GREEN** | **A finding, not a setback.** The behaviour is not covered. Report it; do not touch the harness until you understand why nothing noticed |
| **Did not compile** | Your mutation is invalid. Not evidence about the test |

### Step 5: Report it

The mutation result is a required section of an anomaly-task report and belongs in the task row: the
mutation, the victim, and whether the restore was verified.

## Validation

- [ ] The victim was named **before** the run, not chosen from the output
- [ ] The anchor matched exactly once, and the count was printed
- [ ] The baseline was green before the mutation was applied
- [ ] A compile error was distinguished from "not caught"
- [ ] The restore was **verified** byte-for-byte, not merely attempted
- [ ] A green mutation was reported as a finding rather than retried until it went red

## Common Pitfalls

| Pitfall | Solution |
|---|---|
| A harness killed mid-run leaves the source mutated | Guard 1 refuses to start when the target differs from `HEAD`. Without it the next run treats the mutated file as its baseline and reports *restore verified* |
| Anchor matches twice | On 2026-08-10 an anchor matched twice because `RunCustomTrend` carries a line byte-identical to `RunTrend`'s. Print the count and stop |
| Anchor matches zero times | A multi-line anchor that does not account for CRLF. It reads as "not caught" if the count is not printed |
| Red baseline | Cannot distinguish "the test caught it" from "it was already failing" |
| Treating a compile failure as a caught mutation | By exit code alone they are identical. Filter for `: error ` separately |
| **A `[Theory]` victim reported as survived when it went red** | The runner names a theory `MyTest(arg: 3)` and truncates at the first space, so `EXPECTED in failed` — exact membership — misses it. Match with `startswith`. **This error's direction manufactures findings**: it turns a caught mutation into a "green mutation", which this skill tells you to treat as a defect, so the harness invents work that does not exist. Measured 2026-08-12, two false survivals in one run |
| A guard that cannot apply, skipped instead of replaced | Guard 1 ("clean against `HEAD`") is vacuous when the mutated code is the task's own uncommitted work — the target differs from `HEAD` by design. Do not drop the guard: snapshot the bytes at the start of the run and compare against that instead, and **say in the report that you substituted it** |
| Assuming a broken guard hangs rather than reds | It can go either way. A cycle-detection guard was expected to hang under mutation and produced a clean red — check, do not assume |
| Mutating until something goes red | That is fitting the harness to the answer. One prediction, one run, one verdict |
| Calling a test weak because its assertion looks thin | Shape is not failability. Five `Assert.NotNull`-only tests in `CheckpointedModuleSegmentWalkTests` were reddened by mutation, and `overfit-assertion-quality`'s rubric calls them trivial. **This skill owns that verdict** |

**When you re-run a matrix after adding tests: a victim set that WIDENS is expected, one that SHRINKS is
the finding.** Adding a test that calls the mutated function widens its victim set by construction —
measured 2026-08-14, six of nine arms widened and nothing was wrong. A set that shrinks means a new
test masked an old one. Report widenings with their cause, and say which arms still isolate a single
test, because only those carry evidence about one behaviour.
