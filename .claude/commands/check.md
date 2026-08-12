---
description: Full gate — build the solution with CI flags, then run the whole test suite
argument-hint: (no arguments)
allowed-tools: Write, Edit, Bash(python .claude/do-check.py), Bash(python Scripts/check_agent_definitions.py), Bash(python Scripts/plan_gate_check.py)
---

Run the standard gate for this repository: build everything, then run every test.

Write this into `.claude\do-check.py` and execute it with
`python .claude/do-check.py`. Do not run `dotnet` directly — every command in this repo goes
through `do.py`.

```python
"""Full gate: solution build with CI flags, then the whole test suite."""
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

SLN = r'D:\Overfit\Overfit.sln'
TESTS = r'D:\Overfit\Tests\Tests.csproj'


def run(a):
    return subprocess.run(a, capture_output=True, text=True, encoding="utf-8", errors="replace")


print("=== build ===")
b = run(["dotnet", "build", SLN, "-c", "Release", "--no-incremental", "--nologo"])
errors = sorted({l.strip() for l in (b.stdout + b.stderr).splitlines() if ": error " in l})
print(f"  rc={b.returncode}  errors={len(errors)}")
for e in errors[:15]:
    print("  " + e[:200])
if b.returncode != 0:
    sys.exit(1)

print("\n=== tests ===")
t = run(["dotnet", "test", TESTS, "-c", "Release", "--nologo"])
for line in (t.stdout + t.stderr).splitlines():
    if re.search(r"(owodzenie!|iepowodzenie!|Failed!|Passed!)", line):
        print("  " + line.strip()[:200])
print(f"  rc={t.returncode}")
```

Non-negotiables baked into the script above — do not "simplify" them away:

- **`-c Release` always.** Debug numbers and Debug behaviour are not this project's contract.
- **`--no-incremental`.** An incremental build in this repo has reported `rc=0` without compiling
  anything, which turns a broken change into a green light.
- **No `-p:TreatWarningsAsErrors=true` without `-p:GenerateDocumentationFile=false`.** Omitting the
  second flag produces ~15 phantom XML-doc errors that have nothing to do with the change under test.

If a test fails, re-run with `--logger "console;verbosity=detailed"` and report the **test name** before
diagnosing anything. A bare exit code is not a finding — this repo has twice lost a real failure because
the output filter kept only the summary line.

## Then the two checks the compiler cannot make

Run both and report them beside the suite numbers. They are seconds each and they cover the two places
where this repository has been broken **without anything turning red**:

```
python Scripts/check_agent_definitions.py
python Scripts/plan_gate_check.py
```

- **`check_agent_definitions.py`** — every `.claude/agents/*.md` must still parse as an agent. On
  2026-08-10 one 0x01 byte in a frontmatter silently unregistered `overfit-developer`, the only agent
  permitted to modify source, for two days: the file was on disk, `CLAUDE.md` described eleven agents, the
  session offered ten, and no build or test could see any of it. The script self-tests first, so a clean
  run means the checker still works rather than that it checked nothing.
- **`plan_gate_check.py`** — plans that stopped between gates, and gate manifests that are missing,
  unreadable, or say `NOT_REQUIRED` without a reason. **A non-zero exit here is not a build failure**; it
  is a list of process debts. Report it, do not "fix" it by editing a plan's status line.
