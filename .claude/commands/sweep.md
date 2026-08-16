---
description: List every site an OVERFIT analyzer rule flags, with its source line
argument-hint: <rule id, e.g. OVERFIT026>
allowed-tools: Write, Edit, Bash(python .claude/do-sweep.py)
---

Inventory every place rule `$1` fires across the solution, so the backlog can be judged before the rule
is armed.

Most OVERFIT rules sit at `suggestion` while their backlog is swept, and a plain build does not print
suggestions. So: raise the rule to `warning` in `.editorconfig`, build, collect, **restore the file in a
`finally`** — an interrupted run must not leave the severity changed.

Write this into `.claude\do-sweep.py` (substituting the rule id) and execute it:

```python
"""Inventory one OVERFIT rule across the solution, with the source line for each site."""
import pathlib
import re
import subprocess
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

RULE = "REPLACE_ME"
CONFIG = pathlib.Path(r'D:\Overfit\.editorconfig')
SLN = r'D:\Overfit\Overfit.sln'
PATTERN = re.compile(r'([A-Za-z]:\\[^(]+\.cs)\((\d+),\d+\): warning ' + RULE + r': (.+?)(?: \[|$)')

original = CONFIG.read_text(encoding="utf-8")
try:
    CONFIG.write_text(
        original.replace(f"dotnet_diagnostic.{RULE}.severity = suggestion",
                         f"dotnet_diagnostic.{RULE}.severity = warning"),
        encoding="utf-8")

    build = subprocess.run(["dotnet", "build", SLN, "-c", "Release", "--no-incremental", "--nologo"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(f"build rc={build.returncode}")
    for e in sorted({l.strip() for l in (build.stdout + build.stderr).splitlines() if ": error " in l})[:8]:
        print("  ERR " + e[:180])

    sites = {}
    for line in (build.stdout + build.stderr).splitlines():
        m = PATTERN.search(line.strip())
        if m:
            sites[(m.group(1), int(m.group(2)))] = m.group(3)
finally:
    CONFIG.write_text(original, encoding="utf-8")
    print("(.editorconfig restored)")

cache = {}
per_project = defaultdict(int)
for (path, num) in sites:
    if path not in cache:
        cache[path] = pathlib.Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    rel = path.replace("D:\\Overfit\\", "")
    per_project[rel.split("\\")[1] if rel.startswith(("Sources", "Demo")) else rel.split("\\")[0]] += 1

print(f"\n{RULE}: {len(sites)} site(s)   by project: {dict(per_project)}\n")
for (path, num), msg in sorted(sites.items()):
    rel = path.replace("D:\\Overfit\\", "")
    print(f"  {rel}:{num}\n      {cache[path][num - 1].strip()[:130]}\n      {msg[:120]}")
```

**Then triage each site — do not batch-suppress.** The house rule, from the user:

> *if we have an analyzer, let's use it — it should fail the build, and if the author is convinced that
> e.g. 4 KB is fine, they add a suppression.*

Which means, per site, in this order:

1. **Fix it** if the fix is mechanical and free — a `(long)` cast, a `PooledBuffer<T>` on a cold path.
2. **Bound it** if the input is unvalidated. A `stackalloc` fed by a public parameter is not a style
   issue, it is a way for a caller to kill the host process.
3. **`#pragma` it** only with a comment that **names the bound**: `BOUND: cols is validated to [1,16] by
   the throw above, so 2560 B`. Same contract as OVERFIT022/023. If you cannot write that sentence, the
   code is not safe and the pragma is a lie.

**Never widen a whole directory's budget** (`overfit_max_stackalloc_bytes` and friends) to clear a
backlog. A directory budget also waves through every future file nobody has looked at yet; a per-site
pragma leaves the rule armed.

**Expect false positives from a new rule.** Three of this repo's newest rules each had one on their first
full run (`checked(a*b)`, `<Main>$`, ASP.NET middleware carrying `HttpContext.RequestAborted`). Fix them
in the analyzer, not with a pragma — a suppression hides the same mistake everywhere else too.

Arm the rule as `error` only when the backlog is zero or every survivor carries a stated bound.
