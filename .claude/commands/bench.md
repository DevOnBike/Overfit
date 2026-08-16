---
description: Run a BenchmarkDotNet class and print its table, with the measurement traps pre-checked
argument-hint: <filter, e.g. *MannWhitney* or *ScratchBufferStrategy*>
allowed-tools: Write, Edit, Bash(python .claude/do-bench.py)
---

Run the benchmark(s) matching `$1` and report the result table.

Write this into `.claude\do-bench.py` (substituting the filter) and execute it with
`python D:/Overfit/.claude/do-bench.py`:

```python
"""Run one BenchmarkDotNet class and extract its table."""
import pathlib
import shutil
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

BENCH = r'D:\Overfit\Sources\Benchmark\Benchmarks.csproj'
FILTER = "REPLACE_ME"

result = subprocess.run(
    ["dotnet", "run", "-c", "Release", "--project", BENCH, "--", "--filter", FILTER],
    capture_output=True, text=True, encoding="utf-8", errors="replace")

out = result.stdout + result.stderr
print(f"exit={result.returncode}" + ("  <-- 2 means another benchmark is already running" if result.returncode == 2 else ""))

started = False
for line in out.splitlines():
    if line.startswith("| Method"):
        started = True
    if started and line.startswith("|"):
        print(line[:190])
        continue
    if started and line.strip():
        break

if "| Method" not in out:
    print(out[-2500:])

# BDN writes its artifacts into the repo root; they are build output, not source.
artifacts = pathlib.Path(r'D:\Overfit\BenchmarkDotNet.Artifacts')
if artifacts.exists():
    shutil.rmtree(artifacts)
    print("\n(removed generated BenchmarkDotNet.Artifacts)")
```

**Before reporting any ratio, check these — in this order.** Every one of them has already produced a
wrong conclusion in this repository:

1. **`StdDev` and `RatioSD`.** `LayerNormBenchmark` returns 6–13% spread; nothing under ~15% is
   detectable there. A ratio quoted without its spread is not a measurement.
2. **Is the benchmark on the right job?** The shared `BenchmarkConfig` pins `InvocationCount=1`, which
   fits multi-millisecond model runs and turns a microsecond routine into timer noise. Microbenchmarks
   use `[SimpleJob]`.
3. **Is the lever you are A/B-ing actually live?** Twice now both arms have executed identical code —
   a dead env flag (`OVERFIT_TILED_PREFILL` short-circuited by a `.repack` sidecar) and a bounds check
   the JIT had already hoisted out of both loops. **When the result is a flat 1.00, suspect this before
   concluding "no difference"**, and settle it with `--disasm --disasmDepth 1` or a path counter.
4. **Does the scaffolding outweigh the subject?** A float accumulator chain hides anything cheaper than
   its own latency. If the payload is more expensive than the thing being measured, the benchmark is
   measuring the payload.
5. **Is the machine steady?** Cross-process before/after drifts up to ~30% on this box. Interleave the
   arms in one process (ABAB, not all-A-then-all-B) and keep an untouched path in the run as a canary.

**One benchmark at a time is enforced in code** — `Sources/Benchmark/Program.cs` takes a `Global\` mutex
and a second process exits with code 2 rather than queueing. If you see exit 2, something else is
measuring; do not start a competing run, and do not build anything until it finishes.

A negative result is a result. If the change did not help, say so and record the number — this repo's
most valuable notes are its disproved hypotheses.
