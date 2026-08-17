---
name: verifier-index
description: Index of what this agent has verified and the harness facts it re-learns each run (suite size/duration, machine mutex, the pipe hang)
metadata:
  type: reference
---

**Only entries verified on the date shown. The oracle map, the tolerance table and the
`[LongFact]`/fixture-gated inventory are NOT built yet — do not assume absence means absence.**

## Suite facts (2026-08-14, Release, this box)

- `dotnet test ./Tests/Tests.csproj -c Release --nologo` = **2631 passed / 0 failed / 274 skipped,
  2905 total, 23 s test time, ~34 s wall**. No flake seen in two full runs this day.
- The build refuses to start while a test run holds `Global\DevOnBike.Overfit.MachineMeasurement`
  (`Directory.Build.targets`, error `OVERFITMEASURING`). **An orphaned
  `DevOnBike.Overfit.Tests.exe` keeps holding it** and then every later build fails with a message
  about a concurrent measurement that is not one. Find it with
  `Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'Overfit|Tests' }`.
- **Never pipe `dotnet test` output into `subprocess.run(capture_output=True)` for a run that may
  hang.** Observed 2026-08-14: the child went away, a grandchild kept the pipe open, and
  `communicate(timeout=…)` blocked past its timeout — the harness sat 30 min with the target file
  still mutated. Redirect to a file handle instead; `finally` then restores on time.
- **Quote a coverage figure with the ARM that produced it.** `DecodeChunkClaim` branch rate is
  **0.833 under a three-test filter and 1.000 under the full suite** (both measured 2026-08-14 by
  me): a branch reached only by a race reads as covered when the whole concurrent suite runs. On a
  concurrent path "uncovered" and "unasserted" are different claims and only the second was true.
- `coverlet.runsettings` excludes `…Ops|Kernels|Maths|Intrinsics|Autograd|Optimizers|Tensors|
  LanguageModels.Runtime`. **`DevOnBike.Overfit.Runtime` is NOT excluded** — one word apart from
  `LanguageModels.Runtime`, and code in `Sources/Main/Runtime/` IS measurable.

- **A hanging test has no backstop in CI.** `.github/workflows/ci.yml:52,61` run `dotnet test` with
  no `--blame-hang` and the jobs declare no `timeout-minutes`, so a livelock burns GitHub's 6-hour
  default and reports no test name. Verified 2026-08-14 by reading the workflow. The mutate skill's
  harness passes `--blame-hang`; an ordinary run does not.

## Verified subsystems

- **Decode-pool claim protocol (`XC-50`), 2026-08-14** — see [[xc50-decode-claim-tests]]. Mutation
  matrix executed; one green mutation found.
