---
name: coverage-arm-matters-for-concurrent-branches
description: A branch reached only by a real race shows as partial under a filtered coverage run and 100% under the full suite — quote the arm, and never write "no test reaches it" from a filtered number.
metadata:
  type: project
---

Measured 2026-08-14 on `DevOnBike.Overfit.Runtime.DecodeChunkClaim` (XC-50), coverlet with
`coverlet.runsettings`:

| arm | line rate | branch rate |
|---|---|---|
| `--filter FullyQualifiedName~DecodeChunkClaim` (3 deterministic tests) | 28/28 | **0.833**, single partial branch = the `Interlocked.CompareExchange(...) == current` comparison, 50% (1/2) |
| full suite, 3 samples | 28/28 | **1.000** in 3 of 3 |

**Why:** the lost-CAS retry needs two threads on one word. No test arranges it, but the full suite drives
real `OverfitParallel.ForDecode` dispatches that lose the race, so the branch executes incidentally.

**Why it matters:** a verifier reported "the lost-CAS retry branch is never taken by any test" from the
filtered number. Written into a source comment that would have been a wrong docstring — the branch does
execute; what is true is that **nothing arranges it and no assertion depends on it**, so the coverage is a
timing artefact that can vanish on another box with nothing going red.

**How to apply:** when quoting a coverage figure for anything touching a concurrent path, say which arm
produced it, and prefer "unasserted" to "uncovered". `DevOnBike.Overfit.Runtime` IS measurable — see
[[coverage-runsettings-scope]]; only `LanguageModels.Runtime` is excluded.
