---
name: xc50-decode-claim-tests
description: XC-50 test evidence across three verification rounds — the ten-arm mutation matrix, why AC1's refusal has no independent oracle, the admissibility rule for concurrency tests, and my own coverage-arm mistake
metadata:
  type: project
---

`Sources/Main/Runtime/DecodeChunkClaim.cs`, `Tests/Core/Runtime/DecodeChunkClaimTests.cs`,
`Tests/Core/Runtime/DecodeChunkClaimConcurrencyTests.cs`. Three rounds, all EXECUTED, restores
verified byte-for-byte against a start-of-run snapshot (guard 1 substituted — the targets are the
task's own uncommitted work).

**Round 3 matrix (mine, 10 arms, every victim set predicted in writing first, all EXACT).**
AC1/AC2/AC3 = the three original tests; Z/O/T/G = zero-chunk, one-chunk, top-of-range, tag-collision;
C = concurrency; K = clamp.

| arm | victims |
|---|---|
| M1 publish keeps tag+index | AC1, AC2, O, T, G, C |
| M2 publish `count + 1` | AC2, Z, O, T, C |
| M3 tag `!=` → `<` | AC1, AC3 |
| M4 publish preserves index | **AC1 only** |
| M5 publish `Math.Max(count-1,0)` / M5b raw `count-1` | AC2, O, T, C (identical sets) |
| M6 bound `>=` → `>` | AC2, Z, O, T, C |
| M7 clamp removed | **K only** |
| M8 tag narrowed to 16 bits | **G only** |
| M9 count field narrowed to 15 bits | **T only** |
| M10 CAS → non-atomic read-modify-write | **C only** (total 9671 claims vs 4096) |

**No victim set shrank against round 1** — every round-1 member is still present. Widening is
structural: a new test that calls a mutated function joins that mutation's set by construction.
Shrinking would be the diagnostic signal (a new test masking an old one) and did not occur.

**AC1's refusal assertion has no independent oracle and cannot get one**: post-fix the defect state
is unrepresentable, so every mutation reaching it also reddens AC2 or AC3. Documented on the test.

**Concurrency-test admissibility rule (XC-50 plan §13.1), worth reusing**: *a concurrency test is
admissible iff every assertion holds under every legal schedule.* `TG-T12`/`TG-T13` are cautionary
because they assert an environment-produced quantity (time, CPU), not because they use threads.
Second question always: **when the property is violated, is the result red or hung?** Here: safety
violations are red (M10 proves it), liveness violations (hoisted `Volatile.Read`) **hang** — no
assertion can convert that, and the only real fix is a progress deadline in `ForDecode`'s completion
spin, filed as `XC-52`.

**My own error, round 2**: reported the lost-CAS retry branch "never taken by any test" from
branch-rate **0.833** measured under a three-test filter; the full-suite arm reads **1.000**. A
coverage figure without its arm is not evidence, and "unasserted" ≠ "uncovered" on a concurrent path.

Related: [[verifier-index]].
