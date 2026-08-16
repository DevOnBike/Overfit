---
name: reference-measured-baselines
description: docs/measured-baselines.md is now the canonical, single location for every measured number in this repo — cite it, don't restate figures here.
metadata:
  type: reference
---

`docs/measured-baselines.md` (created before 2026-08-06, confirmed present and current on that date) is the
project's single place for measured numbers: reverted/regressed changes with why, shapes-and-call-costs
(declared-type-on-hot-path lever, `for`/`foreach` non-lever, `OverfitParallelFor` vs `Parallel.For` split by
call site), throughput/memory (Bielik 17 tok/s, Qwen-3B 24.4 tok/s with `OVERFIT_REPACK_GEMV`, llama.cpp gap
~1.13x uniform), anomaly guard false-positive history (112/day old build -> 5/day 2026-08-05, peer-blind-to-
one-OOMKill), and semantic navigator warm-query costs (3-23 ms after first-fault).

**Do not copy figures out of this file into memory.** Read it fresh each time a quality requirement needs
checking against a baseline — it is the pointer, not a cache, and it is maintained by the whole team so it
moves. Every row there already states what it was measured on (model, quantisation, build, box); an
architect memory entry restating a number without that provenance would just be a second, staler copy.
