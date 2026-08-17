---
name: decode-dispatcher-mutation-arms
description: Measured costs and outcomes of mutating OverfitParallel.ForDecode / DecodeChunkClaim, and the shape a client-level test needs before a straggler is observable at all.
metadata:
  type: reference
---

Measured 2026-08-14 on `XC-52` (a), Ryzen 9 9950X3D, decode pool size 10.

**Hang arms are cheap and safe if you pass the right flags.** Mutations that break the claim protocol
(remove `Interlocked.Decrement` from `ExecuteDecodeChunk`'s `finally`; make `DecodeChunkClaim.Publish`
preserve the incoming index; hoist the `Volatile.Read` out of `TryClaim`'s CAS loop) **hang, they do not
redden** — confirmed 3 of 3, exactly as `XC-52` §8 predicted. With
`--blame-hang --blame-hang-timeout 3m --blame-hang-dump-type mini --results-directory Tests/bin/...`
each arm costs **188 s**, prints `Catastrophic failure: Test process crashed with exit code -1`, writes a
~2 MB mini dump, and leaves **no orphan `testhost`** (checked by `tasklist` after every arm). Without
`--results-directory` the dumps land in the working tree.

**A dispatch whose chunk 0 throws is usually drained entirely by the CALLING thread**, so nothing about
workers is observable. The caller claims index 0 first and executes it; if the remaining chunks are light
it finishes all ten before a parked worker can wake. Measured against the M4 mutation (rethrow moved above
the completion spin): 250 000 arithmetic iterations per non-throwing chunk, one dispatch → **caught 2 of
3**; 1 000 000 iterations and three dispatches → **3 of 3**, with 2, 3 and 9 bodies still in flight at the
throw. If a decode test needs a worker to be involved, make the non-caller chunks slow enough that the
caller cannot drain the whole dispatch.

**The capability question comes first.** `_decodePoolSize = min(max(1, ProcessorCount - 1), 10)`; at 1 (a
2-vCPU box) `ForDecode` runs the body inline and a dispatcher test passes without dispatching. The
`internal` accessors `OverfitParallel.DecodePoolSize` / `DecodePoolEnabled` exist for exactly that gate —
`DecodeMaxWorkers` is a settable public property and does **not** track the resolved field.

**Added 2026-08-16 from `XC-52` (b), the concurrent soak.** Two results that change how a mutation arm on
this path must be read and how a soak must bound its own waits.

**The same mutation reddens or hangs depending on WHICH tests are in the filter.** M4 (rethrow hoisted
above the completion spin): sequential test alone → **red in 24 s**; soak alone → **hang, blame-hang killed
it at 3 min**; both in one filter → **hang, and the run named no test at all**. The soak's 600 000
dispatches drive `_decodeRemaining` negative within milliseconds, the completion spin stops terminating,
and `_decodeGate` is held forever, so every decode caller in the process blocks. So: **run a predicted-victim
arm with a filter containing only the predicted victim**, or the verdict is about the process, not the test.
M4 under the soak is also nondeterministic — one sample reddened at iteration 6 on the *rethrow-before-Error-
is-captured* assertion, another hung.

**Bound NO PROGRESS, never total duration.** A whole-soak join deadline has to be set against how slow a
loaded box can make a legitimate 36 s soak, which forces it to ~10 min — and at that value the runner's
`--blame-hang` (5 m in the mutate skill) always fires first, so the test's own named message is unreachable.
Polling a per-thread dispatch counter and failing after **60 s with no advance** is bounded against a single
dispatch (~60 µs, six orders clear) instead, cannot false-red on load because a slow box still advances, and
**converts M5 and M4 from a nameless hang into a named red in 60 s with no hang dump** — where `XC-52` §8
predicted M5 would hang with no victim at all.

Related: [[reference-navigator-index-stale-after-move]] (line numbers in the navigator lag the file),
[[reference-test-output-and-anchors]] (CRLF anchors).
