# program.md — automated calibration of the synthetic cluster

The analogue of `program.md` in [karpathy/autoresearch](https://github.com/karpathy/autoresearch): the
human-authored contract that says what the loop may change, what it is scored on, and what it must never
touch. The loop itself is the easy part. This file is the part that decides whether its output is worth
anything.

## The problem being solved

`SyntheticCluster` generates a healthy Kubernetes deployment with no fault injected. Every false-positive
number this project quotes is measured against it, so the guard's thresholds are only as trustworthy as the
generator's realism.

That realism was tuned by hand: eight constants, six iterations, each one "change a number, run the
comparison, read a table, keep or revert". That is a hill-climb performed by a human at roughly one step per
five minutes. The machine can do the same step in under a second.

## What may be changed

**One file: `Tests/TestSupport/SyntheticCluster.cs`, and within it only the scatter parameters** — now
gathered into `SyntheticClusterShape` so the search does not have to edit source at all.

| Parameter | What it controls |
|---|---|
| `LatencyScatterP50/P95/P99` | per-scrape spread of each latency quantile, independently |
| `TrafficScatter` | per-scrape spread of the per-pod request rate |
| `CpuScatter` | CPU spread beyond what traffic already explains |
| `BurstProbability` | how often a scrape lands on a burst |
| `TrafficBurstFactor`, `CpuBurstFactor` | how far a burst lifts the value |
| `HeapPromotionStep` | size of the gen2 staircase step |

Everything else in the generator is **fixed**: the diurnal curve, the affine CPU cost model, the restart and
warm-up rates, the sawtooth, scrape gaps, the identically-zero counters, the absent CFS series. Those encode
structure that was measured or reasoned about, not fitted, and a search allowed to move them would silently
trade structure for score.

## What it is scored on

```
score = mean over (metric, statistic) of | ln(generator / lab) |
```

Lower is better; zero is exact agreement. Log-ratio so that "twice too large" and "half as large" cost the
same — a plain difference would let the loop buy a good score on the big-magnitude metrics and ignore the
rest.

**Metrics scored:** `CpuUsageRatio`, `MemoryWorkingSetBytes`, `LatencyP50Ms`, `LatencyP95Ms`,
`LatencyP99Ms`, `RequestsPerSecond`, `GcGen2HeapBytes`.

**Statistics scored:** within-pod interquartile spread, and within-pod range — both relative to the pod's
own median, both computed over the same window length on each side.

### What is deliberately NOT scored, and why it matters more than what is

**Between-pod spread is excluded.** It is `(max − min)` over the pods' medians — a range over three draws
on the lab side, which is one of the noisiest statistics available. The same unchanged generator moved that
number from 12% to 28% purely because an unrelated edit shifted the random stream. A loop scored on it would
spend every iteration chasing a coin flip and would report confident convergence.

This is the `val_bpb` lesson, restated: Karpathy's metric is vocabulary-size-independent **so that
architectural changes are compared fairly**. A metric that is not fair does not become fair by being
optimised harder — it becomes a specification of the wrong thing, pursued efficiently.

The consequence is that `LatencyOffsetWidth` and the other per-pod personality widths are **out of scope for
the loop**. They only move a column nobody is scoring, so the search would random-walk them. Resolving them
needs pooled per-pod deviations across several recordings, which is a separate piece of work.

## Gates the score sits behind

The reference must defend itself before anything optimises against it. `LabWindowValidator` rejects a
recorded window that shows any of the failures actually observed: phantom replicas left in Prometheus after
a scale-down, a replica that was scraped but never driven, a window reaching back before the load started, an
operating point too low (empty histograms, quantiles pinned to bucket edges) or too high (saturated node,
fault contrast gone), or a restart inside the window.

`LabWindowFixtureTests` runs that validator against the checked-in fixture on every `dotnet test`, plus one
reproduction of each failure to prove the validator is not vacuous.

**Without this the loop is worse than useless**, because it would fit the generator to a broken recording
quickly, repeatably, and with a falling score the whole way down.

## Budget and loop shape

One evaluation is a generator run over 16 seeds plus a profile comparison: well under a second, entirely
in-process, no rebuild. Karpathy's five-minute budget exists because training is slow; here the budget is
irrelevant and the search can be exhaustive rather than sampled.

Coordinate descent with a shrinking step, deterministic, no model in the loop.

## Where a model earns its place — and where it does not

**Not here.** For fitting nine bounded scalars against a cheap deterministic objective, an LLM is a worse
optimiser than coordinate descent and a more expensive one. Dressing this up as an agent would be cargo cult.

**The structural changes are the opposite case.** The three corrections that mattered most today were not
numbers:

- the three latency quantiles were one series scaled by a constant, which forces identical relative scatter;
  the lab shows 14% / 58% / 35%, and no multiplier can produce that shape
- a uniform draw has a range of exactly twice its interquartile spread, and the lab's CPU sits at 3.3x —
  fat tails, which needed a burst term rather than a wider uniform
- gen2 heap is a staircase with an interquartile spread of exactly zero, not a scaled copy of the working set

None of those is reachable by moving a constant. They came from reading a table and asking why a column was
impossible. **That is the part worth handing to a model**, with the numeric loop as the thing that finishes
the job afterwards.

So the division is: a model proposes structure and writes the code; coordinate descent fits the constants;
the validator decides whether the reference was real; and a human reads the negative results. Points three
and four are not overhead — they are the reason the first two produce something rather than nothing.

## Rules the loop inherits from the repository

- Never `git commit`, `git push`, or any mutating `gh` command. The loop leaves a dirty tree and reports.
- `dotnet test` in `Release` only.
- Every shell command goes through `.claude/do.py`.
- Negative results are recorded, not discarded. A parameter that turns out not to matter is a finding about
  the generator, and this project has already reverted more measured changes than it has kept.
