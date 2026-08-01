# The silence review

A review method for code whose product is **the absence of an alarm**, and the evidence that produced it.

Run it with the `overfit-silence-hunter` agent, or by hand with the checklist below.

---

## Why it exists

On 2026-08-01 the anomaly guard passed **1959 tests** and four hours on a live twelve-replica cluster. A
read-through afterwards found **eleven defects**, and **nine were the same shape**: a failure that produces no
signal, so a broken observer and a healthy system emit identical output — nothing.

That is not bad luck. It is the characteristic defect of observability code. Everywhere else a bug surfaces
as a wrong answer; here it surfaces as no answer, which is also what success looks like. Tests do not catch
it because a test asserts what happens; these defects are about what *fails to* happen. Measurement does not
catch it because the measurement runs on a healthy system, which is exactly the condition under which the
broken path is never taken.

The same day supplied the counter-evidence for why measurement is still indispensable: three separate changes
were **wrong in ways reading did not reveal** and only a measurement exposed them — a step detector gated on
the wrong floor, a seasonal reference that cancelled level instead of slope, and a floor computation repeated
39 times a cycle. Reading and measuring find different bug populations. Neither substitutes for the other.

---

## The question

> **When this fails, what happens — and how would anyone ever find out?**

If the second half has no answer, it is a finding. Report it regardless of probability: this class of defect
is discovered *during* the incident it was meant to catch.

---

## The checklist

Each item lists the shape and the real instance that produced it.

| # | Shape | Found in practice |
|---|---|---|
| 1 | A caught exception that only sets a field or logs at debug | `FileIncidentStore` recorded every write failure into `LastError`; the only reader was a unit test. A read-only volume would discard state for hours in silence, and the operator would learn of it as duplicate pages after a restart. |
| 2 | A throw on a bound, caught by a loop that skips the iteration | Exceeding 1024 findings threw; the host logged "cycle failed; skipping and continuing". A cluster-wide fault — the event the guard exists for — therefore silences it, and the log reads like a transient network hiccup. |
| 3 | A default that means "off" where absent should mean "unknown" | An unset workload name made every maintenance window scoped to a workload unmatchable, and collapsed all deployment-level findings into a single incident identity. |
| 4 | A statistic used outside the regime its own docs state | A detector read a p-value directly instead of through the gate enforcing the sample floor. A shipped preset then put it at seven observations per half, below the eight its dependency documents as the point where the approximation stops being trustworthy. |
| 5 | Accumulated state that only updates on one branch | The seasonal baseline learned only while common-mode decomposition was enabled — one option silently switching off an unrelated subsystem. |
| 6 | A snapshot used without a freshness check | A pod roster retained after a failed refresh, so a pod deleted during the outage was accused of having gone silent. |
| 7 | A comment asserting behaviour the code lacks | A claim that a signal class "drives how the grouper relates findings", in a grouper that never reads signal classes. |
| 8 | Truncation without a count | Restore stopping at a maximum, returning a count indistinguishable from "there were only that many" — and, in the same method, an identifier counter advanced only for the records that survived the filter. |

---

## What counts as a finding

Wrong behaviour, silent blindness, a leak, a crash, or a claim the code does not back.

**Not**: style preferences, documented limitations, hypotheticals. Inflating the list destroys its value —
every entry has to be worth acting on. When two findings share a root cause, say so; two symptoms of one bug
is useful information and pretending otherwise is padding.

---

## Ranking what to fix

1. **Defects that silence the observer during the event it exists to detect.** Everything else degrades
   detection in a corner; these remove it exactly when it is needed.
2. **Defects that invalidate a measurement currently being taken.** A number produced under a known bias is
   a number that has to be caveated for ever, and re-running is usually cheaper than explaining.
3. **Defects that silently degrade a feature already shipped** — worse than one that never shipped, because
   somebody is relying on it.
4. Everything else.

Applied to the eleven from 2026-08-01, that ordering promoted the cycle-discarding bound and the unread
store error above defects that had looked more urgent when they were first written down.

---

## When to run it

- After a burst of changes to any subsystem whose job is to observe something.
- Before shipping a monitoring feature to a customer.
- After any incident where the tooling "saw nothing" — that is the symptom, and this is the differential
  diagnosis.
