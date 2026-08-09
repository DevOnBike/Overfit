# The cluster anomaly guard — should you run it

For an engineering manager or platform lead deciding whether to pilot this, and what it will cost. Every
number on this page names the document it came from; where there isn't a number, that's said too. Two
companion documents cover the other two questions a buyer's team will ask next: **how it works**
(`aiops-architecture.md`) and **what it can touch, see and store** (`aiops-architecture-security.md`). This
one only answers whether it's worth running and what it gives you.

---

## The problem, in plain terms

A cluster tells you everything and therefore nothing. Dashboards are something you read *after* the page
fires, not before. A static threshold — "CPU above 80%", "memory above 2 GB" — is either deaf, because it
was set for the wrong workload, or it's a pager that gets muted within a month, because it fires on
Tuesday's traffic curve as readily as on a real fault. Alert fatigue isn't a personality flaw of the on-call
rotation; it's what a fixed number does when the thing it's watching is never fixed.

What this does instead is compare a replica against its own siblings, right now, in the cluster's own units
— not against a number someone guessed six months ago. That comparison needs no history to be meaningful:
eleven replicas agreeing and one not agreeing is informative on its own, the first minute it's observed.
Nothing has to be labelled, nothing has to be trained on your data before it says anything useful, and
nothing about your traffic pattern, your workload shape, or your incident history has to be handed over in
advance — the only "training data" it ever uses is your own healthy traffic, watched after install.

## Day one is the strongest argument, and here's exactly what day one gets you

| When | What's available | Why |
|---|---|---|
| First cycle (five minutes in) | Comparing replicas against each other; hard-threshold rules (CPU throttling, OOM, restart storms) | Both are rank-based / absolute and need no history to be meaningful |
| Within the first ~20-minute window | A replica's own trend (is it climbing, and how fast) | Needs the window to fill, not training |
| Hours | Noise-floor proposals, in your cluster's own units, that a human still has to accept | It has actually observed your traffic by then |
| Two calendar days, minimum | Comparing a signal against the same hour yesterday | Structural: the guard keeps one observation per hour per day, so a single day literally cannot arm this — a 24-hour trial run can never reach it |
| Not on day one, not promised | The optional learned/model family (a small subset of signals) | This is the only piece that needs training data, and it's trained by us in advance, not by watching your cluster |

(Source: `docs/aiops/aiops-adding-a-metric.md`, Step 4, verified against the code; and
`docs/aiops/aiops-day-one-events.md`.)

The practical upshot for a pilot: **most of what this catches, it can catch from the first cycle.** The
pieces that need time need hours or two days, not weeks — and the one piece that genuinely needs training
data is shipped already trained, not built from your traffic.

**One honest wrinkle for day one specifically:** a fresh install has no "yesterday" to compare against, so
your own daily traffic curve can look like a change on its first pass through. Measured on the lab: two of
five false incidents on day one were exactly the same rising-then-falling traffic curve, counted twice.
Expect roughly one extra incident per rise-and-fall in your traffic on day one, and expect it to stop on day
two. (Source: `aiops-client-readiness.md`.)

## What it costs to run

**The dependency list is one line: an HTTP route to your existing Prometheus.** No agent on any node, no
sidecar, no Kubernetes API-server access, no RBAC beyond reading a Service, no CRDs, nothing installed
inside your application. It runs as one ordinary pod per monitored namespace, configured from a ConfigMap.
That claim is tested by deploying it that way, not asserted.

| | Measured |
|---|---|
| Memory, steady state | **130 MiB resident, 146 MiB peak** — over a 24-hour run, 292 cycles |
| Memory, first minutes | 42 MiB, but don't use this number — it's a guard that hasn't accumulated anything yet |
| CPU | ~1 thousandth of a core |
| Extra load on your Prometheus, per instance | +12 MiB memory on Prometheus, no measurable CPU change |
| Cadence | Evaluates a 20-minute window every 5 minutes |

(Source: `docs/aiops/aiops-client-readiness.md`, "What one instance costs, measured".)

**One instance watches one namespace and one pod selector.** A fleet is one instance per namespace, so
fifty namespaces is arithmetic from the one-instance number — **≈7.3 GB total memory and 50 thousandths of
a core**, plus roughly 850 queries every five minutes (~2.8/second) against your Prometheus. The
per-instance figure is measured; the fifty-namespace figure is arithmetic and should be checked against your
own Prometheus's headroom rather than assumed. (Same source.)

**A pod-ownership source (kube-state-metrics) is strongly recommended but not required.** Without it, there
is no pod roster and no pod age, which removes replica grouping, the check for a pod that's gone silent, and
the grace period for a freshly started pod — everything else still works. (Source:
`docs/aiops/aiops-client-flow.md`, Stage 0.)

**One prerequisite worth stating before a pilot starts, because it fails silently otherwise:** your
Prometheus scrape interval has to be short enough to put at least 30 samples in a 20-minute window. At a
60-second scrape interval that's only 20 samples, and every verdict comes back as "not enough data" —
which looks exactly like a healthy cluster from the outside. (Source: `aiops-client-readiness.md`.)

**A minimum of three replicas per monitored workload.** Below that there's no "rest of the group" to
compare against. (Source: `aiops-client-flow.md`, Stage 0.)

## What it detects — in scenarios, not metric names

Five ways something can go quietly wrong that a static threshold either misses or drowns in noise:

| What's happening | What catches it | Status |
|---|---|---|
| One replica is slower than its siblings, but not slow enough to trip a fixed threshold | Peer comparison, immediately | Measured on an injected-fault population, re-verified 2026-08-03: latency 3× on one replica detected in 0 minutes |
| A memory leak nobody is watching for | Peer + trend comparison | Same population: a 5 MB/min leak caught in 10 minutes, a 20 MB/min leak in 5 |
| A pod is being denied CPU it never asked for (throttled under a limit it's not exceeding) | The absolute-threshold rule family — the only family that can see this, because CPU-throttling accounting only exists on a container with a CPU limit | **Verified live on the cluster, 2026-08-09**: peers stayed at 0% throughout an 8-minute injected fault; the throttled pod rose from 71.6% to 100%; the guard fired on the right pod and accused no peer |
| Requests are piling up unfinished while every latency graph stays flat | In-flight request count, which moves before a request has finished and therefore before it can appear in any latency histogram | **Verified live, 2026-08-09**: during a stall the in-flight count moved 5→40 while p95 latency stayed at its healthy value of 48.75 ms — because a stuck request hasn't finished yet, so it can't yet be counted. **The in-flight signal led the eventual latency spike by two minutes.** |
| Exceptions are being thrown and caught (and retried) before they ever become an error response | A dedicated exceptions channel — error-rate metrics structurally cannot see this, because they count 5xx *responses*, and a caught exception never becomes one | **Verified live, 2026-08-09**: over 8 minutes with half of requests throwing-and-catching, exceptions rose from 0.37 to 0.99/second on the injected pod while peers, the 5xx rate and p95 latency **all stayed at zero or flat throughout** |

One more, because it's the sharpest illustration of why this needs more than one detection method: a real
pod OOM kill is, by construction, invisible to peer comparison — comparing a replica against its siblings
cannot see that replica die. **But the guard is not blind to it.** Verified live, 2026-08-08/09: a pod was
killed (`exit 137`) in under 5 seconds; the guard opened an incident led by a dedicated OOM-events channel
within one cycle, with all eleven other pods reading zero. The lesson generalises — a whole-fleet CPU rise
is likewise invisible to peer comparison (nobody looks different if everybody moves together) and is caught
instead by a level-shift detector; a single stuck-pod throttle is invisible to *that* and caught by the rule
family instead. **A guard with fewer detection methods is not a simpler guard, it's a blind one** — this is
the reasoning cited directly in `aiops-client-readiness.md`.

## What it deliberately does not do

- **No service-dependency graph.** It cannot tell you the root cause was a downstream database, only that
  something changed. The most common real root cause is a dependency, and this will not find it — that's a
  separate, harder product.
- **No cross-service tracing, no log storage, no new time-series database.** It reads Prometheus and writes
  incidents; Grafana and Alertmanager stay exactly what they are.
- **No auto-remediation.** It reports; a human (or your existing automation) decides.
- **One workload technology validated so far.** Everything measured above is on .NET workloads. Container-
  level signals (CPU, memory, restarts) are stack-independent and cover roughly half the monitored signals;
  the rest — how memory *should* behave, for instance — is stack-specific and has only been checked against
  one runtime.
- **One namespace per instance today.** A fleet is N instances, at the cost arithmetic above; several
  namespaces sharing one process is designed but deliberately not built, because the saving is mostly
  runtime overhead and the failure-isolation you get from separate instances is worth more.
- **Rollouts of a genuinely broken new version, a second workload technology, and storage-volume
  saturation are all untested, not merely unsupported** — every rollout exercised so far has been a healthy
  one, and the development cluster doesn't export volume-saturation metrics at all. (Source:
  `aiops-client-flow.md`, "What we will not know until we are there".)

## Where it stands honestly today

**Verified end to end, on a live cluster, this week (2026-08-08/09):** a real pod OOM kill detected within
one cycle; caught-and-retried exceptions detected while error rate, latency and throughput all stayed flat;
an in-flight-request pile-up detected two full minutes before any latency percentile moved; CPU throttling
of a single pod detected correctly with no peer wrongly accused.

**Not yet, and this is the part that decides how you should run it:**

- **The false-positive rate has not passed its own target.** A 24-hour run against a frozen configuration
  measured **10.63 incidents/day against a target band of 1–9/day** — it failed. And honestly, eleven events
  over one day is not enough evidence to judge a band like that at all: the statistical interval around
  eleven events is roughly 5.5 to 19 per day, which straddles both a pass and a fail. **The real limit is the
  event count, not the clock** — a second day of the same rate still wouldn't settle it, which is why the
  next step taken was replaying the historical data Prometheus already retains, to get hundreds of events
  instead of eleven, rather than simply waiting longer.
- **Turning on the learned seasonal-history feature made the guard three times noisier, and why is still
  unknown.** Replaying the same day cold (no learned history) gave 11 incidents; replaying it with the
  guard's normal accumulated history gave 33. Isolating the two halves of that learned state showed
  calibrated noise floors changed nothing — the seasonal comparison itself was the entire effect, and the
  investigation into why is still open.
- **One diagnosed false-positive source is real and unfixed.** A pod that has been consistently, stably
  heavier than its siblings since it started — not getting worse, just different — gets reported by the peer
  comparison every cycle, forever, because peer comparison currently has no concept of "that's just how this
  one always looks." Diagnosed 2026-08-06; independently corroborated on the same run's log shape (91% of
  cycles carried exactly one finding, which is not what a set of independent false alarms looks like); still
  open.

## Recommendation

**Run it in shadow mode first — reporting to a log and its own metrics endpoint, paging nobody — with the
seasonal-history feature switched off.** Everything that makes this safe to try (comparing replicas, the
absolute-threshold rules, trend detection) needs no history and works from the first cycle; the piece that's
currently adding unexplained noise is the one piece that isn't needed for any of the scenarios above. A
shadow week produces exactly the number this document cannot give you in advance: **the false-positive rate
for your own cluster.** That number does not exist for any cluster it hasn't watched, and cannot be
extrapolated from the lab's — it's a measurement your pilot produces, not one this document can promise.

**What would change this recommendation:**

1. The seasonal-history noise is explained and fixed, and re-measured — not merely disabled.
2. The one diagnosed false-positive mechanism (the "stable outlier" pod) is fixed, since it currently
   accounts for a large share of a single cluster's daily false-positive count on its own.
3. The false-positive band itself is re-measured against hundreds of events (via replay of historical data)
   rather than the eleven that produced the current failed result — so that "1–9 a day" is a number anyone
   can trust rather than one with an interval wide enough to contain its own opposite.

None of the three blocks a shadow pilot today. All three affect when it's reasonable to connect the guard to
a pager.

---

## See also

- `docs/aiops/aiops-architecture.md` — how it works, mechanism by mechanism.
- `docs/aiops/aiops-architecture-security.md` — permissions, data handling, supply chain.
- `docs/aiops/aiops-client-flow.md` — the engagement stage by stage: what a client hands over, what happens
  each week, and how each stage is known to have worked.
- `docs/aiops/aiops-client-readiness.md` — the full capability/blind-spot inventory this page draws its
  numbers from.
- `docs/aiops/aiops-day-one-events.md` — measured behaviour under a real rollout, scale-up, scale-down and
  HPA event.
