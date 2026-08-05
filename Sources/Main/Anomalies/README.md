# `Anomalies` — deciding whether a running system is in trouble

This is the AIOps subsystem: it reads metrics off a live cluster, decides which replicas are
misbehaving, groups those decisions into incidents an operator can act on, and remembers them across
restarts. It is the one part of Overfit whose subject is a *deployment* rather than a model.

## Four families, four different questions

The single most important thing to know before editing here is that the detectors are **not**
alternatives to one another. Each answers a question the others cannot, and adding a signal to the
wrong family produces confident nonsense.

| Family | Directory | Question it answers |
|---|---|---|
| Rank statistics | `../Statistics` | *Is this replica unlike its peers, or moving?* |
| Absolute rules | `Rules` | *Has this crossed a level that is bad regardless of peers?* |
| Grouping and lifecycle | `Incidents` | *Are these twelve findings one problem or twelve?* |
| Learned | `Gpt`, `Baseline`, `Neuro`, `Adaptive` | *Does this look like the failures we trained on?* |

The rank family is blind to a fault every replica shares — if all twelve leak, none is an outlier. That
is what `Rules` exists for. The rules family is blind to anything below its threshold. Neither can say
whether a burst of findings is one incident, which is `Incidents`. The learned family needs training
data nobody has on day one at a customer site, which is why the deployed guard runs on the first three.

## The deployed path

`Incidents/AnomalyGuard` is the entry point: one `RunCycle(window, now)` per cadence, and everything
above happens inside it. Around it, `Monitoring` fetches the window from Prometheus and resolves pod
topology from kube-state-metrics; `Contracts` holds every option and result type; the host loop lives
in `Sources/Server.AspNet/Services/AnomalyGuardService.cs`.

## Every threshold here came from a measurement

No number in this subsystem is from the literature, and several contradict it. The two that cost the
most to learn:

- **Cliff's delta is scale-free**, so a 3% spread across healthy replicas produced deltas of 0.52 and
  0.68 against a 0.33 threshold. Relative and absolute gates both exist because neither is sufficient:
  a percentage gate is defenceless on a signal whose magnitude is small (a 3 MB heap moving 2 MB is a
  66% change and nothing at all), and an absolute gate has nobody to set it.
- **A longer window is not a safer window.** Swept on a healthy synthetic population, 20 minutes gave
  234 false incidents a day, 60 minutes gave 93, and 240 minutes gave **2583** — a four-hour window
  sits on the slope of the daily traffic curve, so the trend family finds a real, meaningless drift in
  everything simultaneously.

`Monitoring/FloorCalibrator` is the answer to "what do I put in the absolute floors": it watches a
period believed healthy and proposes floors from it. Measured on a held-out population, its proposal
beat the hand-reasoned floors — 29 false incidents a day against 44 — at identical detection. Its one
failure mode is stated in its own docs and is not hypothetical: a signal whose *unit* is already the
thing you care about, such as a restart count, must never be fitted from data.

`docs/aiops/aiops-detection-pipeline.md` walks the whole pipeline end to end.
