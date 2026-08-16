# `Anomalies/Rules` — the family that can see a fault everybody shares

Absolute thresholds, held apart from the rank statistics in `../../Statistics` because they answer a
question those structurally cannot.

## The blind spot this exists to cover

Peer comparison asks whether a replica is unlike its peers. If **all** of them are wrong together — a
bad rollout, a poisoned dependency, a node under memory pressure — there is no outlier and the rank
family reports nothing, confidently. Trend detection covers part of that case but not a fault that was
already there when the window opened.

`SustainedThresholdRule` is the answer: a level that is bad regardless of what anyone else is doing,
required to hold for a sustained fraction of the window rather than a single sample, so one scrape
spike is not an incident.

## The thresholds are measured, and the literature's are wrong here

The commonly quoted CPU-throttling threshold of 25% **never fires** on this workload — the measured
peak was 19.8%. The rule ships with 5% and 25% bands taken from what the lab actually produced. A
threshold that cannot fire is worse than no rule: it looks like coverage.

## Where the boundary sits

Anything that needs a comparison — against peers, against the pod's own past, against a seasonal
expectation — belongs in `../../Statistics`. Anything that is a statement about a number on its own
terms belongs here. The two run in the same cycle and their findings meet in `../Incidents`, which is
where the question "are these the same problem" gets answered.
