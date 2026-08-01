# `LabLoadDriver` — traffic for the lab, as a pod

Drives `LabWorkload` so the RED signals are not zero. It runs **inside the cluster** rather than from a
workstation, which is not a convenience: three earlier measurement runs were lost to workstation-side
artefacts, and one of them silently drove the wrong pod through a stale port-forward.

## Rate-paced, not fixed-concurrency

With N workers each waiting for a response, a replica that slows down receives **less** traffic — so the
degraded pod appeared idle and the fault hid itself. The driver holds a target requests-per-second
instead, which is what a real load balancer in front of a queue does.

Measured on twelve replicas: distribution even to within 2.5%, no idle pods.

## The daily curve is real time, not compressed

The demand curve has a **1440-minute period**, not a scaled-down one, because the ratio of the detector's
trend window to the signal's period *is itself the object of study* — a compressed day would make the
answer come out whatever the compression chose. This is what showed that a 240-minute window sits on the
slope of the curve and finds a real, meaningless drift in everything at once.

Load varies as drift, not per-request jitter, for the same reason: independent noise per request
averages out over an 80-sample window and would be invisible to exactly the statistics being tested.

## Targeting

Resolves the headless Service's DNS every 30 seconds, so replicas that appear or disappear are picked up
mid-run — which is what a scale-up experiment needs.

**The health probe reports and never decides.** An earlier one-shot probe at startup dropped healthy pods
for an entire run, including the degraded one it was there to observe. Cost: four runs. A component whose
job is to observe must not be able to veto.
