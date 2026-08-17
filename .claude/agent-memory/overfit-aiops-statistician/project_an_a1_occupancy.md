---
name: an-a1-occupancy-is-the-real-number
description: AN-A1's 24h log is on disk and shows 94% incident occupancy / 9 episodes — the rate (10.63/day) is the wrong estimand; also which offline artefacts exist for rate estimation
metadata:
  type: project
---

`Tests/bin/fp-run-guard-log-FINAL.txt` (gitignored `bin/`, present on this box) carries all 298 AN-A1
cycles as `cycle: pods= findings= incidents= opened= ongoing= resolved=`. Derived from it 2026-08-14:
**an incident was open for 280 of 298 cycles = 94.0% of the run**, from only **9 occupancy episodes**
(mean 156 min, longest 130 cycles = 10.8 h, off-runs averaging 10 min). 11 openings, max concurrent 1.
Header also records the config A1 actually ran: *"12 of 13 known features mapped, 0 custom"*.

**Why:** the registry argues about 10.63 incidents/day, but at 94% occupancy a rate cut to 3/day with
the same durations still leaves ~80% red. The rate is measured correctly and describes the wrong thing
(the `AN-D4b` shape). None of this is written in `docs/`; it only exists in that log.

**How to apply:** before any verdict on the acceptance criterion, re-derive occupancy from this log
rather than arguing about the rate. Two caveats found the same day: the other two `fp-run-guard-log-*`
files are the **same run truncated** (identical start 08:04:19.925399), so there is **no independent
replicate** and day-to-day dispersion is unmeasured; and `Tests/Anomalies/Diagnostics/LabFixtureRecorderDiagnostics.cs`
(the raw-window recorder that produced the lab CSV) loops `MetricIndex.Count` only, so it **cannot record
the 6 custom channels** the current config runs. See [[an-a1-power-arithmetic]].
