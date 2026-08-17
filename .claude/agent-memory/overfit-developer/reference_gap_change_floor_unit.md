---
name: gap-change-floor-unit
description: MinAbsoluteGapChange is compared against |Theil-Sen slope| x span of the RETAINED gap ring (8 h at the shipped PeerNovelty profiles), not against one cycle or the guard's 20-minute metric window
metadata:
  type: reference
---

`PeerNoveltyOptions`'s change floor reaches `TrendDetector` as `TrendOptions.MinAbsoluteChangeOverWindow`
(`PeerNoveltyTracker.Classify` builds the options). The quantity it gates is computed at
`Sources/Main/Statistics/TrendDetector.cs:219-220`:

```
var windowSeconds = times[count - 1] - times[0];
var fittedChange = Math.Abs(slope) * windowSeconds;
```

So the unit is **change in the gap across the span of the retained ring**, and that span is
`RetainedCyclesPerSeries x cycle cadence` — 96 x 5 min = **8 hours** for `PerShift`/`Daily`/`Weekly`.
Verified 2026-08-10 by reading the arithmetic.

Consequences when calibrating it:

- A floor fitted over a shorter stretch answers a different question. For a genuine drift the fitted change
  scales with the span (short window understates the floor); for pure wobble it shrinks roughly as
  `sqrt(cadence/T)` (short window overstates). Which way the error goes depends on the regime, so a short
  window cannot be corrected into a long one.
- The relative gate runs alongside it: `MinRelativeChangeOverWindow` (0.10 in `Balanced`) times the gap's own
  median. On a ~10 MB standing gap that is already ~1 MB, so an absolute floor below ~1 MB never binds.
- The tracker only folds gaps for pods where `findings[i].IsOutlier` (`AnomalyGuard.Classify`), so the
  population the floor governs is *reported outliers*, not every replica.

See [[lab-prometheus-measurement-environment]] for why a clean 8 h window was not available on 2026-08-10.
