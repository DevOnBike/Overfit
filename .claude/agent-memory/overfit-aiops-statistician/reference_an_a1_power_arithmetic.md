---
name: an-a1-power-arithmetic
description: Verified Poisson arithmetic behind the AN-A1 acceptance bar — the exact numbers, which approximation the registry used, and the events-not-days invariant
metadata:
  type: reference
---

Reproduced from scratch 2026-08-14 (exact Poisson, no scipy; script pattern in
`.claude/do-overfit-aiops-statistician.py`).

- Exact Garwood 95% CI for k=11: **[5.4912, 19.6820]** — the registry's [5.49, 19.68] is right to 4 dp.
- The registry's power numbers (0.96 / 2.7 / 12.3 / 52.5 days, and **3.09/day** inverted at T=1) reproduce
  exactly under the **sqrt variance-stabilising normal approximation**
  `T = (z_a+z_b)^2 / (4(sqrt(l0)-sqrt(l1))^2)`. That identifies the implementation.
- Exactly, against a 9/day bar in 1 day: critical region **k <= 3** (actual size 2.12%), 80%-power true
  rate **<= 2.297/day**. At 3.0862/day exact power is **62.8%**, not 80%. Spending the full 5.5%
  (`k <= 4`) gives exactly **3.090/day** — so the registry's number is the `k<=4` answer and is defensible
  only at a 5.5% test size.
- Bar **3/day**: T=1d needs k<=0 (l<=0.22), 2d k<=1 (0.41), 3d k<=3 (0.77), 7d k<=13 (1.54), 14d 1.94,
  30d 2.25. Bar **9/day**: 1d 2.30, 2d 4.08, 3d 5.09, 7d 6.28, 14d 7.08.
- **The invariant worth carrying**, independent of clock: demonstrating a true rate of bar/3 needs ~**3**
  expected events; bar/2 ~**10**; 2/3 bar ~**32**; 0.8 bar ~**113**. Days = events / true rate.

**Why:** these get re-derived every time the bar moves, and the sqrt-approximation trap (optimistic at
counts near 3) is invisible unless the exact version is run beside it.

**How to apply:** quote the exact number and say which test size it assumes. See
[[an-a1-occupancy-is-the-real-number]] for why the rate may be the wrong estimand in the first place.
