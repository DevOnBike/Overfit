---
name: coverage-runsettings-scope
description: coverlet.runsettings excludes LanguageModels.Runtime.* but NOT DevOnBike.Overfit.Runtime.* — code in the top-level Runtime namespace IS measurable, so "excluded, cannot measure" is a wrong excuse there
metadata:
  type: reference
---

**`coverlet.runsettings`'s `<Exclude>` names eight namespaces, and one of them is easy to misread.** Read
verbatim on 2026-08-14:

```
[DevOnBike.Overfit]DevOnBike.Overfit.Ops.*, ...Kernels.*, ...Maths.*, ...Intrinsics.*,
...Autograd.*, ...Optimizers.*, ...Tensors.*, ...LanguageModels.Runtime.*
```

The last entry is **`LanguageModels.Runtime`**, not `Runtime`. So `DevOnBike.Overfit.Runtime` —
`OverfitParallel`, `DecodeChunkClaim`, `PooledBuffer` and the rest of that namespace — **is instrumented and
does produce a figure**. Saying "this is in an excluded namespace, cover it with named tests instead" is
wrong for it, and the developer role's own instructions list "Runtime" in a way that invites exactly that
mistake.

Measured for `XC-50` (full fast suite, `--settings coverlet.runsettings`): `DecodeChunkClaim` **28/28 lines
= 100%**, `OverfitParallel` **312/364 = 85.7%**. The instrumented full suite took **1 m 27 s** against 24 s
uninstrumented — perfectly affordable, because the runsettings already excludes the hot loops that cost
10x-900x.

Parse the figure out of `coverage.cobertura.xml` per `<class>` rather than reading the assembly total; the
per-class number is the one the 80%-on-new-code rule actually means.

**Line coverage still does not cover a concurrency branch.** `DecodeChunkClaim.TryClaim`'s lost-CAS retry is
100% "covered" by line count and is never *taken* by any single-threaded test. Say so rather than letting the
number imply it.
