---
name: self-consistency-mutation-blind
description: A self-consistency assertion is blind to any mutation that moves the subject and its oracle through the same local; check the anchor feeds only one side before predicting a victim.
metadata:
  type: reference
---

A test of the form "projecting X reproduces Y" cannot detect a change that moves X and Y together.
Before predicting such a test as a mutation's victim, **read whether the anchor feeds one side or both**.

Measured 2026-08-15 on `XC-55`. `CachedGptStack.PrefillBatchedQuant` reads

```csharp
var lastRow = hidden.Span.Slice((rows - 1) * DModel, DModel);
lastRow.CopyTo(_lastFinalHidden);      // what LastHiddenState exposes
FinalNorm(lastRow, weights, _finalHidden);   // what the logits come from
```

The signed plan specified mutating `rows - 1` -> `0` to pin the exposed row index. It does not:
one local feeds both, so `LogitLens(LastHiddenState)` still equalled `LastLogits` exactly
(`max|logit - lens|` = 0, argmax merely shifted 17 -> 369) and the test stayed **green**. The mutation
that matches the intent desynchronises them — copy row 0 into `_lastFinalHidden` while `FinalNorm`
keeps `lastRow` — and it reddens that one test alone at `max|logit - lens| = 31.7`.

Run **both** arms and report both. The literal mutation is still worth running: it was caught by
`BatchedPrefillParityTests.BatchedPrefill_MatchesSingleToken_OnRealQwen` (`maxAbsLogitDiff` 16.31),
so the answer is "pinned, but by a different test than the plan believed" rather than "unpinned".

Related: [[reference_packed_word_sign_extension]] — the other shape where a predicted victim stays
green for a mechanical reason rather than a coverage one.
