---
name: packed-word-sign-extension
description: In a packed 64-bit word, a negative sub-field sign-extends over its neighbours — so an underflow mutation is refused by the WRONG check and the predicted victim stays green.
metadata:
  type: reference
---

`DecodeChunkClaim`'s word is `[tag:32][count:16][index:16]`. Mutating `Publish` to store
`chunkCount - 1` was predicted to redden `Publish_WithZeroChunks_YieldsNoClaim` (count 0 → -1 →
count field reads 0xFFFF → claims succeed). **It stayed green.** `((long)(-1) << 16)` is
`0xFFFF_FFFF_FFFF_0000`, so the sign bits smear over the *tag* field too; the claim is then refused by
the generation check, not by the bound, and the test passes for a reason it does not intend.

**The rule:** when predicting a mutation's victim on a packed word, work out what the mutated value does
to the NEIGHBOURING fields, not just its own. Signed shifts are the usual culprit.

The `>`-instead-of-`>=` mutation is the one that actually pins an off-by-one bound; it reddened the
zero-count, one-count, top-of-range, exact-count and contention tests, exactly as predicted. See
[[mutation-theory-test-names]] for the other way a victim set is misread.
