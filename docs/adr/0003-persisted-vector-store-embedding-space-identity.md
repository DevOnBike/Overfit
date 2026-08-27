# ADR 0003 — the persisted vector store carries the identity of the embedding space its vectors live in

- **Status**: proposed, 2026-08-27, by `overfit-architect`.
- **Plan**: [`../specs/xc-131-xc-123-embedding-postnorm-and-qwen3-embedding-plan.md`](../specs/xc-131-xc-123-embedding-postnorm-and-qwen3-embedding-plan.md)
- **Scope**: the on-disk format written by
  `Sources/Main/LanguageModels/Retrieval/PersistentVectorStore.cs` (`Save` at `:152`, `Load` at `:179`).

## Context

`XC-131` changes what `OverfitClient.Embed` returns. It pooled the **pre**-final-norm hidden state; it will
pool the **post**-norm state, which is what HuggingFace's `last_hidden_state` and llama.cpp's
`llama-embedding` mean. `output_norm.weight` spans -0.1196 to 15.3125, so these are different directions,
not a rounding difference.

Every vector any caller has stored is therefore invalidated. One caller stores them:
`Demo/LocalAgentAspNetDemo/Rag/RagService.cs` writes a `PersistentVectorStore` and reloads it on start.

**The reload validates four things and none of them moves.** `TryReloadFromCache` (`RagService.cs:206-254`)
checks the file magic and version, the `Dimension`, the `SourceCount`, and each source file's SHA-256 content
hash. The documents are unchanged, the hidden size is unchanged, the file count is unchanged. So a cache
written before the fix reloads cleanly after it, and every query cosine is then computed between a post-norm
query vector and pre-norm document vectors.

**Nothing throws and nothing logs.** Retrieval simply returns worse answers. That is the worst outcome
available here: a wrong result that is indistinguishable from a right one.

The format already has a version lever — `FileMagic = 0x3150_5350` and `FileVersion = 1` at `:25-26`, and
`Load` throws `OverfitFormatException` on a mismatch at `:193`. `RagService` already catches that, logs
*"Ignoring unreadable RAG index cache … rebuilding"*, and rebuilds (`:218-222`).

## Forces

- **The failure must become loud.** A silent cross-space comparison is the thing being removed.
- **A rebuild is affordable.** The demo re-embeds a document folder. This is not a multi-gigabyte migration.
- **The version bump alone fixes only this instance.** The next embedding change — a different model, a
  different pooling, an instruction prefix arriving on the query side — reproduces the whole incident and
  needs another bump. Each bump also invalidates caches written by *any* other embedder, including ones the
  change did not affect.
- **The store cannot compute the identity itself.** It receives `float[]` vectors and has no idea which model,
  pooling or prefix produced them. Only the caller knows.
- **`PersistentVectorStore` has two real consumers** — the demo and its own tests. This is a small surface,
  so a small mechanism is proportionate and a general one is not obviously worth building.

## Options considered

| option | catches this change | catches the next embedding change | cost |
|---|---|---|---|
| **A. Leave it** | no | no | none — and it is the defect |
| **B. Bump `FileVersion` 1 → 2 only** | yes, via the existing throw + rebuild | **no** — needs a bump every time, and each one invalidates unrelated caches | one constant |
| **C. Bump the version **and** store an `EmbeddingSpaceId` string, compared on load** | yes | yes | one constant, one string field, one comparison, one optional constructor parameter |
| **D. Store a structured descriptor** (model digest, pooling enum, prefix strings, quantise flag as separate fields) | yes | yes | a schema to maintain; every future embedder must populate fields it may not have |

## Decision

**Take option C. Bump `FileVersion` to 2 and add one `EmbeddingSpaceId` string to the header, supplied by the
caller and compared on load.**

- The version bump is what rescues caches **already on disk**: they cannot carry the new field, so they must
  be rejected, and the existing `OverfitFormatException` + rebuild path does that with no new mechanism.
- The space id is what rescues every **future** cache, and it is added in the same edit because the version
  must move anyway. One string is close to free; a second incident is not.
- **An empty id means "unknown" and must not match a non-empty one.** An unknown space is exactly the state
  this ADR exists to reject, and treating it as a wildcard would restore the silent path.
- A mismatch **throws** `OverfitFormatException`, the same way a version mismatch does. `RagService` already
  turns that into a logged rebuild, so the operator sees one warning line and a re-index, not a failure.
- **Option D is rejected as disproportionate**, not as wrong. A schema commits every future embedder to
  populating fields it may not have, for two consumers. An opaque string lets each caller decide what
  identifies its own space, and it can be replaced by D later without a further format change if the id is
  simply given more structure.

**What goes into the id is the caller's choice, not this format's.** For `RagService` the recommended content
is: the model file's identity, the pooling mode, the query and passage prefixes, and the quantise flag. That
recommendation belongs in the caller's XML doc, not in the format.

## Consequences

- **Every existing `.psp` cache is invalidated and rebuilt on next start.** For the demo that is one re-embed
  of a document folder and one `LogWarning`. Whether that is acceptable is
  [`Q-C2` in the plan](../specs/xc-131-xc-123-embedding-postnorm-and-qwen3-embedding-plan.md) and is the
  client's call, not this ADR's.
- `PersistentVectorStore` gains one optional constructor parameter — additive to the public surface, so no
  existing caller breaks at compile time. A caller that does not supply an id writes an empty one, and a
  store written with an empty id will not load against a non-empty one. **That is intended**, and it means
  the field only earns its keep once callers populate it.
- The nested `VectorStore` record format (`VectorStore.cs:168-169`, magic `0x3153_564F`, its own
  `FileVersion = 1`) is **unchanged**. Only the outer envelope moves. A future change to the vector encoding
  itself would version that record separately.
- **A version bump is not a migration.** There is no reader for v1 files and none is planned; the product
  reads external formats in and writes only its own caches, which are reproducible from their sources by
  construction.
- Nothing here changes what a *correct* cache does. On a matching id the load path is byte-for-byte the
  behaviour it has today plus one string comparison.
