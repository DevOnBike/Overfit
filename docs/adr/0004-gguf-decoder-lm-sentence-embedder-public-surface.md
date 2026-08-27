# ADR 0004 — the GGUF decoder-LM embedder is a new public type in `Sources/Main`, and its conventions become a contract

- **Status**: proposed, 2026-08-27, by `overfit-architect`.
- **Plan**: [`../specs/xc-131-xc-123-embedding-postnorm-and-qwen3-embedding-plan.md`](../specs/xc-131-xc-123-embedding-postnorm-and-qwen3-embedding-plan.md)
- **Scope**: a new public `sealed class` in `Sources/Main/LanguageModels/Embeddings/`, and the defaults it
  publishes.

## Context

`XC-123` adds Qwen3-Embedding: a decoder LM used as a sentence embedder. Unlike the BERT embedders already
shipped, it needs a pooling choice, an instruction prefix on the query side, and an appended EOS.

Three homes were available and two are wrong:

- **`SentenceEmbedder`** — the existing embedder facade, with exactly the shape wanted (`EmbedQuery` /
  `EmbedPassage`, `QueryPrefix` / `PassagePrefix`, `ForBgeEnV15` / `ForE5`). It is constructed from a
  `WordPieceTokenizer` and a `BertEncoder` (`SentenceEmbedder.cs:29-30`). **It is BERT-only by construction
  and cannot hold a GGUF decoder.**
- **`OverfitClient`** — already exposes `Embed(string, EmbeddingPooling)` at `:318`. But it is the *chat*
  facade: it owns a `ChatSession`, stop sequences and sampling defaults, and its embedding path exists to
  reuse the model *already loaded for chat* (`:343`). A Qwen3-Embedding-specific instruction convention does
  not belong on it.
- **A new type.**

The decisions that are hard to undo are not the file's location. They are: **what becomes public**, and
**which defaults become a contract**. A default published in `DevOnBike.Overfit` is relied on by callers who
cannot see the code and cannot be asked to change.

## Forces

- `Sources/Main` is the only assembly `Mcp`, `Server`, `Cli` and `Demo/*` all consume. Anything they share
  must live there, and dependencies point one way.
- `SentenceEmbedder` is public, in `LanguageModels/Embeddings/`, and sets the naming idiom. Matching an
  existing public idiom is worth more than a better name.
- `OVERFIT034` allows one top-level type per file in `Main`.
- The pooling mode, the prefixes and the quantise flag are all things a caller will store alongside its
  vectors ([ADR 0003](0003-persisted-vector-store-embedding-space-identity.md)). Changing any of them later
  changes every stored vector — the exact failure `XC-131` is.
- **The instruction template text is not verified.** The plan records this as spike S2. A wrong string
  published as a default is a wrong contract.

## Options considered

| option | cost | what it forecloses |
|---|---|---|
| **A. Extend `SentenceEmbedder`** | impossible without breaking its constructor contract | n/a — rejected on feasibility, not taste |
| **B. Put `EmbedQuery` / `EmbedPassage` on `OverfitClient`** | small | puts a model-family convention on the generic chat facade, permanently |
| **C. New public `sealed class` in `Main/LanguageModels/Embeddings/`** | one type, one file | nothing; it can gain an interface later if `/v1/embeddings` ever needs one |
| **D. New type, but `internal` + `InternalsVisibleTo`** | same | the feature — the point of `XC-123` is that callers can use it |

## Decision

**Take option C: one new public `sealed class` in `Sources/Main/LanguageModels/Embeddings/`, mirroring
`SentenceEmbedder`'s public shape.**

- Members: `Embed` / `EmbedQuery` / `EmbedPassage`, `Dimension`, `Pooling`, `QueryPrefix`, `PassagePrefix`,
  `IDisposable`. A `ForQwen3Embedding(path)` factory in the style of `ForBgeEnV15` (`:93`) and `ForE5`
  (`:104`).
- It **owns** its `CachedLlamaInferenceEngine` and session and disposes both, as `SentenceEmbedder` and
  `OverfitClient.Dispose` (`OverfitClient.cs:347-357`) already do.
- **The type name is the developer's choice** and is deliberately not fixed here. It is local and reversible.
- **No interface, and no wiring to `POST /v1/embeddings`.** `EmbeddingsExchange.Handle` takes a concrete
  `SentenceEmbedder` (`EmbeddingsExchange.cs:20`); serving a decoder LM there needs a new abstraction on a
  public type in `Sources/Server` plus a security gate, because that endpoint is externally fed. Neither row
  requires it. Add it when something asks for it, with its own ADR.

**Three defaults become a contract, and each is decided on purpose:**

1. **Pooling.** Set by the `ForQwen3Embedding` factory to the model family's convention. The generic
   constructor takes it as a parameter, as `FromPretrained` does.
2. **`quantize: false`.** An embedder's entire product is the vector. Re-quantising an already-Q8_0 file
   moves the *pairwise* similarity by 8.5e-3 against llama.cpp, 18x the F32 deviation of 4.6e-4 — a
   correctness cost on the quantity callers consume. **Conditional on spike S3**: this dequantizes to F32,
   and peak RAM is the load path's discipline. If the measured peak is unaffordable the default flips and the
   deviation is documented on the type. **The number goes in the plan and in `docs/measured-baselines.md`
   either way.**
3. **The instruction prefix strings.** **Not fixed by this ADR.** Spike S2 reads them from the model's own
   artefacts and the implementer quotes the source in the XML doc. An unverified string must not be
   published as a default.

## Consequences

- **A new public type cannot be withdrawn.** Per `CLAUDE.md`'s versioning policy, removing or changing it
  later is a breaking change and bumps `MINOR` (`MAJOR` is pinned to the .NET target here). That is the price
  of `XC-123` being usable at all, and option D avoids it only by not shipping the feature.
- **The three defaults above are part of that contract.** Anything a caller stores — a vector index, a cache
  — depends on them, and changing one silently repeats `XC-131`. A caller that persists vectors must record
  them in its `EmbeddingSpaceId` (ADR 0003).
- `Sources/Main`'s constraints apply in full: `RS0030` bans `System.Linq`, `System.Reflection`, `Activator`,
  `Expression`, `Array.Copy` and raw `ArrayPool<T>.Shared` at every build; `OVERFIT033` bans `float[][]`;
  `OVERFIT034` allows one top-level type per file; `Stopwatch.StartNew` is banned in favour of
  `ValueStopwatch`.
- **No new dependency enters `Main`.** The type composes `GgufLlamaLoader`, `GgufTokenizer` and
  `CachedLlamaSession`, all already there.
- **Nothing new becomes AOT-reachable.** `Tests/AotSmokeTest/Program.cs` is unchanged.
- **Open side of the moat.** Batch embedding for RAG is offline work plus correctness. No real-time claim
  attaches to this type or its documentation.
