# ADR 0002 — the `.repack` sidecar carries the identity of the GGUF it was built from

- **Status**: proposed, 2026-08-25, by `overfit-architect`. **Amended the same day** after `XC-119`: the
  consequences section described the loss in throughput terms, which `XC-119` refutes as the wrong frame.
  The decision itself is unchanged.
- **Plan**: [`../specs/auto-repack-sidecar-plan.md`](../specs/auto-repack-sidecar-plan.md)
- **Scope**: the on-disk format of `<model>.gguf.repack`, written by
  `Sources/Main/LanguageModels/Loading/RepackedWeightsFile.cs`

## Context

The sidecar holds Q4_K weight matrices pre-converted to `block_q4_Kx8`. `GgufLlamaLoader.TryOpenSidecar:78`
discovers `<model>.repack` unconditionally, and `AttachPrepacked:1068` attaches a blob when the **tensor name**
and the **two dimensions** agree. `Q4KWeight.SetPrepacked:140` then validates the blob **length** and nothing
else.

So the sidecar is bound to the model by **path and shape only**. Nothing binds it to the *bytes* of the GGUF
beside it. `XC-105` recorded this in `docs/TASKS.md:223` as a known property: *"a sidecar built from a
different gguf, or by an older repack, would attach silently and change decode."*

Today the risk is bounded by who creates the file: a human runs `overfit repack` once, and the file appears
next to the model it was made from. The plan above proposes that Overfit create the file itself, and reuse it
across loads. That makes the engine the author of a cache, and a cache without an identity check is a source
of silently wrong numerics: replace `model.gguf` in place with a re-quantisation or a fine-tune of the same
architecture, and every dimension still agrees while every weight has changed.

The failure is silent by construction. The prefill path takes the sidecar's bytes and the decode path takes
the GGUF's, so the model does not crash — it produces different text.

## Forces

- The check must not read the whole model. Time to first token is already 5.56-6.13 s against llama.cpp's
  1.35-1.44 s (`docs/measured-baselines.md`, *"Overfit vs llama.cpp `6d5a910`"*), so a full re-read of
  2,104,932,768 B on every load is not affordable.
- The header alone is not sufficient. Two different quantisations of the same base model can carry identical
  tensor names, types, dimensions and offsets, and can have the same file length.
- The format already has a version lever: the magic is `"OVFRPK" + {1,0}` and `Open` rejects any file whose
  eight magic bytes differ.
- A rejected sidecar must never block a load. `TryOpenSidecar` already treats a bad file as absent.
- Existing sidecars are large and expensive to rebuild. Invalidating them has a real cost to anyone who has
  already run `overfit repack`.

## Options considered

| option | catches a replaced model | cost per load | cost to existing files |
|---|---|---|---|
| **A. Leave it as it is** (name + dims) | no | 0 | none |
| **B. Model file length + mtime** | usually, never a same-size rebuild; mtime does not survive a copy | 1 stat | none if added out of band, all if versioned |
| **C. Digest of the GGUF tensor index** (names, types, dims, offsets) | no — a re-quantisation to the same shapes is identical here | header parse, already done | all |
| **D. sha256 of the whole GGUF** | yes | a full 2.1 GB read | all |
| **E. length + index digest + digest over a sampled set of tensor-data bytes** | yes, to overwhelming probability | one stat plus a bounded read | all |

## Decision

**Take option E, and bump the format magic to `"OVFRPK" + {2,0}`.**

The header gains an identity record written at build time and verified in `RepackedWeightsFile.Open`:

1. the source GGUF's file length, as an `int64`;
2. a digest over the source's tensor index — for every tensor, its name, its `GgmlType`, its dimensions and
   its data offset;
3. a digest over a **deterministic sample of the tensor data**, specified by the format rather than chosen at
   build time, so that a reader can recompute it: a bounded number of fixed-size windows at positions derived
   from the tensor index.

Constraints on the sample, which are part of this decision because a reader must be able to reproduce it:
it reads **no more than 1 MB in total**, it covers **at least 8 distinct tensors** when the model has that
many, and its window positions are a pure function of the tensor index. The exact digest function and window
schedule are the implementer's, subject to those three.

On mismatch, `Open` throws `OverfitFormatException`, which `TryOpenSidecar` already converts to "no sidecar".

## Rationale

Option E is the only one on the table that closes the failure this ADR exists for, at a cost that fits inside
a load. Option C is cheap and does not close it; option D closes it and costs a 2.1 GB read at every load,
which is the one budget already overspent.

The magic bump is preferred to an out-of-band check because a v1 file has no identity record to check — any
scheme that keeps reading v1 files keeps the hazard for exactly the files that have it.

## Consequences

**Every existing `.repack` file stops being used.** It is rejected as bad magic, silently, and the model
falls back to the weight-stationary prefill kernel. This is the real price of the decision and it is paid by
the people who already adopted the feature.

**Stating that price in throughput terms would be the wrong frame.** `XC-119` (`docs/measured-baselines.md`)
established that the tiled *kernel*, not the file, carries the warm prefill gain — 98.2% of it at `pp512`.
What a rejected sidecar costs is: the tiled kernel is unreachable without paying **+1.2 GiB of peak private
commit** for a heap repack instead, and that route measured **9.5% slower end to end on a short CLI
invocation** (6558.2 ± 14.7 against 5991.3 ± 17.1 ms) for **2.4× the memory**. So the loss is a memory and
start-up loss, not a throughput one.

**Two further consequences that the first draft of this ADR did not carry.**

- **Rejection changes generated text.** `IsPrepacked` alone selects the tiled kernel
  (`BatchedQuantProjection.cs:502`), and the repacked kernels are held to coherence rather than
  byte-parity: measured, 48 greedy tokens from a 31-token prompt differ between the two kernels. So a user
  whose sidecar is invalidated by this ADR sees their model's output move. That must be in the release
  note, not only in the log.
- **On a machine without AVX2 the rejection costs nothing at all**, because two gates
  (`Q4KGemvKernel.ResolveFlag`, `BatchedQuantProjection.DispatchQ4K`) make the tiled kernel unreachable
  there regardless — measured 9.539 against 9.535 t/s under `DOTNET_EnableAVX2=0`.

Two things must land with it so that the loss is visible rather than silent:

- `overfit doctor` reports whether a usable sidecar is present, and says why one was rejected. It says nothing
  about the sidecar today.
- The rejection reason is observable from the library, not only from the CLI. See the plan's operability
  section.

The build side gains a dependency on reading a bounded amount of tensor data, which `GgufReader` already does.

A future format change repeats this cost. If a third version becomes likely, the version byte should be read
and reported rather than folded into a magic comparison — but that is a change to make when it is needed, not
in advance.

## Status of the wider question

This ADR settles the format. **It does not decide whether Overfit generates the file automatically, or where.**
Those are open and are recorded as blocking questions in the plan. The identity record is required either way:
it fixes a hazard that exists today with the manual command.
