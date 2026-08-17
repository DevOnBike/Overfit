---
name: harness-pb12
description: The decode-pool audit harness — what it does, where it lives, and how to rebuild it if the scratch copy is gone.
metadata:
  type: reference
---

# PB-12 audit harness

`D:\Overfit\.claude\pb12\` — **gitignored scratch** (`.gitignore:276` covers `.claude/`), NOT in
`Overfit.sln` (which lists projects explicitly, no globs), so it cannot be built or committed by accident.
Left in place 2026-08-14 because re-deriving it costs ~20 minutes; treat its contents as scratch, not
documentation.

Files: `Pb12.csproj` (net10.0 exe, `AllowUnsafeBlocks`, ProjectReference to `Sources/Main/Main.csproj`),
`Program.cs`, and an **empty `Directory.Build.props` + `Directory.Packages.props`** that shield it from the
repo-root props (otherwise the in-repo analyzer makes raw `Parallel.For` — an arm under measurement — a
build error).

Build: `dotnet build D:\Overfit\.claude\pb12\Pb12.csproj -c Release`. Exe at
`.claude\pb12\bin\Release\net10.0\Pb12.exe`. Results append to `.claude\pb12\results.jsonl`.

Two modes:

- `decode <gguf> <tokens> <runs> <label>` — takes the measurement mutex, prints `DecodeMaxWorkers`, runs a
  single-thread FP **canary** before and after, loads the model, warms up 16 tokens, runs the
  **dispatch-count liveness probe**, then best-of-N fixed-length decode reporting tok/s, B/token and an FNV
  hash of the generated token ids (for the bit-identity half of the claim).
- `dispatch <dispatchesPerToken> <reps> <rangeLen> <inner>` — **single-process ABAB** over three arms:
  `ForDecode` (spin pool), capped `Parallel.For` (MaxDoP = `DecodeMaxWorkers`), uncapped `Parallel.For`
  across `WorkerCount` chunks. Allocation via `GC.GetTotalAllocatedBytes(precise: true)` because
  `Parallel.For` allocates on worker threads and the per-thread counter undercounts it.

**The pre-change protocol arm** is `.claude/pb12-old/`: a copy of `Sources/Main` (bin/obj excluded) with
`Runtime/OverfitParallel.cs` replaced by `git show f8dd016:...` and `DecodeChunkClaim.cs` deleted, plus
`harness/Pb12Old.csproj` which compiles **the same `Program.cs`** against it. The empty
`Directory.Build.props` goes in `harness/`, **never** beside the copied `Main` — Main must keep the root
props or it loses the analyzers and CPM and is no longer the same build. `Program.cs` prints the assembly
MVID and infers the protocol from the presence of the `DecodeChunkClaim` type, so an arm cannot be
mislabelled; the summary script asserts label == reported protocol.

Arms are driven by `env=` in `subprocess.run` (`OVERFIT_DECODE_POOL`, `OVERFIT_DECODE_WORKERS`) — both are
read into `static readonly` fields at first touch, so **an in-process A/B of the pool is impossible**; only
the `dispatch` mode can interleave in one process.
