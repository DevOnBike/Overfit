// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;

// ────────────────────────────────────────────────────────────────────────────
// [module: SkipLocalsInit]
// ────────────────────────────────────────────────────────────────────────────
//
// Disables CIL's `.locals init` flag for ALL methods in this assembly. Without
// it, the runtime zero-initializes every local variable (including the entire
// `stackalloc` buffer) on method entry — wasted work when the code is about
// to overwrite those locals immediately.
//
// Why this matters here:
//   - 21+ `stackalloc` sites in Overfit (LayerNorm backward partial buffers,
//     small-batch scratch in PoolingKernels, LoRA weight scratch, GGUF
//     reader, etc.). Each one currently pays a memset of the buffer at
//     entry that we then either fully overwrite or explicitly `.Clear()`.
//   - LayerNorm backward stackalloc's `workerCount × C × 2` floats per call
//     (16-32 KB for typical GPT-1) — visible win.
//
// Safety:
//   - This is the pattern Microsoft uses assembly-wide in every modern
//     .NET 5+ project (Roslyn, Kestrel, BCL — 1068 hits across 134 files
//     in the VS 2022 install per `vs2022-performance-patterns.md`).
//   - Compatible with Native AOT (no runtime reflection; pure CIL flag).
//   - C# language rules still require definite assignment before read of
//     managed locals, so this CANNOT corrupt object references.
//   - The danger surface is `stackalloc` / `Unsafe.SkipInit<T>` style code
//     that reads from a buffer before writing — code must explicitly zero
//     such buffers (e.g. `Span.Clear()`) when zero-init was being relied on.
//     Audit: every `stackalloc` site in this assembly either fully writes
//     before reading OR calls `.Clear()` explicitly. Verified by the
//     `LayerNorm_Backward_GradientCheckNumerical` test (which would diverge
//     under non-zero garbage in the partial buffers) + the new
//     `Stackalloc_AfterSkipLocalsInit_*` regression tests in
//     `SkipLocalsInitTests.cs`.
[module: SkipLocalsInit]
