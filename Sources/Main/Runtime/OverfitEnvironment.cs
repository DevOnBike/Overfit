// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Runtime
{
    /// <summary>
    /// Single source of truth for every <c>OVERFIT_*</c> environment-variable name the engine and CLI read.
    /// Centralised so the names can't drift between the read sites and the docs / <c>overfit doctor</c> output
    /// that surface them. These are opt-in tuning/config switches; the defaults are chosen so nothing here needs
    /// to be set for correct behaviour.
    /// </summary>
    public static class OverfitEnvironment
    {
        // ── Decode parallelism (Sources/Main/Runtime/OverfitParallel.cs) ──────────

        /// <summary>Worker count for the general parallel-for pool.</summary>
        public const string ParallelWorkers = "OVERFIT_PARALLEL_WORKERS";

        /// <summary>Worker count for the decode spin-pool (per-token matmul fan-out).</summary>
        public const string DecodeWorkers = "OVERFIT_DECODE_WORKERS";

        /// <summary>Set to 0/false to opt out of the decode spin-pool (default on).</summary>
        public const string DecodePool = "OVERFIT_DECODE_POOL";

        // ── Quantized-kernel opt-ins (Sources/Main/LanguageModels/Runtime/Q4KGemvKernel.cs) ──

        /// <summary>Set to 1/true to route the FFN gate/up (and LM head) through the repacked 8×8 Q4_K GEMV.</summary>
        public const string RepackGemv = "OVERFIT_REPACK_GEMV";

        /// <summary>Set to 1/true for the whole-matrix Q4_K attention decode path (experimental, off by default).</summary>
        public const string RepackAttn = "OVERFIT_REPACK_ATTN";

        /// <summary>KV-cache element type — e.g. <c>q8</c> for the int8 KV cache (default F32).</summary>
        public const string KvDType = "OVERFIT_KV_DTYPE";

        // ── Training (Sources/Main/Autograd/ComputationGraph.cs) ──────────────────

        /// <summary>Override the autograd tape buffer size (elements).</summary>
        public const string GraphTapeBufferElements = "OVERFIT_GRAPH_TAPE_BUFFER_ELEMENTS";

        // ── CLI path hints (Sources/Cli) ──────────────────────────────────────────

        /// <summary>Default Orpheus GGUF directory for the TTS commands.</summary>
        public const string OrpheusDir = "OVERFIT_ORPHEUS_DIR";

        /// <summary>Default SNAC decoder-weights directory for the TTS commands.</summary>
        public const string SnacDir = "OVERFIT_SNAC_DIR";
    }
}
