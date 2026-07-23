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

        /// <summary>Set to 1/true to route the Q4_K PREFILL projections through the register-tiled 8×NR GEMM
        /// (<c>Q4KGemvKernel.GemmTiled</c>) instead of the weight-stationary kernel. ~3× faster per projection
        /// under real parallelism (measured), but repacks the weight (adds ~model RAM) so it is experimental /
        /// off by default.</summary>
        public const string TiledPrefill = "OVERFIT_TILED_PREFILL";

        /// <summary>KV-cache element type — e.g. <c>q8</c> for the int8 KV cache (default F32).</summary>
        public const string KvDType = "OVERFIT_KV_DTYPE";

        /// <summary>Set to 0 to force the serial im2col patch gather in the conv GEMM path (A/B switch).</summary>
        public const string ParallelIm2Col = "OVERFIT_PARALLEL_IM2COL";

        // ── Prefill kernel switches (all default ON where the hardware allows; set to 0 to opt out) ──
        // These exist so a measured win can be A/B'd against its predecessor without a rebuild, and so a
        // regression on unfamiliar hardware can be bisected in the field rather than only on the dev box.

        /// <summary>Set to 0 to force the 256-bit Q4_K prefill GEMM instead of the AVX-512 two-column kernel.</summary>
        public const string Avx512PrefillQ4K = "OVERFIT_AVX512_Q4K";

        /// <summary>Set to 0 to force the 256-bit Q6_K prefill GEMM instead of the AVX-512 kernel.</summary>
        public const string Avx512PrefillQ6K = "OVERFIT_AVX512_Q6K";

        /// <summary>Set to 0 to project K/V once per KV head instead of one whole-matrix dispatch.</summary>
        public const string WholeMatrixKv = "OVERFIT_WHOLE_KV";

        /// <summary>Set to 0 to hand attention queries to workers in raw order instead of load-balanced pairs.</summary>
        public const string BalancedAttention = "OVERFIT_BALANCED_ATTN";

        /// <summary>Set to 0 to accumulate the attention value sum through memory instead of in registers.</summary>
        public const string AttentionRegisterAccumulate = "OVERFIT_ATTN_REGACC";

        /// <summary>Set to 0 to use scalar <c>MathF.Exp</c> for the softmax instead of the vectorized path.</summary>
        public const string AttentionVectorizedExp = "OVERFIT_ATTN_VEXP";

        /// <summary>Diagnostics only: <c>dot</c> / <c>exp</c> / <c>both</c> removes that part of the attention
        /// kernel to size its share. Produces WRONG results by construction — never set in production.</summary>
        public const string AttentionAblate = "OVERFIT_ATTN_ABLATE";

        /// <summary>Diagnostics A/B switch: set to 1/true to force the scalar Q4_K main-dot (skip AVX2/NEON).
        /// For measuring SIMD-vs-scalar on one device — not a production tuning knob.</summary>
        public const string ForceScalar = "OVERFIT_FORCE_SCALAR";

        // ── Training (Sources/Main/Autograd/ComputationGraph.cs) ──────────────────

        /// <summary>Override the autograd tape buffer size (elements).</summary>
        public const string GraphTapeBufferElements = "OVERFIT_GRAPH_TAPE_BUFFER_ELEMENTS";

        // ── CLI path hints (Sources/Cli) ──────────────────────────────────────────

        /// <summary>Default Orpheus GGUF directory for the TTS commands.</summary>
        public const string OrpheusDir = "OVERFIT_ORPHEUS_DIR";

        /// <summary>Default SNAC decoder-weights directory for the TTS commands.</summary>
        public const string SnacDir = "OVERFIT_SNAC_DIR";

        // ── Model/asset path hints read by the CLI, demos and benchmarks ──────────

        /// <summary>Directory holding the default model for the demos and the local-agent host.</summary>
        public const string ModelDir = "OVERFIT_MODEL_DIR";

        /// <summary>Explicit path to a single model file, where a demo takes a file rather than a directory.</summary>
        public const string ModelPath = "OVERFIT_MODEL_PATH";

        /// <summary>Directory holding the sentence-embedding model used by the RAG demo.</summary>
        public const string EmbeddingDir = "OVERFIT_EMBEDDING_DIR";

        /// <summary>Model used as the judge in the evaluation demo.</summary>
        public const string Judge = "OVERFIT_JUDGE";

        /// <summary>ONNX model path for the large-CNN comparison benchmark.</summary>
        public const string CnnOnnx = "OVERFIT_CNN_ONNX";

        /// <summary>MNIST data directory for the training benchmarks.</summary>
        public const string MnistDir = "OVERFIT_MNIST_DIR";

        // ── Android decode bench (Sources/AndroidBench) ───────────────────────────

        /// <summary>Overrides the decode-pool setting for the on-device bench.</summary>
        public const string BenchPool = "OVERFIT_BENCH_POOL";

        /// <summary>Overrides the worker count for the on-device bench.</summary>
        public const string BenchWorkers = "OVERFIT_BENCH_WORKERS";

        // ── Third-party / host environment (not ours, but read by us) ─────────────

        /// <summary>Hugging Face API endpoint override for the model downloader.</summary>
        public const string HuggingFaceEndpoint = "HF_ENDPOINT";

        /// <summary>Hugging Face access token for gated repositories.</summary>
        public const string HuggingFaceToken = "HF_TOKEN";

        /// <summary>Set by GitHub Actions; used to detect a CI run.</summary>
        public const string GitHubActions = "GITHUB_ACTIONS";
    }
}