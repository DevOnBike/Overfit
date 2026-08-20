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

        /// <summary>N-panels packed and swept together in the conv GEMM; 1 = the original per-panel loop.</summary>
        public const string ConvPanelGroup = "OVERFIT_CONV_PANEL_GROUP";

        /// <summary>Set to 0 to force the AVX2 8×8 conv micro-kernel instead of the AVX-512 8×32 one.</summary>
        public const string ConvAvx512 = "OVERFIT_CONV_AVX512";

        /// <summary>Set to 0 to dispatch the conv GEMM one work item per N-panel, as before the M-split.</summary>
        public const string ConvMSplit = "OVERFIT_CONV_M_SPLIT";

        /// <summary>Set to 0 to build the conv column matrix first, as before the im2col fusion.</summary>
        public const string ConvFusedIm2Col = "OVERFIT_CONV_FUSED_IM2COL";

        /// <summary>Set to 1 to repack conv kernels into MR-major micro-panels before the GEMM sweep.</summary>
        public const string ConvPackA = "OVERFIT_CONV_PACK_A";

        /// <summary>32-column sub-panels per conv-GEMM work item; 1 is the original, 4 gives MLAS's 128.</summary>
        public const string ConvNBlock = "OVERFIT_CONV_NBLOCK";

        /// <summary>Set to 0 to run pool=2 inference on one thread, as before the channel split.</summary>
        public const string ParallelPool = "OVERFIT_PARALLEL_POOL";

        /// <summary>
        /// Set to 0 to keep <c>Relu</c> as its own graph node instead of folding it into the preceding
        /// convolution's bias epilogue.
        ///
        /// <para>It exists to be the second arm. Fusion deletes a node, so "did it help" cannot be answered
        /// by a before-and-after build — the two arms have to run in one sitting against the same box.</para>
        /// </summary>
        public const string FuseConvRelu = "OVERFIT_FUSE_CONV_RELU";

        /// <summary>
        /// Set to 1 to re-enable the M-split inside the <b>fused im2col</b> convolution path.
        ///
        /// <para><b>The polarity is inverted against the four switches around it, and deliberately.</b> Those
        /// default on because they are measured wins turned off for an A/B. This one defaults <b>off</b>
        /// because it is a measured loss kept only so the loss stays reproducible — shipping it on the other
        /// way round is how a switch once carried an unreachable number into the README.</para>
        /// </summary>
        public const string ConvFusedMSplit = "OVERFIT_CONV_FUSED_M_SPLIT";

        /// <summary>
        /// Set to 1 to expand every convolution panel once into a shared buffer instead of gathering each
        /// panel inside its own work item.
        ///
        /// <para><b>Off by default: it is a measured loss.</b> It is kept because the measurement it
        /// produced is the useful part — balanced work items did not help the layers it was built for, which
        /// refutes the claim that those layers are short of parallelism.</para>
        /// </summary>
        public const string ConvExpandPanels = "OVERFIT_CONV_EXPAND_PANELS";

        /// <summary>
        /// Set to 0 to gather im2col one element at a time behind a bounds test, instead of copying the
        /// contiguous runs a unit-stride convolution actually reads.
        ///
        /// <para>It is the A/B arm for the largest single item of per-core work measured in a convolution:
        /// 39% of VGG-16's single-core time before the change.</para>
        /// </summary>
        public const string ConvVectorGather = "OVERFIT_CONV_VECTOR_GATHER";

        /// <summary>
        /// Set to 0 to evaluate GELU with the scalar <c>MathF.Tanh</c> loop that shipped before 2026-08-19
        /// instead of the vectorised identity. Default is the vectorised one.
        ///
        /// <para><b>It exists to keep an A/B possible in ONE process.</b> The activation itself was measured
        /// at 9.4-12.5x, but that says nothing about its share of a token, and comparing two separate
        /// process launches cannot separate the change from the box. Both arms are kept so the question can
        /// be re-asked on another model without rebuilding.</para>
        /// </summary>
        public const string GeluVector = "OVERFIT_GELU_VECTOR";

        /// <summary>
        /// Set to 1 to record what each parallel fan-out cost: occupancy, straggler ratio and join overhead.
        ///
        /// <para>It exists because a hardware profiler cannot see this. Counters key on cycles not in halt,
        /// and a parked worker produces no samples — so the one quantity that separates "the split is
        /// uneven" from "the hand-off is slow" is invisible to uProf by construction.</para>
        /// </summary>
        public const string ParallelOccupancy = "OVERFIT_PARALLEL_OCCUPANCY";

        /// <summary>
        /// Chunks per worker in a parallel fan-out. 1 is the historical behaviour and the default; higher
        /// values give the dynamic claim counter something to rebalance across uneven work.
        ///
        /// <para>Capped at 8 by the dispatcher, which sizes its chunk table from that bound.</para>
        /// </summary>
        public const string ParallelChunkFactor = "OVERFIT_PARALLEL_CHUNK_FACTOR";

        /// <summary>
        /// Set to 1 to lay parallel chunks out region-major, so a worker's successive chunks continue its own
        /// region instead of starting another one. Only has an effect above one chunk per worker.
        /// </summary>
        public const string ParallelRegionMajor = "OVERFIT_PARALLEL_REGION_MAJOR";



        /// <summary>Set to 0 to split a large batch-1 dense layer by output column, as before the row split.</summary>
        public const string LinearRowSplit = "OVERFIT_LINEAR_ROW_SPLIT";

        /// <summary>
        /// Set to 0 to force the portable <c>Vector&lt;T&gt;</c> Linear tile instead of the explicit 512-bit one.
        ///
        /// <para>An A/B switch, and also the only way to reach the portable path from a test on an AVX-512
        /// box. Deliberately an environment variable rather than a test-settable static: a process-wide flag
        /// that two kernels' numerics hang off is the defect `XC-56` and `XC-57` were filed for, and this
        /// keeps the two arms in two processes instead.</para>
        /// </summary>
        public const string LinearAvx512 = "OVERFIT_LINEAR_AVX512";

        /// <summary>Set to 0 to decode Q4_K/Q6_K F16 scales inside the tile loop instead of once per projection.</summary>
        public const string PrecomputedScales = "OVERFIT_PRECOMPUTED_SCALES";

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

        // ── HTTP server (Sources/Server) ──────────────────────────────────────────

        /// <summary>
        /// Set to <c>1</c> to print a per-request phase trace (history replay, prompt-cache reuse, time to
        /// first token) — the attribution used to tell server overhead apart from engine work.
        /// </summary>
        public const string ServerTrace = "OVERFIT_SERVER_TRACE";

        /// <summary>
        /// Set to <c>1</c> to skip the kept-end-of-prompt-logits fast path, so a re-sent prompt costs one
        /// forward pass instead of none. Exists so both configurations can be measured side by side in the
        /// same interleaved run rather than across processes.
        /// </summary>
        public const string DisableLogitsCache = "OVERFIT_DISABLE_LOGITS_CACHE";

        /// <summary>
        /// Set to <c>1</c> to emit each token after the forward pass that follows it rather than before —
        /// the ordering that predates the early-emit change. Same purpose: an in-run A/B.
        /// </summary>
        public const string DisableEarlyEmit = "OVERFIT_DISABLE_EARLY_EMIT";

        /// <summary>
        /// Set to <c>1</c> to force the exact single-token decode loop instead of the speculative path.
        /// Same purpose: an in-run A/B of speculative decode against plain decode.
        /// </summary>
        public const string DisableSpeculative = "OVERFIT_DISABLE_SPECULATIVE";

        /// <summary>
        /// Shared secret the anomaly guard requires on <c>POST /ack</c>, as
        /// <c>Authorization: Bearer &lt;value&gt;</c>.
        ///
        /// <para><b>Unset means the endpoint is refused, not open.</b> <c>/ack</c> suppresses a finding for a
        /// caller-chosen duration, so anyone who can reach the port can silence a real incident and leave
        /// only a log line. The port is shared with the Prometheus scrape, and the `NetworkPolicy` written to
        /// restrict it was **measured inert** on a Docker-Desktop-class cluster — no policy-capable CNI, and
        /// a probe pod still reached the port with the policy applied. A control that depends on the
        /// customer's CNI is not a control.</para>
        ///
        /// <para>Read once at startup. Rotating it needs a restart, which is the right trade for a process
        /// that must not grow a configuration-reload path to hold one string.</para>
        /// </summary>
        public const string GuardAckToken = "OVERFIT_GUARD_ACK_TOKEN";

        /// <summary>
        /// Turns on the per-member peer-decision trace. Unset or empty is off; <c>1</c>, <c>true</c> or
        /// <c>all</c> traces every channel; any other value is read as a channel name and traces only that one.
        ///
        /// <para><b>Why it exists.</b> "No finding" has five different causes that call for opposite fixes —
        /// the rank test, the relative-gap gate, the absolute floor, too few usable samples, the novelty gate —
        /// and only the individual gate values tell them apart. The trace carrying them was built with the
        /// detector and, until 2026-08-10, was reachable only from three diagnostics in Tests: a channel that
        /// went quiet in the cluster could not be told from one that had nothing to say. Measured that day on
        /// a 20 MB injected leak, where MemoryWorkingSetBytes produced no peer finding while every gate the
        /// author could read passed by 2.7x to 5.9x.</para>
        /// </summary>
        public const string GuardPeerTrace = "OVERFIT_GUARD_PEER_TRACE";

        // ── Third-party / host environment (not ours, but read by us) ─────────────

        /// <summary>Hugging Face API endpoint override for the model downloader.</summary>
        public const string HuggingFaceEndpoint = "HF_ENDPOINT";

        /// <summary>Hugging Face access token for gated repositories.</summary>
        public const string HuggingFaceToken = "HF_TOKEN";

        /// <summary>Set by GitHub Actions; used to detect a CI run.</summary>
        public const string GitHubActions = "GITHUB_ACTIONS";
    }
}