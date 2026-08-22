// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Text;
using System.Text.Json;

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// Renders the one block of text that comes back from the friend's machine, and the same content as
    /// JSON beside it. Everything between the two markers is the artefact; nothing outside them matters.
    /// </summary>
    internal sealed class Report
    {
        private const string Begin = "=== BEGIN GPU PROBE REPORT ===";
        private const string End = "=== END GPU PROBE REPORT ===";

        public MachineEcho? Machine { get; set; }

        public string SelectionNote { get; set; } = string.Empty;

        public bool IsRealDevice { get; set; }

        /// <summary>The ILGPU accelerator type the device arms actually ran on, as a word.</summary>
        public string AcceleratorType { get; set; } = "unknown";

        /// <summary>
        /// True only for a real CUDA accelerator. OpenCL and the CPU emulator are both false, and both
        /// suppress the headline ratio: the probe exists to price a port to an NVIDIA card.
        /// </summary>
        public bool IsCuda { get; set; }

        /// <summary>The warm-up stopping rule this run used, in one sentence.</summary>
        public string WarmupRule { get; set; } = string.Empty;

        public CanaryReading? CanaryStart { get; set; }

        public CanaryReading? CanaryEnd { get; set; }

        public bool CanaryMeasured { get; set; }

        public double CanaryStartMs => CanaryStart?.MedianMs ?? 0;

        public double CanaryEndMs => CanaryEnd?.MedianMs ?? 0;

        /// <summary>
        /// True when BOTH canary readings settled. An unsettled canary cannot say whether the machine
        /// moved, so it must not be read as saying that it did not.
        /// </summary>
        public bool CanarySettled => CanaryStart is { Settled: true } && CanaryEnd is { Settled: true };

        public List<string> OracleLines { get; } = [];

        public List<string> OracleFailures { get; } = [];

        public List<CellResult> Cells { get; } = [];

        public List<string> TopBanners { get; } = [];

        /// <summary>Measured FP16 accuracy cost per shape, from --fp16-bound. Empty otherwise.</summary>
        public List<string> Fp16Bounds { get; } = [];

        /// <summary>What the live view costs the host arms, from --live-perturbation. Empty otherwise.</summary>
        public List<string> LivePerturbation { get; } = [];

        /// <summary>What drove the live view this run, or why there was none.</summary>
        public string? LiveViewNote { get; set; }

        /// <summary>
        /// What <see cref="GuardSelfCheck"/> found before anything was timed. It is in the report rather
        /// than only on the console because the report is the artefact a stranger pastes back, and a
        /// drop count of zero means nothing until the reader knows the counter was capable of moving.
        /// </summary>
        public string? SelfCheckNote { get; set; }

        public string? CuBlasSkipReason { get; set; }

        /// <summary>Why the FP16 arm specifically is absent, when cuBLAS itself loaded.</summary>
        public string? Fp16SkipReason { get; set; }

        public double CanaryMove =>
            CanaryStartMs > 0 ? (CanaryEndMs - CanaryStartMs) / CanaryStartMs : 0;

        public string RenderText()
        {
            var run = HeadlineVerdict.ForRun(this);
            var sb = new StringBuilder();
            sb.AppendLine(Begin);
            sb.AppendLine("(copy everything from the line above to the line at the very bottom)");
            sb.AppendLine();

            foreach (var banner in TopBanners)
            {
                sb.AppendLine(banner);
            }

            if (CanaryMeasured)
            {
                var moved = Math.Abs(CanaryMove) > Canary.MoveThreshold;
                var line = string.Create(
                    CultureInfo.InvariantCulture,
                    $"canary 512-cubed CPU GEMM: {CanaryStartMs:F2} ms at the start, {CanaryEndMs:F2} ms at the end ({CanaryMove * 100:+0.0;-0.0} %)");
                sb.AppendLine(moved
                    ? "CANARY MOVED - SITTING SUSPECT. " + line
                    : line + " - within the 5 % threshold");
                sb.AppendLine(string.Create(
                    CultureInfo.InvariantCulture,
                    $"  canary warm-up: start {CanaryStart?.Warmup.Describe()}; end {CanaryEnd?.Warmup.Describe()}"));
            }

            if (!IsRealDevice)
            {
                sb.AppendLine(
                    "NOTE: the 'GPU' arms did NOT run on a GPU. They ran on ILGPU's CPU emulator, so every");
                sb.AppendLine(
                    "      device timing below is a fact about a CPU and about ILGPU's emulator, nothing else.");
            }

            if (Cells.Any(c => c.Timings.Count > 0) && !run.MayPrint)
            {
                sb.AppendLine();
                sb.AppendLine("NO HEADLINE RATIO IN THIS REPORT. The reasons are repeated in every cell, in the place");
                sb.AppendLine("the ratio would have gone, so a number and its caveat cannot be separated by a copy-paste.");
            }

            AppendTensorCoreLine(sb);

            sb.AppendLine();
            sb.AppendLine("MACHINE AND CONFIGURATION");
            sb.AppendLine("  accelerator choice : " + SelectionNote);
            if (WarmupRule.Length > 0)
            {
                sb.AppendLine("  warm-up rule       : " + WarmupRule);
            }

            sb.Append(Machine?.Render());

            sb.AppendLine();
            sb.AppendLine("ORACLES");
            foreach (var line in OracleLines)
            {
                sb.AppendLine(line);
            }

            foreach (var failure in OracleFailures)
            {
                sb.AppendLine("  FAILED: " + failure);
            }

            if (Fp16SkipReason is not null)
            {
                sb.AppendLine();
                sb.AppendLine("ARM X1 (cuBLAS FP16): NOT MEASURED - " + Fp16SkipReason);
                sb.AppendLine("  The FP32 arm X2 may still be below. X1 is the primary arm, so its absence means the");
                sb.AppendLine("  headline question - what this card does at FP16 - was not answered.");
            }

            if (CuBlasSkipReason is not null)
            {
                sb.AppendLine();
                sb.AppendLine("ARMS X1/X2 (cuBLAS): NOT MEASURED - " + CuBlasSkipReason);
                sb.AppendLine(
                    "  So the device arms below are OUR kernels only. They are a FLOOR on what the card can do,");
                sb.AppendLine(
                    "  not the answer: a vendor library is the upper bound and it was not measured here.");
            }

            if (LiveViewNote is not null)
            {
                sb.AppendLine();
                sb.AppendLine("LIVE VIEW: " + LiveViewNote);
            }

            if (SelfCheckNote is not null)
            {
                sb.AppendLine();
                sb.AppendLine("REPAINT GUARD SELF-CHECK: " + SelfCheckNote);
            }

            if (LivePerturbation.Count > 0)
            {
                sb.AppendLine();
                sb.AppendLine("WHAT THE LIVE VIEW COSTS THE HOST ARMS, measured ABAB in one process");
                sb.AppendLine("  The lever is whether the repaint happens; everything else is identical. A ratio near");
                sb.AppendLine("  1.000 means the view is free at this cadence; above it, the view must be suspended.");
                foreach (var line in LivePerturbation)
                {
                    sb.AppendLine("  " + line);
                }
            }

            if (Fp16Bounds.Count > 0)
            {
                sb.AppendLine();
                sb.AppendLine("WHAT FP16 COSTS IN ACCURACY, measured on the host at each shape");
                sb.AppendLine("  relative L2 against the F32 result. 'fp32 acc' is what a TENSOR CORE and cublasGemmEx");
                sb.AppendLine("  with CUBLAS_COMPUTE_32F do; 'fp16 acc' is what cublasHgemm does, and it is the only");
                sb.AppendLine("  FP16 entry point ILGPU's cuBLAS wrapper exposes.");
                foreach (var line in Fp16Bounds)
                {
                    sb.AppendLine("  " + line);
                }
            }

            sb.AppendLine();
            sb.AppendLine("RESULTS");
            sb.AppendLine("  ms per call, min / median / max over the timed repetitions, arms interleaved ABAB.");
            sb.AppendLine("  A cell whose parity check failed shows no timing at all - that is deliberate.");
            sb.AppendLine("  'warm' is the warm-up rounds that arm ran; 'ok' means its timings had stopped moving.");

            foreach (var cell in Cells)
            {
                AppendCell(sb, cell, run);
            }

            sb.AppendLine();
            AppendLimitations(sb);
            sb.AppendLine(End);
            return sb.ToString();
        }

        public string RenderJson()
        {
            var run = HeadlineVerdict.ForRun(this);
            var root = new Dictionary<string, object?>
            {
                ["machine"] = Machine?.Fields.ToDictionary(kv => kv.Key, kv => kv.Value),
                ["acceleratorChoice"] = SelectionNote,
                ["acceleratorType"] = AcceleratorType,
                ["isRealDevice"] = IsRealDevice,
                ["isCuda"] = IsCuda,
                ["warmupRule"] = WarmupRule,
                ["headlineSuppressed"] = !run.MayPrint,
                ["headlineSuppressionReasons"] = run.Blockers,
                ["tensorCores"] = TensorCoreStatus(),
                ["canaryStartMs"] = CanaryMeasured ? CanaryStartMs : null,
                ["canaryEndMs"] = CanaryMeasured ? CanaryEndMs : null,
                ["canaryMoveFraction"] = CanaryMeasured ? CanaryMove : null,
                ["canarySettled"] = CanaryMeasured ? CanarySettled : null,
                ["oracle"] = OracleLines,
                ["oracleFailures"] = OracleFailures,
                ["cuBlasSkipReason"] = CuBlasSkipReason,
                ["fp16Bounds"] = Fp16Bounds,
                ["livePerturbation"] = LivePerturbation,
                ["liveViewNote"] = LiveViewNote,
                ["selfCheckNote"] = SelfCheckNote,
                ["fp16SkipReason"] = Fp16SkipReason,
                ["cells"] = Cells.Select(c => new Dictionary<string, object?>
                {
                    ["cell"] = c.Cell.Name,
                    ["k"] = c.Cell.K,
                    ["m"] = c.Cell.M,
                    ["n"] = c.N,
                    ["callsPerStep"] = c.Cell.CallsPerStep,
                    ["macShareOfStep"] = c.Cell.MacShare,
                    ["forwardFlops"] = c.ForwardFlops,
                    ["weightUploadMs"] = c.WeightUploadMs,
                    ["weightBytes"] = c.Cell.WeightBytesF32,
                    ["warmupRounds"] = c.WarmupRounds,
                    ["warmupStopReason"] = c.WarmupStopReason,
                    ["notes"] = c.Notes,
                    ["arms"] = c.Timings
                        .Where(t => c.MayPrint(t.Key))
                        .ToDictionary(t => t.Key, t => (object)new Dictionary<string, object?>
                        {
                            ["minMs"] = t.Value.MinMs,
                            ["medianMs"] = t.Value.MedianMs,
                            ["maxMs"] = t.Value.MaxMs,
                            ["samplesMs"] = t.Value.Samples,
                            ["gflops"] = t.Value.GFlops(c.ForwardFlops),
                            ["warmupSettled"] = c.Warmups.TryGetValue(t.Key, out var w) && w.Settled,
                            ["warmupSettledAtRound"] = c.Warmups.TryGetValue(t.Key, out var w2) ? w2.SettledAtRound : 0,
                            ["warmupRelativeMove"] = c.Warmups.TryGetValue(t.Key, out var w3) ? w3.RelativeMove : (double?)null,
                        }),
                    ["parity"] = c.Parity.ToDictionary(p => p.Key, p => (object)new Dictionary<string, object?>
                    {
                        ["passed"] = p.Value.Passed,
                        ["cosine"] = p.Value.Cosine,
                        ["maxRelative"] = p.Value.MaxRelative,
                        ["detail"] = p.Value.Detail,
                    }),
                    ["ratios"] = Ratios(c, run),
                }),
                ["ramTypeAndSpeed"] = null,
                ["otherLoadDuringRun"] = null,
            };

            return JsonSerializer.Serialize(root, new JsonSerializerOptions { WriteIndented = true });
        }

        /// <summary>
        /// Amendment 1, requirement N1: whether tensor cores engaged is reported rather than assumed.
        /// The honest answer for this build of the probe is that they cannot engage at all, and the
        /// reason is a property of the arms, not of the card.
        /// </summary>
        private string TensorCoreStatus()
        {
            var capability = Machine?.Fields
                .FirstOrDefault(kv => kv.Key == "cuda compute capability").Value;
            var capabilityText = string.IsNullOrEmpty(capability)
                ? "not applicable, this is not a CUDA device"
                : capability;

            var fp16Measured = Cells.Any(c => c.Timings.ContainsKey(ArmNames.X1));
            if (!fp16Measured)
            {
                return "NOT engaged - no FP16 arm ran, so nothing in this report could have used them. " +
                       "The custom kernels G1/G2/G3 are FP32 because ILGPU exposes no wmma/mma.sync from a " +
                       "custom kernel, and arm X2 is cuBLAS SGEMM. Every device number below is therefore a " +
                       $"FLOOR. Compute capability: {capabilityText}.";
            }

            // N1 of the plan asks whether tensor cores ENGAGED. Nothing in ILGPU or the CUDA driver API
            // reports that, so the preconditions are stated and the inference is left visible rather than
            // dressed up as an observation. Saying "engaged" on the strength of a capability number would
            // be exactly the kind of claim this probe exists to avoid.
            return $"NOT OBSERVABLE, preconditions reported instead. Arm X1 ran cublasHgemm on a device of " +
                   $"compute capability {capabilityText}; every k, m and n here is a multiple of 8. Tensor " +
                   "cores need capability 7.0 or above, and on Pascal cuBLAS falls back to CUDA cores at " +
                   "about FP32 speed. NO API reports whether they were actually used, so treat a large X1 " +
                   "gain as evidence they engaged and a small one as evidence they did not - the X2 arm " +
                   "beside it is the FP32 control that makes that reading possible.";
        }

        private void AppendTensorCoreLine(StringBuilder sb)
        {
            sb.AppendLine();
            sb.AppendLine("TENSOR CORES: " + TensorCoreStatus());
            sb.AppendLine(
                "  A modern NVIDIA card's FP16 tensor rate is roughly ten times its FP32 rate, so every device");
            sb.AppendLine(
                "  number below is a FLOOR for that card and not its ceiling. Read every ratio that way.");
        }

        private void AppendCell(StringBuilder sb, CellResult cell, HeadlineVerdict run)
        {
            var gflop = cell.ForwardFlops / 1e9;
            sb.AppendLine();
            sb.Append(string.Create(
                CultureInfo.InvariantCulture,
                $"  {cell.Cell.Name}  k {cell.Cell.K} -> m {cell.Cell.M}  n={cell.N}  " +
                $"{gflop:F2} GFLOP per call  weight {cell.Cell.WeightBytesF32 / (1024.0 * 1024.0):F1} MiB"));
            sb.AppendLine(cell.WeightUploadMs > 0
                ? string.Create(CultureInfo.InvariantCulture, $"  uploaded in {cell.WeightUploadMs:F1} ms")
                : string.Empty);

            if (cell.Cell.CallsPerStep > 0)
            {
                sb.AppendLine(string.Create(
                    CultureInfo.InvariantCulture,
                    $"    {cell.Cell.CallsPerStep} calls per Qwen-3B training step, {cell.Cell.MacShare * 100:F1} % of the step's forward MACs"));
            }

            if (cell.Timings.Count == 0)
            {
                sb.AppendLine("    nothing was timed in this cell. The correctness verdicts still stand:");
                foreach (var (arm, parity) in cell.Parity)
                {
                    sb.AppendLine($"    {arm,-24} {Describe(parity)}");
                }

                return;
            }

            if (cell.WarmupStopReason.Length > 0)
            {
                sb.AppendLine($"    warm-up: {cell.WarmupRounds} rounds - {cell.WarmupStopReason}");
            }

            sb.AppendLine("    arm                        min ms   median ms      max ms   spread    GFLOP/s  warm   parity");

            foreach (var (arm, measurement) in cell.Timings)
            {
                if (!cell.MayPrint(arm))
                {
                    sb.AppendLine($"    {arm,-24}  PARITY FAILED - TIMING WITHHELD ({Describe(cell.Parity[arm])})");
                    continue;
                }

                var parity = cell.Parity.TryGetValue(arm, out var p) ? Describe(p) : "reference arm";
                var warm = cell.Warmups.TryGetValue(arm, out var w)
                    ? string.Create(CultureInfo.InvariantCulture, $"{w.Rounds}{(w.Settled ? " ok" : " NO")}")
                    : "?";

                sb.AppendLine(string.Create(
                    CultureInfo.InvariantCulture,
                    $"    {arm,-24} {measurement.MinMs,9:F3} {measurement.MedianMs,11:F3} {measurement.MaxMs,11:F3} " +
                    $"{measurement.SpreadFraction * 100,7:F1}% {measurement.GFlops(cell.ForwardFlops),10:F1} {warm,6}   {parity}"));
            }

            foreach (var (arm, w) in cell.Warmups)
            {
                if (!w.Settled)
                {
                    sb.AppendLine($"    WARM-UP NOT SETTLED  {arm}: {w.Describe()}");
                }
            }

            // The headline the plan names, and only that one. C1 includes a dequantize the device arms do
            // not perform, so C1 against G2 would flatter the device and must never be the headline.
            AppendRatio(sb, cell, run, ArmNames.C3, ArmNames.G2, "HEADLINE forward  C3 cpu f32 -> G2 gpu tiled");
            AppendRatio(sb, cell, run, ArmNames.C4, ArmNames.G3, "         backward C4 cpu f32 -> G3 gpu tiled");
            AppendRatio(sb, cell, run, ArmNames.C3, ArmNames.X1, "         forward  C3 cpu f32 -> X1 cuBLAS fp16");
            AppendRatio(sb, cell, run, ArmNames.C3, ArmNames.X2, "         forward  C3 cpu f32 -> X2 cuBLAS fp32");
            AppendRatio(sb, cell, run, ArmNames.G2, ArmNames.X1, "         our kernel gap G2 -> X1 cuBLAS fp16");
            AppendRatio(sb, cell, run, ArmNames.X2, ArmNames.X1, "         fp32 -> fp16 gain, same library, same card");

            if (cell.Timings.TryGetValue(ArmNames.C1, out var c1) &&
                cell.Timings.TryGetValue(ArmNames.C3, out var c3) &&
                cell.MayPrint(ArmNames.C1))
            {
                var cost = c1.MedianMs - c3.MedianMs;
                sb.AppendLine(string.Create(
                    CultureInfo.InvariantCulture,
                    $"         dequantize cost on the CPU, C1 - C3: {cost:F3} ms, {cost / c1.MedianMs * 100:F1} % of C1. A device-side Q4_K dequantize kernel would have to beat this."));
            }

            foreach (var note in cell.Notes)
            {
                sb.AppendLine("         " + note);
            }
        }

        /// <summary>
        /// One ratio line, or the reasons it is absent. A ratio is never printed beside a warning that
        /// would justify ignoring it: the warning takes the ratio's place.
        /// </summary>
        private static void AppendRatio(
            StringBuilder sb,
            CellResult cell,
            HeadlineVerdict run,
            string baseline,
            string candidate,
            string label)
        {
            // An arm nobody asked for - cuBLAS when --x1 was not passed - is silent rather than blocked.
            // Its absence is already stated once, at the top, and repeating it per cell says nothing.
            if (!cell.Timings.ContainsKey(baseline) || !cell.Timings.ContainsKey(candidate))
            {
                return;
            }

            var verdict = HeadlineVerdict.ForRatio(run, cell, baseline, candidate);
            if (!verdict.MayPrint)
            {
                sb.AppendLine($"    {label}: NOT PRINTED, and that is deliberate. Why:");
                foreach (var blocker in verdict.Blockers)
                {
                    AppendWrapped(sb, "      - ", blocker);
                }

                sb.AppendLine(
                    "      The per-arm timings above still stand as facts about the arms. This ratio would not,");
                sb.AppendLine(
                    "      and a ratio printed next to its own warning gets quoted without the warning.");
                return;
            }

            var b = cell.Timings[baseline];
            var c = cell.Timings[candidate];
            if (c.MedianMs <= 0)
            {
                sb.AppendLine($"    {label}: NOT PRINTED - the candidate arm's median is zero, which is not a timing.");
                return;
            }

            sb.AppendLine(string.Create(
                CultureInfo.InvariantCulture,
                $"    {label}: {b.MedianMs / c.MedianMs:F2}x  (medians; range of the ratio over the samples " +
                $"{b.MinMs / c.MaxMs:F2}x to {b.MaxMs / c.MinMs:F2}x)"));
        }

        private static Dictionary<string, object?> Ratios(CellResult cell, HeadlineVerdict run)
        {
            var pairs = new[]
            {
                ("headlineForward", ArmNames.C3, ArmNames.G2),
                ("backward", ArmNames.C4, ArmNames.G3),
                ("forwardCuBlasFp16", ArmNames.C3, ArmNames.X1),
                ("forwardCuBlasFp32", ArmNames.C3, ArmNames.X2),
                ("ourKernelGap", ArmNames.G2, ArmNames.X1),
                ("fp32ToFp16Gain", ArmNames.X2, ArmNames.X1),
            };

            var result = new Dictionary<string, object?>();
            foreach (var (name, baseline, candidate) in pairs)
            {
                if (!cell.Timings.ContainsKey(baseline) || !cell.Timings.ContainsKey(candidate))
                {
                    continue;
                }

                var verdict = HeadlineVerdict.ForRatio(run, cell, baseline, candidate);
                result[name] = verdict.MayPrint && cell.Timings[candidate].MedianMs > 0
                    ? new Dictionary<string, object?>
                    {
                        ["value"] = cell.Timings[baseline].MedianMs / cell.Timings[candidate].MedianMs,
                        ["suppressed"] = false,
                    }
                    : new Dictionary<string, object?>
                    {
                        ["value"] = null,
                        ["suppressed"] = true,
                        ["reasons"] = verdict.Blockers,
                    };
            }

            return result;
        }

        /// <summary>Wraps a long reason at about 100 columns so a terminal paste stays readable.</summary>
        private static void AppendWrapped(StringBuilder sb, string firstPrefix, string text)
        {
            const int Width = 96;
            var continuation = new string(' ', firstPrefix.Length);
            var prefix = firstPrefix;
            var line = new StringBuilder();

            foreach (var word in text.Split(' ', StringSplitOptions.RemoveEmptyEntries))
            {
                if (line.Length > 0 && line.Length + 1 + word.Length > Width)
                {
                    sb.AppendLine(prefix + line);
                    line.Clear();
                    prefix = continuation;
                }

                if (line.Length > 0)
                {
                    line.Append(' ');
                }

                line.Append(word);
            }

            if (line.Length > 0)
            {
                sb.AppendLine(prefix + line);
            }
        }

        private static string Describe(ParityResult parity) => string.Create(
            CultureInfo.InvariantCulture,
            $"cos {parity.Cosine:F7} relL2 {parity.RelativeL2:E1} (max-rel {parity.MaxRelative:E1}, diagnostic only) " +
            $"{(parity.Passed ? "PASS" : "FAIL: " + parity.Detail)}");

        private static void AppendLimitations(StringBuilder sb)
        {
            sb.AppendLine("WHAT THIS PROBE DOES NOT MEASURE - read before quoting any number above");
            sb.AppendLine();
            sb.AppendLine("  1. It is not a fine-tune. It measures one operation family in isolation.");
            sb.AppendLine("  2. The fraction of a real QLoRA training step spent in FrozenQuantizedLinear has NEVER");
            sb.AppendLine("     been measured in this project. Nothing decomposes the step. So a speedup here does");
            sb.AppendLine("     NOT convert into an end-to-end fine-tune speedup, in either direction. This project");
            sb.AppendLine("     has already measured one kernel benchmark under-reporting a real end-to-end effect");
            sb.AppendLine("     by 4.3x, so the gap is not a small correction.");
            sb.AppendLine("  3. No dequantize runs on the device. The device arms receive an F32 weight. A real port");
            sb.AppendLine("     must either dequantize Q4_K/Q6_K on the device or upload F32 - and the F32 Qwen-3B");
            sb.AppendLine("     base is about 11.5 GB, which does not fit most consumer cards. The probe says nothing");
            sb.AppendLine("     about how fast a device-side Q4_K dequantize would be.");
            sb.AppendLine("  4. The FP16 arm X1 is cublasHgemm, which accumulates in FP16. A TENSOR CORE accumulates");
            sb.AppendLine("     in FP32 and is reached through cublasGemmEx with CUBLAS_COMPUTE_32F - an entry point");
            sb.AppendLine("     ILGPU's cuBLAS wrapper does NOT expose. So X1 is neither the fastest nor the most");
            sb.AppendLine("     accurate FP16 path the card offers, and the true FP16 ceiling is above it. Measured");
            sb.AppendLine("     on the host: FP32-accumulate costs a relative L2 of 2.9e-4 and FP16-accumulate costs");
            sb.AppendLine("     6.5e-3 at k=2048 and 1.6e-2 at k=11008 - a factor of 23 to 55 between the two.");
            sb.AppendLine("  4b. Nothing here says FP16 is numerically USABLE for QLoRA training. Speed and numerical");
            sb.AppendLine("     viability are two questions and this probe answers only the first.");
            sb.AppendLine("  5. No optimizer, no LoRA adapter operations, no autograd tape overhead, no host-device");
            sb.AppendLine("     traffic inside a step, no multi-GPU.");
            sb.AppendLine("  6. It measures OUR kernels, not the card. Arm X1 exists so that, when the CUDA toolkit");
            sb.AppendLine("     is present, the gap between our kernel and cuBLAS is a number rather than a guess.");
            sb.AppendLine("  7. Arm C4 is not in the signed plan's arm table. It was added so the device backward has");
            sb.AppendLine("     a CPU counterpart that performs the same work; arm C2 includes a dequantize that G3");
            sb.AppendLine("     never runs.");
            sb.AppendLine();
        }
    }
}
