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

        public double CanaryStartMs { get; set; }

        public double CanaryEndMs { get; set; }

        public bool CanaryMeasured { get; set; }

        public List<string> OracleLines { get; } = [];

        public List<string> OracleFailures { get; } = [];

        public List<CellResult> Cells { get; } = [];

        public List<string> TopBanners { get; } = [];

        public string? CuBlasSkipReason { get; set; }

        public double CanaryMove =>
            CanaryStartMs > 0 ? (CanaryEndMs - CanaryStartMs) / CanaryStartMs : 0;

        public string RenderText()
        {
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
            }

            if (!IsRealDevice)
            {
                sb.AppendLine(
                    "NOTE: the 'GPU' arms did NOT run on a GPU. They ran on ILGPU's CPU emulator, so every");
                sb.AppendLine(
                    "      device timing below is a fact about a CPU and about ILGPU's emulator, nothing else.");
            }

            sb.AppendLine();
            sb.AppendLine("MACHINE AND CONFIGURATION");
            sb.AppendLine("  accelerator choice : " + SelectionNote);
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

            if (CuBlasSkipReason is not null)
            {
                sb.AppendLine();
                sb.AppendLine("ARM X1 (cuBLAS): NOT MEASURED - " + CuBlasSkipReason);
            }

            sb.AppendLine();
            sb.AppendLine("RESULTS");
            sb.AppendLine("  ms per call, min / median / max over the timed repetitions, arms interleaved ABAB.");
            sb.AppendLine("  A cell whose parity check failed shows no timing at all - that is deliberate.");

            foreach (var cell in Cells)
            {
                AppendCell(sb, cell);
            }

            sb.AppendLine();
            AppendLimitations(sb);
            sb.AppendLine(End);
            return sb.ToString();
        }

        public string RenderJson()
        {
            var root = new Dictionary<string, object?>
            {
                ["machine"] = Machine?.Fields.ToDictionary(kv => kv.Key, kv => kv.Value),
                ["acceleratorChoice"] = SelectionNote,
                ["isRealDevice"] = IsRealDevice,
                ["canaryStartMs"] = CanaryMeasured ? CanaryStartMs : null,
                ["canaryEndMs"] = CanaryMeasured ? CanaryEndMs : null,
                ["canaryMoveFraction"] = CanaryMeasured ? CanaryMove : null,
                ["oracle"] = OracleLines,
                ["oracleFailures"] = OracleFailures,
                ["cuBlasSkipReason"] = CuBlasSkipReason,
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
                        }),
                    ["parity"] = c.Parity.ToDictionary(p => p.Key, p => (object)new Dictionary<string, object?>
                    {
                        ["passed"] = p.Value.Passed,
                        ["cosine"] = p.Value.Cosine,
                        ["maxRelative"] = p.Value.MaxRelative,
                        ["detail"] = p.Value.Detail,
                    }),
                }),
                ["ramTypeAndSpeed"] = null,
                ["otherLoadDuringRun"] = null,
            };

            return JsonSerializer.Serialize(root, new JsonSerializerOptions { WriteIndented = true });
        }

        private void AppendCell(StringBuilder sb, CellResult cell)
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

            sb.AppendLine("    arm                        min ms   median ms      max ms   spread    GFLOP/s   parity");

            foreach (var (arm, measurement) in cell.Timings)
            {
                if (!cell.MayPrint(arm))
                {
                    sb.AppendLine($"    {arm,-24}  PARITY FAILED - TIMING WITHHELD ({Describe(cell.Parity[arm])})");
                    continue;
                }

                var parity = cell.Parity.TryGetValue(arm, out var p) ? Describe(p) : "reference arm";
                sb.AppendLine(string.Create(
                    CultureInfo.InvariantCulture,
                    $"    {arm,-24} {measurement.MinMs,9:F3} {measurement.MedianMs,11:F3} {measurement.MaxMs,11:F3} " +
                    $"{measurement.SpreadFraction * 100,7:F1}% {measurement.GFlops(cell.ForwardFlops),10:F1}   {parity}"));
            }

            // The headline the plan names, and only that one. C1 includes a dequantize the device arms do
            // not perform, so C1 against G2 would flatter the device and must never be the headline.
            AppendRatio(sb, cell, ArmNames.C3, ArmNames.G2, "HEADLINE forward  C3 cpu f32 -> G2 gpu tiled");
            AppendRatio(sb, cell, ArmNames.C4, ArmNames.G3, "         backward C4 cpu f32 -> G3 gpu tiled");
            AppendRatio(sb, cell, ArmNames.C3, ArmNames.X1, "         forward  C3 cpu f32 -> X1 cuBLAS");
            AppendRatio(sb, cell, ArmNames.G2, ArmNames.X1, "         our kernel gap G2 -> X1 cuBLAS");

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

        private static void AppendRatio(StringBuilder sb, CellResult cell, string baseline, string candidate, string label)
        {
            if (!cell.Timings.TryGetValue(baseline, out var b) || !cell.Timings.TryGetValue(candidate, out var c))
            {
                return;
            }

            if (!cell.MayPrint(baseline) || !cell.MayPrint(candidate) || c.MedianMs <= 0)
            {
                return;
            }

            sb.AppendLine(string.Create(
                CultureInfo.InvariantCulture,
                $"    {label}: {b.MedianMs / c.MedianMs:F2}x  (medians; range of the ratio over the samples " +
                $"{b.MinMs / c.MaxMs:F2}x to {b.MaxMs / c.MinMs:F2}x)"));
        }

        private static string Describe(ParityResult parity) => string.Create(
            CultureInfo.InvariantCulture,
            $"cos {parity.Cosine:F7} maxRel {parity.MaxRelative:E1} {(parity.Passed ? "PASS" : "FAIL: " + parity.Detail)}");

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
            sb.AppendLine("  4. FP32 CUDA cores only. ILGPU has no tensor-core path. On a modern NVIDIA card the");
            sb.AppendLine("     BF16/TF32 tensor rate is roughly an order of magnitude above the FP32 rate, so this");
            sb.AppendLine("     is the FLOOR of what the hardware can do, not its ceiling.");
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
