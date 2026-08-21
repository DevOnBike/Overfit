// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using ILGPU.Runtime;

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// Entry point. Progress goes to standard error; the report goes to standard output and to two files
    /// beside the executable, so a redirect keeps the paste-back block clean.
    /// </summary>
    internal static class Program
    {
        private const string TextFile = "gpu-probe-report.txt";
        private const string JsonFile = "gpu-probe-report.json";

        private static int Main(string[] args)
        {
            var options = ProbeOptions.Parse(args);
            if (options.Error is not null)
            {
                Console.Error.WriteLine(ProbeOptions.Usage);
                if (options.Error != "help")
                {
                    Console.Error.WriteLine();
                    Console.Error.WriteLine("error: " + options.Error);
                    return 2;
                }

                return 0;
            }

            var log = Console.Error;
            var report = new Report();

            using var selection = DeviceSelection.Open(options);
            var accelerator = selection.Accelerator;

            report.SelectionNote = selection.Note;
            report.IsRealDevice = selection.IsRealDevice;
            report.AcceleratorType = accelerator.AcceleratorType.ToString();
            report.IsCuda = accelerator.AcceleratorType == AcceleratorType.Cuda;
            report.WarmupRule = options.WarmupPolicy.Describe();
            report.Machine = MachineEcho.Collect(accelerator, selection.Context, options);

            log.WriteLine($"accelerator: {accelerator.AcceleratorType} '{accelerator.Name}' ({selection.Note})");

            var groupNeeded = GpuKernels.TileSize * GpuKernels.TileSize;
            if (accelerator.MaxNumThreadsPerGroup < groupNeeded)
            {
                report.TopBanners.Add(
                    $"REFUSED: the tiled kernels need a group of {groupNeeded} threads and this accelerator " +
                    $"allows {accelerator.MaxNumThreadsPerGroup}. Nothing was measured.");
                return Emit(report, 1);
            }

            // The correctness gate comes first and it can veto the whole run. A fast wrong kernel is the
            // failure this probe is most exposed to.
            using (var oracleLog = new StringWriter())
            {
                report.OracleFailures.AddRange(ShapeOracle.Run(accelerator, oracleLog));
                report.OracleLines.AddRange(
                    oracleLog.ToString().Split(Environment.NewLine, StringSplitOptions.RemoveEmptyEntries));
            }

            foreach (var line in report.OracleLines)
            {
                log.WriteLine(line);
            }

            if (report.OracleFailures.Count > 0)
            {
                report.TopBanners.Add(
                    "SHAPE ORACLE FAILED. The device kernels do not compute the right thing at a 3x5x7 shape, " +
                    "so NOTHING was timed. The failures are listed under ORACLES below.");
                return Emit(report, 1);
            }

            if (options.Fp16Bound)
            {
                MeasureFp16Bounds(options, report, log);
                return Emit(report, 0);
            }

            var measure = !options.ParityOnly;
            if (options.ParityOnly)
            {
                report.TopBanners.Add("--parity-only: the oracles ran and nothing was timed.");
            }

            if (!selection.IsRealDevice && !options.AllowCpuAccelerator)
            {
                measure = false;
                report.TopBanners.Add(
                    "NO GPU FOUND, SO NOTHING WAS TIMED. ILGPU offered only its CPU emulator, and a 'GPU' " +
                    "column produced by a CPU emulator is worse than no column at all. The correctness " +
                    "oracles below did run and are valid. Pass --allow-cpu-accelerator to time it anyway.");
            }

            var canary = measure ? new Canary() : null;
            if (canary is not null)
            {
                report.CanaryMeasured = true;
                report.CanaryStart = canary.Measure(options.WarmupPolicy);
                log.WriteLine($"canary at the start: {report.CanaryStart!.MedianMs:F2} ms, {report.CanaryStart.Warmup.Describe()}");
            }

            var cells = SelectCells(options);

            if (cells.Count == 0)
            {
                report.TopBanners.Add($"no cell matched --cells={options.CellFilter}. Nothing ran.");
                return Emit(report, 2);
            }

            foreach (var cell in cells)
            {
                // Once per cell, not once per (cell, token count): the weight does not depend on the
                // batch and quantizing lm_head's 1187 MiB three times dominated the whole run.
                log.WriteLine(
                    $"cell {cell.Name}: quantizing {cell.WeightBytesF32 / (1024.0 * 1024.0):F0} MiB of F32 weight...");
                var weights = new CellWeights(cell, options.Seed);

                foreach (var n in options.Batches)
                {
                    log.WriteLine($"cell {cell.Name} n={n}: running");
                    report.Cells.Add(RunCell(weights, n, options, accelerator, measure, log, report));
                }
            }

            if (canary is not null)
            {
                report.CanaryEnd = canary.Measure(options.WarmupPolicy);
                log.WriteLine($"canary at the end: {report.CanaryEnd!.MedianMs:F2} ms, {report.CanaryEnd.Warmup.Describe()}");
            }

            var anyParityFailure = report.Cells.Any(c => c.Parity.Values.Any(p => !p.Passed));
            return Emit(report, anyParityFailure ? 1 : 0);
        }

        private static CellResult RunCell(
            CellWeights weights,
            int n,
            ProbeOptions options,
            Accelerator accelerator,
            bool measure,
            TextWriter log,
            Report report)
        {
            var cell = weights.Cell;
            var result = new CellResult(cell, n);
            using var fixture = new CellFixture(weights, n, options.Seed);

            // The two CPU references, computed once and outside every clock. Everything else in this cell
            // is compared against them.
            fixture.RunC3Forward();
            fixture.RunC4Backward();

            // Cross-check inside the host: the REAL quantized op against the dequantize-free reference.
            // If these two disagree, the F32 weight handed to the device is not the weight the CPU arm
            // uses, and every parity verdict below would be measuring the wrong thing.
            var hostOutput = new float[(long)n * cell.M];
            fixture.ReadC1Output(hostOutput);
            result.Parity[ArmNames.C1] = ParityResult.Compare(fixture.OutputF32, hostOutput);

            var hostInputGrad = new float[(long)n * cell.K];
            fixture.SeedC2();
            fixture.RunC2Backward();
            fixture.ReadC2InputGrad(hostInputGrad);
            result.Parity[ArmNames.C2] = ParityResult.Compare(fixture.InputGradF32, hostInputGrad);

            using var gpu = new GpuArms(accelerator, n, cell.K, cell.M);
            result.WeightUploadMs = gpu.UploadWeight(fixture.WeightF32);
            gpu.UploadInput(fixture.Input);
            gpu.UploadOutputGrad(fixture.OutputGrad);
            log.WriteLine($"  weight upload: {result.WeightUploadMs:F1} ms");

            gpu.RunNaiveForward();
            accelerator.Synchronize();
            gpu.ReadOutput(hostOutput);
            result.Parity[ArmNames.G1] = ParityResult.Compare(fixture.OutputF32, hostOutput);

            gpu.RunTiledForward();
            accelerator.Synchronize();
            gpu.ReadOutput(hostOutput);
            result.Parity[ArmNames.G2] = ParityResult.Compare(fixture.OutputF32, hostOutput);

            gpu.RunBackward();
            accelerator.Synchronize();
            gpu.ReadInputGrad(hostInputGrad);
            result.Parity[ArmNames.G3] = ParityResult.Compare(fixture.InputGradF32, hostInputGrad);

            using var cuBlas = CreateCuBlas(options, accelerator, report);
            if (cuBlas is not null)
            {
                cuBlas.Forward(gpu.InputView, gpu.WeightView, gpu.OutputView, n, cell.K, cell.M);
                accelerator.Synchronize();
                gpu.ReadOutput(hostOutput);
                result.Parity[ArmNames.X2] = ParityResult.Compare(fixture.OutputF32, hostOutput);

                // X1 is judged against the FP16-ACCUMULATE ceiling, which depends on k and was measured
                // by --fp16-bound. Judging it against the F32 ceiling would fail a correct hgemm.
                if (cuBlas.TryPrepareFp16(fixture.Input, fixture.WeightF32, n, cell.K, cell.M))
                {
                    cuBlas.ForwardFp16(n, cell.K, cell.M);
                    accelerator.Synchronize();
                    cuBlas.ReadFp16Output(hostOutput);
                    result.Parity[ArmNames.X1] = ParityResult.Compare(
                        fixture.OutputF32, hostOutput, ParityResult.Fp16Fp16AccumulateCeiling(cell.K));
                }

                if (cuBlas.Fp16Unavailable is not null)
                {
                    report.Fp16SkipReason ??= cuBlas.Fp16Unavailable;
                }
            }

            foreach (var (arm, parity) in result.Parity)
            {
                log.WriteLine($"  parity {arm,-24} cos {parity.Cosine:F7} maxRel {parity.MaxRelative:E1} " +
                              (parity.Passed ? "PASS" : "FAIL - " + parity.Detail));
            }

            if (!measure)
            {
                return result;
            }

            var arms = new List<Arm>
            {
                Arm.Cpu(ArmNames.C1, fixture.RunC1Forward, after: fixture.ResetC1),
                Arm.Cpu(ArmNames.C2, fixture.RunC2Backward, before: fixture.SeedC2),
                Arm.Cpu(ArmNames.C3, fixture.RunC3Forward),
                Arm.Cpu(ArmNames.C4, fixture.RunC4Backward),
                Arm.Gpu(ArmNames.G1, accelerator, gpu.RunNaiveForward),
                Arm.Gpu(ArmNames.G2, accelerator, gpu.RunTiledForward),
                Arm.Gpu(ArmNames.G3, accelerator, gpu.RunBackward),
            };

            if (cuBlas is not null)
            {
                arms.Add(Arm.Gpu(
                    ArmNames.X2,
                    accelerator,
                    () => cuBlas.Forward(gpu.InputView, gpu.WeightView, gpu.OutputView, n, cell.K, cell.M)));

                if (cuBlas.Fp16Ready)
                {
                    arms.Add(Arm.Gpu(ArmNames.X1, accelerator, () => cuBlas.ForwardFp16(n, cell.K, cell.M)));
                }
            }

            var run = ArmRunner.Interleave(arms, options.WarmupPolicy, options.Reps, log.WriteLine);
            result.WarmupRounds = run.WarmupRounds;
            result.WarmupStopReason = run.WarmupStopReason;

            foreach (var arm in arms)
            {
                result.Timings[arm.Name] = run.Timings[arm.Name];
                result.Warmups[arm.Name] = run.Warmups[arm.Name];
                log.WriteLine(
                    $"  {arm.Name,-24} median {run.Timings[arm.Name].MedianMs,10:F3} ms   " +
                    run.Warmups[arm.Name].Describe());
            }

            return result;
        }

        /// <summary>
        /// Runs the host-side FP16 accuracy measurement at every selected shape and puts the result in
        /// the report. No device is touched, so this is the one part of the FP16 work that can be
        /// answered on a machine with no NVIDIA card.
        /// </summary>
        private static void MeasureFp16Bounds(ProbeOptions options, Report report, TextWriter log)
        {
            report.TopBanners.Add(
                "--fp16-bound: this run measured what FP16 costs in ACCURACY and timed nothing at all. " +
                "Speed and numerical viability are two questions and this answers only the second.");

            foreach (var cell in SelectCells(options))
            {
                log.WriteLine($"cell {cell.Name}: quantizing for the FP16 bound...");
                var weights = new CellWeights(cell, options.Seed);

                foreach (var n in options.Batches)
                {
                    log.WriteLine($"cell {cell.Name} n={n}: measuring the FP16 bound");
                    using var fixture = new CellFixture(weights, n, options.Seed);
                    fixture.RunC3Forward();

                    var fp32Accumulate = new float[(long)n * cell.M];
                    Fp16Reference.ForwardFp32Accumulate(
                        fixture.Input, fixture.WeightF32, fp32Accumulate, n, cell.K, cell.M);
                    var withFp32Acc = ParityResult.Compare(fixture.OutputF32, fp32Accumulate);

                    var withFp16Acc = Fp16Reference.RelativeL2WithFp16Accumulate(
                        fixture.Input, fixture.WeightF32, fixture.OutputF32,
                        n, cell.K, cell.M, options.Seed, out var sampled);

                    var line = string.Create(
                        System.Globalization.CultureInfo.InvariantCulture,
                        $"{cell.Name,-14} k {cell.K,6} -> m {cell.M,6}  n={n,4}   " +
                        $"fp32 acc {withFp32Acc.RelativeL2:E2}   fp16 acc {withFp16Acc:E2} " +
                        $"(from {sampled} sampled output elements)");
                    report.Fp16Bounds.Add(line);
                    log.WriteLine("  " + line);
                }
            }
        }

        private static List<Cell> SelectCells(ProbeOptions options) =>
            (options.Quick ? Cell.Quick : Cell.Production)
            .Where(c => options.CellFilter.Length == 0 ||
                        c.Name.Contains(options.CellFilter, StringComparison.OrdinalIgnoreCase))
            .ToList();

        private static CuBlasArm? CreateCuBlas(ProbeOptions options, Accelerator accelerator, Report report)
        {
            if (!options.EnableCuBlas)
            {
                report.CuBlasSkipReason ??=
                    "not requested. Pass --x1 to measure it; it needs the CUDA REDISTRIBUTABLE installed " +
                    "(cublas64_*.dll), which the rest of this probe deliberately does not.";
                return null;
            }

            var arm = CuBlasArm.TryCreate(accelerator);
            if (arm is null)
            {
                report.CuBlasSkipReason ??= CuBlasArm.Unavailable;
            }

            return arm;
        }

        private static int Emit(Report report, int exitCode)
        {
            var text = report.RenderText();
            Console.Out.WriteLine(text);

            try
            {
                File.WriteAllText(TextFile, text);
                File.WriteAllText(JsonFile, report.RenderJson());
                Console.Error.WriteLine($"written: {TextFile} and {JsonFile}");
            }
            catch (IOException ex)
            {
                Console.Error.WriteLine($"could not write the report files ({ex.Message}); the text above is the report.");
            }
            catch (UnauthorizedAccessException ex)
            {
                Console.Error.WriteLine($"could not write the report files ({ex.Message}); the text above is the report.");
            }

            return exitCode;
        }
    }
}
