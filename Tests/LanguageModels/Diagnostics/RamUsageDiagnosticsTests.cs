// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Diagnostic tests that print precise RAM usage at every stage of model loading
    /// and inference. Run with `-v n` (normal verbosity) to see the output.
    ///
    /// Three metrics reported per checkpoint:
    /// - Managed RAM (GC.GetTotalMemory): allocated managed heap
    /// - Working set (Process.WorkingSet64): physical memory mapped to the process
    /// - Private bytes (Process.PrivateMemorySize64): committed memory (managed + native)
    /// </summary>
    [Trait("Category", "Diagnostics")]
    public sealed class RamUsageDiagnosticsTests
    {
        private static string GgufModelPath   => TestModelPaths.Qwen3B.GgufPath;
        private static string BinaryModelPath => TestModelPaths.Qwen3B.BinaryPath;
        private static string TokenizerDir    => TestModelPaths.Qwen3B.Dir;

        private readonly ITestOutputHelper _output;

        public RamUsageDiagnosticsTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [LongFact]
        public void Diagnose_GgufLoader_3B_RamFootprint()
        {
            TestModelPaths.Qwen3B.RequireGgufPath();

            PrintCheckpoint("00. Test start (cold)");

            using (var engine = CachedLlamaInferenceEngine.LoadGguf(GgufModelPath))
            {
                PrintCheckpoint("01. After LoadGguf returned");

                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
                PrintCheckpoint("02. After GC.Collect (post-load)");

                using (var session = engine.CreateSession(64))
                {
                    PrintCheckpoint("03. After CreateSession(64)");

                    int[] prompt = [151643, 151644, 198];
                    session.Reset(prompt);
                    PrintCheckpoint("04. After session.Reset(3 tokens)");

                    var logits = session.LastLogits.ToArray();
                    PrintCheckpoint("05. After LastLogits.ToArray()");

                    // Generate 5 tokens
                    for (var i = 0; i < 5; i++)
                    {
                        session.GenerateNextToken(SamplingOptions.Greedy);
                    }
                    PrintCheckpoint("06. After 5 tokens generated");
                }

                PrintCheckpoint("07. After session disposed");
            }

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            PrintCheckpoint("08. After engine disposed + GC");
        }

        [LongFact]
        public void Diagnose_BinaryLoader_3B_RamFootprint()
        {
            TestModelPaths.Qwen3B.RequireBinaryPath();

            PrintCheckpoint("00. Test start (cold)");

            using (var engine = CachedLlamaInferenceEngine.Load(BinaryModelPath))
            {
                PrintCheckpoint("01. After Load (binary) returned");

                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
                PrintCheckpoint("02. After GC.Collect (post-load)");

                using (var session = engine.CreateSession(64))
                {
                    PrintCheckpoint("03. After CreateSession(64)");

                    int[] prompt = [151643, 151644, 198];
                    session.Reset(prompt);
                    PrintCheckpoint("04. After session.Reset(3 tokens)");
                }

                PrintCheckpoint("05. After session disposed");
            }

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            PrintCheckpoint("06. After engine disposed + GC");
        }

        private void PrintCheckpoint(string label)
        {
            // Force GC pre-measurement for stable readings
            var managed = GC.GetTotalMemory(forceFullCollection: false);

            var proc = Process.GetCurrentProcess();
            proc.Refresh();
            var workingSet = proc.WorkingSet64;
            var privateBytes = proc.PrivateMemorySize64;

            _output.WriteLine(
                $"{label,-50} | " +
                $"Managed: {managed / 1_000_000_000.0,6:F2} GB | " +
                $"Working: {workingSet / 1_000_000_000.0,6:F2} GB | " +
                $"Private: {privateBytes / 1_000_000_000.0,6:F2} GB");
        }
    }
}
