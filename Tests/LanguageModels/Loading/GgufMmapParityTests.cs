// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// Parity guard for the zero-copy mmap GGUF path: a Q4_K_M model loaded with
    /// <c>mmap: true</c> (verbatim Q4_K/Q6_K weights backed by file-mapped slices)
    /// must produce <b>bit-identical</b> logits to the same model loaded with the
    /// managed-byte-copy path. The block bytes are the same either way — only their
    /// backing store differs — so any drift means the slice offsets/lengths are wrong.
    /// </summary>
    [Trait("Category", "Qwen")]
    [Trait("Category", "Quantization")]
    [Trait("Category", "Parity")]
    public sealed class GgufMmapParityTests
    {
        // Canonical 3-token prompt shared by the Qwen suite: [BOS, im_start, "\n"].
        private static readonly int[] Prompt = [151643, 151644, 198];

        private readonly ITestOutputHelper _out;

        public GgufMmapParityTests(ITestOutputHelper output)
        {
            _out = output;
        }

        [LongFact]
        public void MmapPath_ProducesBitIdenticalLogits_ToCopyPath()
        {
            TestModelPaths.Qwen3B.RequireQ4KmGgufPath();
            var path = TestModelPaths.Qwen3B.Q4KmGgufPath;

            var copyLogits = RunOneStep(path, mmap: false);
            var mmapLogits = RunOneStep(path, mmap: true);

            Assert.Equal(copyLogits.Length, mmapLogits.Length);

            var maxDiff = 0f;
            var firstMismatch = -1;
            for (var i = 0; i < copyLogits.Length; i++)
            {
                var d = MathF.Abs(copyLogits[i] - mmapLogits[i]);
                if (d > maxDiff) { maxDiff = d; }
                if (d != 0f && firstMismatch < 0) { firstMismatch = i; }
            }

            _out.WriteLine($"vocab = {copyLogits.Length}, max abs logit diff = {maxDiff:G9}");
            if (firstMismatch >= 0)
            {
                _out.WriteLine(
                    $"first mismatch @ {firstMismatch}: copy={copyLogits[firstMismatch]:G9} mmap={mmapLogits[firstMismatch]:G9}");
            }

            // Same bytes, same kernels — the result must be exactly equal.
            Assert.Equal(0f, maxDiff);
        }

        [LongFact]
        public void Mmap_MeasuredManagedAllocationAndWorkingSet()
        {
            TestModelPaths.Qwen3B.RequireQ4KmGgufPath();
            var path = TestModelPaths.Qwen3B.Q4KmGgufPath;

            var (copyAlloc, copyWs) = MeasureLoad(path, mmap: false);
            var (mmapAlloc, mmapWs) = MeasureLoad(path, mmap: true);

            _out.WriteLine($"file size           = {new FileInfo(path).Length / (1024.0 * 1024):F0} MB");
            _out.WriteLine($"copy managed alloc  = {copyAlloc / (1024.0 * 1024):F0} MB   working set delta = {copyWs / (1024.0 * 1024):F0} MB");
            _out.WriteLine($"mmap managed alloc  = {mmapAlloc / (1024.0 * 1024):F0} MB   working set delta = {mmapWs / (1024.0 * 1024):F0} MB");
            _out.WriteLine($"managed-heap saved  = {(copyAlloc - mmapAlloc) / (1024.0 * 1024):F0} MB");

            // The mmap path must allocate strictly less managed memory than the copy path:
            // the verbatim Q4_K/Q6_K weights move off the managed heap onto file-mapped pages.
            Assert.True(mmapAlloc < copyAlloc,
                $"mmap managed alloc ({mmapAlloc}) not below copy ({copyAlloc}).");
        }

        private static (long managedAlloc, long workingSetDelta) MeasureLoad(string path, bool mmap)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            var allocBefore = GC.GetTotalAllocatedBytes(precise: true);
            using var proc = System.Diagnostics.Process.GetCurrentProcess();
            proc.Refresh();
            var wsBefore = proc.WorkingSet64;

            using (var engine = CachedLlamaInferenceEngine.LoadGguf(path, mmap: mmap))
            {
                using var session = engine.CreateSession(64);
                session.Reset(Prompt);
                var sink = session.LastLogits[0];
                GC.KeepAlive(sink);

                var allocAfter = GC.GetTotalAllocatedBytes(precise: true);
                proc.Refresh();
                var wsAfter = proc.WorkingSet64;
                return (allocAfter - allocBefore, wsAfter - wsBefore);
            }
        }

        private static float[] RunOneStep(string path, bool mmap)
        {
            using var engine = CachedLlamaInferenceEngine.LoadGguf(path, mmap: mmap);
            using var session = engine.CreateSession(64);
            session.Reset(Prompt);
            return session.LastLogits.ToArray();
        }
    }
}
