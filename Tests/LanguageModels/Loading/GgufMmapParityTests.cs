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



        // ── Default-flip soak: mmap is now the default. These guard that the two
        //    non-K-quant precisions are untouched by the flip. The loader skips the
        //    map entirely when the file has no Q4_K/Q6_K tensors (pure-Q8_0 / pure-FP16),
        //    so mmap:true must be byte-for-byte identical to mmap:false there too.

        [LongFact]
        public void Q8_0Model_DefaultMmap_IdenticalToCopyPath()
        {
            TestModelPaths.Qwen3B.RequireQ8GgufPath();
            AssertBitIdentical(TestModelPaths.Qwen3B.Q8GgufPath);
        }

        [LongFact]
        public void Fp16Model_DefaultMmap_IdenticalToCopyPath()
        {
            TestModelPaths.Qwen3B.RequireGgufPath();
            AssertBitIdentical(TestModelPaths.Qwen3B.GgufPath);
        }

        private void AssertBitIdentical(string path)
        {
            var copyLogits = RunOneStep(path, mmap: false);
            var mmapLogits = RunOneStep(path, mmap: true);

            Assert.Equal(copyLogits.Length, mmapLogits.Length);

            var maxDiff = 0f;
            for (var i = 0; i < copyLogits.Length; i++)
            {
                var d = MathF.Abs(copyLogits[i] - mmapLogits[i]);
                if (d > maxDiff) { maxDiff = d; }
            }

            _out.WriteLine($"{Path.GetFileName(path)}: vocab = {copyLogits.Length}, max abs logit diff = {maxDiff:G9}");
            Assert.Equal(0f, maxDiff);
        }

        [LongFact]
        public void Mmap_MeasuredResidentManagedHeap()
        {
            TestModelPaths.Qwen3B.RequireQ4KmGgufPath();
            var path = TestModelPaths.Qwen3B.Q4KmGgufPath;

            var copyLive = MeasureLiveManagedHeap(path, mmap: false);
            var mmapLive = MeasureLiveManagedHeap(path, mmap: true);

            // The F32 embedding the loader USED to keep on the heap, for context (vocab × dModel × 4).
            using var probe = CachedLlamaInferenceEngine.LoadGguf(path, mmap: true);
            var cfg = probe.Config;
            var f32EmbedMb = (long)cfg.VocabSize * cfg.DModel * sizeof(float) / (1024.0 * 1024);

            _out.WriteLine($"file size                         = {new FileInfo(path).Length / (1024.0 * 1024):F0} MB");
            _out.WriteLine($"OLD F32 embedding (now eliminated)= {f32EmbedMb:F0} MB");
            _out.WriteLine($"live managed heap, copy (mmap off)= {copyLive / (1024.0 * 1024):F0} MB");
            _out.WriteLine($"live managed heap, mmap (mmap on) = {mmapLive / (1024.0 * 1024):F0} MB");
            _out.WriteLine($"copy→mmap resident saved          = {(copyLive - mmapLive) / (1024.0 * 1024):F0} MB");

            // mmap moves the verbatim K-quant weights (incl. the Q6_K embedding) off the managed
            // heap, so the live resident managed heap must be strictly smaller than the copy path.
            Assert.True(mmapLive < copyLive,
                $"mmap live heap ({mmapLive}) not below copy ({copyLive}).");
        }

        /// <summary>Live (post-full-GC) resident managed heap with the model loaded.</summary>
        private static long MeasureLiveManagedHeap(string path, bool mmap)
        {
            using (var engine = CachedLlamaInferenceEngine.LoadGguf(path, mmap: mmap))
            {
                using var session = engine.CreateSession(64);
                session.Reset(Prompt);
                var sink = session.LastLogits[0];
                GC.KeepAlive(sink);

                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
                return GC.GetTotalMemory(forceFullCollection: true);
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
