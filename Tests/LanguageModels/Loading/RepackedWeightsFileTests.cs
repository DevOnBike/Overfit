// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// Pins the offline pre-repack sidecar (<see cref="RepackedWeightsFile"/>): the bytes written offline and
    /// read back (memory-mapped) are byte-for-byte what runtime <see cref="Q4KWeight.EnsureRepacked"/> builds —
    /// so a loader that mmaps them instead of repacking at load changes nothing about the computation (no new
    /// coherence risk), it only avoids the heap copy. Also checks dims round-trip and a missing tensor returns
    /// false.
    /// </summary>
    public sealed class RepackedWeightsFileTests
    {
        [Fact]
        public void Sidecar_RoundTrips_ByteIdenticalToRuntimeRepack()
        {
            // Two Q4_K weights of different shapes (both repackable: outputSize % 8 == 0).
            var specs = new[] { (name: "blk.0.ffn_gate", inputSize: 512, outputSize: 64), (name: "blk.0.ffn_up", inputSize: 256, outputSize: 128) };

            var entries = new List<RepackedWeightsFile.Entry>();
            var runtimeRepacks = new Dictionary<string, byte[]>();
            var rng = new Random(99);

            foreach (var (name, inputSize, outputSize) in specs)
            {
                var f32 = new float[outputSize * inputSize];
                for (var i = 0; i < f32.Length; i++)
                {
                    f32[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
                }
                var q4k = GgmlQuant.QuantizeQ4_K(f32, inputSize, outputSize);

                // "Offline tool" path: repack straight from the Q4_K bytes.
                var offline = Q4KRepack.RepackMatrix(q4k, outputSize, inputSize);
                entries.Add(new RepackedWeightsFile.Entry(name, inputSize, outputSize, offline));

                // "Runtime" path: what EnsureRepacked would build at load.
                runtimeRepacks[name] = new Q4KWeight(q4k, inputSize, outputSize).EnsureRepacked().ToArray();
            }

            var path = Path.Combine(Path.GetTempPath(), $"overfit_repack_{Guid.NewGuid():N}.repack");
            try
            {
                RepackedWeightsFile.Write(path, entries);

                using var file = RepackedWeightsFile.Open(path);
                Assert.Equal(specs.Length, file.Count);

                foreach (var (name, inputSize, outputSize) in specs)
                {
                    Assert.True(file.TryGet(name, out var readIn, out var readOut, out var bytes));
                    Assert.Equal(inputSize, readIn);
                    Assert.Equal(outputSize, readOut);
                    Assert.True(bytes.Span.SequenceEqual(runtimeRepacks[name])); // byte-identical to runtime repack
                }

                Assert.False(file.TryGet("does.not.exist", out _, out _, out _));
            }
            finally
            {
                File.Delete(path);
            }
        }

        [LongFact]
        public void BuildFromGguf_RealQwen3B_ProducesOpenableSidecar()
        {
            const string gguf = @"C:\qwen3b\qwen.q4km.gguf";
            if (!File.Exists(gguf))
            {
                return; // model not present
            }

            var path = Path.Combine(Path.GetTempPath(), $"qwen_repack_{Guid.NewGuid():N}.repack");
            try
            {
                var count = RepackedWeightsFile.BuildFromGguf(gguf, path);
                Assert.True(count > 0, "expected at least one repackable Q4_K matmul weight");

                using var file = RepackedWeightsFile.Open(path);
                Assert.Equal(count, file.Count);
                // a first-layer FFN gate weight should be present with a sane contraction dim
                Assert.True(file.TryGet("blk.0.ffn_gate.weight", out var inSize, out var outSize, out var bytes));
                Assert.True(inSize > 0 && outSize % 8 == 0 && bytes.Length > 0);
            }
            finally
            {
                File.Delete(path);
            }
        }
    }
}
