// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Loading;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// One-shot empirical inspection of the Gemma-2 GGUF: dumps ALL scalar metadata (to catch soft-capping /
    /// sliding-window / head_dim / activation keys) + block-0 + top-level tensor names. Grounds the Gemma loader.
    /// [LongFact] — needs C:\gemma.
    /// </summary>
    public sealed class GemmaGgufInspectTests
    {
        private const string Path = @"C:\gemma\gemma-2-2b-it-Q4_K_M.gguf";
        private readonly ITestOutputHelper _out;
        public GemmaGgufInspectTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Dump_Gemma_Metadata_And_Tensors()
        {
            if (!File.Exists(Path))
            {
                _out.WriteLine("missing C:\\gemma gguf");
                return;
            }

            using var reader = new GgufReader(Path);

            _out.WriteLine("=== SCALAR METADATA ===");
            foreach (var kv in reader.Metadata)
            {
                if (kv.Value is object[])
                {
                    continue;
                } // skip big arrays (vocab/merges)
                _out.WriteLine($"{kv.Key} = {kv.Value}");
            }

            _out.WriteLine("\n=== TENSORS: block 0 + top-level ===");
            foreach (var kv in reader.Tensors)
            {
                var name = kv.Key;
                if (name.StartsWith("blk.0.", StringComparison.Ordinal) || !name.StartsWith("blk.", StringComparison.Ordinal))
                {
                    _out.WriteLine($"{name,-32} [{string.Join("×", kv.Value.Dims)}] type={kv.Value.Type}");
                }
            }
            Assert.NotEmpty(reader.Tensors);
        }
    }
}
