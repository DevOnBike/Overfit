// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Loading;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// One-shot empirical inspection of the Phi-3.5-mini GGUF: dumps arch + rope-related metadata and the tensor
    /// names (block 0 + any top-level rope-factor tensors) so the loader is grounded in what llama.cpp actually
    /// stored — fused vs split QKV, fused gate_up, longrope short/long factor tensors. [LongFact] — needs C:\phi.
    /// </summary>
    public sealed class PhiGgufInspectTests
    {
        private const string Path = @"C:\phi\Phi-3.5-mini-instruct-Q4_K_M.gguf";
        private readonly ITestOutputHelper _out;
        public PhiGgufInspectTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Dump_Phi_Metadata_And_Tensors()
        {
            if (!File.Exists(Path)) { _out.WriteLine("missing C:\\phi gguf"); return; }

            using var reader = new GgufReader(Path);

            _out.WriteLine("=== METADATA (arch + rope + attention) ===");

            foreach (var kv in reader.Metadata)
            {
                var k = kv.Key;
                if (k.Contains("rope", StringComparison.OrdinalIgnoreCase)
                    || k.Contains("architecture", StringComparison.OrdinalIgnoreCase)
                    || k.Contains("attention", StringComparison.OrdinalIgnoreCase)
                    || k.Contains("embedding_length", StringComparison.OrdinalIgnoreCase)
                    || k.Contains("block_count", StringComparison.OrdinalIgnoreCase)
                    || k.Contains("context_length", StringComparison.OrdinalIgnoreCase)
                    || k.Contains("feed_forward", StringComparison.OrdinalIgnoreCase))
                {
                    var v = kv.Value is object[] arr ? $"[array len {arr.Length}]" : kv.Value?.ToString();
                    _out.WriteLine($"{k} = {v}");
                }
            }

            _out.WriteLine("\n=== TENSORS: block 0 + top-level rope/output/token ===");
            foreach (var kv in reader.Tensors)
            {
                var name = kv.Key;
                if (name.StartsWith("blk.0.", StringComparison.Ordinal)
                    || name.Contains("rope", StringComparison.OrdinalIgnoreCase)
                    || (!name.StartsWith("blk.", StringComparison.Ordinal)))
                {
                    var dims = string.Join("×", kv.Value.Dims);
                    _out.WriteLine($"{name,-32} [{dims}] type={kv.Value.Type}");
                }
            }

            Assert.NotEmpty(reader.Tensors);
        }
    }
}
