// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// Regression for non-uniform per-layer quantization in a GGUF. The official Qwen2.5-0.5B Q4_K_M has
    /// hidden=896 (not a multiple of 256), so its attention tensors fall back to Q5_0 / Q8_0 — and
    /// <c>attn_v</c>'s quant VARIES across layers (some Q8_0, some Q5_0). The loader used to peek only
    /// <c>blk.0</c> to decide which F32 scratch to rent, so a later F32-fallback layer hit an un-rented buffer
    /// (ArgumentOutOfRangeException). The fix scans every layer; this test loads the model and checks it still
    /// generates coherently. Pull it first with <c>overfit pull qwen2.5-0.5b</c>. [LongFact].
    /// </summary>
    public sealed class NonUniformQuantGgufLoadTests
    {
        private readonly ITestOutputHelper _out;
        public NonUniformQuantGgufLoadTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void OfficialQwen05B_NonUniformQuant_LoadsAndGenerates()
        {
            var path = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".overfit", "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf");
            if (!File.Exists(path))
            {
                _out.WriteLine($"missing {path} — run 'overfit pull qwen2.5-0.5b'");
                return;
            }

            using var client = OverfitClient.LoadGguf(path, mmap: true);
            var reply = client.Send("What is the capital of France? Answer with one word.");
            _out.WriteLine($"reply: {reply}");

            Assert.Contains("Paris", reply, StringComparison.OrdinalIgnoreCase);
        }
    }
}
