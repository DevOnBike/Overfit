// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    /// <summary>
    /// End-to-end coherence of the <b>Q8 KV cache</b> decode path on a real model. The KV cache can store K/V
    /// as per-vector symmetric int8 + one F32 scale (<c>KvCacheDType.Q8</c>, ~4× less KV RAM and attention read
    /// traffic for long context). This exercises the full opt-in wiring: <c>OVERFIT_KV_DTYPE=q8</c> →
    /// <c>CreateSession</c> builds a Q8 cache → writes quantize (<c>WriteKey</c>) → single-token decode attends
    /// int8 directly (<c>CachedAttentionKernel.ComputeSingleHeadQ8</c>) → batched prefill dequantizes to F32
    /// scratch. The int8 round-trip is not bit-identical to F32, but cosine ≈ 1, so greedy decode stays
    /// coherent. Pull the model first with <c>overfit pull qwen2.5-0.5b</c>. [LongFact].
    /// </summary>
    public sealed class Q8KvCacheCoherenceTests
    {
        private readonly ITestOutputHelper _out;
        public Q8KvCacheCoherenceTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Q8KvCache_RealModel_GeneratesCoherently()
        {
            var path = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".overfit", "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf");
            if (!File.Exists(path))
            {
                _out.WriteLine($"missing {path} — run 'overfit pull qwen2.5-0.5b'");
                return;
            }

            var previous = Environment.GetEnvironmentVariable("OVERFIT_KV_DTYPE");
            Environment.SetEnvironmentVariable("OVERFIT_KV_DTYPE", "q8");
            try
            {
                using var client = OverfitClient.LoadGguf(path, mmap: true);
                var reply = client.Send("What is the capital of France? Answer with one word.");
                _out.WriteLine($"Q8-KV reply: {reply}");

                Assert.Contains("Paris", reply, StringComparison.OrdinalIgnoreCase);
            }
            finally
            {
                Environment.SetEnvironmentVariable("OVERFIT_KV_DTYPE", previous);
            }
        }
    }
}
