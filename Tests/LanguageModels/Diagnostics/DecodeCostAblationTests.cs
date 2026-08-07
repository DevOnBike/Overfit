// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Diagnostics;
using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.LanguageModels.Chat;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Isolates which of the two TTFT changes cost decode throughput.
    ///
    /// <para><b>The observation.</b> Through the server, the full prompt-cache stack drove time to first
    /// token from 74.8 ms to 0.6 ms — and simultaneously pushed inter-token latency from 37.20 to 55.20 ms
    /// and end-to-end from 1228 to 1711 ms. Counting passes says that should be impossible: the old path ran
    /// 33 forward passes for 32 tokens (two before the first emit), the new one runs 32. Less work, more
    /// time, so something per-token got slower and the pass count is not the explanation.</para>
    ///
    /// <para>Two changes landed between the last good measurement and the bad one — emitting the token
    /// before the forward pass that follows it, and keeping the end-of-prompt logits. This runs all four
    /// combinations <b>in one process, interleaved</b>, because cross-process before/after on this box has
    /// already produced a phantom 32% swing on an untouched path.</para>
    /// </summary>
    public sealed class DecodeCostAblationTests
    {
        private const int Rounds = 3;
        private const int NewTokens = 24;

        private readonly ITestOutputHelper _out;

        public DecodeCostAblationTests(ITestOutputHelper output) => _out = output;

        [LongFact("16s")]
        public void DecodeCost_ByEarlyEmitAndLogitsCache()
        {
            var path = TestModelPaths.Qwen3B.Q4KmGgufPath;
            if (!File.Exists(path))
            {
                _out.WriteLine($"missing {path}");
                return;
            }

            using var client = OverfitClient.LoadGguf(
                path, maxContextLength: 1024, maxNewTokens: NewTokens, sampling: SamplingOptions.Greedy);

            const string Prompt = "Explain, in a short paragraph, why running language models locally "
                + "can be useful.";

            // Warm: first turn pays the cold prefill and page-in, which is not what is being compared.
            var options = client.Options;
            client.Chat.Send(Prompt, in options, onText: null, constraint: null);

            var arms = new (string Label, bool EarlyEmit, bool LogitsCache)[]
            {
                ("early-emit OFF, logits-cache OFF", false, false),
                ("early-emit ON,  logits-cache OFF", true, false),
                ("early-emit OFF, logits-cache ON", false, true),
                ("early-emit ON,  logits-cache ON", true, true),
            };

            var best = new double[arms.Length];
            for (var i = 0; i < best.Length; i++)
            {
                best[i] = double.MaxValue;
            }

            // Interleaved: every arm is measured once per round, so a machine drift moves all four together
            // instead of favouring whichever ran first.
            for (var round = 0; round < Rounds; round++)
            {
                for (var a = 0; a < arms.Length; a++)
                {
                    ChatSession.DisableEarlyEmit = !arms[a].EarlyEmit;
                    CachedLlamaSession.DisableLogitsCache = !arms[a].LogitsCache;

                    try
                    {
                        // Re-send the identical prompt: this is the shape the load driver uses and the only
                        // one where the logits cache can fire at all.
                        client.Reset();
                        var started = ValueStopwatch.StartNew();
                        client.Chat.Send(Prompt, in options, onText: null, constraint: null);
                        var elapsed = started.GetElapsedTime().TotalMilliseconds;
                        var generated = client.Chat.LastStats.GeneratedTokens;

                        best[a] = Math.Min(best[a], elapsed / Math.Max(1, generated));
                    }
                    finally
                    {
                        ChatSession.DisableEarlyEmit = false;
                        CachedLlamaSession.DisableLogitsCache = false;
                    }
                }
            }

            _out.WriteLine($"  {"arm",-36}{"ms/token",12}");
            for (var a = 0; a < arms.Length; a++)
            {
                _out.WriteLine($"  {arms[a].Label,-36}{best[a],10:F2}");
            }

            Assert.All(best, b => Assert.True(b > 0));
        }
    }
}
