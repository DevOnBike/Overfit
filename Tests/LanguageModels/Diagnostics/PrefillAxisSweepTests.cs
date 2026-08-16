// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Diagnostics;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Both parallelisation axes and both tile widths, at the prompt lengths a chat server actually sees,
    /// against the floor set by a single decode step.
    ///
    /// <para><b>The gap being chased.</b> Measured through an OpenAI-protocol load test, a competing pure-.NET
    /// engine answers a chat-sized prompt with ~44 ms to first token; we take ~300 ms. A server phase trace
    /// showed only ~26 ms of that is HTTP and JSON — the rest is prefill of the chat-templated prompt (roughly
    /// 50 tokens once role markers and the system turn are added).</para>
    ///
    /// <para><b>The floor.</b> One decode step reads every weight once and costs ~56 ms. A 50-token prefill
    /// should cost about one such pass plus arithmetic; theirs does, and ours costs about five. Whatever the
    /// cause, it is structural rather than kernel quality — the same kernels decode at a competitive rate.</para>
    ///
    /// <para>This measures the three candidate configurations side by side so the choice is made on numbers:
    /// the automatic axis rule, forced banding, and forced column tiling, each at the tile width in use and at
    /// the widest the kernel supports.</para>
    /// </summary>
    public sealed class PrefillAxisSweepTests
    {
        private static readonly int[] Lengths = [24, 48, 96];

        private const int Repeats = 3;

        private readonly ITestOutputHelper _out;

        public PrefillAxisSweepTests(ITestOutputHelper output) => _out = output;

        [LongFact("24s")]
        public void Prefill_AxisAndTileWidth_AtChatPromptLengths()
        {
            var path = TestModelPaths.Qwen3B.Q4KmGgufPath;
            if (!File.Exists(path))
            {
                _out.WriteLine($"missing {path}");
                return;
            }

            using var engine = CachedLlamaInferenceEngine.LoadGguf(path);
            var tok = GgufTokenizer.Load(path);
            var sampling = SamplingOptions.Greedy;

            var paragraph = string.Join(" ",
                Enumerable.Repeat(
                    "The history of computing began with mechanical calculators and evolved through vacuum tubes, "
                    + "transistors, integrated circuits and finally the microprocessor era.", 40));
            var allIds = tok.Encode(paragraph);

            using (var warm = engine.CreateSession(1024))
            {
                warm.Reset(allIds.AsSpan(0, 64).ToArray());
                warm.GenerateNextToken(in sampling);
            }

            // The floor: one decode step is one full pass over the weights.
            double decodeMs;
            {
                using var session = engine.CreateSession(1024);
                session.Reset(allIds.AsSpan(0, 16).ToArray());
                session.GenerateNextToken(in sampling);

                var started = ValueStopwatch.StartNew();
                for (var i = 0; i < 5; i++)
                {
                    session.GenerateNextToken(in sampling);
                }

                decodeMs = started.GetElapsedTime().TotalMilliseconds / 5;
            }

            _out.WriteLine($"one decode step (= one weight pass): {decodeMs:F1} ms");
            _out.WriteLine(string.Empty);
            _out.WriteLine($"  {"tokens",7}{"NR",4}{"auto",11}{"banded",11}{"tiled",11}{"best/decode",13}");

            foreach (var length in Lengths)
            {
                var ids = allIds.AsSpan(0, length).ToArray();

                foreach (var nr in new[] { 0, 16 })
                {
                    BatchedQuantProjection.TileColsOverride = nr;

                    var auto = Best(engine, ids, forceBanding: null);
                    var banded = Best(engine, ids, forceBanding: true);
                    var tiled = Best(engine, ids, forceBanding: false);
                    var best = Math.Min(auto, Math.Min(banded, tiled));

                    _out.WriteLine(
                        $"  {length,7}{(nr == 0 ? 8 : nr),4}{auto,8:F1} ms{banded,8:F1} ms{tiled,8:F1} ms"
                        + $"{best / decodeMs,12:F1}x");
                }

                BatchedQuantProjection.TileColsOverride = 0;
            }

            Assert.True(decodeMs > 0);
        }

        /// <summary>
        /// <paramref name="forceBanding"/> null leaves the automatic rule in charge; true or false pins the
        /// axis. Forcing "tiled" needs the rule disabled, which is what the negative <c>TileColsOverride</c>
        /// path cannot express — hence the explicit switch.
        /// </summary>
        private static double Best(CachedLlamaInferenceEngine engine, int[] ids, bool? forceBanding)
        {
            BatchedQuantProjection.UseOutputBlocking = forceBanding == true;
            BatchedQuantProjection.DisableAutoAxisSelection = forceBanding == false;

            try
            {
                var best = double.MaxValue;

                for (var r = 0; r < Repeats; r++)
                {
                    using var session = engine.CreateSession(1024);
                    var started = ValueStopwatch.StartNew();
                    session.Reset(ids);
                    best = Math.Min(best, started.GetElapsedTime().TotalMilliseconds);
                }

                return best;
            }
            finally
            {
                BatchedQuantProjection.UseOutputBlocking = false;
                BatchedQuantProjection.DisableAutoAxisSelection = false;
            }
        }
    }
}
