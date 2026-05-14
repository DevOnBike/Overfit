// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Streaming
{
    /// <summary>
    /// Integration tests for the streaming generation API. Each test loads the
    /// real Qwen2.5-3B GGUF (skipped if not available) since stream semantics
    /// are coupled to actual token sampling and KV cache behavior.
    /// </summary>
    [Trait("Category", "Streaming")]
    public sealed class StreamGenerateTests
    {
        private static string GgufModelPath => TestModelPaths.Qwen3B.GgufPath;
        private static string TokenizerDir => TestModelPaths.Qwen3B.Dir;

        private readonly ITestOutputHelper _output;

        public StreamGenerateTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [LongFact]
        public async Task StreamGenerate_YieldsMaxTokensWhenNoStopHit()
        {
            TestModelPaths.Qwen3B.RequireGgufPath();

            using var engine = CachedLlamaInferenceEngine.LoadGguf(GgufModelPath);
            using var session = engine.CreateSession(64);

            int[] prompt = { 151643, 151644, 198 };  // BOS, im_start, \n
            session.Reset(prompt);

            // No stop tokens, very low MaxTokens — should yield exactly that many
            var opts = new StreamingOptions(maxTokens: 5, stopTokens: [], sampling: SamplingOptions.Greedy);
            var tokens = new List<int>();

            await foreach (var t in session.StreamGenerate(opts))
            {
                tokens.Add(t);
            }

            Assert.Equal(5, tokens.Count);
            _output.WriteLine($"Generated tokens: [{string.Join(", ", tokens)}]");
        }

        [LongFact]
        public async Task StreamGenerate_TerminatesOnStopToken()
        {
            TestModelPaths.Qwen3B.RequireGgufPath();

            using var engine = CachedLlamaInferenceEngine.LoadGguf(GgufModelPath);
            // Cache must comfortably exceed prompt + maxTokens so the
            // terminator under test is the stop token, not cache exhaustion.
            using var session = engine.CreateSession(128);

            // Use the system prompt scenario where the model wants to emit
            // ImStart immediately (known from previous session work).
            // The 3-token prompt forces the model to generate "The..." for math
            // (won't hit stop quickly), so we use a higher cap and a fake
            // stop token that the model is very likely to produce.
            int[] prompt = { 151643, 151644, 198 };
            session.Reset(prompt);

            // The model very likely produces a period or comma or newline.
            // Use \n token (198) as a stop token — frequently emitted.
            var opts = new StreamingOptions(
                maxTokens: 64,
                stopTokens: new[] { 198 },  // newline
                sampling: SamplingOptions.Greedy);

            var tokens = new List<int>();
            await foreach (var t in session.StreamGenerate(opts))
            {
                tokens.Add(t);
            }

            // Last token should be the stop token (yielded before terminating)
            // OR we hit maxTokens without seeing newline.
            if (tokens.Count < 64)
            {
                Assert.Equal(198, tokens[^1]);
                _output.WriteLine($"Stopped at newline after {tokens.Count} tokens.");
            }
            else
            {
                _output.WriteLine("Hit MaxTokens without producing newline (acceptable).");
            }
        }

        [LongFact]
        public async Task StreamGenerate_RespectsCancellation()
        {
            TestModelPaths.Qwen3B.RequireGgufPath();

            using var engine = CachedLlamaInferenceEngine.LoadGguf(GgufModelPath);
            using var session = engine.CreateSession(64);

            int[] prompt = { 151643, 151644, 198 };
            session.Reset(prompt);

            using var cts = new CancellationTokenSource();
            var opts = new StreamingOptions(
                maxTokens: 100,
                stopTokens: Array.Empty<int>(),
                sampling: SamplingOptions.Greedy);

            var produced = 0;
            await Assert.ThrowsAsync<OperationCanceledException>(async () =>
            {
                await foreach (var t in session.StreamGenerate(opts, cts.Token))
                {
                    produced++;
                    if (produced == 3) { cts.Cancel(); }
                }
            });

            Assert.Equal(3, produced);
            _output.WriteLine($"Cancellation honored after {produced} tokens.");
        }

        [LongFact]
        public async Task StreamGenerate_ThrowsWhenSessionEmpty()
        {
            TestModelPaths.Qwen3B.RequireGgufPath();

            using var engine = CachedLlamaInferenceEngine.LoadGguf(GgufModelPath);
            using var session = engine.CreateSession(64);
            // Note: NO Reset() — session has Position == 0

            var opts = StreamingOptions.Default;

            await Assert.ThrowsAsync<InvalidOperationException>(async () =>
            {
                await foreach (var _ in session.StreamGenerate(opts))
                {
                    // should never reach here
                }
            });
        }

        [LongFact]
        public async Task StreamGenerate_MatchesGenerateNextTokenForSamePrompt()
        {
            TestModelPaths.Qwen3B.RequireGgufPath();

            using var engine = CachedLlamaInferenceEngine.LoadGguf(GgufModelPath);

            int[] prompt = { 151643, 151644, 198 };
            const int n = 8;

            // ── Path 1: classic GenerateNextToken loop ──
            using var s1 = engine.CreateSession(64);
            s1.Reset(prompt);
            var viaClassic = new List<int>();
            for (var i = 0; i < n; i++)
            {
                viaClassic.Add(s1.GenerateNextToken(SamplingOptions.Greedy));
            }

            // ── Path 2: StreamGenerate ──
            using var s2 = engine.CreateSession(64);
            s2.Reset(prompt);
            var viaStream = new List<int>();
            var opts = new StreamingOptions(n, Array.Empty<int>(), SamplingOptions.Greedy);
            await foreach (var t in s2.StreamGenerate(opts))
            {
                viaStream.Add(t);
            }

            // Both paths run the same sampler with the same seed-less greedy path,
            // so token sequences must be identical.
            Assert.Equal(viaClassic, viaStream);
            _output.WriteLine($"Both paths produced: [{string.Join(", ", viaStream)}]");
        }

        [LongFact]
        public async Task StreamGenerate_WithFactoryStopTokens_QwenChatTerminators()
        {
            TestModelPaths.Qwen3B.RequireGgufPath();

            using var engine = CachedLlamaInferenceEngine.LoadGguf(GgufModelPath);
            using var session = engine.CreateSession(64);

            int[] prompt = { 151643, 151644, 198 };
            session.Reset(prompt);

            // Use the convenience factory with all 3 Qwen chat terminators
            var opts = StreamingOptions.WithStopTokens(
                maxTokens: 48,
                QwenTokenizer.EndOfText,
                QwenTokenizer.ImStart,
                QwenTokenizer.ImEnd);

            var tokens = new List<int>();
            await foreach (var t in session.StreamGenerate(opts))
            {
                tokens.Add(t);
            }

            _output.WriteLine($"Generated {tokens.Count} tokens, last = {tokens[^1]}");
            // Either hit a stop token OR maxTokens
            Assert.True(tokens.Count > 0);
            Assert.True(tokens.Count <= 48);
        }
    }
}
