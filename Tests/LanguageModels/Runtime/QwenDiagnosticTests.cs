// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    [Trait("Category", "QwenDiag")]
    [Trait("Category", "Qwen")]
    public sealed class QwenDiagnosticTests
    {
        private readonly ITestOutputHelper _out;

        public QwenDiagnosticTests(ITestOutputHelper output)
        {
            _out = output;
        }

        private static string ModelPath => TestModelPaths.Qwen3B.BinaryPath;
        private static string TokenizerDir => TestModelPaths.Qwen3B.Dir;

        /// <summary>
        /// Loads the engine + tokenizer or throws <see cref="FileNotFoundException"/>
        /// if the required model file is missing. Tokenizer is optional — its absence
        /// leaves <paramref name="tok"/> null. Returns true unconditionally for source
        /// compatibility with callers that still use <c>if (!TryLoad(...)) return;</c>.
        /// </summary>
        private bool TryLoad(out CachedLlamaInferenceEngine? engine, out QwenTokenizer? tok)
        {
            TestModelPaths.Qwen3B.RequireBinaryPath();
            engine = CachedLlamaInferenceEngine.Load(ModelPath);
            tok = File.Exists(Path.Combine(TokenizerDir, "tokenizer.json"))
                ? QwenTokenizer.Load(TokenizerDir)
                : null;
            return true;
        }

        /// <summary>
        /// After BOS-only prompt, top predicted tokens should be common words/punctuation.
        /// Broken model: top tokens are high-ID garbage (>100000).
        /// Working model: top tokens are low-ID common tokens (<5000).
        /// </summary>
        [LongFact("7s")]
        public void Diag_BosOnly_TopLogits()
        {
            if (!TryLoad(out var engine, out var tok))
            {
                return;
            }
            using (engine)
            {
                using var session = engine!.CreateSession(64);
                session.Reset([QwenTokenizer.EndOfText]);
                var sampling = SamplingOptions.Greedy;
                var firstToken = session.GenerateNextToken(in sampling);

                var logits = session.LastLogits.ToArray();

                // Top 10 by logit value
                var top10 = logits
                    .Select((v, i) => (v, i))
                    .OrderByDescending(x => x.v)
                    .Take(10)
                    .ToArray();

                _out.WriteLine($"Greedy token: {firstToken}" +
                    (tok is not null ? $" = '{tok.DecodeToken(firstToken)}'" : ""));
                _out.WriteLine("Top-10 logits:");
                foreach (var (v, id) in top10)
                {
                    _out.WriteLine($"  [{id,7}] {v,8:F3}" +
                                   (tok is not null ? $"  '{tok.DecodeToken(id)}'" : ""));
                }

                var maxLogit = top10[0].v;
                var minTop10 = top10[9].v;
                _out.WriteLine($"\nLogit spread (top1 - top10): {maxLogit - minTop10:F3}");
                _out.WriteLine("Working model: spread > 1.0, top tokens < 5000");
                _out.WriteLine("Broken model:  spread ≈ 0 OR top tokens > 100000");

                Assert.True(maxLogit > minTop10 + 0.01f, "Logits are completely flat — something is wrong");
            }
        }

        /// <summary>
        /// BOS-only logits vs full-chat-prompt logits should differ.
        /// If model ignores context, both will be identical (avg diff ≈ 0).
        /// </summary>
        [LongFact("11s")]
        public void Diag_ContextChangesLogits()
        {
            if (!TryLoad(out var engine, out var tok))
            {
                return;
            }
            if (tok is null)
            {
                _out.WriteLine("SKIPPED: no tokenizer");
                return;
            }
            using (engine)
            {
                var sampling = SamplingOptions.Greedy;

                using var s1 = engine!.CreateSession(64);
                s1.Reset([QwenTokenizer.EndOfText]);
                s1.GenerateNextToken(in sampling);
                var logits1 = s1.LastLogits.ToArray();

                using var s2 = engine.CreateSession(64);
                var prompt = tok.BuildChatPrompt("What is 2+2?");
                s2.Reset(prompt);
                s2.GenerateNextToken(in sampling);
                var logits2 = s2.LastLogits.ToArray();

                var diff = 0f;
                for (var i = 0; i < logits1.Length; i++)
                {
                    diff += MathF.Abs(logits1[i] - logits2[i]);
                }
                diff /= logits1.Length;

                _out.WriteLine($"Avg |logit1 - logit2| across {logits1.Length} vocab: {diff:F6}");
                _out.WriteLine("diff ≈ 0      → model ignores context (attention broken)");
                _out.WriteLine("diff > 0.01   → model conditions on context (working)");

                // This is the critical diagnostic
                _out.WriteLine(diff < 0.001f ? "\n!!! ATTENTION APPEARS BROKEN !!!" : "\nAttention seems OK");
            }
        }

        /// <summary>
        /// Binary file date/size — confirms re-conversion actually happened.
        ///
        /// <para><b>The 2000–3000 MB assertion this used to carry was removed on 2026-08-07, and what it
        /// caught is worth recording.</b> It failed with <c>Actual: 12963</c>: the file at
        /// <see cref="TestModelPaths.Qwen3B.BinaryPath"/> is a <b>13.6 GB FP32</b> checkpoint, while the
        /// range encoded an assumption that it would be a 2–3 GB quantised one. That is not a defect in the
        /// loader or in the file; it is a test pinning which artifact happens to sit on one developer's
        /// disk. An absolute megabyte window cannot survive a re-conversion at a different precision, which
        /// is the very event this diagnostic exists to report.</para>
        ///
        /// <para>It also corroborates the separate finding in <c>docs/test-gate-backlog.md</c> (T9) that
        /// <c>GgufLlamaLoaderIntegrationTests</c> compares two files that are not the same weights: the
        /// <c>.gguf</c> is 6.18 GB and this <c>.bin</c> is 13.59 GB.</para>
        ///
        /// <para>What is asserted instead is the only thing this can honestly check without loading the
        /// model: that the path resolves to something large enough to be a checkpoint at all. The size and
        /// date are <i>reported</i> — that is the diagnostic.</para>
        /// </summary>
        [LongFact("4ms")]
        public void Diag_ModelFileInfo()
        {
            var info = new FileInfo(ModelPath);
            if (!info.Exists)
            {
                _out.WriteLine($"NOT FOUND: {ModelPath}");
                return;
            }
            _out.WriteLine($"Path     : {info.FullName}");
            _out.WriteLine($"Size     : {info.Length / 1024.0 / 1024.0:F1} MB");
            _out.WriteLine($"Modified : {info.LastWriteTime:yyyy-MM-dd HH:mm:ss}");

            // A truncated or half-written conversion is the failure worth catching here, and it does not
            // need a magic window to catch: no usable checkpoint of this model is under a gigabyte.
            Assert.True(info.Length > 1024L * 1024L * 1024L,
                $"{ModelPath} is only {info.Length / 1024.0 / 1024.0:F1} MB — too small to be a converted "
                + "Qwen-3B checkpoint. A truncated or partial conversion looks exactly like this.");
        }
    }
}
