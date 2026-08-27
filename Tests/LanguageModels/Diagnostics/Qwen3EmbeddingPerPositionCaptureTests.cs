// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using System.Text.Json;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// <b>Capture, not assertion — the instrument <c>XC-132</c> needs and did not have.</b> Writes the
    /// per-token hidden state this engine produces for the four reference texts, under BOTH
    /// <c>quantize</c> arms, to a JSON file that <c>Scripts/xc132_compare.py</c> reads beside llama.cpp's
    /// <c>--pooling none</c> output.
    ///
    /// <para><b>Why it exists as a committed file.</b> The 2026-08-27 measurement that produced `XC-132`
    /// captured these states from a scratch script that is gone, so its numbers cannot be re-derived and
    /// the next person starts from zero. This repository has already recorded what re-derivation costs.</para>
    ///
    /// <para><b>How a per-position state is obtained, and why it is exact rather than an approximation.</b>
    /// The model is causal: the hidden state at position <c>k</c> depends only on tokens <c>0..k</c>. So
    /// embedding the first <c>k+1</c> tokens with <see cref="EmbeddingPooling.LastToken"/> returns exactly
    /// the state at position <c>k</c>. No hook, no instrumentation, no change to
    /// <c>Sources/Main</c>. The cost is quadratic in token count and irrelevant at these lengths — 56
    /// tokens across four texts.</para>
    ///
    /// <para><b>Normalisation is OFF on purpose.</b> A per-position row must be the raw state: the mean of
    /// L2-normalised rows is not the L2-normalised mean, and pooling averages raw states. The same trap
    /// exists on llama.cpp's side, where <c>--embd-normalize 2</c> is the default and rebuilds
    /// <c>--pooling mean</c> at cosine 0.999502 instead of 1.000000. Use <c>-1</c> there.</para>
    ///
    /// <para><b>The two arms are loaded one at a time, not together.</b> <c>quantize:false</c> peaks at
    /// 3270 MB against 1608 MB for <c>true</c> on this 0.6B file; holding both would peak near 4.9 GB for
    /// no reason.</para>
    /// </summary>
    public sealed class Qwen3EmbeddingPerPositionCaptureTests
    {
        private readonly ITestOutputHelper _out;

        public Qwen3EmbeddingPerPositionCaptureTests(ITestOutputHelper output)
        {
            _out = output;
        }

        /// <summary>
        /// Writes <c>Tests/bin/xc132-per-position.json</c>: for each reference text, the token ids and the
        /// per-position hidden states under both <c>quantize</c> arms.
        ///
        /// <para><b>The self-check is the part that makes the output trustworthy.</b> The mean of the
        /// captured rows must reproduce <c>Embed(Mean)</c> on the same token stream. If it does not, the
        /// prefix reconstruction is wrong and every downstream comparison would be measuring this file
        /// rather than the model. It is asserted, not printed.</para>
        /// </summary>
        [FixtureFact(TestFixture.Qwen3EmbeddingGguf, "90s")]
        public void CapturePerPositionStates_ForBothQuantizeArms()
        {
            var path = TestModelPaths.Qwen3Embedding.RequireGgufPath();
            var tokenizer = GgufTokenizer.Load(path);

            // The four reference texts by default. OVERFIT_XC132_TEXTS overrides them with a JSON array so
            // a probe can be designed for a specific question without editing this file — `XC-134` needs a
            // sequence of REPEATED identical tokens, where content is constant and only the position index
            // varies, which no natural-language text provides. OVERFIT_XC132_OUT names the output file so
            // a probe run does not overwrite the reference capture.
            var texts = new[]
            {
                "The capital of France is Paris.",
                "Stolica Francji to Paryz.",
                "Stolica Francji to Paryż.",
                "Instruct: Given a web search query, retrieve relevant passages that answer the query "
                + "Query: What is the capital of France?",
            };

            var textsOverride = Environment.GetEnvironmentVariable("OVERFIT_XC132_TEXTS");
            if (!string.IsNullOrWhiteSpace(textsOverride))
            {
                texts = JsonSerializer.Deserialize<string[]>(textsOverride)
                    ?? throw new InvalidOperationException(
                        "OVERFIT_XC132_TEXTS is set but did not parse as a JSON array of strings.");
                Assert.NotEmpty(texts);
                _out.WriteLine($"OVERFIT_XC132_TEXTS overrode the probe set: {texts.Length} text(s)");
            }

            var streams = new int[texts.Length][];
            for (var t = 0; t < texts.Length; t++)
            {
                var ids = tokenizer.Encode(texts[t]);
                var withEos = new int[ids.Length + (tokenizer.AddEosByDefault ? 1 : 0)];
                ids.CopyTo(withEos.AsSpan());
                if (tokenizer.AddEosByDefault)
                {
                    withEos[^1] = tokenizer.EosId;
                }

                streams[t] = withEos;
                _out.WriteLine($"text {t}: {withEos.Length} tokens (add_eos={tokenizer.AddEosByDefault})");
            }

            var capture = new Dictionary<string, object>
            {
                ["// what this is"] =
                    "XC-132 diagnostic. Per-position hidden states from DevOnBike.Overfit for both quantize "
                    + "arms, obtained by last-token pooling over each token prefix (exact for a causal "
                    + "model). Unnormalised. Compare against llama-embedding --pooling none "
                    + "--embd-normalize -1.",
                ["texts"] = texts,
                ["token_ids"] = streams,
                ["model_file"] = path,
            };

            foreach (var quantize in new[] { true, false })
            {
                capture["quantize_" + (quantize ? "true" : "false")] = Capture(path, streams, quantize);
            }

            var outputPath = Path.Combine(
                AppContext.BaseDirectory,
                Environment.GetEnvironmentVariable("OVERFIT_XC132_OUT") ?? "xc132-per-position.json");
            File.WriteAllText(
                outputPath,
                JsonSerializer.Serialize(capture, new JsonSerializerOptions { WriteIndented = false }),
                Encoding.UTF8);

            _out.WriteLine($"wrote {outputPath} ({new FileInfo(outputPath).Length / 1024} KB)");
            Assert.True(File.Exists(outputPath));
        }

        private float[][][] Capture(string path, int[][] streams, bool quantize)
        {
            using var engine = GgufLlamaLoader.Load(path, quantize: quantize, mmap: true);
            using var session = engine.CreateSession(1024);

            var perText = new float[streams.Length][][];

            for (var t = 0; t < streams.Length; t++)
            {
                var tokens = streams[t];
                var rows = new float[tokens.Length][];

                for (var k = 0; k < tokens.Length; k++)
                {
                    rows[k] = session.Embed(
                        tokens.AsSpan(0, k + 1), EmbeddingPooling.LastToken, normalize: false);
                }

                perText[t] = rows;

                // The self-check: mean of the captured rows must equal Embed(Mean) on the same stream.
                // Without it a silent off-by-one in the prefix walk looks exactly like a model finding.
                var pooled = session.Embed(tokens, EmbeddingPooling.Mean, normalize: false);
                var rebuilt = new float[pooled.Length];
                for (var k = 0; k < rows.Length; k++)
                {
                    for (var j = 0; j < rebuilt.Length; j++)
                    {
                        rebuilt[j] += rows[k][j] / rows.Length;
                    }
                }

                var cos = Cosine(rebuilt, pooled);
                _out.WriteLine($"quantize={quantize} text {t}: rebuild cos={cos:F6} over {rows.Length} rows");
                Assert.True(
                    cos >= 0.99999,
                    $"quantize={quantize} text {t}: the mean of the captured per-position rows rebuilds "
                    + $"Embed(Mean) at only {cos:F6}. The prefix reconstruction is wrong, so every number "
                    + "downstream would describe this capture rather than the model.");
            }

            return perText;
        }

        private static double Cosine(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
        {
            double dot = 0, na = 0, nb = 0;
            for (var i = 0; i < a.Length; i++)
            {
                dot += (double)a[i] * b[i];
                na += (double)a[i] * a[i];
                nb += (double)b[i] * b[i];
            }

            return dot / (Math.Sqrt(na) * Math.Sqrt(nb));
        }
    }
}
