// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Security.Cryptography;
using System.Text.Json;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Embeddings;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.LanguageModels.Embeddings
{
    /// <summary>
    /// Qwen3-Embedding-0.6B as a sentence embedder, against two independent external references.
    ///
    /// <para><b>Why two.</b> llama.cpp reads the same quantised bytes we do, so it isolates our arithmetic
    /// from quantisation — but it can be wrong too, and it shares one assumption with us (that a trailing
    /// end-of-text token belongs on the input). The model card publishes its own similarity matrix from the
    /// full-precision HuggingFace model, which shares nothing with either engine, so it is the tiebreak. The
    /// EOS question is settled only by the second one.</para>
    ///
    /// <para>[LongFact] via [FixtureFact] — loads a 639 MB model.</para>
    /// </summary>
    [Trait("Category", "Qwen")]
    public sealed class Qwen3EmbeddingParityTests
    {
        // The model card's `transformers` usage block publishes the similarity matrix for exactly these
        // four texts, so they are copied verbatim rather than paraphrased.
        private const string RetrievalTask = GgufSentenceEmbedder.Qwen3EmbeddingRetrievalTask;

        private static readonly string[] ModelCardQueries =
        [
            "What is the capital of China?",
            "Explain gravity",
        ];

        private static readonly string[] ModelCardDocuments =
        [
            "The capital of China is Beijing.",
            "Gravity is a force that attracts two bodies towards each other. It gives weight to physical "
            + "objects and is responsible for the movement of planets around the sun.",
        ];

        private static readonly double[][] ModelCardScores =
        [
            [0.7645568251609802, 0.14142508804798126],
            [0.13549736142158508, 0.5999549627304077],
        ];

        private readonly ITestOutputHelper _out;

        public Qwen3EmbeddingParityTests(ITestOutputHelper output) => _out = output;

        /// <summary>
        /// The reference vectors were produced from ONE specific file. A different quantisation of the same
        /// model would still load, still embed, and still be compared against them — and the cosines would
        /// be wrong for a reason no assertion below could name. So the fixture is identified before it is
        /// used. Cheap: a length and the first mebibyte, no model load.
        /// </summary>
        [FixtureFact(TestFixture.Qwen3EmbeddingGguf, "1s")]
        public void TheFixtureIsTheFileTheReferenceVectorsWereProducedFrom()
        {
            using var reference = LoadReference();
            var root = reference.RootElement;

            var path = TestModelPaths.Qwen3Embedding.RequireGgufPath();
            var expectedName = root.GetProperty("model_file").GetString();
            var expectedSize = root.GetProperty("model_size_bytes").GetInt64();
            var expectedDigest = root.GetProperty("model_sha256_first_1mib").GetString();

            Assert.Equal(expectedName, Path.GetFileName(path));
            Assert.Equal(expectedSize, new FileInfo(path).Length);
            Assert.Equal(expectedDigest, FirstMebibyteDigest(path));

            _out.WriteLine($"reference produced by: {root.GetProperty("producer_build").GetString()}");
        }

        /// <summary>
        /// <b>O1, O3 and O4 in one pass</b>, because they are three predicates over one set of vectors and
        /// splitting them would load the model three times.
        ///
        /// <list type="bullet">
        /// <item>O1 — per-vector cosine against llama.cpp's, last-token pooling, ≥ 0.999.</item>
        /// <item>O3 — our token count equals llama.cpp's for each text. This is the end-of-text half: the
        /// file sets <c>tokenizer.ggml.add_eos_token</c>, llama.cpp honours it, and without the appended
        /// token the cosine falls to about 0.80 while every other assertion here still passes.</item>
        /// <item>O4 — the PAIRWISE similarity between two of our vectors matches the pairwise similarity
        /// between llama.cpp's, to 1e-3. Per-vector parity does not imply this: re-quantising the weights
        /// moves per-vector cosine by 1.4e-4 and pairwise similarity by 8.5e-3, sixty times more, and
        /// similarity is the quantity every consumer of an embedding actually uses.</item>
        /// </list>
        /// </summary>
        [FixtureFact(TestFixture.Qwen3EmbeddingGguf, "20s")]
        public void LastTokenEmbeddings_MatchLlamaCpp_PerVectorAndPairwise()
        {
            using var reference = LoadReference();
            var texts = ReadStrings(reference.RootElement.GetProperty("texts"));
            var expectedCounts = ReadInts(reference.RootElement.GetProperty("token_counts"));
            var expectedVectors = ReadVectors(reference.RootElement.GetProperty("last"));

            // O3 first: it needs no model, only the GGUF's embedded vocabulary, and if the token streams
            // disagree then every cosine below is comparing two different inputs.
            var tokenizer = GgufTokenizer.Load(TestModelPaths.Qwen3Embedding.RequireGgufPath());
            Assert.True(tokenizer.AddEosByDefault, "the fixture should set tokenizer.ggml.add_eos_token");
            for (var t = 0; t < texts.Length; t++)
            {
                // +1 for the end-of-text token the embedder appends because the file asks for it.
                var count = tokenizer.Encode(texts[t], addBos: false).Length + 1;
                Assert.Equal(expectedCounts[t], count);
            }

            using var embedder = GgufSentenceEmbedder.ForQwen3Embedding(
                TestModelPaths.Qwen3Embedding.RequireGgufPath());

            var ours = new float[texts.Length][];
            for (var t = 0; t < texts.Length; t++)
            {
                ours[t] = embedder.Embed(texts[t]);   // no prefix: the reference texts carry none
                var cos = Cosine(ours[t], expectedVectors[t]);
                _out.WriteLine($"[{t}] cos={cos:F6}  {texts[t][..Math.Min(48, texts[t].Length)]}");
                Assert.True(cos >= 0.999, $"text {t}: cosine against llama.cpp is {cos:F6}, expected >= 0.999.");
            }

            var oursPairwise = Cosine(ours[0], ours[2]);
            var referencePairwise = Cosine(expectedVectors[0], expectedVectors[2]);
            _out.WriteLine($"pairwise cos(text0,text2): ours={oursPairwise:F6} llama.cpp={referencePairwise:F6}");
            Assert.True(
                Math.Abs(oursPairwise - referencePairwise) <= 1e-3,
                $"pairwise similarity drifted: ours {oursPairwise:F6} vs llama.cpp {referencePairwise:F6}.");
        }

        /// <summary>
        /// Mean pooling against llama.cpp's <c>--pooling mean</c> on the same token streams. Not the mode
        /// Qwen3-Embedding is trained for — that is last-token — but it is the DEFAULT of
        /// <c>CachedLlamaSession.Embed</c> and what <c>OverfitClient.Embed</c> uses for the
        /// embed-with-your-chat-model path, so it is the mode most vectors this library has produced were
        /// pooled with. Before 2026-08-27 it averaged pre-norm states and scored 0.827-0.891 here.
        ///
        /// <para><b>Why this arm quantises and the last-token arm does not.</b> Mean pooling reads EVERY
        /// position, including position 0, and the reference was produced by llama.cpp reading the Q8_0
        /// weights with Q8 arithmetic. <c>quantize:true</c> is the like-for-like comparison. The dequantised
        /// path is measured separately below, because at position 0 the two numeric paths genuinely part
        /// company and that is a finding rather than a tolerance to widen.</para>
        /// </summary>
        [FixtureFact(TestFixture.Qwen3EmbeddingGguf, "20s")]
        public void MeanPooledEmbeddings_MatchLlamaCppMeanPooling()
        {
            using var reference = LoadReference();
            var texts = ReadStrings(reference.RootElement.GetProperty("texts"));
            var expectedVectors = ReadVectors(reference.RootElement.GetProperty("mean"));

            using var embedder = GgufSentenceEmbedder.FromGguf(
                TestModelPaths.Qwen3Embedding.RequireGgufPath(),
                pooling: EmbeddingPooling.Mean,
                quantize: true);

            for (var t = 0; t < texts.Length; t++)
            {
                var cos = Cosine(embedder.Embed(texts[t]), expectedVectors[t]);
                _out.WriteLine($"[{t}] mean cos={cos:F6}");
                Assert.True(cos >= 0.999, $"text {t}: mean-pooled cosine against llama.cpp is {cos:F6}.");
            }
        }

        /// <summary>
        /// <b>Records a measured divergence that this task found and did not resolve.</b> On the dequantised
        /// (<c>quantize:false</c>) path the FIRST token's hidden state disagrees with llama.cpp far more than
        /// any other position, so mean pooling — which reads every position — lands short of the 0.999 the
        /// quantized path reaches, while last-token pooling on the same run is unaffected because it never
        /// looks at position 0.
        ///
        /// <para><b>What was measured</b>, per-position cosine against llama.cpp's <c>--pooling none</c>
        /// output on the four reference texts. Position 0: <c>0.9786 / 0.9064 / 0.9064 / 0.9741</c> at
        /// <c>quantize:false</c> against <c>0.999997 / 0.999979 / 0.999979 / 0.999981</c> at
        /// <c>quantize:true</c>. Every other position is above 0.99 in both arms. Per-layer capture at
        /// position 0 puts the whole difference in the LAST transformer block: the absolute difference
        /// between the two arms' residual streams is flat at 31.2 from layer 2 to layer 26 and jumps to
        /// 765.5 at layer 27. Block 27 is where the model cancels its massive-activation channel — channel
        /// 35 carries 5704 at layer 26 and the block subtracts about 6700 — so a 0.52% input difference
        /// there emerges as a 10.2% difference in the block's output magnitude, with its direction still
        /// agreeing at cosine 0.9976.</para>
        ///
        /// <para><b>The attribution to position 0 is measured, not inferred</b>, which is what licenses this
        /// test's name. Removing position 0 from BOTH arms' means collapses them together: mean-excluding-
        /// position-0 is <c>0.999846 / 0.999682 / 0.999728 / 0.999595</c> dequantised against
        /// <c>0.999875 / 0.999727 / 0.999726 / 0.999547</c> quantised — largest gap 4.5e-5, and on two of
        /// the four texts the dequantised arm is the better one. Both reconstructions were checked first:
        /// our mean-of-per-position rebuilt <c>Embed(Mean)</c> at cosine 1.000000, and llama.cpp's
        /// mean-of-per-token rebuilt its own <c>--pooling mean</c> output at 1.000000, so the ablation is
        /// measuring the model and not the harness.</para>
        ///
        /// <para><b>It is NOT length that decides.</b> Text 0 is the shortest at 8 tokens and has the
        /// second-SMALLEST gap, because the size of the position-0 error varies by token (0.9786 there
        /// against 0.9064 on texts 1 and 2) and outweighs the 1/n dilution.</para>
        ///
        /// <para><b>What was NOT established:</b> whether that 20x amplification is entirely the arithmetic
        /// of cancelling a channel ten times larger than the result, or whether the dequantised path has a
        /// defect of its own. Settling it needs instrumentation inside the block, which is outside this
        /// task. Until it is settled this test states the measured band rather than a target.</para>
        ///
        /// <para>The second assertion is the half that gives the first one meaning: on the SAME arm,
        /// last-token pooling still clears 0.999. Without it a reader cannot tell a position-0 problem from
        /// a dequantised path that is simply worse everywhere.</para>
        /// </summary>
        [FixtureFact(TestFixture.Qwen3EmbeddingGguf, "20s")]
        public void OnTheDequantizedPath_MeanPoolingTrailsLastTokenPooling_BecauseOfPositionZero()
        {
            using var reference = LoadReference();
            var texts = ReadStrings(reference.RootElement.GetProperty("texts"));
            var expectedMean = ReadVectors(reference.RootElement.GetProperty("mean"));
            var expectedLast = ReadVectors(reference.RootElement.GetProperty("last"));

            var path = TestModelPaths.Qwen3Embedding.RequireGgufPath();

            using (var mean = GgufSentenceEmbedder.FromGguf(path, pooling: EmbeddingPooling.Mean))
            {
                for (var t = 0; t < texts.Length; t++)
                {
                    var cos = Cosine(mean.Embed(texts[t]), expectedMean[t]);
                    _out.WriteLine($"[{t}] f32 mean cos={cos:F6}");
                    Assert.True(cos >= 0.997, $"text {t}: dequantised mean cosine fell to {cos:F6}.");
                }
            }

            using var last = GgufSentenceEmbedder.FromGguf(path, pooling: EmbeddingPooling.LastToken);
            for (var t = 0; t < texts.Length; t++)
            {
                var cos = Cosine(last.Embed(texts[t]), expectedLast[t]);
                _out.WriteLine($"[{t}] f32 last cos={cos:F6}");
                Assert.True(
                    cos >= 0.999,
                    $"text {t}: dequantised LAST-token cosine is {cos:F6}. The gap above is supposed to be "
                    + "position 0 only; if this fails too, the dequantised path is worse everywhere and the "
                    + "explanation on this test is wrong.");
            }
        }

        /// <summary>
        /// <b>The HuggingFace tiebreak, and the only oracle here that shares nothing with llama.cpp.</b> The
        /// model card publishes the similarity matrix its own full-precision PyTorch model produces for four
        /// specific texts. Reproducing it end to end exercises the query instruction prefix, the passage side
        /// staying bare, last-token pooling, the appended end-of-text token and L2 normalisation at once.
        ///
        /// <para><b>What the tolerance is doing.</b> 0.01 is loose against the measured 0.000988 because the
        /// reference is full precision and the fixture is Q8_0 — two different numeric paths, so the gap is
        /// real and is not ours to remove. It is nowhere near loose enough to admit the failures that matter:
        /// dropping the end-of-text token moves the worst pair by 0.106, and pooling the pre-norm state moves
        /// it further. The measured worst deviation is printed on every run.</para>
        /// </summary>
        [FixtureFact(TestFixture.Qwen3EmbeddingGguf, "20s")]
        public void ReproducesTheModelCardsOwnPublishedSimilarityMatrix()
        {
            using var embedder = GgufSentenceEmbedder.ForQwen3Embedding(
                TestModelPaths.Qwen3Embedding.RequireGgufPath());

            Assert.Equal($"Instruct: {RetrievalTask}\nQuery:", embedder.QueryPrefix);
            Assert.Null(embedder.PassagePrefix);
            Assert.Equal(EmbeddingPooling.LastToken, embedder.Pooling);

            var queries = new float[ModelCardQueries.Length][];
            for (var q = 0; q < queries.Length; q++)
            {
                queries[q] = embedder.EmbedQuery(ModelCardQueries[q]);
            }

            var documents = new float[ModelCardDocuments.Length][];
            for (var d = 0; d < documents.Length; d++)
            {
                documents[d] = embedder.EmbedPassage(ModelCardDocuments[d]);
            }

            var worst = 0.0;
            for (var q = 0; q < queries.Length; q++)
            {
                for (var d = 0; d < documents.Length; d++)
                {
                    var got = Cosine(queries[q], documents[d]);
                    var delta = Math.Abs(got - ModelCardScores[q][d]);
                    worst = Math.Max(worst, delta);
                    _out.WriteLine($"s[{q},{d}]={got:F6}  model card {ModelCardScores[q][d]:F6}  delta {delta:F6}");
                }
            }

            _out.WriteLine($"worst |delta| = {worst:F6}");
            Assert.True(worst <= 0.01, $"worst deviation from the model card's own matrix is {worst:F6}.");
        }

        private static JsonDocument LoadReference()
            => JsonDocument.Parse(File.ReadAllText(
                TestModelPaths.Qwen3Embedding.RequireLlamaCppReferenceJsonPath()));

        private static string FirstMebibyteDigest(string path)
        {
            using var stream = File.OpenRead(path);
            var buffer = new byte[1024 * 1024];
            var read = stream.ReadAtLeast(buffer, buffer.Length, throwOnEndOfStream: false);
            return Convert.ToHexString(SHA256.HashData(buffer.AsSpan(0, read))).ToLowerInvariant();
        }

        private static string[] ReadStrings(JsonElement element)
        {
            var result = new string[element.GetArrayLength()];
            var i = 0;
            foreach (var item in element.EnumerateArray())
            {
                result[i++] = item.GetString()!;
            }
            return result;
        }

        private static int[] ReadInts(JsonElement element)
        {
            var result = new int[element.GetArrayLength()];
            var i = 0;
            foreach (var item in element.EnumerateArray())
            {
                result[i++] = item.GetInt32();
            }
            return result;
        }

        private static float[][] ReadVectors(JsonElement element)
        {
            var result = new float[element.GetArrayLength()][];
            var i = 0;
            foreach (var row in element.EnumerateArray())
            {
                result[i++] = ReadFloats(row);
            }
            return result;
        }

        private static float[] ReadFloats(JsonElement element)
        {
            var result = new float[element.GetArrayLength()];
            var i = 0;
            foreach (var item in element.EnumerateArray())
            {
                result[i++] = item.GetSingle();
            }
            return result;
        }

        private static double Cosine(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
        {
            Assert.Equal(a.Length, b.Length);
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
