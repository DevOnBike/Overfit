// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using System.Text.Json.Serialization;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tokenization;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Manual GPT-2 attention parity diagnostic.
    /// </summary>
    /// <remarks>
    /// Purpose:
    /// Determine whether the first GPT-2 mismatch is:
    /// - Q/K/V projection weight orientation/split,
    /// - Q/K/V bias loading/application,
    /// - attention context / output projection mapping.
    ///
    /// Generate/update reference JSON first:
    /// python3 Scripts/debug_gpt2_reference.py --size small --fixtures Tests/test_fixtures --out Tests/test_fixtures/gpt2_reference_small.json
    ///
    /// Run:
    /// dotnet test -c Release --filter "Gpt2Small_CompareAttentionInternalsAgainstPyTorchReference"
    /// </remarks>
    public sealed class Gpt2ImportAttentionParityDiagnostics
    {
        private const string ModelPath = "test_fixtures/gpt2_small.bin";
        private const string VocabPath = "test_fixtures/vocab.json";
        private const string MergesPath = "test_fixtures/merges.txt";
        private const string ReferencePath = "test_fixtures/gpt2_reference_small.json";
        private const int DefaultArenaSize = 1_500_000_000;

        private readonly ITestOutputHelper _output;

        public Gpt2ImportAttentionParityDiagnostics(ITestOutputHelper output)
        {
            _output = output;
        }

        //[Fact(Skip = "Manual GPT-2 attention parity diagnostic. Generate PyTorch reference JSON first, remove Skip locally, run once, then restore Skip.")]
        [Fact]
        [Trait("Category", "Diagnostics")]
        [Trait("Category", "Manual")]
        public void Gpt2Small_CompareAttentionInternalsAgainstPyTorchReference()
        {
            SkipIfMissing(ModelPath, VocabPath, MergesPath, ReferencePath);

            var reference = LoadReference(ReferencePath);
            var tokenizer = BytePairEncoder.Load(VocabPath, MergesPath);

            Assert.Equal(reference.VocabSize, tokenizer.VocabSize);

            var overfitTokens = tokenizer.Encode(reference.Prompt);
            Assert.Equal(reference.Tokens, overfitTokens);

            _output.WriteLine("=== GPT-2 Attention Parity Diagnostic ===");
            _output.WriteLine($"Prompt: {reference.Prompt}");
            _output.WriteLine($"Tokens: {string.Join(", ", reference.Tokens)}");
            _output.WriteLine($"Stage shape: {string.Join(" x ", reference.StageShape)}");
            _output.WriteLine("");

            using var model = new GPT1Model(Gpt2Config.Small);
            model.Eval();

            using (var stream = File.OpenRead(ModelPath))
            using (var reader = new BinaryReader(stream))
            {
                model.Load(reader);
            }

            _output.WriteLine(
                $"block0 head0 bias L1: " +
                $"Bq={L1(model.Blocks[0].Attention.BqHeads[0].DataReadOnlySpan):F6}, " +
                $"Bk={L1(model.Blocks[0].Attention.BkHeads[0].DataReadOnlySpan):F6}, " +
                $"Bv={L1(model.Blocks[0].Attention.BvHeads[0].DataReadOnlySpan):F6}");
            _output.WriteLine("");

            var arenaSize = GetIntEnvironmentVariable(
                "OVERFIT_GPT2_ATTENTION_ARENA",
                DefaultArenaSize);

            using var graph = new ComputationGraph(arenaSize);
            model.InvalidateAllCaches();

            var ln1 = ComputeBlock0Ln1(
                graph,
                model,
                overfitTokens);

            var attention = ComputeOverfitAttentionInternals(
                ln1,
                model.Blocks[0].Attention,
                seqLen: overfitTokens.Length,
                dModel: model.Config.DModel);

            Compare(
                "q raw vs PyTorch q WITHOUT bias",
                reference.Attention["block0_q_without_bias"],
                attention["block0_q_raw"]);

            Compare(
                "k raw vs PyTorch k WITHOUT bias",
                reference.Attention["block0_k_without_bias"],
                attention["block0_k_raw"]);

            Compare(
                "v raw vs PyTorch v WITHOUT bias",
                reference.Attention["block0_v_without_bias"],
                attention["block0_v_raw"]);

            Compare(
                "q WITH bias vs PyTorch q WITH bias",
                reference.Attention["block0_q_with_bias"],
                attention["block0_q_with_bias"]);

            Compare(
                "k WITH bias vs PyTorch k WITH bias",
                reference.Attention["block0_k_with_bias"],
                attention["block0_k_with_bias"]);

            Compare(
                "v WITH bias vs PyTorch v WITH bias",
                reference.Attention["block0_v_with_bias"],
                attention["block0_v_with_bias"]);

            Compare(
                "context vs PyTorch context WITHOUT qkv bias",
                reference.Attention["block0_context_without_qkv_bias"],
                attention["block0_context"]);

            Compare(
                "context vs PyTorch context WITH qkv bias",
                reference.Attention["block0_context_with_bias"],
                attention["block0_context_with_bias"]);

            Compare(
                "attn output vs PyTorch c_proj WITHOUT qkv bias",
                reference.Attention["block0_attn_manual_cproj_without_qkv_bias"],
                attention["block0_attn_output"]);

            Compare(
                "attn output vs PyTorch c_proj WITH qkv bias",
                reference.Attention["block0_attn_manual_cproj_with_bias"],
                attention["block0_attn_output_with_bias"]);

            Compare(
                "attn output vs PyTorch module block0_attn",
                reference.Stages["block0_attn"],
                attention["block0_attn_output_with_bias"]);
        }

        private static float[] ComputeBlock0Ln1(
            ComputationGraph graph,
            GPT1Model model,
            int[] tokenIds)
        {
            var seqLen = tokenIds.Length;
            var dModel = model.Config.DModel;

            var tokenEmbeddingNode = model.TokenEmbedding.Forward(
                graph,
                tokenIds);

            var tokenEmbedding = graph.Reshape(
                tokenEmbeddingNode,
                1,
                seqLen,
                dModel);

            var positionIds = new int[seqLen];
            for (var i = 0; i < seqLen; i++)
            {
                positionIds[i] = i;
            }

            var positionEmbeddingNode = model.PositionEmbedding.Forward(
                graph,
                positionIds);

            var positionEmbedding = graph.Reshape(
                positionEmbeddingNode,
                1,
                seqLen,
                dModel);

            var x = TensorMath.Add(
                graph,
                tokenEmbedding,
                positionEmbedding);

            var block0Ln1 = model.Blocks[0].Norm1.Forward(
                graph,
                x);

            return block0Ln1
                .DataView
                .AsReadOnlySpan()
                .ToArray();
        }

        private static Dictionary<string, float[]> ComputeOverfitAttentionInternals(
            float[] ln1,
            MultiHeadAttentionLayer attention,
            int seqLen,
            int dModel)
        {
            var nHeads = attention.NHeads;
            var dHead = attention.DHead;

            var q = new float[seqLen * dModel];
            var k = new float[seqLen * dModel];
            var v = new float[seqLen * dModel];

            var qWithBias = new float[seqLen * dModel];
            var kWithBias = new float[seqLen * dModel];
            var vWithBias = new float[seqLen * dModel];

            for (var h = 0; h < nHeads; h++)
            {
                ProjectHeadIntoConcat(
                    ln1,
                    attention.WqHeads[h].DataReadOnlySpan,
                    seqLen,
                    dModel,
                    dHead,
                    h,
                    q);

                ProjectHeadIntoConcat(
                    ln1,
                    attention.WkHeads[h].DataReadOnlySpan,
                    seqLen,
                    dModel,
                    dHead,
                    h,
                    k);

                ProjectHeadIntoConcat(
                    ln1,
                    attention.WvHeads[h].DataReadOnlySpan,
                    seqLen,
                    dModel,
                    dHead,
                    h,
                    v);

                ProjectHeadIntoConcatWithBias(
                    ln1,
                    attention.WqHeads[h].DataReadOnlySpan,
                    attention.BqHeads[h].DataReadOnlySpan,
                    seqLen,
                    dModel,
                    dHead,
                    h,
                    qWithBias);

                ProjectHeadIntoConcatWithBias(
                    ln1,
                    attention.WkHeads[h].DataReadOnlySpan,
                    attention.BkHeads[h].DataReadOnlySpan,
                    seqLen,
                    dModel,
                    dHead,
                    h,
                    kWithBias);

                ProjectHeadIntoConcatWithBias(
                    ln1,
                    attention.WvHeads[h].DataReadOnlySpan,
                    attention.BvHeads[h].DataReadOnlySpan,
                    seqLen,
                    dModel,
                    dHead,
                    h,
                    vWithBias);
            }

            var context = ComputeCausalAttentionContext(
                q,
                k,
                v,
                seqLen,
                nHeads,
                dHead,
                dModel);

            var contextWithBias = ComputeCausalAttentionContext(
                qWithBias,
                kWithBias,
                vWithBias,
                seqLen,
                nHeads,
                dHead,
                dModel);

            var output = ProjectContextThroughWo(
                context,
                attention,
                seqLen,
                dModel,
                nHeads,
                dHead);

            var outputWithBias = ProjectContextThroughWo(
                contextWithBias,
                attention,
                seqLen,
                dModel,
                nHeads,
                dHead);

            return new Dictionary<string, float[]>(StringComparer.Ordinal)
            {
                ["block0_q_raw"] = q,
                ["block0_k_raw"] = k,
                ["block0_v_raw"] = v,
                ["block0_q_with_bias"] = qWithBias,
                ["block0_k_with_bias"] = kWithBias,
                ["block0_v_with_bias"] = vWithBias,
                ["block0_context"] = context,
                ["block0_context_with_bias"] = contextWithBias,
                ["block0_attn_output"] = output,
                ["block0_attn_output_with_bias"] = outputWithBias,
            };
        }

        private static void ProjectHeadIntoConcat(
            float[] input,
            ReadOnlySpan<float> weight,
            int seqLen,
            int dModel,
            int dHead,
            int head,
            float[] destination)
        {
            var headOffset = head * dHead;

            for (var t = 0; t < seqLen; t++)
            {
                var inputOffset = t * dModel;
                var destOffset = t * dModel + headOffset;

                for (var o = 0; o < dHead; o++)
                {
                    var sum = 0.0;

                    for (var i = 0; i < dModel; i++)
                    {
                        sum += input[inputOffset + i] * weight[i * dHead + o];
                    }

                    destination[destOffset + o] = (float)sum;
                }
            }
        }

        private static void ProjectHeadIntoConcatWithBias(
            float[] input,
            ReadOnlySpan<float> weight,
            ReadOnlySpan<float> bias,
            int seqLen,
            int dModel,
            int dHead,
            int head,
            float[] destination)
        {
            var headOffset = head * dHead;

            for (var t = 0; t < seqLen; t++)
            {
                var inputOffset = t * dModel;
                var destOffset = t * dModel + headOffset;

                for (var o = 0; o < dHead; o++)
                {
                    var sum = (double)bias[o];

                    for (var i = 0; i < dModel; i++)
                    {
                        sum += input[inputOffset + i] * weight[i * dHead + o];
                    }

                    destination[destOffset + o] = (float)sum;
                }
            }
        }

        private static float[] ComputeCausalAttentionContext(
            float[] q,
            float[] k,
            float[] v,
            int seqLen,
            int nHeads,
            int dHead,
            int dModel)
        {
            var context = new float[seqLen * dModel];
            var scores = new double[seqLen];
            var scale = 1.0 / Math.Sqrt(dHead);

            for (var h = 0; h < nHeads; h++)
            {
                var headOffset = h * dHead;

                for (var t = 0; t < seqLen; t++)
                {
                    var maxScore = double.NegativeInfinity;

                    for (var source = 0; source <= t; source++)
                    {
                        var dot = 0.0;
                        var qOffset = t * dModel + headOffset;
                        var kOffset = source * dModel + headOffset;

                        for (var d = 0; d < dHead; d++)
                        {
                            dot += q[qOffset + d] * k[kOffset + d];
                        }

                        var score = dot * scale;
                        scores[source] = score;

                        if (score > maxScore)
                        {
                            maxScore = score;
                        }
                    }

                    var denominator = 0.0;

                    for (var source = 0; source <= t; source++)
                    {
                        var weight = Math.Exp(scores[source] - maxScore);
                        scores[source] = weight;
                        denominator += weight;
                    }

                    var outOffset = t * dModel + headOffset;

                    for (var d = 0; d < dHead; d++)
                    {
                        var sum = 0.0;

                        for (var source = 0; source <= t; source++)
                        {
                            var probability = scores[source] / denominator;
                            var vOffset = source * dModel + headOffset;
                            sum += probability * v[vOffset + d];
                        }

                        context[outOffset + d] = (float)sum;
                    }
                }
            }

            return context;
        }

        private static float[] ProjectContextThroughWo(
            float[] context,
            MultiHeadAttentionLayer attention,
            int seqLen,
            int dModel,
            int nHeads,
            int dHead)
        {
            var output = new float[seqLen * dModel];

            for (var h = 0; h < nHeads; h++)
            {
                var headOffset = h * dHead;
                var weight = attention.WoHeads[h].DataReadOnlySpan;

                for (var t = 0; t < seqLen; t++)
                {
                    var contextOffset = t * dModel + headOffset;
                    var outputOffset = t * dModel;

                    for (var o = 0; o < dModel; o++)
                    {
                        var sum = 0.0;

                        for (var i = 0; i < dHead; i++)
                        {
                            sum += context[contextOffset + i] * weight[i * dModel + o];
                        }

                        output[outputOffset + o] += (float)sum;
                    }
                }
            }

            var bias = attention.Bo.DataReadOnlySpan;

            for (var t = 0; t < seqLen; t++)
            {
                var outputOffset = t * dModel;

                for (var i = 0; i < dModel; i++)
                {
                    output[outputOffset + i] += bias[i];
                }
            }

            return output;
        }

        private void Compare(
            string name,
            IReadOnlyList<float> reference,
            IReadOnlyList<float> actual)
        {
            if (reference.Count != actual.Count)
            {
                _output.WriteLine($"{name}: length mismatch reference={reference.Count}, overfit={actual.Count}");
                return;
            }

            var maxAbs = 0.0f;
            var meanAbs = 0.0;
            var rms = 0.0;
            var dot = 0.0;
            var refNorm = 0.0;
            var actualNorm = 0.0;
            var maxIndex = 0;

            for (var i = 0; i < reference.Count; i++)
            {
                var diff = MathF.Abs(reference[i] - actual[i]);

                if (diff > maxAbs)
                {
                    maxAbs = diff;
                    maxIndex = i;
                }

                meanAbs += diff;
                rms += diff * diff;
                dot += reference[i] * actual[i];
                refNorm += reference[i] * reference[i];
                actualNorm += actual[i] * actual[i];
            }

            meanAbs /= reference.Count;
            rms = Math.Sqrt(rms / reference.Count);

            var cosine = refNorm <= 0.0 || actualNorm <= 0.0
                ? 0.0
                : dot / Math.Sqrt(refNorm * actualNorm);

            _output.WriteLine(
                $"{name,-52} maxAbs={maxAbs,12:F6} meanAbs={meanAbs,12:F6} rms={rms,12:F6} cosine={cosine,10:F6} maxIndex={maxIndex}");

            _output.WriteLine(
                $"  reference[{maxIndex}]={reference[maxIndex]:F6}, overfit[{maxIndex}]={actual[maxIndex]:F6}");
        }

        private static double L1(ReadOnlySpan<float> values)
        {
            var sum = 0.0;

            for (var i = 0; i < values.Length; i++)
            {
                sum += Math.Abs(values[i]);
            }

            return sum;
        }

        private static Gpt2AttentionReference LoadReference(string path)
        {
            var json = File.ReadAllText(path);
            var reference = JsonSerializer.Deserialize<Gpt2AttentionReference>(
                json,
                new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true,
                });

            return reference ?? throw new InvalidOperationException(
                $"Could not deserialize reference JSON: {path}");
        }

        private static int GetIntEnvironmentVariable(
            string name,
            int defaultValue)
        {
            var raw = Environment.GetEnvironmentVariable(name);
            return int.TryParse(raw, out var parsed) && parsed > 0
                ? parsed
                : defaultValue;
        }

        private static void SkipIfMissing(params string[] paths)
        {
            foreach (var path in paths)
            {
                if (!File.Exists(path))
                {
                    throw new Exception(
                        $"Fixture '{path}' not found.\n" +
                        "Run:\n" +
                        "  python3 Scripts/convert_gpt2.py --size small --out Tests/test_fixtures/\n" +
                        "  python3 Scripts/debug_gpt2_reference.py --size small --fixtures Tests/test_fixtures --out Tests/test_fixtures/gpt2_reference_small.json");
                }
            }
        }

        private sealed class Gpt2AttentionReference
        {
            [JsonPropertyName("prompt")]
            public string Prompt { get; set; } = string.Empty;

            [JsonPropertyName("tokens")]
            public int[] Tokens { get; set; } = [];

            [JsonPropertyName("vocab_size")]
            public int VocabSize { get; set; }

            [JsonPropertyName("stage_shape")]
            public int[] StageShape { get; set; } = [];

            [JsonPropertyName("stages")]
            public Dictionary<string, float[]> Stages { get; set; } = new(StringComparer.Ordinal);

            [JsonPropertyName("attention")]
            public Dictionary<string, float[]> Attention { get; set; } = new(StringComparer.Ordinal);
        }
    }
}
