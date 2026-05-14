// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using System.Text.Json.Serialization;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tests.TestSupport;
using DevOnBike.Overfit.Tokenization;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Manual GPT-2 stage parity diagnostic.
    ///
    /// Generate fixtures:
    ///   python3 Scripts/convert_gpt2.py --size small --out Tests/test_fixtures/
    ///
    /// Generate reference:
    ///   python3 Scripts/debug_gpt2_reference.py --size small --fixtures Tests/test_fixtures --out Tests/test_fixtures/gpt2_reference_small.json
    ///
    /// Remove Skip locally and run:
    ///   dotnet test -c Release --filter "Gpt2Small_CompareStagesAgainstPyTorchReference"
    ///
    /// This test is intentionally diagnostic-only. It prints stage deltas and does not
    /// require parity unless OVERFIT_GPT2_STAGE_ASSERT_CLOSE=true.
    /// </summary>
    public sealed class Gpt2ImportStageParityDiagnostics
    {
        private static string ModelPath     => TestModelPaths.Gpt2Small.BinaryPath;
        private static string VocabPath     => TestModelPaths.Gpt2Small.VocabPath;
        private static string MergesPath    => TestModelPaths.Gpt2Small.MergesPath;
        private static string ReferencePath => TestModelPaths.Gpt2Small.ReferenceJsonPath;

        private const int DefaultArenaSize = 1_500_000_000;

        private readonly ITestOutputHelper _output;

        public Gpt2ImportStageParityDiagnostics(
            ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        [Trait("Category", "Diagnostics")]
        [Trait("Category", "Parity")]
        public void Gpt2Small_CompareStagesAgainstPyTorchReference()
        {
            SkipIfMissing(
                ModelPath,
                VocabPath,
                MergesPath,
                ReferencePath);

            var reference = LoadReference(ReferencePath);

            var tokenizer = BytePairEncoder.Load(
                VocabPath,
                MergesPath);

            Assert.Equal(
                reference.VocabSize,
                tokenizer.VocabSize);

            var overfitTokens = tokenizer.Encode(reference.Prompt);

            Assert.Equal(
                reference.Tokens,
                overfitTokens);

            _output.WriteLine("=== GPT-2 Stage Parity Diagnostic ===");
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

            var arenaSize = GetIntEnvironmentVariable(
                "OVERFIT_GPT2_STAGE_ARENA",
                DefaultArenaSize);

            using var graph = new ComputationGraph(arenaSize);

            model.InvalidateAllCaches();

            var overfitStages = ComputeOverfitStages(
                graph,
                model,
                overfitTokens);

            var stageOrder = new[]
            {
                "embedding",
                "block0_ln1",
                "block0_attn",
                "block0_after_attn_residual",
                "block0_ln2",
                "block0_mlp",
                "block0_output",
                "final_norm"
            };

            foreach (var stage in stageOrder)
            {
                if (!reference.Stages.TryGetValue(stage, out var referenceStage))
                {
                    _output.WriteLine($"Missing reference stage: {stage}");
                    continue;
                }

                if (!overfitStages.TryGetValue(stage, out var overfitStage))
                {
                    _output.WriteLine($"Missing Overfit stage: {stage}");
                    continue;
                }

                PrintVectorComparison(
                    stage,
                    referenceStage,
                    overfitStage);
            }

            var assertClose = GetBoolEnvironmentVariable(
                "OVERFIT_GPT2_STAGE_ASSERT_CLOSE",
                defaultValue: false);

            if (assertClose)
            {
                foreach (var stage in stageOrder)
                {
                    var referenceStage = reference.Stages[stage];
                    var overfitStage = overfitStages[stage];

                    AssertStageClose(
                        stage,
                        referenceStage,
                        overfitStage,
                        maxAllowedAbsDiff: 1e-3f);
                }
            }
        }

        private Dictionary<string, float[]> ComputeOverfitStages(
            ComputationGraph graph,
            GPT1Model model,
            int[] tokenIds)
        {
            var seqLen = tokenIds.Length;
            var dModel = model.Config.DModel;
            var stages = new Dictionary<string, float[]>(
                StringComparer.Ordinal);

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

            stages["embedding"] = CopyNodeData(x);

            var block0 = model.Blocks[0];

            var block0Ln1 = block0.Norm1.Forward(
                graph,
                x);

            stages["block0_ln1"] = CopyNodeData(block0Ln1);

            var block0Attn = block0.Attention.Forward(
                graph,
                block0Ln1);

            stages["block0_attn"] = CopyNodeData(block0Attn);

            var block0AfterAttn = TensorMath.Add(
                graph,
                x,
                block0Attn);

            stages["block0_after_attn_residual"] = CopyNodeData(block0AfterAttn);

            var block0Ln2 = block0.Norm2.Forward(
                graph,
                block0AfterAttn);

            stages["block0_ln2"] = CopyNodeData(block0Ln2);

            var block0Mlp = block0.FFN.Forward(
                graph,
                block0Ln2);

            stages["block0_mlp"] = CopyNodeData(block0Mlp);

            var current = TensorMath.Add(
                graph,
                block0AfterAttn,
                block0Mlp);

            stages["block0_output"] = CopyNodeData(current);

            for (var i = 1; i < model.Blocks.Length; i++)
            {
                current = model.Blocks[i].Forward(
                    graph,
                    current);
            }

            var finalNorm = model.FinalNorm.Forward(
                graph,
                current);

            stages["final_norm"] = CopyNodeData(finalNorm);

            return stages;
        }

        private void PrintVectorComparison(
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

            var cosine =
                refNorm <= 0.0 || actualNorm <= 0.0
                    ? 0.0
                    : dot / Math.Sqrt(refNorm * actualNorm);

            _output.WriteLine(
                $"{name,-30} maxAbs={maxAbs,12:F6} meanAbs={meanAbs,12:F6} rms={rms,12:F6} cosine={cosine,10:F6} maxIndex={maxIndex}");

            _output.WriteLine(
                $"  reference[{maxIndex}]={reference[maxIndex]:F6}, overfit[{maxIndex}]={actual[maxIndex]:F6}");
        }

        private static void AssertStageClose(
            string name,
            IReadOnlyList<float> reference,
            IReadOnlyList<float> actual,
            float maxAllowedAbsDiff)
        {
            Assert.Equal(
                reference.Count,
                actual.Count);

            for (var i = 0; i < reference.Count; i++)
            {
                var diff = MathF.Abs(reference[i] - actual[i]);

                if (diff > maxAllowedAbsDiff)
                {
                    throw new InvalidOperationException(
                        $"Stage '{name}' mismatch at {i}: reference={reference[i]}, overfit={actual[i]}, diff={diff}");
                }
            }
        }

        private static float[] CopyNodeData(
            AutogradNode node)
        {
            return node
                .DataView
                .AsReadOnlySpan()
                .ToArray();
        }

        private static Gpt2StageReference LoadReference(
            string path)
        {
            var json = File.ReadAllText(path);

            var reference = JsonSerializer.Deserialize<Gpt2StageReference>(
                json,
                new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
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

        private static bool GetBoolEnvironmentVariable(
            string name,
            bool defaultValue)
        {
            var raw = Environment.GetEnvironmentVariable(name);

            if (string.IsNullOrWhiteSpace(raw))
            {
                return defaultValue;
            }

            return raw.Equals("1", StringComparison.OrdinalIgnoreCase) ||
                   raw.Equals("true", StringComparison.OrdinalIgnoreCase) ||
                   raw.Equals("yes", StringComparison.OrdinalIgnoreCase);
        }

        private static void SkipIfMissing(
            params string[] paths)
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

        private sealed class Gpt2StageReference
        {
            [JsonPropertyName("size")]
            public string Size { get; set; } = string.Empty;

            [JsonPropertyName("prompt")]
            public string Prompt { get; set; } = string.Empty;

            [JsonPropertyName("tokens")]
            public int[] Tokens { get; set; } = [];

            [JsonPropertyName("vocab_size")]
            public int VocabSize { get; set; }

            [JsonPropertyName("stage_shape")]
            public int[] StageShape { get; set; } = [];

            [JsonPropertyName("stages")]
            public Dictionary<string, float[]> Stages { get; set; } = new(
                StringComparer.Ordinal);

            [JsonPropertyName("logits")]
            public float[] Logits { get; set; } = [];
        }
    }
}
