// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
//
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using System.Text.Json.Serialization;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Tokenization;
using Xunit;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Manual GPT-2 import parity diagnostic.
    ///
    /// This is intentionally skipped by default because it depends on large GPT-2 fixtures
    /// and a Python-generated PyTorch reference JSON.
    ///
    /// Generate fixtures:
    ///   python3 Scripts/convert_gpt2.py --size small --out Tests/test_fixtures/
    ///
    /// Generate reference:
    ///   python3 Scripts/debug_gpt2_reference.py --size small --fixtures Tests/test_fixtures --out Tests/test_fixtures/gpt2_reference_small.json
    ///
    /// Then remove Skip locally and run:
    ///   dotnet test -c Release --filter "Gpt2Small_CompareFinalLogitsAgainstPyTorchReference"
    ///
    /// Current expectation:
    /// - tokenizer ids should match,
    /// - logits are likely not close yet,
    /// - output should reveal whether LM head / attention / layer layout is badly mismatched.
    /// </summary>
    public sealed class Gpt2ImportParityDiagnostics
    {
        private const string ModelPath = "test_fixtures/gpt2_small.bin";
        private const string VocabPath = "test_fixtures/vocab.json";
        private const string MergesPath = "test_fixtures/merges.txt";
        private const string ReferencePath = "test_fixtures/gpt2_reference_small.json";

        private const int DefaultArenaSize = 1_500_000_000;

        private readonly ITestOutputHelper _output;

        public Gpt2ImportParityDiagnostics(
            ITestOutputHelper output)
        {
            _output = output;
        }

        //[Fact(Skip = "Manual GPT-2 import parity diagnostic. Generate PyTorch reference JSON first, remove Skip locally, run once, then restore Skip.")]
        [Fact]
        [Trait("Category", "Diagnostics")]
        [Trait("Category", "Manual")]
        public void Gpt2Small_CompareFinalLogitsAgainstPyTorchReference()
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

            _output.WriteLine("=== GPT-2 Import Parity Diagnostic ===");
            _output.WriteLine($"Prompt: {reference.Prompt}");
            _output.WriteLine($"Tokens: {string.Join(", ", reference.Tokens)}");
            _output.WriteLine($"Vocab: {reference.VocabSize}");
            _output.WriteLine("");

            using var model = new GPT1Model(Gpt2Config.Small);
            model.Eval();

            using (var stream = File.OpenRead(ModelPath))
            using (var reader = new BinaryReader(stream))
            {
                model.Load(reader);
            }

            var arenaSize = GetIntEnvironmentVariable(
                "OVERFIT_GPT2_DIAG_ARENA",
                DefaultArenaSize);

            using var graph = new ComputationGraph(arenaSize);

            model.InvalidateAllCaches();

            var logitsNode = model.Forward(
                graph,
                overfitTokens,
                batchSize: 1,
                seqLen: overfitTokens.Length);

            try
            {
                var overfitFinalLogits = ExtractLastPositionLogits(
                    logitsNode,
                    overfitTokens.Length,
                    reference.VocabSize);

                CompareTopLogits(
                    reference,
                    overfitFinalLogits,
                    topK: 10);

                var referenceTop = reference.TopLogits[0].Token;
                var overfitTop = ArgMax(overfitFinalLogits);

                _output.WriteLine("");
                _output.WriteLine($"Reference next token: {referenceTop} '{tokenizer.Decode(new[] { referenceTop })}'");
                _output.WriteLine($"Overfit next token:   {overfitTop} '{tokenizer.Decode(new[] { overfitTop })}'");

                var assertClose = GetBoolEnvironmentVariable(
                    "OVERFIT_GPT2_DIAG_ASSERT_CLOSE",
                    defaultValue: false);

                if (assertClose)
                {
                    Assert.Equal(
                        referenceTop,
                        overfitTop);
                }
            }
            finally
            {
                logitsNode.Dispose();
            }
        }

        private void CompareTopLogits(
            Gpt2Reference reference,
            float[] overfitFinalLogits,
            int topK)
        {
            var overfitTop = TopK(
                overfitFinalLogits,
                topK);

            var referenceTopTokens = reference
                .TopLogits
                .Take(topK)
                .Select(item => item.Token)
                .ToArray();

            var overfitTopTokens = overfitTop
                .Select(item => item.Token)
                .ToArray();

            var overlap = referenceTopTokens
                .Intersect(overfitTopTokens)
                .Count();

            _output.WriteLine("Reference top logits:");
            foreach (var item in reference.TopLogits.Take(topK))
            {
                _output.WriteLine($"  {item.Token,6} {item.Logit,14:F6}");
            }

            _output.WriteLine("");
            _output.WriteLine("Overfit top logits:");
            foreach (var item in overfitTop)
            {
                _output.WriteLine($"  {item.Token,6} {item.Logit,14:F6}");
            }

            _output.WriteLine("");
            _output.WriteLine($"Top-{topK} overlap: {overlap}/{topK}");

            var unionTokens = referenceTopTokens
                .Concat(overfitTopTokens)
                .Distinct()
                .ToArray();

            var maxAbsDiffOnUnion = 0.0f;
            var sumAbsDiffOnUnion = 0.0f;

            foreach (var token in unionTokens)
            {
                var referenceLogit = reference.Logits[token];
                var overfitLogit = overfitFinalLogits[token];
                var diff = MathF.Abs(referenceLogit - overfitLogit);

                maxAbsDiffOnUnion = MathF.Max(
                    maxAbsDiffOnUnion,
                    diff);

                sumAbsDiffOnUnion += diff;
            }

            _output.WriteLine($"Max abs diff on top-token union: {maxAbsDiffOnUnion:F6}");
            _output.WriteLine($"Mean abs diff on top-token union: {sumAbsDiffOnUnion / unionTokens.Length:F6}");
        }

        private static float[] ExtractLastPositionLogits(
            AutogradNode logitsNode,
            int seqLen,
            int vocabSize)
        {
            var data = logitsNode.DataView.AsReadOnlySpan();

            var expected = seqLen * vocabSize;

            if (data.Length < expected)
            {
                throw new InvalidOperationException(
                    $"Logits data length {data.Length} is smaller than expected {expected}.");
            }

            var lastOffset = (seqLen - 1) * vocabSize;
            var result = new float[vocabSize];

            data
                .Slice(lastOffset, vocabSize)
                .CopyTo(result);

            return result;
        }

        private static int ArgMax(
            IReadOnlyList<float> values)
        {
            if (values.Count == 0)
            {
                throw new ArgumentException("Values cannot be empty.", nameof(values));
            }

            var index = 0;
            var max = values[0];

            for (var i = 1; i < values.Count; i++)
            {
                if (values[i] > max)
                {
                    max = values[i];
                    index = i;
                }
            }

            return index;
        }

        private static List<TopLogit> TopK(
            IReadOnlyList<float> values,
            int k)
        {
            return Enumerable
                .Range(0, values.Count)
                .Select(index => new TopLogit
                {
                    Token = index,
                    Logit = values[index]
                })
                .OrderByDescending(item => item.Logit)
                .Take(k)
                .ToList();
        }

        private static Gpt2Reference LoadReference(
            string path)
        {
            var json = File.ReadAllText(path);

            var reference = JsonSerializer.Deserialize<Gpt2Reference>(
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

        private sealed class Gpt2Reference
        {
            [JsonPropertyName("size")]
            public string Size { get; set; } = string.Empty;

            [JsonPropertyName("prompt")]
            public string Prompt { get; set; } = string.Empty;

            [JsonPropertyName("tokens")]
            public int[] Tokens { get; set; } = Array.Empty<int>();

            [JsonPropertyName("vocab_size")]
            public int VocabSize { get; set; }

            [JsonPropertyName("next_token")]
            public int NextToken { get; set; }

            [JsonPropertyName("next_token_text")]
            public string NextTokenText { get; set; } = string.Empty;

            [JsonPropertyName("top_logits")]
            public TopLogit[] TopLogits { get; set; } = Array.Empty<TopLogit>();

            [JsonPropertyName("logits")]
            public float[] Logits { get; set; } = Array.Empty<float>();
        }

        private sealed class TopLogit
        {
            [JsonPropertyName("token")]
            public int Token { get; set; }

            [JsonPropertyName("logit")]
            public float Logit { get; set; }
        }
    }
}
