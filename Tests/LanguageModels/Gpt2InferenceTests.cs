// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Tests.TestSupport;
using DevOnBike.Overfit.Tokenization;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels
{
    /// <summary>
    /// GPT-2 inference smoke tests.
    ///
    /// Fixtures resolved via <c>TestModelPaths.Gpt2Small</c>
    /// (defaults: <c>C:\gpt2\gpt2_small.bin</c>, <c>vocab.json</c>, <c>merges.txt</c>;
    ///  override with the <c>OVERFIT_GPT2_DIR</c> env var).
    ///
    /// Generate fixtures once with:
    ///   python3 Scripts/convert_gpt2.py --size small --out C:\gpt2\
    ///
    /// These are integration/smoke tests for the imported checkpoint path, tokenizer,
    /// and cached generation API. Sampler probability behavior is covered separately
    /// in TokenSamplerTests; numerical parity against PyTorch (top-10 logit overlap 10/10,
    /// maxAbsDiff ≈ 0.000107) lives in <c>Gpt2ImportParityDiagnostics</c>.
    /// </summary>
    public class Gpt2InferenceTests
    {
        private readonly ITestOutputHelper _output;

        private static string ModelPath  => TestModelPaths.Gpt2Small.BinaryPath;
        private static string VocabPath  => TestModelPaths.Gpt2Small.VocabPath;
        private static string MergesPath => TestModelPaths.Gpt2Small.MergesPath;

        public Gpt2InferenceTests(
            ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        public void Gpt2Small_LoadAndGenerate_ProducesText()
        {
            SkipIfMissing(
                ModelPath,
                VocabPath,
                MergesPath);

            var tokenizer = BytePairEncoder.Load(
                VocabPath,
                MergesPath);

            _output.WriteLine($"Tokenizer: {tokenizer.VocabSize} tokens");

            Assert.Equal(
                50257,
                tokenizer.VocabSize);

            using var model = new GPT1Model(Gpt2Config.Small);
            model.Eval();

            using (var fs = File.OpenRead(ModelPath))
            using (var br = new BinaryReader(fs))
            {
                var loadWatch = System.Diagnostics.Stopwatch.StartNew();

                model.Load(br);

                _output.WriteLine(
                    $"Loaded in {loadWatch.ElapsedMilliseconds}ms ({new FileInfo(ModelPath).Length / 1e6:F0}MB)");
            }

            var prompts = new[]
            {
                "The future of software development is",
                "In C#, the best way to handle memory is",
                "Kubernetes pod anomaly detection works by"
            };

            foreach (var prompt in prompts)
            {
                var promptTokens = tokenizer.Encode(prompt);

                _output.WriteLine(string.Empty);
                _output.WriteLine($"Prompt: \"{prompt}\"");
                _output.WriteLine($"Tokens: {promptTokens.Length}");

                Assert.NotEmpty(promptTokens);
                Assert.All(
                    promptTokens,
                    token => Assert.InRange(token, 0, tokenizer.VocabSize - 1));

                var generated = GenerateWithCachedRuntime(
                    model,
                    promptTokens,
                    maxNewTokens: 16,
                    SamplingOptions.Greedy);

                var generatedText = tokenizer.Decode(generated);
                var text = prompt + generatedText;

                _output.WriteLine($"Output: \"{text}\"");

                Assert.True(
                    generated.Length > 0,
                    "No tokens generated");

                Assert.All(
                    generated,
                    token => Assert.InRange(token, 0, tokenizer.VocabSize - 1));

                AssertNoNullCharacters(text);
            }

            _output.WriteLine(string.Empty);
            _output.WriteLine("✓ GPT-2 Small fixture loads and generates through cached runtime.");
        }

        [Fact]
        public void Gpt2Small_TopPSampling_ExecutesThroughCachedRuntime()
        {
            SkipIfMissing(
                ModelPath,
                VocabPath,
                MergesPath);

            var tokenizer = BytePairEncoder.Load(
                VocabPath,
                MergesPath);

            Assert.Equal(
                50257,
                tokenizer.VocabSize);

            using var model = new GPT1Model(Gpt2Config.Small);
            model.Eval();

            using (var fs = File.OpenRead(ModelPath))
            using (var br = new BinaryReader(fs))
            {
                model.Load(br);
            }

            var promptTokens = tokenizer.Encode("Once upon a time");

            Assert.NotEmpty(promptTokens);

            var outputs = new List<string>();

            for (var seed = 0; seed < 3; seed++)
            {
                var sampling = new SamplingOptions(
                    SamplingStrategy.TopP,
                    temperature: 0.8f,
                    topK: 0,
                    topP: 0.95f,
                    seed: seed);

                var generated = GenerateWithCachedRuntime(
                    model,
                    promptTokens,
                    maxNewTokens: 16,
                    sampling);

                var text = tokenizer.Decode(generated);

                outputs.Add(text);
                _output.WriteLine($"Seed {seed}: \"{text}\"");

                Assert.True(
                    generated.Length > 0,
                    "No tokens generated");

                Assert.All(
                    generated,
                    token => Assert.InRange(token, 0, tokenizer.VocabSize - 1));

                AssertNoNullCharacters(text);
            }

            Assert.Equal(
                3,
                outputs.Count);

            _output.WriteLine(string.Empty);
            _output.WriteLine("✓ Top-P sampling executes through cached GPT-2 runtime.");
        }

        private static int[] GenerateWithCachedRuntime(
            GPT1Model model,
            int[] promptTokens,
            int maxNewTokens,
            SamplingOptions sampling)
        {
            using var handle = SlmRuntimeFactory.CreateGpt1(
                model,
                SlmRuntimeMode.Cached);

            handle.Session.Reset(promptTokens);

            var generated = new int[maxNewTokens];

            for (var i = 0; i < generated.Length; i++)
            {
                generated[i] = handle.Session.GenerateNextToken(
                    in sampling);
            }

            return generated;
        }

        private static void AssertNoNullCharacters(
            string text)
        {
            for (var i = 0; i < text.Length; i++)
            {
                if (text[i] == '\0')
                {
                    throw new InvalidOperationException(
                        $"Generated text contains a null character at index {i}.");
                }
            }
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
                        "Run: python3 Scripts/convert_gpt2.py --size small --out Tests/test_fixtures/");
                }
            }
        }
    }
}
