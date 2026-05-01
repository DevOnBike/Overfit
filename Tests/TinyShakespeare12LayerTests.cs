// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Tokenization;
using Xunit;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests
{
    /// <summary>
    /// Opcja A: 12 warstw, 256d, 8 głów — ~9.5M parametrów.
    ///
    /// Cel: val loss poniżej 2.0 po 5000 krokach.
    /// Punkt odniesienia: nanoGPT (Karpathy) osiąga 1.47 przy SeqLen=256.
    /// My używamy SeqLen=64 (limit areny 50MB) więc cel to ~1.8-2.0.
    ///
    /// Co pokazuje:
    ///   Pełna głębokość GPT-1 (12 warstw) uczy się struktury języka angielskiego
    ///   na Tiny Shakespeare. Gradient przepływa przez cały stos bez dywergencji.
    ///
    /// Czas: ~5-8 minut na Ryzen 9 9950X3D.
    ///
    /// Fixture:
    ///   curl -o Tests/test_fixtures/tiny_shakespeare.txt \
    ///     https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    /// </summary>
    public class TinyShakespeare12LayerTests
    {
        private readonly ITestOutputHelper _output;
        private const string FixturePath = "test_fixtures/tiny_shakespeare.txt";

        public TinyShakespeare12LayerTests(ITestOutputHelper output)
        {
            _output = output;
        }

        //[Fact]
        public void Shakespeare_12Layer_256d_LossBelow200_After5000Steps()
        {
            SkipIfMissing(FixturePath);

            // ── Konfiguracja ─────────────────────────────────────────────────
            // 12 warstw, 256d, 8 głów, dFF=1024, SeqLen=64
            // Params: ~9.5M | Arena: ~8MB (mieści się w 50MB)
            const int seqLen = 64;
            const int totalSteps = 5_000;
            const int reportEvery = 500;
            const float lr = 3e-4f;

            // ── Dane ─────────────────────────────────────────────────────────
            var text = File.ReadAllText(FixturePath);
            var tokenizer = CharacterTokenizer.FromCorpus(text);
            var allIds = tokenizer.Encode(text);

            var trainSize = (int)(allIds.Length * 0.9);
            var trainIds = allIds.AsSpan(0, trainSize).ToArray();
            var valIds = allIds.AsSpan(trainSize).ToArray();

            _output.WriteLine($"Corpus: {text.Length:N0} chars, vocab={tokenizer.VocabSize}");
            _output.WriteLine(string.Empty);

            // ── Model ─────────────────────────────────────────────────────────
            var config = new GPT1Config
            {
                VocabSize = tokenizer.VocabSize,
                ContextLength = seqLen,
                DModel = 256,
                NHeads = 8,
                NLayers = 12,
                DFF = 1024,
                TieWeights = false,
                PreLayerNorm = true,
            };

            _output.WriteLine($"Model: {config}");
            _output.WriteLine($"Parameters: {config.ParameterCount:N0} (~{config.ParameterCount / 1e6:F1}M)");
            _output.WriteLine($"Training: {totalSteps:N0} steps, seqLen={seqLen}, lr={lr}");
            _output.WriteLine(string.Empty);

            using var model = new GPT1Model(config);
            using var optimizer = new Adam(model.TrainableParameters(), lr)
            {
                UseAdamW = true,
                WeightDecay = 0.1f,
            };

            model.Train();

            var rng = new Random(42);
            var sw = System.Diagnostics.Stopwatch.StartNew();
            float windowLoss = 0f;
            float initialLoss = 0f;
            float finalValLoss = 0f;

            // ── Pętla treningowa ──────────────────────────────────────────────
            for (var step = 0; step < totalSteps; step++)
            {
                var (inputIds, targetIds) = SampleSequence(trainIds, seqLen, rng);

                optimizer.ZeroGrad();
                using var graph = new ComputationGraph();
                using var logits = model.Forward(graph, inputIds, batchSize: 1, seqLen);

                var loss = ComputeLossAndSeedGrad(logits, targetIds, seqLen, config.VocabSize);
                windowLoss += loss;

                graph.BackwardFromGrad(logits);
                optimizer.Step();

                if (step == 0)
                {
                    initialLoss = loss;
                    _output.WriteLine(
                        $"Step {step,5} | Loss: {loss:F4} | " +
                        $"baseline ≈ ln({tokenizer.VocabSize}) = {MathF.Log(tokenizer.VocabSize):F4}");
                }

                if ((step + 1) % reportEvery == 0)
                {
                    var avgLoss = windowLoss / reportEvery;
                    windowLoss = 0f;

                    var valLoss = EvaluateLoss(model, valIds, config, seqLen, rng, valSteps: 50);
                    var elapsed = sw.Elapsed;

                    _output.WriteLine(
                        $"Step {step + 1,5} | Train: {avgLoss:F4} | Val: {valLoss:F4} | " +
                        $"{elapsed:mm\\:ss} | {elapsed.TotalMilliseconds / (step + 1):F0}ms/step");

                    // Sample text po każdych 2500 krokach
                    if ((step + 1) % 2500 == 0)
                    {
                        var sample = GenerateSample(model, tokenizer, "ROMEO:", maxTokens: 120);
                        _output.WriteLine($"  Sample: \"{sample}\"");
                        _output.WriteLine(string.Empty);
                    }

                    if (step + 1 == totalSteps)
                    {
                        finalValLoss = valLoss;
                    }
                }
            }

            sw.Stop();

            _output.WriteLine(string.Empty);
            _output.WriteLine($"Trening zakończony: {sw.Elapsed:mm\\:ss}");
            _output.WriteLine($"Initial loss: {initialLoss:F4}");
            _output.WriteLine($"Final val loss: {finalValLoss:F4}");
            _output.WriteLine($"nanoGPT reference (SeqLen=256): 1.4697");
            _output.WriteLine($"Nasz cel (SeqLen=64): < 2.0");
            _output.WriteLine(string.Empty);

            // Finalna generacja
            var finalSample = GenerateSample(model, tokenizer, "ROMEO:", maxTokens: 200);
            _output.WriteLine("Finalna generacja:");
            _output.WriteLine($"  \"{finalSample}\"");
            _output.WriteLine(string.Empty);

            // ── Asercje ───────────────────────────────────────────────────────

            // 1. Val loss poniżej 2.0 — realny wynik uczenia na 12 warstwach
            Assert.True(finalValLoss < 2.0f,
                $"Val loss {finalValLoss:F4} ≥ 2.0. " +
                "Model nie nauczył się wystarczająco — sprawdź backward pass.");

            // 2. Poprawa względem startu
            Assert.True(finalValLoss < initialLoss * 0.60f,
                $"Za mała poprawa: initial={initialLoss:F4}, val={finalValLoss:F4}. " +
                "Oczekiwano 40%+ redukcji loss.");

            // 3. Brak NaN
            Assert.False(float.IsNaN(finalValLoss) || float.IsInfinity(finalValLoss),
                "Val loss jest NaN/Inf — problem numeryczny.");

            _output.WriteLine("✓ Val loss < 2.0 — 12-warstwowy GPT uczy się języka angielskiego.");
            _output.WriteLine("✓ Porównanie z nanoGPT: osiągalny ~1.5 przy SeqLen=256.");
        }

        // ── Helpers ──────────────────────────────────────────────────────────

        private static (int[] inputIds, int[] targetIds) SampleSequence(
            int[] corpus, int seqLen, Random rng)
        {
            var start = rng.Next(0, corpus.Length - seqLen - 1);
            var inputIds = corpus.AsSpan(start, seqLen).ToArray();
            var targetIds = corpus.AsSpan(start + 1, seqLen).ToArray();
            return (inputIds, targetIds);
        }

        private static float ComputeLossAndSeedGrad(
            AutogradNode logits,
            int[] targetIds,
            int seqLen,
            int vocabSize)
        {
            var logitS = logits.DataView.AsReadOnlySpan();
            var gradS = logits.GradView.AsSpan();
            gradS.Clear();

            var totalLoss = 0f;

            for (var t = 0; t < seqLen; t++)
            {
                var row = logitS.Slice(t * vocabSize, vocabSize);
                var gradRow = gradS.Slice(t * vocabSize, vocabSize);
                var targetId = targetIds[t];

                var maxVal = row[0];
                for (var v = 1; v < vocabSize; v++)
                    if (row[v] > maxVal) maxVal = row[v];

                var sumExp = 0f;
                for (var v = 0; v < vocabSize; v++)
                    sumExp += MathF.Exp(row[v] - maxVal);

                totalLoss += maxVal + MathF.Log(sumExp) - row[targetId];

                var scale = 1f / seqLen;
                for (var v = 0; v < vocabSize; v++)
                {
                    var sm = MathF.Exp(row[v] - maxVal) / sumExp;
                    gradRow[v] = (sm - (v == targetId ? 1f : 0f)) * scale;
                }
            }

            return totalLoss / seqLen;
        }

        private static float EvaluateLoss(
            GPT1Model model,
            int[] valCorpus,
            GPT1Config config,
            int seqLen,
            Random rng,
            int valSteps)
        {
            model.Eval();
            var total = 0f;

            for (var s = 0; s < valSteps; s++)
            {
                var (inputIds, targetIds) = SampleSequence(valCorpus, seqLen, rng);
                using var graph = new ComputationGraph();
                using var logits = model.Forward(graph, inputIds, batchSize: 1, seqLen);
                total += ComputeLossAndSeedGrad(logits, targetIds, seqLen, config.VocabSize);
            }

            model.Train();
            return total / valSteps;
        }

        private static string GenerateSample(
            GPT1Model model,
            CharacterTokenizer tokenizer,
            string prompt,
            int maxTokens)
        {
            model.Eval();
            var ids = tokenizer.Encode(prompt);
            var generated = model.Generate(ids, maxTokens);
            model.Train();
            return prompt + tokenizer.Decode(generated);
        }

        private static void SkipIfMissing(string path)
        {
            if (!File.Exists(path))
            {
                throw new Exception(
                    $"Fixture '{path}' not found. " +
                    "curl -o Tests/test_fixtures/tiny_shakespeare.txt " +
                    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt");
            }
        }
    }
}