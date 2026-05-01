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
    /// GPT-1 training z SeqLen=256 — pełny kontekst, poprawny gradient przez residuals.
    ///
    /// Naprawiono:
    ///   TransformerBlock.Residual() był off-tape → gradient nie przepływał przez bloki.
    ///   Teraz używamy TensorMath.Add(graph, x, sublayer) — ON TAPE.
    ///   Gradient przepływa przez oba ramiona residual (identity + sublayer).
    ///
    /// SeqLen=256 mieści się w domyślnej arenie (200MB) dla vocab=68 (char-level).
    ///
    /// Cel: val loss poniżej 2.0 po 5000 krokach.
    /// nanoGPT reference: 1.47 przy SeqLen=256, batch=64, 5K iteracji.
    ///
    /// Czas: ~15-20 minut na Ryzen 9 9950X3D.
    /// </summary>
    public class TinyShakespeareCheckpointTests
    {
        private readonly ITestOutputHelper _output;
        private const string FixturePath = "test_fixtures/tiny_shakespeare.txt";

        public TinyShakespeareCheckpointTests(ITestOutputHelper output)
        {
            _output = output;
        }

        // [Fact]
        public void Shakespeare_12Layer_Checkpointed_SeqLen256_LossBelow200()
        {
            SkipIfMissing(FixturePath);

            const int   seqLen      = 256;
            const int   totalSteps  = 5_000;
            const int   reportEvery = 500;
            const float lrMax       = 3e-4f;
            const float lrMin       = 3e-5f;

            var text      = File.ReadAllText(FixturePath);
            var tokenizer = CharacterTokenizer.FromCorpus(text);
            var allIds    = tokenizer.Encode(text);
            var trainSize = (int)(allIds.Length * 0.9);
            var trainIds  = allIds.AsSpan(0, trainSize).ToArray();
            var valIds    = allIds.AsSpan(trainSize).ToArray();

            _output.WriteLine($"Corpus: {text.Length:N0} chars, vocab={tokenizer.VocabSize}");
            _output.WriteLine(string.Empty);

            var config = new GPT1Config
            {
                VocabSize     = tokenizer.VocabSize,
                ContextLength = seqLen,
                DModel        = 256,
                NHeads        = 8,
                NLayers       = 12,
                DFF           = 1024,
                TieWeights    = false,
                PreLayerNorm  = true,
            };

            _output.WriteLine($"Model: {config}");
            _output.WriteLine($"Parameters: {config.ParameterCount:N0} (~{config.ParameterCount / 1e6:F1}M)");
            _output.WriteLine($"SeqLen: {seqLen} | Residuals ON TAPE (TensorMath.Add)");
            _output.WriteLine($"Training: {totalSteps:N0} steps, lr={lrMax}→{lrMin} (cosine)");
            _output.WriteLine(string.Empty);

            using var model     = new GPT1Model(config);
            using var optimizer = new Adam(model.TrainableParameters(), lrMax)
            {
                UseAdamW    = true,
                WeightDecay = 0.1f,
            };

            model.Train();

            var rng           = new Random(42);
            var sw            = System.Diagnostics.Stopwatch.StartNew();
            float windowLoss  = 0f;
            float initialLoss = 0f;
            float finalValLoss = 0f;

            for (var step = 0; step < totalSteps; step++)
            {
                var (inputIds, targetIds) = SampleSequence(trainIds, seqLen, rng);

                optimizer.ZeroGrad();
                using var graph  = new ComputationGraph();
                using var logits = model.Forward(graph, inputIds, batchSize: 1, seqLen);

                var loss = ComputeLossAndSeedGradParallel(logits, targetIds, seqLen, config.VocabSize);
                windowLoss += loss;

                graph.BackwardFromGrad(logits);
                ClipGradNorm(model.TrainableParameters(), maxNorm: 1.0f);

                var cosine = 0.5f * (1f + MathF.Cos(MathF.PI * (float)step / totalSteps));
                optimizer.LearningRate = lrMin + (lrMax - lrMin) * cosine;
                optimizer.Step();

                if (step == 0)
                {
                    initialLoss = loss;
                    _output.WriteLine(
                        $"Step {step,5} | Loss: {loss:F4} | baseline=ln({tokenizer.VocabSize})={MathF.Log(tokenizer.VocabSize):F4}");
                }

                if ((step + 1) % reportEvery == 0)
                {
                    var avgLoss = windowLoss / reportEvery;
                    windowLoss  = 0f;
                    var elapsed = sw.Elapsed;
                    var valStr  = string.Empty;

                    if ((step + 1) % 1_000 == 0)
                    {
                        var valLoss = EvaluateLoss(model, valIds, config, seqLen, rng, valSteps: 20);
                        valStr = $"| Val: {valLoss:F4} ";
                        if (step + 1 == totalSteps) finalValLoss = valLoss;
                    }

                    _output.WriteLine(
                        $"Step {step + 1,5} | Train: {avgLoss:F4} {valStr}| {elapsed:mm\\:ss} | {elapsed.TotalMilliseconds / (step + 1):F0}ms/step");

                    if ((step + 1) % 2_500 == 0 || step + 1 == totalSteps)
                    {
                        var sample = GenerateSample(model, tokenizer, "ROMEO:", maxTokens: 150);
                        _output.WriteLine($"  Sample: \"{sample}\"");
                        _output.WriteLine(string.Empty);
                    }
                }
            }

            sw.Stop();
            _output.WriteLine($"Trening zakończony: {sw.Elapsed:mm\\:ss}");
            _output.WriteLine($"Initial loss: {initialLoss:F4}");
            _output.WriteLine($"Final val loss: {finalValLoss:F4}");
            _output.WriteLine($"nanoGPT reference (SeqLen=256, batch=64): 1.4697");

            var finalSample = GenerateSample(model, tokenizer, "ROMEO:", maxTokens: 250);
            _output.WriteLine($"Finalna generacja: \"{finalSample}\"");
            _output.WriteLine(string.Empty);

            Assert.False(float.IsNaN(finalValLoss) || float.IsInfinity(finalValLoss),
                "Val loss NaN/Inf.");

            Assert.True(finalValLoss < 2.0f,
                $"Val loss {finalValLoss:F4} >= 2.0. " +
                "Z poprawnymi residual gradientami i SeqLen=256 oczekujemy < 2.0.");
        }

        private static (int[] inputIds, int[] targetIds) SampleSequence(
            int[] corpus, int seqLen, Random rng)
        {
            var start = rng.Next(0, corpus.Length - seqLen - 1);
            return (corpus.AsSpan(start, seqLen).ToArray(),
                    corpus.AsSpan(start + 1, seqLen).ToArray());
        }

        private static float ComputeLossAndSeedGradParallel(
            AutogradNode logits, int[] targetIds, int seqLen, int vocabSize)
        {
            var logitArr = logits.DataView.AsReadOnlySpan().ToArray();
            var gradArr  = new float[seqLen * vocabSize];
            var losses   = new float[seqLen];

            Parallel.For(0, seqLen, t =>
            {
                var offset   = t * vocabSize;
                var targetId = targetIds[t];
                var maxVal   = logitArr[offset];
                for (var v = 1; v < vocabSize; v++)
                    if (logitArr[offset + v] > maxVal) maxVal = logitArr[offset + v];
                var sumExp = 0f;
                for (var v = 0; v < vocabSize; v++)
                    sumExp += MathF.Exp(logitArr[offset + v] - maxVal);
                losses[t] = maxVal + MathF.Log(sumExp) - logitArr[offset + targetId];
                var scale = 1f / seqLen;
                for (var v = 0; v < vocabSize; v++)
                {
                    var sm = MathF.Exp(logitArr[offset + v] - maxVal) / sumExp;
                    gradArr[offset + v] = (sm - (v == targetId ? 1f : 0f)) * scale;
                }
            });

            gradArr.AsSpan().CopyTo(logits.GradView.AsSpan());
            var total = 0f;
            for (var t = 0; t < seqLen; t++) total += losses[t];
            return total / seqLen;
        }

        private static void ClipGradNorm(
            IEnumerable<DevOnBike.Overfit.Parameters.Parameter> parameters, float maxNorm)
        {
            var totalNormSq = 0f;
            var paramList   = parameters.ToList();
            foreach (var p in paramList)
            {
                var g = p.GradSpan;
                for (var i = 0; i < g.Length; i++) totalNormSq += g[i] * g[i];
            }
            var norm = MathF.Sqrt(totalNormSq);
            if (norm <= maxNorm) return;
            var scale = maxNorm / (norm + 1e-6f);
            foreach (var p in paramList)
            {
                var g = p.GradSpan;
                for (var i = 0; i < g.Length; i++) g[i] *= scale;
            }
        }

        private static float EvaluateLoss(
            GPT1Model model, int[] valCorpus, GPT1Config config,
            int seqLen, Random rng, int valSteps)
        {
            model.Eval();
            var total = 0f;
            for (var s = 0; s < valSteps; s++)
            {
                var (inputIds, targetIds) = SampleSequence(valCorpus, seqLen, rng);
                using var graph  = new ComputationGraph();
                using var logits = model.Forward(graph, inputIds, batchSize: 1, seqLen);
                total += ComputeLossAndSeedGradParallel(logits, targetIds, seqLen, config.VocabSize);
            }
            model.Train();
            return total / valSteps;
        }

        private static string GenerateSample(
            GPT1Model model, CharacterTokenizer tokenizer, string prompt, int maxTokens)
        {
            model.Eval();
            var ids       = tokenizer.Encode(prompt);
            var generated = model.Generate(ids, maxTokens);
            model.Train();
            return prompt + tokenizer.Decode(generated);
        }

        private static void SkipIfMissing(string path)
        {
            if (!File.Exists(path))
                throw new Exception(
                    $"Fixture '{path}' not found. curl -o Tests/test_fixtures/tiny_shakespeare.txt " +
                    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt");
        }
    }
}
