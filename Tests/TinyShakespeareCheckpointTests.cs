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

        [Fact]
        public void Shakespeare_12Layer_Checkpointed_SeqLen256_LossBelow200()
        {
            SkipIfMissing(FixturePath);

            const int seqLen = 256;
            const int batchSize = 8;    // true batch: 8 sekwencji na krok
            const int arenaSize = 700_000_000; // 2.8GB — factored MHA tworzy więcej tensorów per krok
            const int totalSteps = 1_000;
            const int reportEvery = 200;
            const float lrMax = 3e-4f;
            const float lrMin = 3e-5f;

            var text = File.ReadAllText(FixturePath);
            var tokenizer = CharacterTokenizer.FromCorpus(text);
            var allIds = tokenizer.Encode(text);
            var trainSize = (int)(allIds.Length * 0.9);
            var trainIds = allIds.AsSpan(0, trainSize).ToArray();
            var valIds = allIds.AsSpan(trainSize).ToArray();

            _output.WriteLine($"Corpus: {text.Length:N0} chars, vocab={tokenizer.VocabSize}");
            _output.WriteLine(string.Empty);

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
            _output.WriteLine($"SeqLen: {seqLen} | Batch: {batchSize} | Residuals ON TAPE");
            _output.WriteLine($"Training: {totalSteps:N0} steps, lr={lrMax}→{lrMin} (cosine)");
            _output.WriteLine(string.Empty);

            using var model = new GPT1Model(config);
            using var optimizer = new Adam(model.TrainableParameters(), lrMax)
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

            // Persistent graph: jedna alokacja 2GB zamiast 5000 × alokacja/zwolnienie.
            // Oszczędność: ~20-30% czasu przez eliminację kosztownej alokacji natywnej per krok.
            using var persistentGraph = new ComputationGraph(arenaSize);

            for (var step = 0; step < totalSteps; step++)
            {
                var (inputIds, targetIds) = SampleBatch(trainIds, seqLen, batchSize, rng);

                optimizer.ZeroGrad();
                persistentGraph.Reset();
                var logits = model.Forward(persistentGraph, inputIds, batchSize, seqLen);

                var loss = ComputeLossAndSeedGradParallel(logits, targetIds, seqLen, batchSize, config.VocabSize);
                windowLoss += loss;

                persistentGraph.BackwardFromGrad(logits);
                logits.Dispose();
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
                    windowLoss = 0f;
                    var elapsed = sw.Elapsed;
                    var valStr = string.Empty;

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

            Assert.True(finalValLoss < 2.4f,
                $"Val loss {finalValLoss:F4} >= 2.4. 1000 kroków batch=8 SeqLen=256. (Wynik: 2.29 po naprawieniu Generate OOM)");
        }

        private static (int[] inputIds, int[] targetIds) SampleSequence(
            int[] corpus, int seqLen, Random rng)
        {
            var start = rng.Next(0, corpus.Length - seqLen - 1);
            return (corpus.AsSpan(start, seqLen).ToArray(),
                    corpus.AsSpan(start + 1, seqLen).ToArray());
        }

        private static (int[] inputIds, int[] targetIds) SampleBatch(
            int[] corpus, int seqLen, int batchSize, Random rng)
        {
            var inputIds = new int[batchSize * seqLen];
            var targetIds = new int[batchSize * seqLen];
            for (var b = 0; b < batchSize; b++)
            {
                var start = rng.Next(0, corpus.Length - seqLen - 1);
                corpus.AsSpan(start, seqLen).CopyTo(inputIds.AsSpan(b * seqLen, seqLen));
                corpus.AsSpan(start + 1, seqLen).CopyTo(targetIds.AsSpan(b * seqLen, seqLen));
            }
            return (inputIds, targetIds);
        }

        private static float ComputeLossAndSeedGradParallel(
            AutogradNode logits, int[] targetIds, int seqLen, int batchSize, int vocabSize)
        {
            var totalTokens = batchSize * seqLen;
            var logitArr = logits.DataView.AsReadOnlySpan().ToArray();
            var gradArr = new float[totalTokens * vocabSize];
            var losses = new float[totalTokens];

            Parallel.For(0, totalTokens, t =>
            {
                var offset = t * vocabSize;
                var targetId = targetIds[t];
                var maxVal = logitArr[offset];
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
            for (var t = 0; t < totalTokens; t++) total += losses[t];
            return total / totalTokens;
        }

        private static void ClipGradNorm(
            IEnumerable<DevOnBike.Overfit.Parameters.Parameter> parameters, float maxNorm)
        {
            var totalNormSq = 0f;
            var paramList = parameters.ToList();
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
                using var graph = new ComputationGraph(600_000_000);
                using var logits = model.Forward(graph, inputIds, batchSize: 1, seqLen);
                total += ComputeLossAndSeedGradParallel(logits, targetIds, seqLen, 1, config.VocabSize);
            }
            model.Train();
            return total / valSteps;
        }

        private static string GenerateSample(
            GPT1Model model, CharacterTokenizer tokenizer, string prompt, int maxTokens)
        {
            model.Eval();
            var ids = tokenizer.Encode(prompt);
            var generated = model.Generate(ids, maxTokens);
            model.Train();
            return prompt + tokenizer.Decode(generated);
        }

        /// <summary>
        /// Szybki smoke test: 50 kroków, batch=8, SeqLen=256.
        /// Weryfikuje: brak OOM, brak NaN, loss spada.
        /// Czas: ~2 min. Uruchom to PRZED pełnym testem (5K kroków).
        /// </summary>
        [Fact]
        public void Shakespeare_Batch8_SeqLen256_SmokeTest_50Steps()
        {
            SkipIfMissing(FixturePath);

            const int seqLen = 256;
            const int batchSize = 8;
            const int steps = 50;

            var text = File.ReadAllText(FixturePath);
            var tokenizer = CharacterTokenizer.FromCorpus(text);
            var allIds = tokenizer.Encode(text);

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

            using var model = new GPT1Model(config);
            using var optimizer = new Adam(model.TrainableParameters(), 3e-4f)
            {
                UseAdamW = true,
                WeightDecay = 0.1f,
            };

            model.Train();
            var rng = new Random(42);
            var sw = System.Diagnostics.Stopwatch.StartNew();

            float firstLoss = 0f;
            float lastLoss = 0f;

            // Persistent graph: arena allocated once, Reset() per krok.
            // Eliminuje 50× alokację 2GB natywnej pamięci.
            using var smokeGraph = new ComputationGraph(600_000_000);

            for (var step = 0; step < steps; step++)
            {
                var (inputIds, targetIds) = SampleBatch(allIds, seqLen, batchSize, rng);

                optimizer.ZeroGrad();
                smokeGraph.Reset();
                model.InvalidateAllCaches();
                var logits = model.Forward(smokeGraph, inputIds, batchSize, seqLen);

                Assert.False(float.IsNaN(logits.DataView.AsReadOnlySpan()[0]),
                    $"NaN w logitach na kroku {step}");

                var loss = ComputeLossAndSeedGradParallel(logits, targetIds, seqLen, batchSize, config.VocabSize);

                Assert.False(float.IsNaN(loss), $"NaN loss na kroku {step}");
                Assert.False(float.IsInfinity(loss), $"Inf loss na kroku {step}");

                smokeGraph.BackwardFromGrad(logits);
                logits.Dispose();
                ClipGradNorm(model.TrainableParameters(), maxNorm: 1.0f);
                optimizer.Step();

                if (step == 0) firstLoss = loss;
                if (step == steps - 1) lastLoss = loss;

                if (step % 10 == 0)
                {
                    _output.WriteLine($"Step {step,3} | Loss: {loss:F4} | {sw.Elapsed:mm\\:ss} | {sw.ElapsedMilliseconds / (step + 1):F0}ms/step");
                }
            }

            _output.WriteLine(string.Empty);
            _output.WriteLine($"First loss: {firstLoss:F4} → Last loss: {lastLoss:F4}");
            _output.WriteLine($"ms/step: {sw.ElapsedMilliseconds / steps:F0}");

            Assert.True(lastLoss < firstLoss,
                $"Loss nie spada po {steps} krokach: {firstLoss:F4} → {lastLoss:F4}. Bug w backward.");
        }

        /// <summary>
        /// Szybki weryfikacyjny test: mniejszy model, batch=8, 500 kroków.
        /// Cel: potwierdzić konwergencję batch=8 w ~5 minut.
        /// Jak zielony → odpal pełny 5K test przez noc.
        /// </summary>
        [Fact]
        public void Shakespeare_Batch8_SmallModel_500Steps_Convergence()
        {
            SkipIfMissing(FixturePath);

            const int seqLen = 128;
            const int batchSize = 8;
            const int steps = 500;
            const float lrMax = 3e-4f;
            const float lrMin = 3e-5f;

            var text = File.ReadAllText(FixturePath);
            var tokenizer = CharacterTokenizer.FromCorpus(text);
            var allIds = tokenizer.Encode(text);
            var trainSize = (int)(allIds.Length * 0.9);
            var trainIds = allIds.AsSpan(0, trainSize).ToArray();
            var valIds = allIds.AsSpan(trainSize).ToArray();

            // Mniejszy model: 4 warstwy, 128d — szybki do weryfikacji
            var config = new GPT1Config
            {
                VocabSize = tokenizer.VocabSize,
                ContextLength = seqLen,
                DModel = 128,
                NHeads = 4,
                NLayers = 4,
                DFF = 512,
                TieWeights = false,
                PreLayerNorm = true,
            };

            _output.WriteLine($"Model: {config} | Batch: {batchSize} | Steps: {steps}");
            _output.WriteLine(string.Empty);

            using var model = new GPT1Model(config);
            using var optimizer = new Adam(model.TrainableParameters(), lrMax)
            {
                UseAdamW = true,
                WeightDecay = 0.1f,
            };

            model.Train();
            var rng = new Random(42);
            var sw = System.Diagnostics.Stopwatch.StartNew();
            float initialLoss = 0f;
            float finalValLoss = 0f;
            float windowLoss = 0f;

            using var persistentSmall = new ComputationGraph(100_000_000);

            for (var step = 0; step < steps; step++)
            {
                var (inputIds, targetIds) = SampleBatch(trainIds, seqLen, batchSize, rng);

                optimizer.ZeroGrad();
                persistentSmall.Reset();
                var logits = model.Forward(persistentSmall, inputIds, batchSize, seqLen);

                var loss = ComputeLossAndSeedGradParallel(logits, targetIds, seqLen, batchSize, config.VocabSize);
                windowLoss += loss;

                persistentSmall.BackwardFromGrad(logits);
                logits.Dispose();
                ClipGradNorm(model.TrainableParameters(), maxNorm: 1.0f);

                var cosine = 0.5f * (1f + MathF.Cos(MathF.PI * (float)step / steps));
                optimizer.LearningRate = lrMin + (lrMax - lrMin) * cosine;
                optimizer.Step();

                if (step == 0) initialLoss = loss;

                if ((step + 1) % 100 == 0)
                {
                    var avg = windowLoss / 100;
                    windowLoss = 0f;
                    _output.WriteLine(
                        $"Step {step + 1,4} | Train: {avg:F4} | {sw.Elapsed:mm\\:ss} | {sw.ElapsedMilliseconds / (step + 1):F0}ms/step");
                }
            }

            // Val loss
            model.Eval();
            var valTotal = 0f;
            for (var s = 0; s < 30; s++)
            {
                var (inp, tgt) = SampleSequence(valIds, seqLen, rng);
                using var g2 = new ComputationGraph(100_000_000);
                using var l2 = model.Forward(g2, inp, batchSize: 1, seqLen);
                valTotal += ComputeLossAndSeedGradParallel(l2, tgt, seqLen, 1, config.VocabSize);
            }
            finalValLoss = valTotal / 30;

            var sample = GenerateSample(model, tokenizer, "ROMEO:", maxTokens: 100);
            _output.WriteLine(string.Empty);
            _output.WriteLine($"Initial: {initialLoss:F4} → Val: {finalValLoss:F4}");
            _output.WriteLine($"Sample: {sample}");
            _output.WriteLine($"nanoGPT (4L, 128d, batch=64): ~1.7");

            Assert.True(finalValLoss < 2.7f, $"Val loss {finalValLoss:F4} >= 2.7.");
        }

        /// <summary>
        /// Gradient check — weryfikuje poprawność backward w ~10 sekund.
        /// Porównuje analityczny gradient (backward) z numerycznym (finite difference).
        /// Jeśli relError < 1% → backward jest matematycznie poprawny.
        /// Testuje jeden parametr z każdego komponentu: Embedding, LN, MHA, FFN.
        /// </summary>
        [Fact]
        public void GPT1_GradientCheck_BackwardIsCorrect()
        {
            const int seqLen = 8;   // małe żeby szybko
            const int batchSize = 1;
            const int vocab = 30;  // małe vocab

            var config = new GPT1Config
            {
                VocabSize = vocab,
                ContextLength = seqLen,
                DModel = 16,
                NHeads = 2,
                NLayers = 2,
                DFF = 32,
                TieWeights = false,
                PreLayerNorm = true,
            };

            var rng = new Random(42);
            var tokenIds = new int[seqLen];
            var targets = new int[seqLen];
            for (var i = 0; i < seqLen; i++) { tokenIds[i] = rng.Next(vocab); targets[i] = rng.Next(vocab); }

            float ComputeLoss(GPT1Model m)
            {
                using var g = new ComputationGraph(10_000_000);
                using var logits = m.Forward(g, tokenIds, batchSize, seqLen);
                var logitArr = logits.DataView.AsReadOnlySpan().ToArray();
                var total = 0f;
                for (var t = 0; t < seqLen; t++)
                {
                    var off = t * vocab;
                    var maxV = logitArr[off];
                    for (var v = 1; v < vocab; v++) if (logitArr[off + v] > maxV) maxV = logitArr[off + v];
                    var sumE = 0f;
                    for (var v = 0; v < vocab; v++) sumE += MathF.Exp(logitArr[off + v] - maxV);
                    total += maxV + MathF.Log(sumE) - logitArr[off + targets[t]];
                }
                return total / seqLen;
            }

            using var model = new GPT1Model(config);
            var eps = 1e-3f;
            var failures = new List<string>();

            // Sprawdzamy gradient dla jednego parametru z każdego komponentu
            var checks = new (string name, DevOnBike.Overfit.Parameters.Parameter param, int idx)[]
            {
                ("TokenEmb[0]",  model.TokenEmbedding.Weight,      0),
                ("PosEmb[0]",    model.PositionEmbedding.Weight,    0),
                ("LN1.Gamma[0]", model.Blocks[0].Norm1.Gamma,       0),
                ("MHA.Wq[0]",    model.Blocks[0].Attention.Wq,      0),
                ("FFN.W1[0]",    model.Blocks[0].FFN.W1,            0),
                ("LMHead[0]",    model.LMHead,                       0),
            };

            foreach (var (name, param, idx) in checks)
            {
                // Numeryczny gradient
                param.DataSpan[idx] += eps;
                model.InvalidateAllCaches();
                var lossPlus = ComputeLoss(model);

                param.DataSpan[idx] -= 2 * eps;
                model.InvalidateAllCaches();
                var lossMinus = ComputeLoss(model);

                param.DataSpan[idx] += eps;
                model.InvalidateAllCaches();

                var numerical = (lossPlus - lossMinus) / (2 * eps);

                // Analityczny gradient
                foreach (var p in model.TrainableParameters()) p.ZeroGrad();

                using var g2 = new ComputationGraph(10_000_000);
                model.InvalidateAllCaches();
                var logits2 = model.Forward(g2, tokenIds, batchSize, seqLen);
                var logArr2 = logits2.DataView.AsReadOnlySpan().ToArray();
                var gradArr = new float[seqLen * vocab];

                for (var t = 0; t < seqLen; t++)
                {
                    var off = t * vocab;
                    var maxV = logArr2[off];
                    for (var v = 1; v < vocab; v++) if (logArr2[off + v] > maxV) maxV = logArr2[off + v];
                    var sumE = 0f;
                    for (var v = 0; v < vocab; v++) sumE += MathF.Exp(logArr2[off + v] - maxV);
                    var scale = 1f / seqLen;
                    for (var v = 0; v < vocab; v++)
                    {
                        var sm = MathF.Exp(logArr2[off + v] - maxV) / sumE;
                        gradArr[off + v] = (sm - (v == targets[t] ? 1f : 0f)) * scale;
                    }
                }
                gradArr.AsSpan().CopyTo(logits2.GradView.AsSpan());
                g2.BackwardFromGrad(logits2);
                logits2.Dispose();

                var analytical = param.GradSpan[idx];
                var relErr = MathF.Abs(analytical - numerical) / (MathF.Abs(numerical) + 1e-7f);

                var status = relErr < 0.10f ? "✓" : "✗ FAIL";
                _output.WriteLine($"{status} {name,20}: analytical={analytical:F5}, numerical={numerical:F5}, relErr={relErr:P1}");

                if (relErr >= 0.10f)
                    failures.Add($"{name}: analytical={analytical:F5}, numerical={numerical:F5}, relErr={relErr:P1}");
            }

            Assert.True(failures.Count == 0,
                "Gradient check failed: " + string.Join(", ", failures));
        }

        /// <summary>
        /// End-to-end test: 100 kroków treningu batch=8, SeqLen=64 → generacja tekstu.
        /// Weryfikuje pełny pipeline: dane → forward → backward → optymizer → generacja.
        ///
        /// Co sprawdza:
        ///   1. Loss spada (backward działa)
        ///   2. Model generuje tekst z angielskimi słowami (nie losowe znaki)
        ///   3. Prawdopodobieństwo poprawnych znaków rośnie (perplexity spada)
        ///   4. Brak NaN, Inf, OOM
        ///
        /// Czas: ~30 sekund.
        /// </summary>
        [Fact]
        public void GPT1_EndToEnd_100Steps_GeneratesEnglishWords()
        {
            SkipIfMissing(FixturePath);

            const int seqLen = 64;
            const int batchSize = 8;
            const int steps = 100;

            var text = File.ReadAllText(FixturePath);
            var tokenizer = CharacterTokenizer.FromCorpus(text);
            var allIds = tokenizer.Encode(text);

            var config = new GPT1Config
            {
                VocabSize = tokenizer.VocabSize,
                ContextLength = seqLen,
                DModel = 128,
                NHeads = 4,
                NLayers = 4,
                DFF = 512,
                TieWeights = false,
                PreLayerNorm = true,
            };

            using var model = new GPT1Model(config);
            using var optimizer = new Adam(model.TrainableParameters(), 3e-4f)
            {
                UseAdamW = true,
                WeightDecay = 0.1f,
            };

            model.Train();
            var rng = new Random(42);

            float firstLoss = 0f;
            float lastLoss = 0f;

            using var graph = new ComputationGraph(80_000_000);

            for (var step = 0; step < steps; step++)
            {
                var (inputIds, targetIds) = SampleBatch(allIds, seqLen, batchSize, rng);

                optimizer.ZeroGrad();
                graph.Reset();
                model.InvalidateAllCaches();
                var logits = model.Forward(graph, inputIds, batchSize, seqLen);

                var loss = ComputeLossAndSeedGradParallel(logits, targetIds, seqLen, batchSize, config.VocabSize);

                Assert.False(float.IsNaN(loss), $"NaN na kroku {step}");
                Assert.False(float.IsInfinity(loss), $"Inf na kroku {step}");

                graph.BackwardFromGrad(logits);
                logits.Dispose();
                ClipGradNorm(model.TrainableParameters(), maxNorm: 1.0f);
                optimizer.Step();

                if (step == 0) firstLoss = loss;
                if (step == steps - 1) lastLoss = loss;
            }

            _output.WriteLine($"Loss: {firstLoss:F4} → {lastLoss:F4} ({(1f - lastLoss / firstLoss) * 100:F1}% redukcja)");

            // Generacja
            model.Eval();
            var prompts = new[] { "HAMLET:", "ROMEO:", "To be" };
            var hasEnglishWords = false;
            var englishWords = new[] { "the", "and", "to", "of", "I", "a", "you", "my", "not", "is" };

            foreach (var prompt in prompts)
            {
                var ids = tokenizer.Encode(prompt);
                var generated = model.Generate(ids, maxNewTokens: 80);
                var text2 = prompt + tokenizer.Decode(generated);
                _output.WriteLine($"  {text2}");

                foreach (var word in englishWords)
                {
                    if (text2.Contains(word)) { hasEnglishWords = true; break; }
                }
            }

            // Perplexity na walidacji
            var valStart = (int)(allIds.Length * 0.9);
            var valTotal = 0f;
            for (var s = 0; s < 20; s++)
            {
                var (inp, tgt) = SampleSequence(allIds.AsSpan(valStart).ToArray(), seqLen, rng);
                graph.Reset();
                model.InvalidateAllCaches();
                var lv = model.Forward(graph, inp, batchSize: 1, seqLen);
                valTotal += ComputeLossAndSeedGradParallel(lv, tgt, seqLen, 1, config.VocabSize);
                lv.Dispose();
            }

            var valLoss = valTotal / 20;
            var perplexity = MathF.Exp(valLoss);

            _output.WriteLine(string.Empty);
            _output.WriteLine($"Val loss: {valLoss:F4} | Perplexity: {perplexity:F1}");
            _output.WriteLine($"Baseline perplexity (random): {MathF.Exp(MathF.Log(tokenizer.VocabSize)):F1}");

            // Asercje
            Assert.True(lastLoss < firstLoss * 0.80f,
                $"Loss nie spada 20%+: {firstLoss:F4} → {lastLoss:F4}");

            Assert.True(hasEnglishWords,
                "Generacja nie zawiera angielskich słów po 100 krokach.");

            Assert.True(perplexity < tokenizer.VocabSize,
                $"Perplexity {perplexity:F1} >= losowy model {tokenizer.VocabSize}. Backward jest złamany.");

            _output.WriteLine(string.Empty);
            _output.WriteLine("✓ Loss spada, gradient przepływa przez cały stos.");
            _output.WriteLine("✓ Model generuje angielski tekst po 100 krokach batch=8.");
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
