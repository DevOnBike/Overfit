// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// Measures the QLoRA training step time on the real Qwen2.5-3B at realistic chunk lengths, to
    /// estimate how long a real fine-tune (e.g. tens of pages of text) would take. <see cref="LongFact"/>.
    /// </summary>
    public sealed class QwenGgufTrainStepTimeTests
    {
        private readonly ITestOutputHelper _out;
        public QwenGgufTrainStepTimeTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void TrainingStepTime_VsSequenceLength()
        {
            var path = TestModelPaths.Qwen3B.RequireQ4KmGgufPath();
            using var engine = CachedLlamaInferenceEngine.LoadGguf(path);
            var vocab = engine.Config.VocabSize;
            using var model = TrainableLlamaModel.FromEngine(engine, loraRank: 8, rng: new Random(1), maxSeqLen: 1024, loraOnLmHead: true);
            using var graph = new ComputationGraph(320_000_000);
            var trainable = ToList(model.TrainableParameters());
            using var opt = new Adam(trainable, learningRate: 0.002f) { WeightDecay = 0f, Epsilon = 1e-4f };

            var rng = new Random(5);
            foreach (var T in new[] { 128, 256 })
            {
                var input = new int[T];
                var target = new int[T];
                for (var i = 0; i < T; i++) { input[i] = rng.Next(vocab); target[i] = rng.Next(vocab); }

                // 1 warm-up + 3 timed steps.
                double totalMs = 0;
                for (var step = 0; step < 4; step++)
                {
                    var sw = Stopwatch.StartNew();
                    graph.Reset();
                    opt.ZeroGrad();
                    var logits = model.Forward(graph, input, useCheckpoint: true);
                    TrainableLlamaModel.CrossEntropyLossAndSeed(logits, target, vocab);
                    graph.BackwardFromGrad(logits);
                    opt.Step();
                    sw.Stop();
                    if (step > 0) { totalMs += sw.ElapsedMilliseconds; }
                }
                _out.WriteLine($"T={T,4}: {totalMs / 3.0 / 1000.0:F2} s/step (avg of 3)");
            }
        }

        private static System.Collections.Generic.List<AutogradNode> ToList(System.Collections.Generic.IEnumerable<AutogradNode> e)
        {
            var l = new System.Collections.Generic.List<AutogradNode>();
            foreach (var x in e) { l.Add(x); }
            return l;
        }
    }
}
