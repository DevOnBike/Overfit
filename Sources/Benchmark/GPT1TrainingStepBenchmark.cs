// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Parameters;

namespace Benchmarks
{
    /// <summary>
    /// Diagnostic GPT-1 training-step benchmark.
    ///
    /// Purpose:
    /// identify where training time is going before adding parallel kernels.
    ///
    /// This benchmark intentionally uses synthetic token IDs, not TinyShakespeare
    /// file IO, so the measured path is model/training compute rather than data
    /// loading.
    ///
    /// Shape matches the current xUnit TinyShakespeare demo:
    ///
    /// vocab = 68
    /// context = 128
    /// dModel = 128
    /// heads = 4
    /// layers = 4
    /// dFF = 512
    ///
    /// Benchmark methods:
    ///
    /// - SampleBatchOnly
    /// - ForwardOnly
    /// - LossSeedOnly_Standalone
    /// - ForwardLossBackwardNoStep
    /// - OptimizerStepOnly
    /// - FullTrainStep
    ///
    /// Interpretation:
    ///
    /// FullTrainStep - ForwardLossBackwardNoStep approximates optimizer cost.
    /// ForwardLossBackwardNoStep - ForwardOnly approximates loss+backward cost.
    /// LossSeedOnly_Standalone shows whether softmax/CE seed is relevant.
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    public class GPT1TrainingStepBenchmark : IDisposable
    {
        private const int VocabSize = 68;
        private const int ContextLength = 128;
        private const int DModel = 128;
        private const int HeadCount = 4;
        private const int LayerCount = 4;
        private const int DFF = 512;

        private const int CorpusLength = 1_115_394;
        private const int ArenaSize = 180_000_000;

        private const float LearningRate = 3e-4f;
        private const float WeightDecay = 0.1f;
        private const float MaxGradNorm = 1.0f;

        private GPT1Model _model = null!;
        private Adam _optimizer = null!;
        private ComputationGraph _graph = null!;
        private int[] _corpus = null!;
        private int[] _inputIds = null!;
        private int[] _targetIds = null!;
        private Random _rng = null!;
        private List<Parameter> _parameters = null!;

        private float[] _standaloneLogits = null!;
        private float[] _standaloneGrad = null!;
        private int[] _standaloneTargets = null!;

        private float _loss;
        private int _checksum;
        private bool _disposed;

        [Params(8, 16, 32)]
        public int BatchSize
        {
            get; set;
        }

        [Params(128)]
        public int SeqLen
        {
            get; set;
        }

        [GlobalSetup]
        public void Setup()
        {
            var config = new GPT1Config
            {
                VocabSize = VocabSize,
                ContextLength = ContextLength,
                DModel = DModel,
                NHeads = HeadCount,
                NLayers = LayerCount,
                DFF = DFF,
                TieWeights = false,
                PreLayerNorm = true
            };

            _model = new GPT1Model(config);
            _model.Train();

            _parameters = _model
                .TrainableParameters()
                .ToList();

            _optimizer = new Adam(_parameters, LearningRate)
            {
                UseAdamW = true,
                WeightDecay = WeightDecay
            };

            _graph = new ComputationGraph(ArenaSize);

            _rng = new Random(42);
            _corpus = CreateSyntheticCorpus(CorpusLength, VocabSize);
            _inputIds = new int[BatchSize * SeqLen];
            _targetIds = new int[BatchSize * SeqLen];

            _standaloneLogits = new float[BatchSize * SeqLen * VocabSize];
            _standaloneGrad = new float[BatchSize * SeqLen * VocabSize];
            _standaloneTargets = new int[BatchSize * SeqLen];

            FillStandaloneLossInputs(
                _standaloneLogits,
                _standaloneTargets,
                VocabSize);
        }

        [Benchmark]
        public int SampleBatchOnly()
        {
            SampleBatch(
                _corpus,
                SeqLen,
                BatchSize,
                _rng,
                _inputIds,
                _targetIds);

            _checksum ^= _inputIds[0];
            _checksum ^= _targetIds[^1];

            return _checksum;
        }

        [Benchmark]
        public float ForwardOnly()
        {
            SampleBatch(
                _corpus,
                SeqLen,
                BatchSize,
                _rng,
                _inputIds,
                _targetIds);

            _graph.Reset();
            _model.InvalidateAllCaches();

            var logits = _model.Forward(
                _graph,
                _inputIds,
                BatchSize,
                SeqLen);

            var value = logits.DataView[0];

            logits.Dispose();

            _loss = value;
            return value;
        }

        [Benchmark]
        public float LossSeedOnly_Standalone()
        {
            var loss = ComputeLossAndSeedGradStandalone(
                _standaloneLogits,
                _standaloneTargets,
                _standaloneGrad,
                SeqLen,
                BatchSize,
                VocabSize);

            _loss = loss;
            return loss;
        }

        [Benchmark]
        public float ForwardLossBackwardNoStep()
        {
            SampleBatch(
                _corpus,
                SeqLen,
                BatchSize,
                _rng,
                _inputIds,
                _targetIds);

            _optimizer.ZeroGrad();
            _graph.Reset();
            _model.InvalidateAllCaches();

            var logits = _model.Forward(
                _graph,
                _inputIds,
                BatchSize,
                SeqLen);

            var loss = ComputeLossAndSeedGradParallel(
                logits,
                _targetIds,
                SeqLen,
                BatchSize,
                VocabSize);

            _graph.BackwardFromGrad(logits);
            logits.Dispose();

            ClipGradNorm(
                _parameters,
                MaxGradNorm);

            _loss = loss;
            return loss;
        }

        [Benchmark]
        public float OptimizerStepOnly()
        {
            SeedSyntheticGradients(_parameters);

            ClipGradNorm(
                _parameters,
                MaxGradNorm);

            _optimizer.Step();

            var checksum = _parameters[0].DataSpan[0];

            _loss = checksum;
            return checksum;
        }

        [Benchmark]
        public float FullTrainStep()
        {
            SampleBatch(
                _corpus,
                SeqLen,
                BatchSize,
                _rng,
                _inputIds,
                _targetIds);

            _optimizer.ZeroGrad();
            _graph.Reset();
            _model.InvalidateAllCaches();

            var logits = _model.Forward(
                _graph,
                _inputIds,
                BatchSize,
                SeqLen);

            var loss = ComputeLossAndSeedGradParallel(
                logits,
                _targetIds,
                SeqLen,
                BatchSize,
                VocabSize);

            _graph.BackwardFromGrad(logits);
            logits.Dispose();

            ClipGradNorm(
                _parameters,
                MaxGradNorm);

            _optimizer.Step();

            _loss = loss;
            return loss;
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;

            _graph.Dispose();
            _optimizer.Dispose();
            _model.Dispose();

            GC.KeepAlive(_loss);
            GC.KeepAlive(_checksum);
        }

        public void Dispose()
        {
            Cleanup();
        }

        private static int[] CreateSyntheticCorpus(
            int length,
            int vocabSize)
        {
            var result = new int[length];
            var rng = new Random(123);

            for (var i = 0; i < result.Length; i++)
            {
                result[i] = rng.Next(0, vocabSize);
            }

            return result;
        }

        private static void SampleBatch(
            int[] corpus,
            int seqLen,
            int batchSize,
            Random rng,
            int[] inputIds,
            int[] targetIds)
        {
            for (var b = 0; b < batchSize; b++)
            {
                var start = rng.Next(0, corpus.Length - seqLen - 1);

                corpus
                    .AsSpan(start, seqLen)
                    .CopyTo(inputIds.AsSpan(b * seqLen, seqLen));

                corpus
                    .AsSpan(start + 1, seqLen)
                    .CopyTo(targetIds.AsSpan(b * seqLen, seqLen));
            }
        }

        private static float ComputeLossAndSeedGradParallel(
            AutogradNode logits,
            int[] targetIds,
            int seqLen,
            int batchSize,
            int vocabSize)
        {
            var totalTokens = batchSize * seqLen;
            var logitArr = logits.DataView.AsReadOnlySpan().ToArray();
            var gradArr = new float[totalTokens * vocabSize];
            var losses = new float[totalTokens];

            Parallel.For(0, totalTokens, tokenIndex =>
            {
                var offset = tokenIndex * vocabSize;
                var targetId = targetIds[tokenIndex];

                var maxVal = logitArr[offset];

                for (var v = 1; v < vocabSize; v++)
                {
                    if (logitArr[offset + v] > maxVal)
                    {
                        maxVal = logitArr[offset + v];
                    }
                }

                var sumExp = 0f;

                for (var v = 0; v < vocabSize; v++)
                {
                    sumExp += MathF.Exp(logitArr[offset + v] - maxVal);
                }

                losses[tokenIndex] =
                    maxVal +
                    MathF.Log(sumExp) -
                    logitArr[offset + targetId];

                var scale = 1f / seqLen;

                for (var v = 0; v < vocabSize; v++)
                {
                    var softmax =
                        MathF.Exp(logitArr[offset + v] - maxVal) /
                        sumExp;

                    gradArr[offset + v] =
                        (softmax - (v == targetId ? 1f : 0f)) *
                        scale;
                }
            });

            gradArr
                .AsSpan()
                .CopyTo(logits.GradView.AsSpan());

            var total = 0f;

            for (var t = 0; t < totalTokens; t++)
            {
                total += losses[t];
            }

            return total / totalTokens;
        }

        private static float ComputeLossAndSeedGradStandalone(
            float[] logits,
            int[] targetIds,
            float[] grad,
            int seqLen,
            int batchSize,
            int vocabSize)
        {
            var totalTokens = batchSize * seqLen;
            var losses = new float[totalTokens];

            Parallel.For(0, totalTokens, tokenIndex =>
            {
                var offset = tokenIndex * vocabSize;
                var targetId = targetIds[tokenIndex];

                var maxVal = logits[offset];

                for (var v = 1; v < vocabSize; v++)
                {
                    if (logits[offset + v] > maxVal)
                    {
                        maxVal = logits[offset + v];
                    }
                }

                var sumExp = 0f;

                for (var v = 0; v < vocabSize; v++)
                {
                    sumExp += MathF.Exp(logits[offset + v] - maxVal);
                }

                losses[tokenIndex] =
                    maxVal +
                    MathF.Log(sumExp) -
                    logits[offset + targetId];

                var scale = 1f / seqLen;

                for (var v = 0; v < vocabSize; v++)
                {
                    var softmax =
                        MathF.Exp(logits[offset + v] - maxVal) /
                        sumExp;

                    grad[offset + v] =
                        (softmax - (v == targetId ? 1f : 0f)) *
                        scale;
                }
            });

            var total = 0f;

            for (var t = 0; t < totalTokens; t++)
            {
                total += losses[t];
            }

            return total / totalTokens;
        }

        private static void FillStandaloneLossInputs(
            float[] logits,
            int[] targets,
            int vocabSize)
        {
            var rng = new Random(456);

            for (var i = 0; i < logits.Length; i++)
            {
                logits[i] = ((float)rng.NextDouble() * 2f - 1f) * 0.1f;
            }

            for (var i = 0; i < targets.Length; i++)
            {
                targets[i] = rng.Next(0, vocabSize);
            }
        }

        private static void SeedSyntheticGradients(
            IReadOnlyList<Parameter> parameters)
        {
            for (var p = 0; p < parameters.Count; p++)
            {
                var grad = parameters[p].GradSpan;

                for (var i = 0; i < grad.Length; i++)
                {
                    grad[i] = ((i + p) % 17 - 8) * 0.0001f;
                }
            }
        }

        private static void ClipGradNorm(
            IReadOnlyList<Parameter> parameters,
            float maxNorm)
        {
            var totalNormSq = 0f;

            foreach (var parameter in parameters)
            {
                var grad = parameter.GradSpan;

                for (var i = 0; i < grad.Length; i++)
                {
                    totalNormSq += grad[i] * grad[i];
                }
            }

            var norm = MathF.Sqrt(totalNormSq);

            if (norm <= maxNorm)
            {
                return;
            }

            var scale = maxNorm / (norm + 1e-6f);

            foreach (var parameter in parameters)
            {
                var grad = parameter.GradSpan;

                for (var i = 0; i < grad.Length; i++)
                {
                    grad[i] *= scale;
                }
            }
        }
    }
}
