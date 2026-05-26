// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Optimizers.Abstractions;
using DevOnBike.Overfit.Parameters;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Training;

namespace DevOnBike.Overfit.Tests.Training
{
    /// <summary>
    /// Fast correctness tests for <see cref="DataParallelTrainer"/>'s gradient all-reduce, broadcast,
    /// and grad-norm clip — isolated from any model by driving worker gradients directly and using a
    /// trivial SGD-style optimizer (data -= lr·grad), so only the trainer's math is under test.
    /// </summary>
    public sealed class DataParallelTrainerTests
    {
        // data -= lr * grad, applied to the parameters it was constructed over.
        private sealed class StepByGradOptimizer : IOptimizer
        {
            private readonly Parameter[] _parameters;
            public StepByGradOptimizer(Parameter[] parameters) => _parameters = parameters;
            public float LearningRate { get; set; } = 1f;
            public void ZeroGrad()
            {
                foreach (var p in _parameters) { p.GradSpan.Clear(); }
            }
            public void Step()
            {
                foreach (var p in _parameters)
                {
                    var d = p.DataSpan;
                    var g = p.GradSpan;
                    for (var i = 0; i < d.Length; i++) { d[i] -= LearningRate * g[i]; }
                }
            }
        }

        private static Parameter Vec(int n, float fill)
        {
            var p = new Parameter(new TensorShape(n));
            p.DataSpan.Fill(fill);
            return p;
        }

        [Fact]
        public void Step_AveragesWorkerGradients_StepsMaster_AndBroadcasts()
        {
            using var m0 = Vec(3, 1f);
            using var w0p0 = Vec(3, 0f);
            using var w1p0 = Vec(3, 0f);

            var master = new[] { m0 };
            var workers = new IReadOnlyList<Parameter>[] { new[] { w0p0 }, new[] { w1p0 } };
            var trainer = new DataParallelTrainer(master, workers);
            trainer.BroadcastParameters();   // workers ← master = [1,1,1]

            var optimizer = new StepByGradOptimizer(master) { LearningRate = 1f };

            // Worker 0 grads = [2,4,6], worker 1 grads = [4,8,12]. Average = [3,6,9].
            var loss = trainer.Step(optimizer, w =>
            {
                var g = workers[w][0].GradSpan;
                if (w == 0) { g[0] = 2; g[1] = 4; g[2] = 6; }
                else { g[0] = 4; g[1] = 8; g[2] = 12; }
                return w == 0 ? 1.0f : 3.0f;
            });

            // Master grad = average; master data = 1 - 1·avg.
            Assert.Equal(3f, m0.GradSpan[0]); Assert.Equal(6f, m0.GradSpan[1]); Assert.Equal(9f, m0.GradSpan[2]);
            Assert.Equal(-2f, m0.DataSpan[0]); Assert.Equal(-5f, m0.DataSpan[1]); Assert.Equal(-8f, m0.DataSpan[2]);

            // Workers were re-broadcast to the updated master.
            for (var w = 0; w < workers.Length; w++)
            {
                Assert.Equal(-2f, workers[w][0].DataSpan[0]);
                Assert.Equal(-5f, workers[w][0].DataSpan[1]);
                Assert.Equal(-8f, workers[w][0].DataSpan[2]);
            }

            Assert.Equal(2.0f, loss); // mean of {1,3}
        }

        [Fact]
        public void Step_ClipsMasterGradientNorm_WhenAboveMax()
        {
            using var m0 = Vec(2, 0f);
            using var w0 = Vec(2, 0f);

            var master = new[] { m0 };
            var workers = new IReadOnlyList<Parameter>[] { new[] { w0 } };
            var trainer = new DataParallelTrainer(master, workers);

            var optimizer = new StepByGradOptimizer(master) { LearningRate = 0f }; // isolate clipping

            // Single worker grad = [3,4] (norm 5). Average over 1 worker = [3,4]. Clip to maxNorm 1 → /5.
            trainer.Step(optimizer, w =>
            {
                var g = workers[w][0].GradSpan;
                g[0] = 3; g[1] = 4;
                return 0f;
            }, maxGradNorm: 1f);

            // lr=0 so Step() leaves data unchanged; assert the master grad was scaled to unit norm.
            var gn = MathF.Sqrt(m0.GradSpan[0] * m0.GradSpan[0] + m0.GradSpan[1] * m0.GradSpan[1]);
            Assert.True(MathF.Abs(gn - 1f) < 1e-4f, $"expected clipped grad-norm ≈ 1, got {gn}");
        }

        [Fact]
        public void Constructor_Rejects_MismatchedParameterShapes()
        {
            using var m0 = Vec(3, 0f);
            using var w0 = Vec(4, 0f); // wrong length

            var master = new[] { m0 };
            var workers = new IReadOnlyList<Parameter>[] { new[] { w0 } };
            Assert.Throws<ArgumentException>(() => new DataParallelTrainer(master, workers));
        }
    }
}
