// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Linq;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Parameters;
using DevOnBike.Overfit.Training;

namespace DevOnBike.Overfit.Tests.Training
{
    /// <summary>
    /// Fast tests for <see cref="DataParallelSession{TModel}"/>: it should build the requested number of
    /// replicas, broadcast the master weights into every replica, and delegate <c>Step</c> to the
    /// underlying trainer — all on a tiny real <see cref="GPT1Model"/> (no full training run).
    /// </summary>
    public sealed class DataParallelSessionTests
    {
        private static GPT1Config TinyConfig() => new()
        {
            VocabSize = 32,
            ContextLength = 8,
            DModel = 16,
            NHeads = 2,
            NLayers = 1,
            DFF = 32,
            TieWeights = false,
            PreLayerNorm = true,
        };

        [Fact]
        public void Builds_Replicas_Broadcasts_And_StepsThroughTrainer()
        {
            var config = TinyConfig();
            using var master = new GPT1Model(config);
            master.Train();

            // Perturb the master so its weights differ from a fresh replica's init —
            // a successful broadcast must overwrite the replica with these values.
            var masterParameters = master.TrainableParameters().ToList();
            masterParameters[0].DataSpan.Fill(0.1234f);

            using var session = new DataParallelSession<GPT1Model>(
                masterParameters,
                workerCount: 3,
                modelFactory: _ =>
                {
                    var m = new GPT1Model(config);
                    m.Train();
                    return m;
                },
                parameterSelector: m => m.TrainableParameters(),
                arenaElementsPerReplica: 1 << 16);

            Assert.Equal(3, session.WorkerCount);
            Assert.Equal(masterParameters.Count, session.Replicas[0].Parameters.Count);

            // Broadcast (run in the ctor) copied the master's perturbed weights into every replica.
            foreach (var replica in session.Replicas)
            {
                Assert.Equal(0.1234f, replica.Parameters[0].DataSpan[0], 5);
                Assert.Equal(0.1234f, replica.Parameters[0].DataSpan[^1], 5);
            }

            // Step delegates to the trainer: workers clear their grads, so the averaged master grad is
            // zero and the optimizer moves nothing regardless of lr; assert the mean loss is the average
            // of the per-worker losses we return, and nothing throws.
            var optimizer = new SGD(masterParameters, 0.01f);
            var meanLoss = session.Step(optimizer, workerIndex =>
            {
                // No real forward — just clear this replica's grads and report a known loss.
                foreach (var p in session.Replicas[workerIndex].Parameters)
                {
                    p.GradSpan.Clear();
                }
                return workerIndex; // losses 0,1,2 → mean 1
            });

            Assert.Equal(1f, meanLoss, 5);
        }
    }
}
