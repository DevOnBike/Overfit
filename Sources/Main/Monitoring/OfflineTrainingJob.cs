// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Optimizers;

namespace DevOnBike.Overfit.Monitoring
{
    /// <summary>
    /// Trains an <see cref="AnomalyAutoencoder"/> on historical normal-traffic feature vectors
    /// and calibrates a <see cref="ReconstructionScorer"/> threshold from the resulting MSE distribution.
    ///
    /// Responsibilities:
    ///   1. Validate inputs and configuration.
    ///   2. Run the Adam optimisation loop for the configured number of epochs
    ///      using MSE reconstruction loss (autoencoder target == input).
    ///   3. Switch the model to Eval mode and calibrate the scorer (p99 by default).
    ///   4. Return a <see cref="OfflineTrainingResult"/> with per-epoch loss history.
    ///
    /// The job does NOT own the autoencoder or scorer — callers are responsible for
    /// saving them after a successful run:
    /// <code>
    ///   var result = job.Run(autoencoder, scorer, trainingData);
    ///   autoencoder.Save("model.bin");
    ///   scorer.Save("scorer.bin");
    /// </code>
    ///
    /// Training data is expected to be already normalised (output of RobustScaler
    /// or the FeatureExtractor pipeline). Raw metric values will produce poor results.
    /// </summary>
    public sealed class OfflineTrainingJob
    {
        private readonly OfflineTrainingConfig _config;

        public OfflineTrainingJob(OfflineTrainingConfig? config = null)
        {
            _config = config ?? new OfflineTrainingConfig();
        }

        // -------------------------------------------------------------------------
        // Run
        // -------------------------------------------------------------------------

        /// <summary>
        /// Executes the full training and calibration pipeline.
        /// Blocks until all epochs complete or the token is cancelled.
        /// </summary>
        /// <param name="autoencoder">
        ///   The autoencoder to train. Will be left in Eval mode on success.
        /// </param>
        /// <param name="scorer">
        ///   The scorer to calibrate. Threshold is set from the p99 of training MSE values.
        /// </param>
        /// <param name="trainingData">
        ///   Normalised feature vectors representing normal behaviour.
        ///   Each element must have length == <see cref="AnomalyAutoencoder.InputSize"/>.
        ///   Minimum 1 sample required.
        /// </param>
        /// <param name="progress">
        ///   Optional sink for per-epoch progress reports.
        ///   Called on the calling thread — use a thread-safe implementation for UI scenarios.
        /// </param>
        /// <param name="ct">Optional cancellation token. Checked at the start of each epoch.</param>
        /// <returns>Loss history, final threshold, and wall-clock duration.</returns>
        /// <exception cref="ArgumentNullException">When autoencoder, scorer, or trainingData is null.</exception>
        /// <exception cref="ArgumentException">When trainingData is empty or contains wrong-sized samples.</exception>
        /// <exception cref="OperationCanceledException">When the token is cancelled.</exception>
        public OfflineTrainingResult Run(
            AnomalyAutoencoder autoencoder,
            ReconstructionScorer scorer,
            IReadOnlyList<float[]> trainingData,
            IProgress<TrainingProgress>? progress = null,
            CancellationToken ct = default)
        {
            ArgumentNullException.ThrowIfNull(autoencoder);
            ArgumentNullException.ThrowIfNull(scorer);
            ArgumentNullException.ThrowIfNull(trainingData);

            if (trainingData.Count == 0)
            {
                throw new ArgumentException(
                    "Training data must contain at least one sample.", nameof(trainingData));
            }

            ValidateSampleDimensions(trainingData, autoencoder.InputSize);

            var sw = Stopwatch.StartNew();
            var epochLosses = new float[_config.Epochs];
            var rng = _config.Seed.HasValue
                ? new Random(_config.Seed.Value)
                : new Random();

            // Index array used for shuffling — avoids copying the data itself
            var indices = BuildIndices(trainingData.Count);

            autoencoder.Train();

            using var optimizer = new Adam(autoencoder.Parameters(), _config.LearningRate);

            // Pre-allocated input node [1, InputSize].
            // Shape [1, x] activates the zero-allocation SIMD inference path in LinearLayer,
            // but here we run with graph != null so the full autograd path is used.
            // Reusing the node across samples avoids per-sample heap allocations.
            using var inputNode = new AutogradNode(
                new FastTensor<float>(1, autoencoder.InputSize),
                requiresGrad: false);

            for (var epoch = 0; epoch < _config.Epochs; epoch++)
            {
                ct.ThrowIfCancellationRequested();

                if (_config.ShuffleEachEpoch)
                {
                    Shuffle(indices, rng);
                }

                var totalLoss = 0f;

                foreach (var idx in indices)
                {
                    // Overwrite the reused input tensor with the current sample
                    trainingData[idx].CopyTo(inputNode.Data.AsSpan());

                    // New graph per sample — records the forward pass for Backward.
                    // Must be disposed AFTER Backward so intermediate tensors
                    // (stored by graph.Record) remain valid during backpropagation.
                    var graph = new ComputationGraph();

                    var reconstruction = autoencoder.Forward(graph, inputNode);

                    // Autoencoder loss: MSE(reconstruction, input) — target equals input
                    var loss = TensorMath.MSELoss(graph, reconstruction, inputNode);

                    // Read loss value before Backward modifies gradient buffers
                    totalLoss += loss.Data.AsSpan()[0];

                    // Backward is on ComputationGraph, not on AutogradNode.
                    // Traverses the recorded tape in reverse and accumulates gradients.
                    graph.Backward(loss);

                    loss.Dispose();

                    optimizer.Step();
                    optimizer.ZeroGrad();
                }

                epochLosses[epoch] = totalLoss / trainingData.Count;

                progress?.Report(new TrainingProgress
                {
                    Epoch = epoch + 1,
                    TotalEpochs = _config.Epochs,
                    EpochLoss = epochLosses[epoch]
                });
            }

            // Switch to eval mode — activates running statistics in BatchNorm
            // and enables the SIMD fast-path in LinearLayer for inference
            autoencoder.Eval();

            // Calibrate the scorer threshold from the p99 of training MSE values
            scorer.CalibrateFromModel(autoencoder, trainingData, _config.CalibrationPercentile);

            sw.Stop();

            return new OfflineTrainingResult
            {
                EpochLosses = epochLosses,
                FinalThreshold = scorer.Threshold,
                Duration = sw.Elapsed
            };
        }

        // -------------------------------------------------------------------------
        // Private helpers
        // -------------------------------------------------------------------------

        private static void ValidateSampleDimensions(
            IReadOnlyList<float[]> trainingData,
            int expectedSize)
        {
            for (var i = 0; i < trainingData.Count; i++)
            {
                if (trainingData[i].Length != expectedSize)
                {
                    throw new ArgumentException(
                        $"Sample at index {i} has length {trainingData[i].Length}, " +
                        $"expected {expectedSize} (autoencoder.InputSize).",
                        nameof(trainingData));
                }
            }
        }

        private static int[] BuildIndices(int count)
        {
            var indices = new int[count];
            for (var i = 0; i < count; i++) { indices[i] = i; }
            return indices;
        }

        /// <summary>Fisher-Yates in-place shuffle.</summary>
        private static void Shuffle(int[] indices, Random rng)
        {
            for (var i = indices.Length - 1; i > 0; i--)
            {
                var j = rng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
        }
    }
}