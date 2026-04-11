// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Optimizers
{
    /// <summary>
    ///     Manages the learning rate during training and provides safety mechanisms against numerical instability.
    ///     Implements "Reduce LR on Plateau" with an integrated RAM checkpointing system.
    /// </summary>
    /// <remarks>
    ///     This scheduler monitors the loss and automatically restores the last "safe" weight state
    ///     if NaN or Infinity is detected, preventing total model collapse.
    /// </remarks>
    public sealed class LRScheduler : IDisposable
    {
        private readonly FastTensor<float>[] _checkpoint; // In-RAM weight backup

        private readonly float _factor;
        private readonly Action<string> _log;
        private readonly float _minDelta;
        private readonly float _minLR;
        private readonly IOptimizer _optimizer;
        private readonly AutogradNode[] _parameters;
        private readonly int _patience;
        private int _badEpochs;

        private float _bestLoss = float.MaxValue;

        /// <param name="optimizer">The optimizer whose LearningRate will be managed.</param>
        /// <param name="parameters">The model parameters to monitor and backup.</param>
        /// <param name="log">Logging callback for scheduler events.</param>
        /// <param name="factor">Multiplicative factor for learning rate reduction (0.0 - 1.0).</param>
        /// <param name="patience">Number of epochs to wait before reducing LR after stagnation.</param>
        /// <param name="minLR">Lower bound for the learning rate.</param>
        /// <param name="minDelta">Threshold for measuring new best loss (relative change).</param>
        public LRScheduler(
            IOptimizer optimizer,
            AutogradNode[] parameters,
            Action<string> log,
            float factor = 0.5f,
            int patience = 2,
            float minLR = 1e-6f,
            float minDelta = 0.005f)
        {
            ArgumentNullException.ThrowIfNull(optimizer);
            ArgumentNullException.ThrowIfNull(parameters);
            ArgumentNullException.ThrowIfNull(log);

            if (factor is <= 0f or >= 1f)
            {
                throw new ArgumentOutOfRangeException(nameof(factor), "Reduction factor must be in the range (0, 1).");
            }

            _optimizer = optimizer;
            _parameters = parameters;
            _log = log;
            _factor = factor;
            _patience = patience;
            _minLR = minLR;
            _minDelta = minDelta;

            _checkpoint = new FastTensor<float>[_parameters.Length];
            for (var i = 0; i < _parameters.Length; i++)
            {
                _checkpoint[i] = FastTensor<float>.SameShape(_parameters[i].Data);
                _parameters[i].Data.AsSpan().CopyTo(_checkpoint[i].AsSpan());
            }
        }

        public void Dispose()
        {
            if (_checkpoint != null)
            {
                for (var i = 0; i < _checkpoint.Length; i++)
                {
                    _checkpoint[i]?.Dispose();
                }
            }
        }

        /// <summary>
        ///     Evaluates the current loss and updates the learning rate or restores weights if necessary.
        /// </summary>
        /// <param name="currentLoss">The loss value from the current epoch.</param>
        public void Step(float currentLoss)
        {
            if (float.IsNaN(currentLoss) || float.IsInfinity(currentLoss))
            {
                _log($">>> NUMERICAL CATASTROPHE (Loss={currentLoss}). Restoring weights to the last safe state!");

                RestoreCheckpoint();
                ReduceLR();

                return;
            }

            if (currentLoss < _bestLoss * (1f - _minDelta))
            {
                _bestLoss = currentLoss;
                _badEpochs = 0;

                SaveCheckpoint();
            }
            else
            {
                _badEpochs++;
            }

            if (_badEpochs >= _patience)
            {
                ReduceLR();
            }
        }

        private void ReduceLR()
        {
            var oldLR = _optimizer.LearningRate;
            var newLR = MathF.Max(oldLR * _factor, _minLR);

            if (newLR < oldLR)
            {
                _optimizer.LearningRate = newLR;
                _log($">>> LR SCHEDULER: Reducing LR {oldLR:F6} → {newLR:F6}");
            }

            _badEpochs = 0;
        }

        /// <summary> Performs a high-speed copy of current weights into the RAM checkpoint. </summary>
        private void SaveCheckpoint()
        {
            for (var i = 0; i < _parameters.Length; i++)
            {
                _parameters[i].Data.AsSpan().CopyTo(_checkpoint[i].AsSpan());
            }
        }

        /// <summary> Restores weights from the RAM checkpoint using SIMD-accelerated CopyTo. </summary>
        private void RestoreCheckpoint()
        {
            for (var i = 0; i < _parameters.Length; i++)
            {
                _checkpoint[i].AsSpan().CopyTo(_parameters[i].Data.AsSpan());
            }
        }

        public void Reset()
        {
            _bestLoss = float.MaxValue;
            _badEpochs = 0;
        }
    }
}