// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Optimizers.Abstractions;
using DevOnBike.Overfit.Tensors;

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
                var view = _parameters[i].DataView;

                // Alokujemy kopię zapasową na stercie na podstawie wymiarów widoku
                _checkpoint[i] = view.Rank switch
                {
                    1 => new FastTensor<float>(view.GetDim(0), clearMemory: false),
                    2 => new FastTensor<float>(view.GetDim(0), view.GetDim(1), clearMemory: false),
                    3 => new FastTensor<float>(view.GetDim(0), view.GetDim(1), view.GetDim(2), clearMemory: false),
                    4 => new FastTensor<float>(view.GetDim(0), view.GetDim(1), view.GetDim(2), view.GetDim(3), clearMemory: false),
                    _ => throw new InvalidOperationException("Wymiar nieobsługiwany przez LRScheduler")
                };

                view.AsReadOnlySpan().CopyTo(_checkpoint[i].GetView().AsSpan());
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
                _parameters[i].DataView.AsReadOnlySpan().CopyTo(_checkpoint[i].GetView().AsSpan());
            }
        }

        /// <summary> Restores weights from the RAM checkpoint using SIMD-accelerated CopyTo. </summary>
        private void RestoreCheckpoint()
        {
            for (var i = 0; i < _parameters.Length; i++)
            {
                _checkpoint[i].GetView().AsReadOnlySpan().CopyTo(_parameters[i].DataView.AsSpan());
            }
        }

        public void Reset()
        {
            _bestLoss = float.MaxValue;
            _badEpochs = 0;
        }
    }
}