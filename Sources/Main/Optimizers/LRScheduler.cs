using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Optimizers
{
    public sealed class LRScheduler : IDisposable
    {
        private readonly IOptimizer _optimizer;
        private readonly AutogradNode[] _parameters;
        private readonly FastTensor<float>[] _checkpoint; // Zrzut pamięci (RAM)

        private readonly float _factor;
        private readonly int _patience;
        private readonly float _minLR;
        private readonly float _minDelta;
        private readonly Action<string> _log;

        private float _bestLoss = float.MaxValue;
        private int _badEpochs = 0;

        public LRScheduler(
            IOptimizer optimizer,
            AutogradNode[] parameters, // Sieć przekazuje swoje wagi do monitorowania
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
                throw new ArgumentOutOfRangeException(nameof(factor), "Mnożnik redukcji musi być w przedziale (0, 1).");
            }

            _optimizer = optimizer;
            _parameters = parameters;
            _log = log;
            _factor = factor;
            _patience = patience;
            _minLR = minLR;
            _minDelta = minDelta;

            // PRE-ALOKACJA: Tworzymy bliźniacze tensory w pamięci
            _checkpoint = new FastTensor<float>[_parameters.Length];
            for (var i = 0; i < _parameters.Length; i++)
            {
                _checkpoint[i] = FastTensor<float>.SameShape(_parameters[i].Data);
                // Zapisujemy stan początkowy (na wypadek wybuchu w 1. epoce)
                _parameters[i].Data.AsSpan().CopyTo(_checkpoint[i].AsSpan());
            }
        }

        public void Step(float currentLoss)
        {
            // 1. KATASTROFA: Wykryto NaN/Inf
            if (float.IsNaN(currentLoss) || float.IsInfinity(currentLoss))
            {
                _log($">>> KATASTROFA NUMERYCZNA (Loss={currentLoss}). Cofam wagi do ostatniego bezpiecznego stanu!");
                RestoreCheckpoint();
                ReduceLR(); // Od razu tniemy LR, by nie powtórzyć błędu
                return;
            }

            // 2. SUKCES: Prawdziwy progres
            if (currentLoss < _bestLoss * (1f - _minDelta))
            {
                _bestLoss = currentLoss;
                _badEpochs = 0;
                
                SaveCheckpoint(); // SIMD CopyTo (Błyskawiczny zrzut)
            }
            else
            {
                _badEpochs++;
            }

            // 3. STAGNACJA: Cierpliwość wyczerpana
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
                _log($">>> LR SCHEDULER: Redukcja LR {oldLR:F6} → {newLR:F6}");
            }

            _badEpochs = 0;
        }

        private void SaveCheckpoint()
        {
            for (var i = 0; i < _parameters.Length; i++)
            {
                _parameters[i].Data.AsSpan().CopyTo(_checkpoint[i].AsSpan());
            }
        }

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

        public void Dispose()
        {
            // Zwalniamy bufory pamięci
            for (var i = 0; i < _checkpoint.Length; i++)
            {
                _checkpoint[i]?.Dispose();
            }
        }
    }
}