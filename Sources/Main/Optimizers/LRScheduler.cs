namespace DevOnBike.Overfit.Optimizers
{
    public sealed class LRScheduler
    {
        private readonly IOptimizer _optimizer;
        private readonly float _factor;
        private readonly int _patience;
        private readonly float _minLR;
        private readonly float _minDelta;
        private readonly Action<string> _log;

        private float _bestLoss = float.MaxValue;
        private int _badEpochs = 0;

        /// <param name="optimizer">Optymalizator do sterowania LR.</param>
        /// <param name="log">Callback logowania (np. Console.WriteLine).</param>
        /// <param name="factor">Mnożnik redukcji LR (domyślnie 0.5).</param>
        /// <param name="patience">Liczba epok bez poprawy przed redukcją (domyślnie 2).</param>
        /// <param name="minLR">Dolna granica LR (domyślnie 1e-6).</param>
        /// <param name="minDelta">Minimalna relative poprawa licząca się jako progres (domyślnie 0.005 = 0.5%).</param>
        public LRScheduler(
            IOptimizer optimizer,
            Action<string> log,
            float factor = 0.5f,
            int patience = 2,
            float minLR = 1e-6f,
            float minDelta = 0.005f)
        {
            ArgumentNullException.ThrowIfNull(optimizer);
            ArgumentNullException.ThrowIfNull(log);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(factor);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(patience);

            _optimizer = optimizer;
            _log = log;
            _factor = factor;
            _patience = patience;
            _minLR = minLR;
            _minDelta = minDelta;
        }

        public void Step(float currentLoss)
        {
            // NaN/Inf = divergencja treningu — ignoruj, nie zmieniaj stanu
            if (float.IsNaN(currentLoss) || float.IsInfinity(currentLoss))
            {
                _log($">>> LR SCHEDULER: Ostrzeżenie — loss={currentLoss}, ignoruję epokę.");
                return;
            }

            // Poprawa względna o więcej niż minDelta = prawdziwy progres
            if (currentLoss < _bestLoss * (1f - _minDelta))
            {
                _bestLoss = currentLoss;
                _badEpochs = 0;
            }
            else
            {
                _badEpochs++;
            }

            if (_badEpochs >= _patience)
            {
                var oldLR = _optimizer.LearningRate;
                var newLR = MathF.Max(oldLR * _factor, _minLR);

                if (newLR < oldLR)
                {
                    _optimizer.LearningRate = newLR;
                    _log($">>> LR SCHEDULER: Redukcja LR {oldLR:F6} → {newLR:F6}");

                    // Reset bestLoss — nowy LR startuje z czystą kartą
                    _bestLoss = float.MaxValue;
                }

                _badEpochs = 0;
            }
        }

        /// <summary>Ręczny reset stanu — użyteczny przy fine-tuningu lub zmianie datasetu.</summary>
        public void Reset()
        {
            _bestLoss = float.MaxValue;
            _badEpochs = 0;
        }
    }
}