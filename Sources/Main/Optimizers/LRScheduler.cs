namespace DevOnBike.Overfit.Optimizers
{
    public class LRScheduler
    {
        private readonly IOptimizer _optimizer;
        private readonly float _factor;
        private readonly int _patience;
        private float _bestLoss = float.MaxValue;
        private int _badEpochs = 0;
        private readonly float _minLR;
        private readonly Action<string> _log;

        // Możesz wstrzyknąć Adam, SGD lub cokolwiek implementującego IOptimizer
        public LRScheduler(IOptimizer optimizer, Action<string> log, float factor = 0.5f, int patience = 2, float minLR = 1e-6f)
        {
            _optimizer = optimizer;
            _log = log;
            _factor = factor;
            _patience = patience;
            _minLR = minLR;
        }

        public void Step(float currentLoss)
        {
            if (currentLoss < _bestLoss * 0.995f)
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
                    _log($">>> LR SCHEDULER: Redukcja LR z {oldLR:F6} na {newLR:F6}");
                }
                _badEpochs = 0;
            }
        }
    }
}