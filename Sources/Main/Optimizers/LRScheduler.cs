using System.Diagnostics;

namespace DevOnBike.Overfit.Optimizers
{
    public class LRScheduler
    {
        private readonly Adam _optimizer;
        private readonly double _factor;     // Przez ile mnożymy LR (np. 0.5)
        private readonly int _patience;      // Ile epok czekamy na poprawę
        private double _bestLoss = double.MaxValue;
        private int _badEpochs = 0;
        private readonly double _minLR;

        public LRScheduler(Adam optimizer, double factor = 0.5, int patience = 2, double minLR = 1e-6)
        {
            _optimizer = optimizer;
            _factor = factor;
            _patience = patience;
            _minLR = minLR;
        }

        public void Step(double currentLoss)
        {
            if (currentLoss < _bestLoss * 0.995) // Margines 0.5% żeby uznać poprawę
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
                var newLR = Math.Max(oldLR * _factor, _minLR);
            
                if (newLR < oldLR)
                {
                    _optimizer.LearningRate = newLR;
                    Debug.WriteLine($">>> LR SCHEDULER: Redukcja LR z {oldLR:F6} na {newLR:F6}");
                }
                _badEpochs = 0;
            }
        }
    }
}