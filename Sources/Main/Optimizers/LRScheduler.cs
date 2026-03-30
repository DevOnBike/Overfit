namespace DevOnBike.Overfit.Optimizers
{
    public class LRScheduler
    {
        private readonly Adam _optimizer; //
        private readonly double _factor; // Przez ile mnożymy LR (np. 0.5)[cite: 3]
        private readonly int _patience; // Ile epok czekamy na poprawę[cite: 3]
        private double _bestLoss = double.MaxValue; //[cite: 3]
        private int _badEpochs = 0; //[cite: 3]
        private readonly double _minLR; //[cite: 3]
        private readonly Action<string> _log; // Dodano dla xUnit

        public LRScheduler(Adam optimizer, Action<string> log, double factor = 0.5, int patience = 2, double minLR = 1e-6) //[cite: 3]
        {
            _optimizer = optimizer; //[cite: 3]
            _log = log;
            _factor = factor; //[cite: 3]
            _patience = patience; //[cite: 3]
            _minLR = minLR; //[cite: 3]
        }

        public void Step(double currentLoss) //[cite: 3]
        {
            if (currentLoss < _bestLoss * 0.995) // Margines 0.5% żeby uznać poprawę[cite: 3]
            {
                _bestLoss = currentLoss; //[cite: 3]
                _badEpochs = 0; //[cite: 3]
            }
            else
            {
                _badEpochs++; //[cite: 3]
            }

            if (_badEpochs >= _patience) //[cite: 3]
            {
                var oldLR = _optimizer.LearningRate; //[cite: 3]
                var newLR = Math.Max(oldLR * _factor, _minLR); //[cite: 3]

                if (newLR < oldLR) //[cite: 3]
                {
                    _optimizer.LearningRate = newLR; //[cite: 3]
                    _log($">>> LR SCHEDULER: Redukcja LR z {oldLR:F6} na {newLR:F6}"); // Używamy wstrzykniętego logera
                }
                _badEpochs = 0; //[cite: 3]
            }
        }
    }
}