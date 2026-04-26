// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Training
{
    public sealed class DelegateTrainingOptimizer : ITrainingOptimizer
    {
        private readonly Action _zeroGrad;
        private readonly Action _step;

        public DelegateTrainingOptimizer(
            Action zeroGrad,
            Action step)
        {
            _zeroGrad = zeroGrad ?? throw new ArgumentNullException(nameof(zeroGrad));
            _step = step ?? throw new ArgumentNullException(nameof(step));
        }

        public void ZeroGrad()
        {
            _zeroGrad();
        }

        public void Step()
        {
            _step();
        }
    }
}
