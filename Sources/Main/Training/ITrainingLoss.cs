// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;

namespace DevOnBike.Overfit.Training
{
    public interface ITrainingLoss
    {
        AutogradNode Forward(
            ComputationGraph graph,
            AutogradNode prediction,
            AutogradNode target);

        void Backward(
            ComputationGraph graph,
            AutogradNode loss);

        float ReadScalar(
            AutogradNode loss);
    }
}
