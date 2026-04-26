// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;

namespace DevOnBike.Overfit.Training
{
    public sealed class DelegateTrainingLoss : ITrainingLoss
    {
        private readonly Func<ComputationGraph, AutogradNode, AutogradNode, AutogradNode> _forward;
        private readonly Action<ComputationGraph, AutogradNode> _backward;
        private readonly Func<AutogradNode, float> _readScalar;

        public DelegateTrainingLoss(
            Func<ComputationGraph, AutogradNode, AutogradNode, AutogradNode> forward,
            Action<ComputationGraph, AutogradNode> backward,
            Func<AutogradNode, float>? readScalar = null)
        {
            _forward = forward ?? throw new ArgumentNullException(nameof(forward));
            _backward = backward ?? throw new ArgumentNullException(nameof(backward));
            _readScalar = readScalar ?? DefaultReadScalar;
        }

        public AutogradNode Forward(
            ComputationGraph graph,
            AutogradNode prediction,
            AutogradNode target)
        {
            return _forward(graph, prediction, target);
        }

        public void Backward(
            ComputationGraph graph,
            AutogradNode loss)
        {
            _backward(graph, loss);
        }

        public float ReadScalar(
            AutogradNode loss)
        {
            return _readScalar(loss);
        }

        private static float DefaultReadScalar(
            AutogradNode loss)
        {
            var data = loss.DataView.AsReadOnlySpan();

            if (data.Length == 0)
            {
                throw new InvalidOperationException("Loss node has empty data.");
            }

            return data[0];
        }
    }
}
