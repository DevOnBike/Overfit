using System.Numerics.Tensors;
using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Optimizers
{
    public sealed class SGD : IOptimizer
    {
        private readonly AutogradNode[] _parameters;

        public float LearningRate { get; set; }

        public SGD(IEnumerable<AutogradNode> parameters, float learningRate)
        {
            if (parameters == null) throw new ArgumentNullException(nameof(parameters));

            _parameters = parameters.Where(p => p.RequiresGrad).ToArray();
            LearningRate = learningRate;
        }

        public void Step()
        {
            var negativeLr = -LearningRate;

            foreach (var p in _parameters)
            {
                TensorPrimitives.MultiplyAdd(
                    x: p.Grad.AsReadOnlySpan(),
                    y: negativeLr,
                    addend: p.Data.AsReadOnlySpan(),
                    destination: p.Data.AsSpan()
                );
            }
        }

        public void ZeroGrad()
        {
            foreach (var p in _parameters)
            {
                p.Grad.Clear();
            }
        }
    }
}