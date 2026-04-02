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
                if (p.Grad == null) continue;

                TensorPrimitives.MultiplyAdd(
                    x: p.Grad.AsSpan(),
                    y: negativeLr,
                    addend: p.Data.AsSpan(),
                    destination: p.Data.AsSpan()
                );
            }
        }

        public void ZeroGrad()
        {
            foreach (var p in _parameters)
            {
                // Bezpieczne zerowanie dzięki Span
                if (p.Grad != null) p.Grad.AsSpan().Clear();
            }
        }
    }
}