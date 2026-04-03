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
            ArgumentNullException.ThrowIfNull(parameters);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(learningRate);

            _parameters = parameters.Where(p => p.RequiresGrad).ToArray();
            LearningRate = learningRate;
        }

        public void Step()
        {
            // Lokalna kopia — eliminuje getter per iteracja pętli
            var negativeLr = -LearningRate;

            foreach (var p in _parameters)
            {
                // Grad jest zawsze non-null — ctor filtruje przez RequiresGrad
                // w = grad * (-lr) + w  — jeden AVX-512 VFMADD, zero alokacji
                TensorPrimitives.MultiplyAdd(
                    x: p.Grad.AsSpan(),
                    y: negativeLr,
                    addend: p.Data.AsSpan(),
                    destination: p.Data.AsSpan());
            }
        }

        public void ZeroGrad()
        {
            // Grad zawsze non-null po filtracji w ctor — guard zbędny
            foreach (var p in _parameters)
            {
                p.Grad.AsSpan().Clear();
            }
        }
    }
}