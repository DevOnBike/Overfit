using System.Numerics.Tensors;
using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Optimizers
{
    public sealed class SGD : IOptimizer // Dodano implementację interfejsu
    {
        // 1. Zmiana List<T> na płaską tablicę dla szybszej iteracji w pamięci
        private readonly AutogradNode[] _parameters;

        public double LearningRate { get; set; }

        public SGD(IEnumerable<AutogradNode> parameters, double learningRate)
        {
            if (parameters == null) throw new ArgumentNullException(nameof(parameters));

            // 2. Filtrujemy RequiresGrad TYLKO RAZ w konstruktorze!
            _parameters = parameters.Where(p => p.RequiresGrad).ToArray();
            LearningRate = learningRate;
        }

        public void Step()
        {
            // 3. Prekalkulacja zmiennej poza pętlą (choć to tylko znak, oszczędza cykle)
            var negativeLr = -LearningRate;

            // Iteracja po tablicy (JIT w C# uwielbia takie pętle)
            foreach (var p in _parameters)
            {
                // Czyste, sprzętowe FMA, zero instrukcji warunkowych (if)
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
                // Czyste zerowanie pamięci (memset), bez pytania o RequiresGrad
                p.Grad.Clear();
            }
        }
    }
}