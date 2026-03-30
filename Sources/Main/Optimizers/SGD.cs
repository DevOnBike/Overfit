using System.Numerics.Tensors;
using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Optimizers
{
    /// <summary>
    /// Stochastic Gradient Descent (SGD) Optimizer.
    /// Updates tensor weights using their computed gradients and a learning rate.
    /// Fully SIMD-accelerated and allocation-free during the training loop.
    /// </summary>
    public sealed class SGD
    {
        private readonly List<AutogradNode> _parameters;
        
        public double LearningRate { get; set; }

        /// <summary>
        /// Creates a new SGD optimizer.
        /// </summary>
        /// <param name="parameters">The list of tensors (weights/biases) to optimize.</param>
        /// <param name="learningRate">The step size for weight updates.</param>
        public SGD(IEnumerable<AutogradNode> parameters, double learningRate)
        {
            if (parameters == null)
            {
                throw new ArgumentNullException(nameof(parameters));
            }
            
            _parameters = new List<AutogradNode>(parameters);
            LearningRate = learningRate;
        }

        /// <summary>
        /// Performs a single optimization step (weight update).
        /// Executes W = W - LR * Grad using hardware FMA.
        /// </summary>
        public void Step()
        {
            foreach (var p in _parameters)
            {
                if (!p.RequiresGrad) continue;

                // Hardware AVX-512 FMA: destination = (x * y) + addend
                // We do: Data = (Grad * -LearningRate) + Data
                TensorPrimitives.MultiplyAdd(
                    x: p.Grad.AsReadOnlySpan(), 
                    y: -LearningRate, 
                    addend: p.Data.AsReadOnlySpan(), 
                    destination: p.Data.AsSpan()
                );
            }
        }

        /// <summary>
        /// Clears the gradients of all optimized parameters.
        /// MUST be called before every new forward/backward pass, 
        /// otherwise gradients will accumulate infinitely.
        /// </summary>
        public void ZeroGrad()
        {
            foreach (var p in _parameters)
            {
                if (p.RequiresGrad)
                {
                    // Calls Span<T>.Clear() under the hood - blisteringly fast memset
                    p.Grad.Clear(); 
                }
            }
        }
    }
}