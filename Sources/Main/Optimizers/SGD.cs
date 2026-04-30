// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Parameters;
using DevOnBike.Overfit.Kernels;
using DevOnBike.Overfit.Optimizers.Abstractions;

namespace DevOnBike.Overfit.Optimizers
{
    /// <summary>
    /// Implements the Stochastic Gradient Descent (SGD) optimizer.
    /// </summary>
    public sealed class SGD : IOptimizer
    {
        // Parallel arrays — exactly one of _params[i] / _nodes[i] is non-null per slot.
        // Parameter path: direct Data/Grad access, no AutogradNode overhead.
        // Legacy path: AutogradNode (used when constructed from IEnumerable<AutogradNode>).
        private readonly Parameter?[] _params;
        private readonly AutogradNode?[] _nodes;
        private readonly int _count;

        /// <summary>
        /// Initializes a new SGD optimizer.
        /// Only parameters with RequiresGrad=true are tracked.
        /// </summary>
        /// <summary>Preferred constructor — accepts <see cref="Parameter"/> directly.</summary>
        public SGD(IEnumerable<Parameter> parameters, float learningRate)
        {
            ArgumentNullException.ThrowIfNull(parameters);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(learningRate);

            var list = new List<Parameter>();
            
            foreach (var p in parameters)
            {
                if (p.RequiresGrad)
                {
                    list.Add(p);
                }
            }

            _count  = list.Count;
            _params = list.ToArray();
            _nodes  = new AutogradNode?[_count];
            LearningRate = learningRate;
        }

        /// <summary>Compatibility shim — prefer the <see cref="Parameter"/> overload.</summary>
        [Obsolete("Pass IEnumerable<Parameter> via module.TrainableParameters() instead.")]
        public SGD(IEnumerable<AutogradNode> parameters, float learningRate)
        {
            ArgumentNullException.ThrowIfNull(parameters);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(learningRate);

            var list = new List<AutogradNode>();
            foreach (var p in parameters)
            {
                if (p.RequiresGrad)
                {
                    list.Add(p);
                }
            }

            _count  = list.Count;
            _nodes  = list.ToArray();
            _params = new Parameter?[_count];
            
            LearningRate = learningRate;
        }

        /// <summary>
        /// Learning rate used by the optimizer.
        /// </summary>
        public float LearningRate { get; set; }

        /// <summary>
        /// Performs a single optimization step.
        /// Applies the update rule:
        ///
        ///     w = w - learningRate * grad
        ///
        /// Implemented as:
        ///
        ///     weights = grad * (-learningRate) + weights
        ///
        /// The same weights buffer is used as both addend and destination.
        /// This requires ElementwiseKernels.MultiplyAdd to support:
        ///
        ///     addend == destination
        ///
        /// Keep the aliasing test before changing this method.
        /// </summary>
        public void Step()
        {
            var negativeLearningRate = -LearningRate;

            for (var i = 0; i < _count; i++)
            {
                if (_params[i] != null)
                {
                    var p = _params[i]!;
                    
                    ElementwiseKernels.MultiplyAdd(
                        p.Grad!.AsReadOnlySpan(),
                        negativeLearningRate,
                        p.Data.AsReadOnlySpan(),
                        p.Data.AsSpan());
                }
                else
                {
                    var n = _nodes[i]!;
                    ElementwiseKernels.MultiplyAdd(
                        n.GradView.AsReadOnlySpan(),
                        negativeLearningRate,
                        n.DataView.AsReadOnlySpan(),
                        n.DataView.AsSpan());
                }
            }
        }

        /// <summary>
        /// Resets the gradients of all managed parameters to zero.
        /// Should be called before each forward pass.
        /// </summary>
        public void ZeroGrad()
        {
            for (var i = 0; i < _count; i++)
            {
                if (_params[i] != null) { _params[i]!.ZeroGrad(); }
                else { _nodes[i]!.ZeroGrad(); }
            }
        }
    }
}