using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Evolutionary.Abstractions;

namespace DevOnBike.Overfit.Evolutionary.Adapters
{
    /// <summary>
    ///     Projects the trainable parameters of an <see cref="IModule"/> to and from a single
    ///     flat vector of <see cref="float"/>. Bridges the evolutionary pipeline (which operates
    ///     on genome vectors) with the deep-learning module stack (which operates on autograd
    ///     nodes backed by <c>FastTensor&lt;float&gt;</c>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         The parameter order is frozen at construction time and exactly matches the
    ///         enumeration order of <see cref="IModule.Parameters"/> for the wrapped module.
    ///         Subsequent structural changes to the module (adding layers, etc.) are not
    ///         reflected — dispose the adapter and build a new one in that case.
    ///     </para>
    ///     <para>
    ///         Steady-state <see cref="WriteToVector"/> and <see cref="ReadFromVector"/> calls
    ///         perform zero managed allocations. The constructor performs one allocation for
    ///         the parameter-table snapshot.
    ///     </para>
    ///     <para>
    ///         <see cref="ReadFromVector"/> invokes <see cref="IModule.InvalidateParameterCaches"/>
    ///         after writing parameters, so modules with inference-time caches derived from
    ///         their weights (e.g. <c>LinearLayer</c>'s transposed-weight buffer) rebuild those
    ///         caches lazily on the next inference call.
    ///     </para>
    ///     <para>
    ///         Not thread-safe. For parallel fitness evaluation, construct one adapter per
    ///         evaluator thread, each wrapping its own <see cref="IModule"/> instance.
    ///     </para>
    /// </remarks>
    public sealed class NeuralNetworkParameterAdapter : IParameterVectorAdapter
    {
        private readonly IModule _module;
        private readonly AutogradNode[] _parameters;
        private readonly int[] _offsets;
        private readonly int[] _lengths;

        public NeuralNetworkParameterAdapter(IModule module)
        {
            ArgumentNullException.ThrowIfNull(module);

            _module = module;

            // Snapshot the parameter list once. IModule.Parameters() is implemented with
            // `yield return` in every current layer, so each call allocates an iterator
            // state machine — unacceptable on a fitness-evaluation hot path where this
            // may run thousands of times per generation.
            var count = 0;

            foreach (var _ in module.Parameters())
            {
                count++;
            }

            _parameters = new AutogradNode[count];
            _offsets = new int[count];
            _lengths = new int[count];

            var index = 0;
            var runningOffset = 0;

            foreach (var parameter in module.Parameters())
            {
                if (parameter is null)
                {
                    throw new InvalidOperationException(
                        $"Module produced a null parameter at index {index}.");
                }

                var length = parameter.DataView.AsReadOnlySpan().Length;

                _parameters[index] = parameter;
                _offsets[index] = runningOffset;
                _lengths[index] = length;

                runningOffset += length;
                index++;
            }

            ParameterCount = runningOffset;
        }

        public int ParameterCount { get; }

        /// <summary>
        ///     Copies the current parameter values of the wrapped module into
        ///     <paramref name="destination"/>, preserving declared parameter order.
        /// </summary>
        public void WriteToVector(Span<float> destination)
        {
            if (destination.Length != ParameterCount)
            {
                throw new ArgumentException(
                    $"destination length must be {ParameterCount}.",
                    nameof(destination));
            }

            for (var i = 0; i < _parameters.Length; i++)
            {
                var source = _parameters[i].DataView.AsReadOnlySpan();
                var slice = destination.Slice(_offsets[i], _lengths[i]);
                source.CopyTo(slice);
            }
        }

        /// <summary>
        ///     Overwrites the wrapped module's parameters with values from <paramref name="source"/>,
        ///     preserving declared parameter order. Triggers
        ///     <see cref="IModule.InvalidateParameterCaches"/> so any inference-time caches
        ///     derived from the old weights are rebuilt lazily on the next forward pass.
        /// </summary>
        public void ReadFromVector(ReadOnlySpan<float> source)
        {
            if (source.Length != ParameterCount)
            {
                throw new ArgumentException(
                    $"source length must be {ParameterCount}.",
                    nameof(source));
            }

            for (var i = 0; i < _parameters.Length; i++)
            {
                var destination = _parameters[i].DataView.AsSpan();
                var slice = source.Slice(_offsets[i], _lengths[i]);
                slice.CopyTo(destination);
            }

            _module.InvalidateParameterCaches();
        }
    }
}