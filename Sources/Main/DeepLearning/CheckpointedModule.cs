// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// Wraps an inner <see cref="IModule"/> so its forward pass runs under
    /// <see cref="ComputationGraph.Checkpoint"/> during training: the inner module's activations are NOT
    /// kept on the tape but recomputed in the backward pass — trading one extra forward for a much lower
    /// peak activation footprint. Drop one (or several) into a <see cref="Sequential"/> to checkpoint the
    /// heavy segments of a model. Transparent: identical result (bit-close), all parameters / lifecycle
    /// delegated to the inner module. Inference (<c>graph == null</c> / <see cref="ForwardInference"/>) is
    /// passed straight through (checkpointing is a training-only memory trade).
    /// </summary>
    public sealed class CheckpointedModule : IModule
    {
        private readonly IModule _inner;
        private readonly int _subArenaElements;

        /// <param name="inner">The segment to recompute rather than keep on the tape.</param>
        /// <param name="subArenaElements">Working set for the recomputed forward.</param>
        /// <param name="allowNonDeterministic">
        /// Accepts a segment whose forward pass draws randomness. <b>Off by default, and the default is the
        /// point.</b>
        /// </param>
        /// <exception cref="ArgumentException">
        /// The segment contains a layer that draws randomness on the forward path - dropout - and
        /// <paramref name="allowNonDeterministic"/> was not set.
        ///
        /// <para><b>Checkpointing requires a deterministic segment, and nothing used to check.</b>
        /// <c>ComputationGraph.Checkpoint</c> documents the requirement; this type accepted any
        /// <see cref="IModule"/>, and <c>TensorMath.Dropout</c> draws from an unseeded
        /// <c>Random.Shared</c>. The recomputation during backward then uses a <i>different mask</i> than the
        /// forward pass did, so the gradients belong to a network that was never evaluated. Nothing throws,
        /// the loss goes down, and the model quietly optimises a different objective - which is why this is
        /// refused at construction rather than reported afterwards.</para>
        ///
        /// <para>No shipped path was wrong when this was found: <c>GPT1Model</c>'s transformer block carries
        /// no dropout. The composition was simply unguarded.</para>
        /// </exception>
        public CheckpointedModule(
            IModule inner, int subArenaElements = 1 << 20, bool allowNonDeterministic = false)
        {
            _inner = inner ?? throw new ArgumentNullException(nameof(inner));
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(subArenaElements);
            _subArenaElements = subArenaElements;

            if (!allowNonDeterministic && FindNonDeterministic(inner) is { } offender)
            {
                throw new ArgumentException(
                    $"{offender} draws randomness on its forward path, and a checkpointed segment is run "
                    + "twice: the backward recomputation would use a different mask than the forward pass "
                    + "did, producing gradients for a network that was never evaluated. Nothing would throw. "
                    + "Move the dropout outside the checkpointed segment, or pass allowNonDeterministic: true "
                    + "if the randomness is seeded per call.",
                    nameof(inner));
            }
        }

        /// <summary>
        /// The first layer in <paramref name="module"/> that draws randomness on the forward path, or null.
        ///
        /// <para>Structural rather than exhaustive: it recognises this project's dropout layers and looks one
        /// composition level down through <see cref="Sequential"/>, which is the shape a checkpointed segment
        /// actually has. A custom module that draws internally is not detectable from here, which is what
        /// <c>allowNonDeterministic</c> documents rather than hides.</para>
        /// </summary>
        private static string? FindNonDeterministic(IModule module)
        {
            if (module is DropoutLayer or Dropout2DLayer)
            {
                return module.GetType().Name;
            }

            if (module is not Sequential sequential)
            {
                return null;
            }

            for (var i = 0; i < sequential.Modules.Count; i++)
            {
#pragma warning disable OVERFIT022 // Bounded: recursion follows Sequential nesting, which a caller builds explicitly and is a handful of levels at most.
                var offender = FindNonDeterministic(sequential.Modules[i]);
#pragma warning restore OVERFIT022

                if (offender is not null)
                {
                    return offender;
                }
            }

            return null;
        }

        public bool IsTraining => _inner.IsTraining;

        public void Train() => _inner.Train();

        public void Eval() => _inner.Eval();

        public AutogradNode Forward(ComputationGraph? graph, AutogradNode input)
        {
            // No graph (inference) → run the inner module directly; checkpointing only pays off in training.
            if (graph is null)
            {
                return _inner.Forward(graph, input);
            }

            return graph.Checkpoint((g, x) => _inner.Forward(g, x), input, _subArenaElements);
        }

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
            => _inner.ForwardInference(input, output);

        public IEnumerable<AutogradNode> Parameters() => _inner.Parameters();

        public void Save(BinaryWriter bw) => _inner.Save(bw);

        public void Load(BinaryReader br) => _inner.Load(br);

        public void InvalidateParameterCaches() => _inner.InvalidateParameterCaches();

        public void Dispose() => _inner.Dispose();
    }
}
