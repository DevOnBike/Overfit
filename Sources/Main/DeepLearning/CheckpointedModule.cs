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
        /// <para>Structural rather than exhaustive: it recognises this project's dropout layers and walks down
        /// through <see cref="Sequential"/>, which is the shape a checkpointed segment actually has. A custom
        /// module that draws internally is not detectable from here, which is what
        /// <c>allowNonDeterministic</c> documents rather than hides.</para>
        ///
        /// <para><b>Iterative with a visited set, and both halves are load-bearing.</b> This was recursion
        /// under a <c>#pragma warning disable OVERFIT022</c> whose stated bound — "Sequential nesting, which a
        /// caller builds explicitly and is a handful of levels at most" — was an assumption about how the API
        /// gets used, not a proof. <c>Sequential.Add</c> null-checks its argument and nothing else, so
        /// <c>s.Add(s)</c> compiles, runs, and takes the host process down with a
        /// <c>StackOverflowException</c> that .NET cannot catch: no exception to report, no stack trace, no
        /// log line, and Overfit runs inside somebody else's application.
        ///
        /// <b>An explicit stack alone would have made that worse.</b> It converts the overflow into an
        /// infinite loop — a hang, which is harder to diagnose than a crash precisely because nothing happens,
        /// and the reason <c>OVERFIT023</c> exists alongside <c>OVERFIT022</c>. The visited set is what
        /// terminates a cycle; the stack only moves the growth off the call stack. A cycle is then an
        /// <see cref="ArgumentException"/> the caller can catch.</para>
        ///
        /// <para><b>A cycle is "on the current path", not "seen before", and the difference is a regression
        /// this nearly shipped.</b> The first version of this fix used one global visited set and threw on any
        /// repeat — which rejects <c>s.Add(relu); s.Add(relu)</c>, a legal model that reuses one stateless
        /// instance at two positions. Only a container reached while it is still an ancestor of the walk is a
        /// cycle, so the walk carries the ancestor set explicitly; a separate <c>finished</c> set stops a
        /// shared sub-model being re-walked once per position.</para>
        /// </summary>
        private static string? FindNonDeterministic(IModule module)
        {
            if (module is DropoutLayer or Dropout2DLayer)
            {
#pragma warning disable RS0030 // AOT-safe: the layer's own type name for a diagnostic message; no metadata is looked up.
                return module.GetType().Name;
#pragma warning restore RS0030
            }

            if (module is not Sequential root)
            {
                return null;
            }

            // `path` is the set of containers between the root and where the walk currently is — the
            // iterative spelling of "what is on the call stack". `finished` keeps a shared sub-model from
            // being walked once per position, which is what stops a wide DAG costing exponential time.
            var path = new HashSet<IModule>(ReferenceEqualityComparer.Instance) { root };
            var finished = new HashSet<IModule>(ReferenceEqualityComparer.Instance);
            var frames = new Stack<(Sequential Node, int Index)>();
            frames.Push((root, 0));

            while (frames.Count > 0)
            {
                var (node, index) = frames.Pop();

                if (index >= node.Modules.Count)
                {
                    path.Remove(node);
                    finished.Add(node);

                    continue;
                }

                // Resumed at the next child when this frame comes back up, so the walk is depth-first in
                // model order and reports the FIRST offending layer — which is what the constructor's message
                // claims and what a caller reads as "the one to move".
                frames.Push((node, index + 1));

                var child = node.Modules[index];

                if (child is DropoutLayer or Dropout2DLayer)
                {
#pragma warning disable RS0030 // AOT-safe: the layer's own type name for a diagnostic message; no metadata is looked up.
                    return child.GetType().Name;
#pragma warning restore RS0030
                }

                if (child is not Sequential nested || finished.Contains(nested))
                {
                    continue;
                }

                if (!path.Add(nested))
                {
                    throw new ArgumentException(
                        "The segment contains a cycle: a Sequential contains itself, directly or through "
                        + "another Sequential. Sequential.Add accepts any module, so a graph like s.Add(s) is "
                        + "legal to build, and walking it would otherwise recurse until the process is "
                        + "terminated by a StackOverflowException that .NET cannot catch.",
                        nameof(module));
                }

                frames.Push((nested, 0));
            }

            return null;
        }

        public bool IsTraining => _inner.IsTraining;

        public void Train() => _inner.Train();

        public void Eval() => _inner.Eval();

        public AutogradNode Forward(ComputationGraph? graph, AutogradNode input)
        {
            // No graph (inference) → run the inner module directly; checkpointing only pays off in training.
            if (graph == null)
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
