using System.Runtime.InteropServices;
using DevOnBike.Overfit.Evolutionary.Abstractions;

namespace DevOnBike.Overfit.Evolutionary.Evaluators
{
    /// <summary>
    ///     Parallel dispatcher over an <see cref="ICandidateEvaluator{TContext}"/>.
    ///     Fans population evaluation across the thread pool, giving each worker thread
    ///     an independent, lazily-initialized instance of <typeparamref name="TContext"/>
    ///     (a "brain slot" — typically a neural network, scratch buffer, or other
    ///     heavyweight per-thread resource).
    /// </summary>
    /// <typeparam name="TContext">
    ///     Per-thread mutable state carried between consecutive <see cref="ICandidateEvaluator{TContext}.Evaluate"/>
    ///     calls on the same thread. May be a reference type (e.g. a pre-initialized
    ///     <c>IModule</c> wrapped together with its parameter adapter) or a mutable struct
    ///     containing scratch buffers.
    /// </typeparam>
    /// <remarks>
    ///     <para>
    ///         The evaluator preserves the wrapped <see cref="ICandidateEvaluator{TContext}"/>
    ///         contract exactly: for each genome in the population, the wrapped evaluator is
    ///         called once, receiving that genome as a <see cref="ReadOnlySpan{T}"/> and its
    ///         thread-local <typeparamref name="TContext"/> by reference.
    ///     </para>
    ///     <para>
    ///         Context instances are allocated lazily on first touch per thread via
    ///         <see cref="ThreadLocal{T}"/> and reused across every subsequent
    ///         <see cref="Evaluate"/> invocation on that thread. This matters when the
    ///         context is expensive to build (e.g. a neural network with tens of thousands
    ///         of parameters) — a typical evolutionary run makes millions of evaluations,
    ///         and paying the context-construction cost once per worker thread (rather than
    ///         once per genome) is the entire point of this class.
    ///     </para>
    ///     <para>
    ///         Expected allocation profile: BCL's <see cref="Parallel.For(int, int, Action{int})"/>
    ///         allocates closures and scheduling state per call, typically in the 0.5–2 KB range.
    ///         This is placed at the fitness-evaluation boundary intentionally, where the per-call
    ///         work is measured in milliseconds and the dispatch overhead is negligible. The
    ///         overall pipeline remains zero-alloc in the GA core (<c>GenerationalGeneticAlgorithm</c>).
    ///     </para>
    ///     <para>
    ///         AOT-safe. Uses <c>unsafe</c> with pinned spans to thread population and fitness
    ///         buffers through the closure; no reflection, no dynamic code.
    ///     </para>
    /// </remarks>
    public sealed class ParallelPopulationEvaluator<TContext> : IPopulationEvaluator, IDisposable
    {
        private readonly ICandidateEvaluator<TContext> _evaluator;
        private readonly ThreadLocal<TContext> _contextLocal;
        private readonly Action<TContext>? _contextDispose;
        private readonly ParallelOptions _parallelOptions;
        private int _disposed;

        /// <summary>
        ///     Creates a parallel evaluator.
        /// </summary>
        /// <param name="evaluator">The per-candidate evaluator. Must be thread-safe for concurrent
        ///     invocation: multiple threads will call <see cref="ICandidateEvaluator{TContext}.Evaluate"/>
        ///     on this instance simultaneously, each with its own context. The instance itself
        ///     should hold no mutable shared state.</param>
        /// <param name="contextFactory">Factory invoked lazily, once per worker thread, to
        ///     construct that thread's persistent context. Called at most
        ///     <paramref name="degreeOfParallelism"/> times over the lifetime of this evaluator.</param>
        /// <param name="contextDispose">Optional per-context cleanup, invoked for every live
        ///     thread-local context when this evaluator itself is disposed.</param>
        /// <param name="degreeOfParallelism">Maximum worker count. Defaults to
        ///     <see cref="Environment.ProcessorCount"/> when null.</param>
        public ParallelPopulationEvaluator(
            ICandidateEvaluator<TContext> evaluator,
            Func<TContext> contextFactory,
            Action<TContext>? contextDispose = null,
            int? degreeOfParallelism = null)
        {
            ArgumentNullException.ThrowIfNull(evaluator);
            ArgumentNullException.ThrowIfNull(contextFactory);

            var dop = degreeOfParallelism ?? Environment.ProcessorCount;

            if (dop < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(degreeOfParallelism), "Degree of parallelism must be at least 1.");
            }

            _evaluator = evaluator;
            _contextLocal = new ThreadLocal<TContext>(contextFactory, trackAllValues: true);
            _contextDispose = contextDispose;
            _parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = dop };
        }

        public unsafe void Evaluate(
            ReadOnlySpan<float> populationData,
            Span<float> fitnessOut,
            int populationSize,
            int parameterCount)
        {
            ThrowIfDisposed();

            if (populationSize < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(populationSize));
            }

            if (parameterCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(parameterCount));
            }

            var expectedPopulation = populationSize * parameterCount;

            if (populationData.Length != expectedPopulation)
            {
                throw new ArgumentException(
                    $"populationData length must be {expectedPopulation}.",
                    nameof(populationData));
            }

            if (fitnessOut.Length != populationSize)
            {
                throw new ArgumentException(
                    $"fitnessOut length must be {populationSize}.",
                    nameof(fitnessOut));
            }

            if (populationSize == 0)
            {
                return;
            }

            // Span<T> cannot cross the closure boundary because it's a ref struct. Pin the
            // backing memory of both spans for the duration of the Parallel.For dispatch
            // and smuggle raw pointers through the closure instead. Workers reconstitute
            // fresh spans on each iteration from (ptr + i*paramCount).
            //
            // MemoryMarshal.GetReference is zero-cost for array-backed spans (the most common
            // case here — population is stored in a FastTensor's managed array). For
            // stack-allocated or unmanaged-backed spans the `fixed` statement degenerates
            // to a no-op pin, which is also correct.
            fixed (float* populationPtr = &MemoryMarshal.GetReference(populationData))
            fixed (float* fitnessPtr = &MemoryMarshal.GetReference(fitnessOut))
            {
                // Capture into locals so the closure captures the locals (single small class)
                // rather than `this` plus indirections through instance fields on every iteration.
                var popPtr = populationPtr;
                var fitPtr = fitnessPtr;
                var evaluator = _evaluator;
                var contextLocal = _contextLocal;
                var paramCount = parameterCount;

                Parallel.For(0, populationSize, _parallelOptions, i =>
                {
                    // Read thread-local context. First touch on a worker thread triggers the
                    // factory from the constructor; subsequent touches are a cheap lookup.
                    var context = contextLocal.Value!;

                    // Reconstitute a non-allocating span over this genome's slice.
                    var parameters = new ReadOnlySpan<float>(popPtr + (i * paramCount), paramCount);

                    fitPtr[i] = evaluator.Evaluate(parameters, ref context);

                    // Write back in case the evaluator replaced the reference (harmless for
                    // reference types; required for struct TContext whose fields were mutated
                    // since `context` is a local copy on the worker stack).
                    contextLocal.Value = context;
                });
            }
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
            {
                return;
            }

            // Run user-supplied cleanup for every live thread-local context before disposing
            // the ThreadLocal container itself.
            if (_contextDispose is not null)
            {
                foreach (var context in _contextLocal.Values)
                {
                    _contextDispose(context);
                }
            }

            _contextLocal.Dispose();
        }

        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(_disposed == 1, this);
        }
    }
}