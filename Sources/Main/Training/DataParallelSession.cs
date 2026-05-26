// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Optimizers.Abstractions;
using DevOnBike.Overfit.Parameters;

namespace DevOnBike.Overfit.Training
{
    /// <summary>
    /// Turnkey data-parallel training session: builds <c>workerCount</c> model replicas (each with its
    /// own training-mode model + private <see cref="ComputationGraph"/> + trainable-parameter list),
    /// wires a <see cref="DataParallelTrainer"/> over them, and broadcasts the master weights into every
    /// replica — so the caller no longer hand-rolls the replica/graph/parameter/broadcast boilerplate.
    ///
    /// <para>The caller still owns the master model + optimizer and supplies the task-specific
    /// per-replica body to <see cref="Step"/> (sample batch → reset <c>Replicas[i].Graph</c> → forward
    /// on <c>Replicas[i].Model</c> → loss + backward → return loss). Disposing the session disposes all
    /// replicas (graphs + models); the master is the caller's.</para>
    /// </summary>
    public sealed class DataParallelSession<TModel> : IDisposable
        where TModel : IDisposable
    {
        private readonly DataParallelReplica<TModel>[] _replicas;

        /// <param name="masterParameters">
        /// The authoritative parameters the optimizer updates (the master model's trainable parameters).
        /// </param>
        /// <param name="workerCount">Number of replicas to build (typically ≈ core count).</param>
        /// <param name="modelFactory">
        /// Builds replica <c>i</c>'s model (same architecture/config as the master), training-ready
        /// (call the model's <c>Train()</c> in the factory if it is not in training mode by default).
        /// Called once per replica.
        /// </param>
        /// <param name="parameterSelector">
        /// Extracts a model's trainable parameters in the same order as <paramref name="masterParameters"/>
        /// (e.g. <c>m =&gt; m.TrainableParameters()</c>).
        /// </param>
        /// <param name="arenaElementsPerReplica">Tape arena size (floats) for each replica's graph.</param>
        /// <param name="runWorkerOpsInline">
        /// Forwarded to <see cref="DataParallelTrainer"/> — keep replica inner ops single-threaded
        /// (default <c>true</c>; the right choice when worker count ≈ core count).
        /// </param>
        public DataParallelSession(
            IReadOnlyList<Parameter> masterParameters,
            int workerCount,
            Func<int, TModel> modelFactory,
            Func<TModel, IEnumerable<Parameter>> parameterSelector,
            int arenaElementsPerReplica,
            bool runWorkerOpsInline = true)
        {
            ArgumentNullException.ThrowIfNull(masterParameters);
            ArgumentNullException.ThrowIfNull(modelFactory);
            ArgumentNullException.ThrowIfNull(parameterSelector);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(workerCount);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(arenaElementsPerReplica);

            _replicas = new DataParallelReplica<TModel>[workerCount];
            var sets = new IReadOnlyList<Parameter>[workerCount];

            try
            {
                for (var i = 0; i < workerCount; i++)
                {
                    var model = modelFactory(i);
                    var graph = new ComputationGraph(arenaElementsPerReplica);
                    var parameters = Materialize(parameterSelector(model));
                    _replicas[i] = new DataParallelReplica<TModel>(model, parameters, graph);
                    sets[i] = parameters;
                }
            }
            catch
            {
                DisposeReplicas();
                throw;
            }

            Trainer = new DataParallelTrainer(masterParameters, sets, runWorkerOpsInline);
            Trainer.BroadcastParameters();
        }

        /// <summary>The underlying trainer (all-reduce / clip / optimizer step / broadcast).</summary>
        public DataParallelTrainer Trainer { get; }

        /// <summary>The worker replicas (model + graph + parameters), indexed by worker.</summary>
        public IReadOnlyList<DataParallelReplica<TModel>> Replicas => _replicas;

        /// <summary>Number of replicas.</summary>
        public int WorkerCount => _replicas.Length;

        /// <summary>
        /// Runs one data-parallel step. Convenience pass-through to <see cref="DataParallelTrainer.Step"/>
        /// — see it for the phase order and the <paramref name="trainWorker"/> contract.
        /// </summary>
        public float Step(IOptimizer optimizer, Func<int, float> trainWorker, float maxGradNorm = 0f)
            => Trainer.Step(optimizer, trainWorker, maxGradNorm);

        public void Dispose() => DisposeReplicas();

        private void DisposeReplicas()
        {
            for (var i = 0; i < _replicas.Length; i++)
            {
                _replicas[i]?.Dispose();
            }
        }

        private static Parameter[] Materialize(IEnumerable<Parameter> parameters)
        {
            if (parameters is IReadOnlyList<Parameter> list)
            {
                var array = new Parameter[list.Count];
                for (var i = 0; i < list.Count; i++)
                {
                    array[i] = list[i];
                }
                return array;
            }

            var collected = new List<Parameter>();
            foreach (var parameter in parameters)
            {
                collected.Add(parameter);
            }
            return collected.ToArray();
        }
    }
}
