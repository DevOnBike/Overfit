// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Parameters;

namespace DevOnBike.Overfit.Training
{
    /// <summary>
    /// One data-parallel worker: a model replica, its own recording <see cref="ComputationGraph"/>
    /// (replicas must not share a tape — they train concurrently), and its trainable parameters
    /// (aligned 1:1 with the master). Built and owned by <see cref="DataParallelSession{TModel}"/>,
    /// which disposes the graph and the model together. <typeparamref name="TModel"/> is any
    /// <see cref="IDisposable"/> model (e.g. <c>GPT1Model</c>).
    /// </summary>
    public sealed class DataParallelReplica<TModel> : IDisposable
        where TModel : IDisposable
    {
        internal DataParallelReplica(TModel model, IReadOnlyList<Parameter> parameters, ComputationGraph graph)
        {
            Model = model;
            Parameters = parameters;
            Graph = graph;
        }

        /// <summary>The replica model (already switched to training mode).</summary>
        public TModel Model
        {
            get;
        }

        /// <summary>This replica's private autograd tape — never shared with other replicas.</summary>
        public ComputationGraph Graph
        {
            get;
        }

        /// <summary>This replica's trainable parameters, aligned 1:1 with the master.</summary>
        public IReadOnlyList<Parameter> Parameters
        {
            get;
        }

        public void Dispose()
        {
            Graph.Dispose();
            Model.Dispose();
        }
    }
}
