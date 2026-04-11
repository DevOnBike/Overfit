// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.Data.Abstractions;
using DevOnBike.Overfit.Data.Contracts;

namespace DevOnBike.Overfit.Data.Prepare
{
    /// <summary>
    ///     Coordinates the sequential execution of data processing layers.
    ///     Tracks dimension changes and execution time for each step in the pipeline.
    /// </summary>
    public class DataPipeline
    {
        private readonly List<IDataLayer> _layers = [];
        private readonly Action<string> _log;

        /// <param name="log">
        ///     Optional logging callback (e.g., ILogger.LogInformation).
        /// </param>
        public DataPipeline(Action<string> log = null)
        {
            _log = log;
        }

        /// <summary>
        ///     Appends a processing layer to the end of the pipeline.
        /// </summary>
        public DataPipeline AddLayer(IDataLayer layer)
        {
            _layers.Add(layer);
            return this;
        }

        /// <summary>
        ///     Executes all registered layers sequentially on the provided features and targets.
        /// </summary>
        public PipelineContext Execute(FastTensor<float> features, FastTensor<float> targets)
        {
            if (features.GetDim(0) != targets.GetDim(0))
            {
                throw new InvalidOperationException($"Mismatched row counts: Features={features.GetDim(0)}, Targets={targets.GetDim(0)}.");
            }

            var current = new PipelineContext(features, targets);

            foreach (var layer in _layers)
            {
                var rowsBefore = current.Features.GetDim(0);
                var colsBefore = current.Features.GetDim(1);

                var sw = Stopwatch.StartNew();
                current = layer.Process(current);
                sw.Stop();

                var rowsAfter = current.Features.GetDim(0);
                var colsAfter = current.Features.GetDim(1);

                // Validate dimension consistency after each layer processing
                if (current.Features.GetDim(0) != current.Targets.GetDim(0))
                {
                    throw new InvalidOperationException($"[{layer.GetType().Name}] Desynchronized dimensions: Features={current.Features.GetDim(0)} rows, Targets={current.Targets.GetDim(0)} rows.");
                }

                // Collect diagnostic metadata
                var diag = new LayerDiagnostic
                {
                    LayerName = layer.GetType().Name,
                    RowsBefore = rowsBefore,
                    ColsBefore = colsBefore,
                    RowsAfter = rowsAfter,
                    ColsAfter = colsAfter,
                    ElapsedMs = sw.ElapsedMilliseconds
                };

                current.Diagnostics.Add(diag);

                _log?.Invoke($"[{diag.LayerName}] {rowsBefore}x{colsBefore} → {rowsAfter}x{colsAfter} ({sw.ElapsedMilliseconds}ms)");
            }

            return current;
        }
    }
}