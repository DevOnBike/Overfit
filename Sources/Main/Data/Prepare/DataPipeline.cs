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
    public class DataPipeline
    {
        private readonly List<IDataLayer> _layers = [];
        private readonly Action<string> _log;

        public DataPipeline(Action<string> log = null)
        {
            _log = log;
        }

        public DataPipeline AddLayer(IDataLayer layer)
        {
            _layers.Add(layer);
            return this;
        }

        public PipelineContext Execute(FastTensor<float> features, FastTensor<float> targets)
        {
            if (features.GetView().GetDim(0) != targets.GetView().GetDim(0))
            {
                throw new InvalidOperationException($"Mismatched row counts: Features={features.GetView().GetDim(0)}, Targets={targets.GetView().GetDim(0)}.");
            }

            var current = new PipelineContext(features, targets);

            foreach (var layer in _layers)
            {
                var rowsBefore = current.Features.GetView().GetDim(0);
                var colsBefore = current.Features.GetView().GetDim(1);

                var sw = Stopwatch.StartNew();
                current = layer.Process(current);
                sw.Stop();

                var rowsAfter = current.Features.GetView().GetDim(0);
                var colsAfter = current.Features.GetView().GetDim(1);

                if (current.Features.GetView().GetDim(0) != current.Targets.GetView().GetDim(0))
                {
                    throw new InvalidOperationException($"[{layer.GetType().Name}] Desynchronized dimensions: Features={current.Features.GetView().GetDim(0)} rows, Targets={current.Targets.GetView().GetDim(0)} rows.");
                }

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