using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.Data.Contracts;
using System.Diagnostics;

namespace DevOnBike.Overfit.Data.Prepare
{
    public class DataPipeline
    {
        private readonly List<IDataLayer> _layers = [];
        private readonly Action<string> _log;

        /// <param name="log">
        /// Opcjonalny callback logowania (np. ITestOutputHelper.WriteLine, ILogger.LogInformation).
        /// Null = diagnostyka zbierana w PipelineContext.Diagnostics, ale bez logowania na żywo.
        /// </param>
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
            // Walidacja wejścia
            if (features.GetDim(0) != targets.GetDim(0))
            {
                throw new InvalidOperationException(
                    $"Niezgodna liczba wierszy: Features={features.GetDim(0)}, Targets={targets.GetDim(0)}.");
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

                // Walidacja spójności wymiarów po każdej warstwie
                if (current.Features.GetDim(0) != current.Targets.GetDim(0))
                {
                    throw new InvalidOperationException(
                        $"[{layer.GetType().Name}] Rozsynchronizowane wymiary: " +
                        $"Features={current.Features.GetDim(0)} wierszy, " +
                        $"Targets={current.Targets.GetDim(0)} wierszy.");
                }

                // Diagnostyka
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

                _log?.Invoke(
                    $"[{diag.LayerName}] {rowsBefore}x{colsBefore} → {rowsAfter}x{colsAfter} ({sw.ElapsedMilliseconds}ms)");
            }

            return current;
        }
    }
}