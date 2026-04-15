using System.Diagnostics;
using DevOnBike.Overfit.Diagnostics.Contracts;
using DevOnBike.Overfit.Diagnostics.DevOnBike.Overfit.Diagnostics;

namespace DevOnBike.Overfit.Diagnostics
{
    /// <summary>
    /// Zlew (Sink) w pełni kompatybilny z OpenTelemetry w .NET 10.
    /// Tłumaczy lekkie struktury wewnętrzne na potężne metryki systemowe.
    /// </summary>
    public sealed class OpenTelemetrySink : IOverfitDiagnosticsSink
    {
        public static readonly OpenTelemetrySink Instance = new();

        public void OnKernelCompleted(in KernelDiagnosticEvent evt)
        {
            var tags = new TagList
            {
                {
                    "category", evt.Category
                },
                {
                    "kernel", evt.Name
                }
            };

            OverfitTelemetry.InferenceDuration.Record(evt.DurationMs, tags);
            OverfitTelemetry.InferenceCount.Add(1, tags);
        }

        public void OnModuleCompleted(in ModuleDiagnosticEvent evt)
        {
            // Obsługa całych warstw
        }

        public void OnGraphCompleted(in GraphDiagnosticEvent evt)
        {
            // Możemy tu wysyłać podsumowanie Backpropagation
        }

        public void OnAllocation(in AllocationDiagnosticEvent evt)
        {
            // Aktualizacja wskaźnika natywnego RAM-u w czasie rzeczywistym
            OverfitTelemetry.NativeMemoryBytes.Add(evt.Bytes, new KeyValuePair<string, object?>("owner", evt.Owner));
        }

        public void OnCounter(string name, long value)
        {
            // Customowe liczniki
        }
    }
}