// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using DevOnBike.Overfit.Anomalies.Incidents;

namespace Benchmarks
{
    /// <summary>
    /// The question an IDE raised about <c>SignalCatalog.MatchesAny</c>: should its indexed <c>for</c> be a
    /// <c>foreach</c>?
    ///
    /// <para><b>This arm exists to establish that the question does not matter here</b>, which is a different
    /// and more useful answer than "foreach is fine". The loop body is
    /// <c>string.Contains(marker, OrdinalIgnoreCase)</c> — an ordinal-ignore-case substring search — and the
    /// loop itself is an increment and a compare. If the two shapes differ at all, the difference is buried
    /// under the body by orders of magnitude. Measuring only this would produce a confident "no difference"
    /// that says nothing about <c>for</c> versus <c>foreach</c>; see <see cref="LoopShapeBenchmark"/> for the
    /// isolated version, which is where the shapes are actually visible.</para>
    ///
    /// <para><see cref="Classify_Shipped"/> is present for proportion: it is what the caller really pays, so
    /// the reader can see what fraction of it the loop shape could ever have been.</para>
    ///
    /// <para>The marker tables are copied from <c>SignalCatalog</c> rather than exposed, because widening a
    /// type's surface to benchmark it changes the thing being benchmarked. They are only inputs here; the
    /// classification logic under test lives in the two <c>MatchesAny_*</c> arms.</para>
    /// </summary>
    [SimpleJob]
    [MemoryDiagnoser]
    public class SignalCatalogLoopBenchmark
    {
        private static readonly string[] Markers =
        [
            "restart", "oom", "throttl", "evict", "terminat", "kubelet", "imagepull", "image_pull",
            "probe_fail", "node_condition", "unschedulable", "pod_status_ready", "container_status_ready",
            "crashloop", "preempt", "disruption"
        ];

        /// <summary>
        /// Real metric names from the cluster lab, deliberately mixed: one that matches early, one that
        /// matches late, and two that match nothing and therefore walk the whole table — the worst case, and
        /// the common one, since two of the three catalog tiers always miss.
        /// </summary>
        private static readonly string[] Signals =
        [
            "kube_pod_container_status_restarts_total",
            "container_cpu_cfs_throttled_periods_total",
            "process_resident_memory_bytes",
            "overfit_chat_response_time_seconds"
        ];

        [Benchmark(Baseline = true)]
        public int MatchesAny_For()
        {
            var hits = 0;

            for (var s = 0; s < Signals.Length; s++)
            {
                if (MatchesAnyFor(Signals[s], Markers))
                {
                    hits++;
                }
            }

            return hits;
        }

        [Benchmark]
        public int MatchesAny_Foreach()
        {
            var hits = 0;

            foreach (var signal in Signals)
            {
                if (MatchesAnyForeach(signal, Markers))
                {
                    hits++;
                }
            }

            return hits;
        }

        /// <summary>What the caller actually pays: three marker tables and the tier dispatch around them.</summary>
        [Benchmark]
        public int Classify_Shipped()
        {
            var total = 0;

            for (var s = 0; s < Signals.Length; s++)
            {
                total += (int)SignalCatalog.Classify(Signals[s]);
            }

            return total;
        }

        private static bool MatchesAnyFor(string signal, string[] markers)
        {
            for (var i = 0; i < markers.Length; i++)
            {
                if (signal.Contains(markers[i], StringComparison.OrdinalIgnoreCase))
                {
                    return true;
                }
            }

            return false;
        }

        private static bool MatchesAnyForeach(string signal, string[] markers)
        {
            foreach (var marker in markers)
            {
                if (signal.Contains(marker, StringComparison.OrdinalIgnoreCase))
                {
                    return true;
                }
            }

            return false;
        }
    }
}
