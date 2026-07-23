// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Reflection;
using BenchmarkDotNet.Columns;
using BenchmarkDotNet.Reports;
using BenchmarkDotNet.Running;
using DevOnBike.Overfit.Diagnostics;

namespace Benchmarks.Helpers
{
    /// <summary>
    /// Adds a <c>TFLOP/s</c> and a <c>GB/s</c> column to the summary table, derived from BenchmarkDotNet's own
    /// measured mean and from a <see cref="WorkAmount"/> the benchmark class declares for itself.
    ///
    /// <para><b>Opting in.</b> A benchmark class declares a public static method
    /// <c>WorkAmount GetWorkAmount(BenchmarkCase)</c>. The column finds it by reflection — the Benchmark
    /// project is not subject to <c>Sources/Main</c>'s reflection ban, and reflection is what lets the lookup
    /// work without relying on a static constructor having run in the host process. Classes that do not
    /// declare the method simply get empty cells.</para>
    ///
    /// <para>Cells stay empty when the relevant part of the declared work is zero, so a benchmark that moves
    /// memory without doing arithmetic can never be reported as if it had done arithmetic.</para>
    /// </summary>
    internal sealed class ThroughputColumn : IColumn
    {
        public static readonly IColumn Teraflops = new ThroughputColumn(compute: true);
        public static readonly IColumn Gigabytes = new ThroughputColumn(compute: false);

        private readonly bool _compute;

        private ThroughputColumn(bool compute)
        {
            _compute = compute;
        }

        public string Id => nameof(ThroughputColumn) + (_compute ? ".Compute" : ".Memory");

        public string ColumnName => _compute ? "TFLOP/s" : "GB/s";

        public string Legend => _compute
            ? "Logical multiply-accumulates per second (MAC counted as 2 ops), as declared by the benchmark"
            : "Bytes read plus written per second, as declared by the benchmark";

        public bool AlwaysShow => false;

        public ColumnCategory Category => ColumnCategory.Custom;

        public int PriorityInCategory => _compute ? 0 : 1;

        public bool IsNumeric => true;

        public UnitType UnitType => UnitType.Dimensionless;

        public bool IsAvailable(Summary summary)
        {
            return true;
        }

        public bool IsDefault(Summary summary, BenchmarkCase benchmarkCase)
        {
            return false;
        }

        public string GetValue(Summary summary, BenchmarkCase benchmarkCase)
        {
            var statistics = summary[benchmarkCase]?.ResultStatistics;

            if (statistics is null || statistics.Mean <= 0.0)
            {
                return "-";
            }

            if (!TryGetWorkAmount(benchmarkCase, out var work))
            {
                return "-";
            }

            // BenchmarkDotNet reports the mean in nanoseconds. The rates themselves come from the shared
            // Throughput helper so the benchmark table and the runtime profiler cannot drift apart.
            var elapsed = TimeSpan.FromTicks((long)(statistics.Mean / 100.0));

            if (_compute)
            {
                return work.Flops <= 0L
                    ? "-"
                    : Throughput.TeraflopsPerSecond(work.Flops, elapsed)
                        .ToString("F2", CultureInfo.InvariantCulture);
            }

            return work.Bytes <= 0L
                ? "-"
                : Throughput.GigabytesPerSecond(work.Bytes, elapsed)
                    .ToString("F1", CultureInfo.InvariantCulture);
        }

        public string GetValue(Summary summary, BenchmarkCase benchmarkCase, SummaryStyle style)
        {
            return GetValue(summary, benchmarkCase);
        }

        private static bool TryGetWorkAmount(BenchmarkCase benchmarkCase, out WorkAmount work)
        {
            work = default;

            var provider = benchmarkCase.Descriptor.Type.GetMethod(
                "GetWorkAmount",
                BindingFlags.Public | BindingFlags.Static,
                [typeof(BenchmarkCase)]);

            if (provider is null || provider.ReturnType != typeof(WorkAmount))
            {
                return false;
            }

            work = (WorkAmount)provider.Invoke(null, [benchmarkCase])!;

            return true;
        }
    }
}
