// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Runtime;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.Core.Runtime
{
    /// <summary>
    /// Pins the one property the straggler ratio exists to have: <b>balanced work reports 1.00, whatever
    /// mix of chunk counts the dispatches produce</b>.
    ///
    /// <para><b>The defect this was written for, found 2026-08-21 by reading the arithmetic.</b>
    /// <c>RecordOccupancy</c> summed each dispatch's worst chunk RAW and <c>OccupancyReport</c> multiplied
    /// that sum by the GLOBAL mean chunk count. For a balanced dispatch <c>worst == busy / chunkCount</c>,
    /// so it cancels to 1.00 only when every dispatch has the same chunk count — and on the 60.9 MB CNN
    /// they do not, which the report itself shows as <c>mean chunks 26.4</c> on a 32-worker pool. It read
    /// 1.5x at the shipping default where the corrected form reads 1.37x, and a closed row (<c>XC-93</c>)
    /// had already built an argument on the broken figure.</para>
    ///
    /// <para><b>These assert arithmetic, not wall time, and that is the second lesson from the same day.</b>
    /// The first version of this class drove real <c>OverfitParallel.For</c> dispatches and asserted the
    /// measured ratio stayed under a bound. It passed alone and <b>failed in one full-suite run out of
    /// two</b> — a ratio of parallel chunk times is inflated by whatever else the box is doing, which is
    /// exactly the load dependence <c>TG-T12</c> is filed for. A test that passes half the time is worse
    /// than no test. The property is arithmetic, so it is tested as arithmetic, through
    /// <c>OverfitParallel.AddOccupancySample</c>, with exact integers and no threads.</para>
    ///
    /// <para><b>Membership of <see cref="ExclusiveProcessMeasurementCollection"/> is still load-bearing</b>,
    /// even though nothing here is timed: these tests write the process-wide occupancy totals, and a
    /// concurrent test in another class that measured occupancy would see them. The flag cannot be made
    /// per-thread — it is read on the worker threads, not the caller — see
    /// <see cref="OccupancyMeasurementScope"/>.</para>
    /// </summary>
    [Collection(ExclusiveProcessMeasurementCollection.Name)]
    public sealed class OccupancyStragglerTests
    {
        private readonly ITestOutputHelper _out;

        public OccupancyStragglerTests(ITestOutputHelper output)
        {
            _out = output;
        }

        /// <summary>
        /// Two dispatches, both perfectly balanced, carrying the SAME total busy time through DIFFERENT
        /// chunk counts. This is the shape the broken form got wrong; equal work per CHUNK cancels to 1.00
        /// in both forms and would not discriminate.
        /// </summary>
        [Fact]
        public void BalancedDispatches_WithDifferentChunkCounts_ReportExactlyOne()
        {
            using var measuring = new OccupancyMeasurementScope();

            OverfitParallel.ResetOccupancy();

            // 4 chunks x 12 ticks = 48 busy; 48 chunks x 1 tick = 48 busy. Both perfectly balanced.
            OverfitParallel.AddOccupancySample(chunkCount: 4, wallTicks: 100, busy: 48, worst: 12);
            OverfitParallel.AddOccupancySample(chunkCount: 48, wallTicks: 100, busy: 48, worst: 1);

            var straggler = OverfitParallel.OccupancyStragglerRatio;

            _out.WriteLine($"straggler {straggler:F4} — the pre-2026-08-21 form read "
                           + $"(12 + 1) * 26 / 96 = {(12 + 1) * 26.0 / 96:F2}");

            // Exact, because the inputs are exact: (12*4 + 1*48) / (48 + 48) = 96 / 96.
            Assert.Equal(1.0, straggler, 12);
        }

        /// <summary>
        /// The same balanced pair with the chunk counts EQUAL. Both the broken and the corrected form read
        /// 1.00 here, so this is not a discriminator — it is the control that shows the test above fails for
        /// the reason claimed rather than because any two samples happen to disagree.
        /// </summary>
        [Fact]
        public void BalancedDispatches_WithEqualChunkCounts_ReportExactlyOne()
        {
            using var measuring = new OccupancyMeasurementScope();

            OverfitParallel.ResetOccupancy();

            OverfitParallel.AddOccupancySample(chunkCount: 8, wallTicks: 100, busy: 80, worst: 10);
            OverfitParallel.AddOccupancySample(chunkCount: 8, wallTicks: 100, busy: 40, worst: 5);

            Assert.Equal(1.0, OverfitParallel.OccupancyStragglerRatio, 12);
        }

        /// <summary>
        /// A genuinely unbalanced dispatch must still be reported. A metric that reads 1.00 for everything
        /// would pass both tests above and be useless, so this pins the other direction.
        /// </summary>
        [Fact]
        public void UnbalancedDispatch_ReportsTheRatioOfWorstToMean()
        {
            using var measuring = new OccupancyMeasurementScope();

            OverfitParallel.ResetOccupancy();

            // 4 chunks, 100 busy, worst 40: the mean chunk is 25, so the worst is 1.6x the mean.
            OverfitParallel.AddOccupancySample(chunkCount: 4, wallTicks: 100, busy: 100, worst: 40);

            Assert.Equal(1.6, OverfitParallel.OccupancyStragglerRatio, 12);
        }

        /// <summary>
        /// The worst-dispatch tracking must follow the largest PER-DISPATCH ratio, which is not the same
        /// thing as the largest dispatch or the slowest one. Measured 2026-08-21 on the 60.9 MB CNN, the
        /// worst fan-out reads 6.73x and accounts for 0.2% of busy time — so a reader who mistook it for
        /// the dominant term would chase nothing. The histogram exists to make that visible, and this pins
        /// that it points at the right dispatch.
        /// </summary>
        [Fact]
        public void WorstDispatch_TracksTheLargestRatio_NotTheLargestOrSlowestDispatch()
        {
            using var measuring = new OccupancyMeasurementScope();

            OverfitParallel.ResetOccupancy();

            // Big and slow, but perfectly balanced: 10 chunks x 100 = 1000 busy, worst 100 -> ratio 1.00.
            OverfitParallel.AddOccupancySample(chunkCount: 10, wallTicks: 500, busy: 1000, worst: 100);

            // Tiny and fast, but badly unbalanced: 4 chunks, 20 busy, worst 14 -> ratio 2.80.
            OverfitParallel.AddOccupancySample(chunkCount: 4, wallTicks: 20, busy: 20, worst: 14);

            _out.WriteLine(OverfitParallel.OccupancyHistogram());

            Assert.Equal(2.8, OverfitParallel.OccupancyWorstDispatchRatio, 12);
        }

        /// <summary>With no samples the ratio is 0 rather than a division by zero.</summary>
        [Fact]
        public void NoSamples_ReportsZeroRatherThanDividingByZero()
        {
            using var measuring = new OccupancyMeasurementScope();

            OverfitParallel.ResetOccupancy();

            Assert.Equal(0.0, OverfitParallel.OccupancyStragglerRatio, 12);
        }
    }
}
