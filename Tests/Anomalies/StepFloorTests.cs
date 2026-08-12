// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// The separate step floor — <c>AN-D4b</c>.
    ///
    /// <para><b>What it is for.</b> The step gate used to read the TREND table. A trend floor is fitted to
    /// how far one pod's series travels across a window; a step floor governs how far the median across pods
    /// moves between the halves of one, and they are not the same distribution. Measured over 30.8 h of lab
    /// data: sharing them made the step gate demand <b>40%</b> of the level on
    /// <c>MemoryWorkingSetBytes</c> — so its 25% relative gate never bound, 100% of the time — and
    /// <b>123%</b> at the low decile of <c>GcGen2HeapBytes</c>, meaning the heap had to more than double
    /// before a step was reportable.</para>
    ///
    /// <para><b>The compatibility rule is the load-bearing part, and it is what most of these tests pin.</b>
    /// Adding the field must move no deployed threshold on its own: absent, the step floor is the trend
    /// floor, exactly as before. Falling back to the CALIBRATOR instead would silently change what every
    /// existing deployment reports on the day it upgraded — the reason this field did not exist until
    /// now.</para>
    /// </summary>
    public sealed class StepFloorTests
    {
        private const MetricIndex Cpu = MetricIndex.CpuUsageRatio;
        private const MetricIndex Heap = MetricIndex.GcGen2HeapBytes;

        /// <summary><b>The compatibility contract.</b> No step floor configured: the trend floor is used.</summary>
        [Fact]
        public void WithNoStepFloorConfiguredTheTrendFloorIsUsed()
        {
            var floors = Source(trend: (Cpu, 0.000326), step: null);

            Assert.Equal(0.000326, floors.MinAbsoluteLevelShift(Cpu));
        }

        [Fact]
        public void AConfiguredStepFloorWins()
        {
            var floors = Source(trend: (Cpu, 0.000326), step: (Cpu, 0.00008));

            Assert.Equal(0.00008, floors.MinAbsoluteLevelShift(Cpu));

            // And the trend gate is untouched — the whole point is that the two can differ.
            Assert.Equal(0.000326, floors.MinAbsoluteTrendChange(Cpu));
        }

        /// <summary>
        /// The fallback is <b>per metric</b>, not per table. An operator who fixes one channel must not
        /// silently move every other one — which is what a whole-table fallback would do.
        /// </summary>
        [Fact]
        public void AStepFloorOnOneMetricLeavesTheOthersOnTheirTrendFloor()
        {
            var trend = new double[(int)MetricIndex.Count];
            var step = new double[(int)MetricIndex.Count];

            trend[(int)Cpu] = 0.000326;
            trend[(int)Heap] = 3.93e6;
            step[(int)Heap] = 0.9e6;

            var floors = new ConfiguredFloorSource(null, trend, step, null, applyCalibrated: false);

            Assert.Equal(0.9e6, floors.MinAbsoluteLevelShift(Heap));
            Assert.Equal(0.000326, floors.MinAbsoluteLevelShift(Cpu));
        }

        /// <summary>
        /// A zero entry means "not configured", not "no floor". Reading it as an explicit zero would turn
        /// the gate off for that metric — the configuration measured at 209 false incidents a day.
        /// </summary>
        [Fact]
        public void AZeroStepEntryFallsBackRatherThanDisablingTheGate()
        {
            var floors = Source(trend: (Cpu, 0.000326), step: (Cpu, 0.0));

            Assert.Equal(0.000326, floors.MinAbsoluteLevelShift(Cpu));
        }

        /// <summary>With neither configured the gate is off, which is the pre-existing behaviour.</summary>
        [Fact]
        public void WithNeitherConfiguredTheGateIsOff()
        {
            var floors = new ConfiguredFloorSource(null, null, null, null, applyCalibrated: false);

            Assert.Equal(0.0, floors.MinAbsoluteLevelShift(Cpu));
        }

        /// <summary>The four-argument constructor still exists and still means "no step floor".</summary>
        [Fact]
        public void TheOlderConstructorKeepsItsMeaning()
        {
            var trend = new double[(int)MetricIndex.Count];
            trend[(int)Cpu] = 0.000326;

            var floors = new ConfiguredFloorSource(null, trend, null, applyCalibrated: false);

            Assert.Equal(0.000326, floors.MinAbsoluteLevelShift(Cpu));
        }

        /// <summary>
        /// The config file's <c>minStepChange</c> reaches the floor source. Parsed with a unit, like every
        /// other quantity in that file.
        /// </summary>
        [Fact]
        public void TheConfigFileFieldIsRead()
        {
            var file = new AnomalyGuardConfigFile
            {
                Prometheus = "http://localhost:9090",
                Thresholds =
                {
                    ["GcGen2HeapBytes"] = new AnomalyGuardConfigFile.ThresholdEntry
                    {
                        MinTrendChange = "3.93MB",
                        MinStepChange = "0.93MB",
                    },
                },
            };

            var (_, trend, step, _) = AnomalyGuardConfigReader.ReadThresholds(file, out var problems);

            Assert.Empty(problems);
            Assert.Equal(3.93e6, trend[(int)Heap]);
            Assert.Equal(0.93e6, step[(int)Heap]);
        }

        /// <summary>An omitted <c>minStepChange</c> parses to zero, i.e. to the fallback.</summary>
        [Fact]
        public void AnOmittedConfigFieldParsesToTheFallback()
        {
            var file = new AnomalyGuardConfigFile
            {
                Prometheus = "http://localhost:9090",
                Thresholds =
                {
                    ["GcGen2HeapBytes"] = new AnomalyGuardConfigFile.ThresholdEntry
                    {
                        MinTrendChange = "3.93MB",
                    },
                },
            };

            var (_, trend, step, _) = AnomalyGuardConfigReader.ReadThresholds(file, out var problems);

            Assert.Empty(problems);
            Assert.Equal(0.0, step[(int)Heap]);

            var floors = new ConfiguredFloorSource(null, trend, step, null, applyCalibrated: false);

            Assert.Equal(3.93e6, floors.MinAbsoluteLevelShift(Heap));
        }

        private static ConfiguredFloorSource Source(
            (MetricIndex Metric, double Value) trend, (MetricIndex Metric, double Value)? step)
        {
            var trendTable = new double[(int)MetricIndex.Count];
            trendTable[(int)trend.Metric] = trend.Value;

            double[]? stepTable = null;

            if (step is { } configured)
            {
                stepTable = new double[(int)MetricIndex.Count];
                stepTable[(int)configured.Metric] = configured.Value;
            }

            return new ConfiguredFloorSource(null, trendTable, stepTable, null, applyCalibrated: false);
        }
    }
}
