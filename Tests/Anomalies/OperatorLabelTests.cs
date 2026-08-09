// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// The half of operator feedback that pulls against the other half.
    ///
    /// <para><b>Every other response to "this was noise" makes the guard quieter.</b> Suppression rules,
    /// labels folded into calibration, thresholds nudged upward — all one-signed. A hundred honest dismissals
    /// then converge on a detector that reports nothing, and it arrives gradually enough that nobody notices
    /// the day it stopped working, which is the exact failure this subsystem exists to remove. A
    /// <c>--real</c> label is the constraint that stops it, so these tests are about a silence that must NOT
    /// be reachable rather than about a feature that works.</para>
    /// </summary>
    public sealed class OperatorLabelTests
    {
        private static readonly DateTimeOffset T0 = new(2026, 8, 2, 12, 0, 0, TimeSpan.Zero);

        /// <summary>The property everything else leans on.</summary>
        [Fact]
        public void AConfirmedFindingCannotBeSilencedByALaterProposal()
        {
            var calibrator = Fitted();
            var uncapped = calibrator.Propose()[(int)MetricIndex.GcGen2HeapBytes];

            Assert.True(uncapped.IsUsable);
            Assert.False(uncapped.WasCapped);

            // An operator confirms a finding SMALLER than what the healthy period suggested was normal.
            var confirmed = uncapped.ProposedMinAbsoluteGap / 2.0;
            var labels = new OperatorLabelStore();

            labels.Add(new OperatorLabel(
                42, nameof(MetricIndex.GcGen2HeapBytes), OperatorLabelKind.Real, confirmed, T0, "the leak"));

            calibrator.UseLabels(labels);

            var capped = calibrator.Propose()[(int)MetricIndex.GcGen2HeapBytes];

            Assert.True(capped.WasCapped);
            Assert.True(capped.ProposedMinAbsoluteGap < confirmed,
                $"proposed {capped.ProposedMinAbsoluteGap} would gate out a confirmed {confirmed}");
        }

        [Fact]
        public void ANoiseLabelConstrainsNothing()
        {
            var calibrator = Fitted();
            var before = calibrator.Propose()[(int)MetricIndex.GcGen2HeapBytes];
            var labels = new OperatorLabelStore();

            labels.Add(new OperatorLabel(
                1, nameof(MetricIndex.GcGen2HeapBytes), OperatorLabelKind.Noise, 1.0, T0, "sawtooth"));

            calibrator.UseLabels(labels);

            var after = calibrator.Propose()[(int)MetricIndex.GcGen2HeapBytes];

            Assert.False(after.WasCapped);
            Assert.Equal(before.ProposedMinAbsoluteGap, after.ProposedMinAbsoluteGap, 6);
        }

        /// <summary>A label about one signal must not quieten the calibrator on another.</summary>
        [Fact]
        public void ALabelOnOneSignalDoesNotConstrainAnother()
        {
            var calibrator = Fitted();
            var labels = new OperatorLabelStore();

            labels.Add(new OperatorLabel(
                7, nameof(MetricIndex.CpuUsageRatio), OperatorLabelKind.Real, 1e-9, T0, "tiny but real"));

            calibrator.UseLabels(labels);

            Assert.False(calibrator.Propose()[(int)MetricIndex.GcGen2HeapBytes].WasCapped);
            Assert.True(calibrator.Propose()[(int)MetricIndex.CpuUsageRatio].WasCapped);
        }

        /// <summary>
        /// Labels have to survive a restart with the rest of the learned state, or the constraint lasts until
        /// the next rolling update of the monitoring tool and no longer.
        /// </summary>
        [Fact]
        public void LabelsSurviveTheLearnedStateRoundTrip()
        {
            var calibrator = Fitted();
            var labels = new OperatorLabelStore();

            labels.Add(new OperatorLabel(
                42, nameof(MetricIndex.GcGen2HeapBytes), OperatorLabelKind.Real, 1234.5, T0,
                "tab\there and newline\nthere"));

            var payload = LearnedState.Write(new MetricHistory(), calibrator, labels);
            var (_, restoredCalibrator, restoredLabels, _) = LearnedState.Read(payload);

            Assert.Equal(1, restoredLabels.Count);
            Assert.Equal(1234.5, restoredLabels.SmallestRealMagnitude(nameof(MetricIndex.GcGen2HeapBytes)), 6);
            Assert.Equal("tab\there and newline\nthere", restoredLabels.Labels[0].Reason);

            // And the restored calibrator is already wired to them — a store that survives but is not
            // consulted is the same as no store at all.
            Assert.True(restoredCalibrator.Propose()[(int)MetricIndex.GcGen2HeapBytes].WasCapped
                        || !restoredCalibrator.Propose()[(int)MetricIndex.GcGen2HeapBytes].IsUsable);
        }

        /// <summary>
        /// The store is bounded, and what it drops matters: forgetting a dismissal costs a repeated alert,
        /// forgetting a confirmation costs the ability to see something an operator said was real.
        /// </summary>
        [Fact]
        public void EvictionDropsDismissalsBeforeConfirmations()
        {
            var labels = new OperatorLabelStore();

            labels.Add(new OperatorLabel(
                1, "signal", OperatorLabelKind.Real, 5.0, T0, "keep me"));

            for (var i = 0; i < OperatorLabelStore.MaxLabels + 50; i++)
            {
                labels.Add(new OperatorLabel(
                    i + 2, "signal", OperatorLabelKind.Noise, 1.0, T0.AddMinutes(i), "noise"));
            }

            Assert.Equal(OperatorLabelStore.MaxLabels, labels.Count);
            Assert.Equal(5.0, labels.SmallestRealMagnitude("signal"), 6);
        }

        [Fact]
        public void WithNoLabelsThereIsNoConstraint()
        {
            Assert.Equal(double.PositiveInfinity, new OperatorLabelStore().SmallestRealMagnitude("anything"));
        }

        /// <summary>
        /// Twelve replicas doing nothing wrong, observed long enough for a proposal to be usable.
        ///
        /// <para>Bound to <see cref="FloorProposal.MinimumWindows"/> rather than a literal, because the
        /// sentence above is a claim about the gate and a literal lets the two drift: this helper said
        /// "long enough" while observing twenty windows against a minimum of twenty-four.</para>
        /// </summary>
        private static FloorCalibrator Fitted()
        {
            var calibrator = new FloorCalibrator();

            for (var cycle = 0; cycle < FloorProposal.MinimumWindows; cycle++)
            {
                var names = new List<string>(12);

                for (var p = 0; p < 12; p++)
                {
                    names.Add($"pod-{p}");
                }

                var window = new MetricWindow(
                    names, 80, T0.AddMinutes(5 * cycle), TimeSpan.FromSeconds(15));
                var rng = new Random(20260802 + cycle);

                for (var pod = 0; pod < names.Count; pod++)
                {
                    var heap = window.Series(pod, MetricIndex.GcGen2HeapBytes);
                    var cpu = window.Series(pod, MetricIndex.CpuUsageRatio);
                    var heapLevel = 4.0e6 * (1.0 + ((rng.NextDouble() - 0.5) * 0.5));
                    var cpuLevel = 0.02 * (1.0 + ((rng.NextDouble() - 0.5) * 0.5));

                    for (var i = 0; i < window.Length; i++)
                    {
                        heap[i] = heapLevel * (1.0 + ((rng.NextDouble() - 0.5) * 0.04));
                        cpu[i] = cpuLevel * (1.0 + ((rng.NextDouble() - 0.5) * 0.06));
                    }
                }

                calibrator.Observe(window);
            }

            return calibrator;
        }
    }
}
