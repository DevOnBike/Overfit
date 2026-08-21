// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// Runs a set of arms ABAB — one repetition of every arm, then the next repetition — rather than all
    /// of A and then all of B. Drift over the sitting then falls on every arm equally instead of wearing
    /// the costume of the variable (plan, section 3.5 rule 5).
    /// <para>
    /// The warm-up phase is interleaved for the same reason, and it runs until <see cref="WarmupPolicy"/>
    /// says every arm has stopped moving rather than for a fixed count. An arm that has already settled
    /// keeps running to the end of the phase: dropping it would change what the other arms are
    /// interleaved against half way through the measurement.
    /// </para>
    /// <para>
    /// <b>The live view, when there is one, is frozen for both phases and repaints only between them.</b>
    /// That is measured rather than assumed: with a repaint between every round, the C3 host arm's median
    /// moved by 12 % to 42 % across three runs of <c>--live-perturbation --quick</c>, and the host arms are
    /// the baseline every device ratio is divided by. The repaint is not concurrent with the arm - nothing
    /// here is threaded - so what it costs is the state it leaves behind: a 2 MiB F32 weight evicted from
    /// cache by a frame's worth of allocation, which is also why the quantized arm C1, whose weight is
    /// eight times smaller, is unaffected at 0.970 to 0.983.
    /// </para>
    /// </summary>
    internal static class ArmRunner
    {
        public static ArmRunResult Interleave(
            IReadOnlyList<Arm> arms,
            WarmupPolicy policy,
            int reps,
            Action<string>? log = null,
            LiveView? view = null)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(reps, 5);

            var warm = Warm(arms, policy, log, view, out var rounds, out var stopReason);

            var samples = new Dictionary<string, List<double>>(arms.Count);
            foreach (var arm in arms)
            {
                samples[arm.Name] = new List<double>(reps);
            }

            // Painted here, between the two phases, and then held still until every repetition is done.
            view?.FreezeFor("timing - the display holds still until this phase ends, so that a repaint " +
                            "cannot move the numbers it is about to show");

            for (var r = 0; r < reps; r++)
            {
                foreach (var arm in arms)
                {
                    samples[arm.Name].Add(arm.TimeOnce());
                }
            }

            view?.Resume();

            var timings = new Dictionary<string, Measurement>(arms.Count);
            foreach (var arm in arms)
            {
                timings[arm.Name] = new Measurement(arm.Name, samples[arm.Name]);
            }

            return new ArmRunResult(timings, warm, rounds, stopReason);
        }

        /// <summary>
        /// The untimed phase. The first ILGPU launch of a kernel includes PTX or OpenCL C generation and
        /// the driver's compile of it, the host arms start against a cold cache, and a device that has
        /// been idle starts clocked down. All three decay over repetitions and none of them decays on a
        /// schedule this probe can know in advance, which is why the count is measured and not chosen.
        /// </summary>
        private static IReadOnlyDictionary<string, WarmupOutcome> Warm(
            IReadOnlyList<Arm> arms,
            WarmupPolicy policy,
            Action<string>? log,
            LiveView? view,
            out int rounds,
            out string stopReason)
        {
            // Frozen for this phase as well as for the timed one, and for a second reason on top of the
            // measured perturbation: these readings feed the stopping rule and its noise band. A repaint
            // between rounds widens that band, and a wider band accepts a drift it was built to reject.
            view?.FreezeFor("warming up - the display holds still until the arms have settled");

            var history = new Dictionary<string, List<double>>(arms.Count);
            foreach (var arm in arms)
            {
                history[arm.Name] = new List<double>(policy.MinRounds);
            }

            var firstSettledRound = new Dictionary<string, int>(arms.Count);
            var started = Stopwatch.GetTimestamp();
            var round = 0;
            stopReason = string.Empty;

            // BOUND: at most policy.MaxRounds iterations. The three exits below are checked in an order
            // that guarantees termination: the cap is unconditional and is tested last.
            while (stopReason.Length == 0)
            {
                round++;
                foreach (var arm in arms)
                {
                    history[arm.Name].Add(arm.TimeOnce());
                }

                var everySettled = round >= policy.MinRounds;
                foreach (var arm in arms)
                {
                    if (!IsSettled(history[arm.Name], policy))
                    {
                        everySettled = false;
                        continue;
                    }

                    if (round >= policy.MinRounds && !firstSettledRound.ContainsKey(arm.Name))
                    {
                        firstSettledRound[arm.Name] = round;
                    }
                }

                if (everySettled)
                {
                    stopReason = $"every arm satisfied the stopping rule at round {round}";
                    continue;
                }

                if (round >= policy.MinRounds &&
                    Stopwatch.GetElapsedTime(started).TotalMilliseconds >= policy.BudgetMs)
                {
                    stopReason =
                        $"the {policy.BudgetMs / 1000:F0} s warm-up budget for this cell ran out at round {round} " +
                        "before every arm settled. Raise --warmup-budget-ms.";
                    continue;
                }

                if (round >= policy.MaxRounds)
                {
                    stopReason =
                        $"the cap of {policy.MaxRounds} warm-up rounds was reached before every arm settled. " +
                        "Raise --warmup-max.";
                    continue;
                }

                // The budget above is only checked once MinRounds have run, so that the stopping rule is
                // always evaluable. On a slow device those first rounds alone can take far longer than the
                // budget, so this is the limit that actually bounds the phase. An arm stopped here reports
                // NOT SETTLED, which suppresses the headline - the honest outcome, and a bounded one.
                if (Stopwatch.GetElapsedTime(started).TotalMilliseconds >= 10 * policy.BudgetMs)
                {
                    stopReason =
                        $"the hard limit of {10 * policy.BudgetMs / 1000:F0} s stopped the warm-up at round {round}, " +
                        $"before the minimum of {policy.MinRounds} rounds could finish. This cell is too slow for " +
                        "the budget. Raise --warmup-budget-ms.";
                }
            }

            if (view is not null)
            {
                view.State.WarmupRound = round;
            }

            log?.Invoke($"  warm-up: {round} rounds - {stopReason}");

            var outcomes = new Dictionary<string, WarmupOutcome>(arms.Count);
            foreach (var arm in arms)
            {
                var readings = history[arm.Name];
                var (previous, last) = WindowMedians(readings, policy);
                outcomes[arm.Name] = new WarmupOutcome(
                    arm.Name,
                    round,
                    IsSettled(readings, policy),
                    firstSettledRound.GetValueOrDefault(arm.Name),
                    previous,
                    last,
                    readings.Count >= 2 * policy.WindowSize ? NoiseBand(readings, policy) : 0);
            }

            rounds = round;
            return outcomes;
        }

        /// <summary>
        /// The stopping rule itself, in one place: the median of the last window against the median of
        /// the window before it. An arm is settled when that move is inside the tolerance, OR inside the
        /// scatter of the readings themselves.
        /// <para>
        /// The second clause is not a loophole, it is the difference between a rule and a rule that
        /// cannot terminate. A trend smaller than a measurement's own noise is not resolvable by that
        /// measurement, so demanding it produces an arm that warms for ever and is then reported as
        /// "did not settle" whatever the code does. Measured on 2026-08-21 with the tolerance clause
        /// alone: the C3 host arm of quick_wide at n=16 runs at about 0.09 ms, and after 100 rounds its
        /// two window medians were 0.099 and 0.078 ms - 21.6 % apart - while its timed repetitions
        /// spread over 200 %. It had warmed long before; the criterion was simply below the noise.
        /// </para>
        /// </summary>
        private static bool IsSettled(IReadOnlyList<double> readings, WarmupPolicy policy)
        {
            if (readings.Count < 2 * policy.WindowSize)
            {
                return false;
            }

            var (previous, last) = WindowMedians(readings, policy);
            if (previous <= 0)
            {
                return false;
            }

            var move = Math.Abs(last - previous);
            return move <= policy.Tolerance * previous || move <= NoiseBand(readings, policy);
        }

        /// <summary>
        /// How far two window medians of the SAME stationary arm are expected to sit apart, in ms.
        /// <para>
        /// Built from the readings rather than assumed. The median absolute deviation estimates the
        /// standard deviation as <c>1.4826 * MAD</c>; the standard error of a median of <c>W</c> samples
        /// is about <c>1.253 * sigma / sqrt(W)</c>; two independent medians differ with
        /// <c>sqrt(2)</c> times that; and a two-sigma band doubles it once more. The product of those
        /// four factors is 5.25, which is the only constant here and is arithmetic, not a fitted value.
        /// </para>
        /// <para>
        /// The scatter is pooled from deviations about EACH WINDOW'S OWN median, never about the median
        /// of both windows together. Measured on 2026-08-21, and it is the difference between a guard
        /// and a rubber stamp: an arm mutated to take one extra pass every four calls - a drift of about
        /// 100 % between the two windows - was reported SETTLED at round 10 and printed a headline. A
        /// combined median puts the drift itself into the deviations, so the drift inflates the very
        /// band that is supposed to reject it. Per-window medians remove the between-window step first,
        /// which is exactly the quantity under test.
        /// </para>
        /// </summary>
        private static double NoiseBand(IReadOnlyList<double> readings, WarmupPolicy policy)
        {
            const double TwoSigmaOfMedianDifference = 5.25;

            var w = policy.WindowSize;
            var deviations = new double[2 * w];
            var offset = readings.Count - (2 * w);

            for (var half = 0; half < 2; half++)
            {
                var start = offset + (half * w);
                var centre = Median(readings, start, w);
                for (var i = 0; i < w; i++)
                {
                    deviations[(half * w) + i] = Math.Abs(readings[start + i] - centre);
                }
            }

            Array.Sort(deviations);
            var mad = Median(deviations, 0, deviations.Length);

            return TwoSigmaOfMedianDifference * mad / Math.Sqrt(w);
        }

        private static (double Previous, double Last) WindowMedians(
            IReadOnlyList<double> readings,
            WarmupPolicy policy)
        {
            if (readings.Count < 2 * policy.WindowSize)
            {
                return (0, 0);
            }

            var w = policy.WindowSize;
            return (Median(readings, readings.Count - 2 * w, w), Median(readings, readings.Count - w, w));
        }

        private static double Median(IReadOnlyList<double> readings, int offset, int count)
        {
            var window = new double[count];
            for (var i = 0; i < count; i++)
            {
                window[i] = readings[offset + i];
            }

            Array.Sort(window);
            return count % 2 == 1
                ? window[count / 2]
                : 0.5 * (window[count / 2 - 1] + window[count / 2]);
        }
    }
}
