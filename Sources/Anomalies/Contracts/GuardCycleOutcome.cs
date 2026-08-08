// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// What one evaluation cycle ended as, and — only when it completed — what it decided.
    ///
    /// <para><b>The result is not reachable without asking the kind first, and that is the whole design.</b>
    /// The obvious shape for this type is a <see cref="GuardCycleKind"/> beside a public
    /// <see cref="GuardCycleResult"/> field, and it recreates the bug it was written to fix one field over: a
    /// blind or failed cycle would carry an all-zero result, which is indistinguishable from a healthy cycle
    /// that found nothing to any caller who forgot the check. So there is no property to read — only
    /// <see cref="TryGetResult"/>, which answers <c>false</c> for anything but
    /// <see cref="GuardCycleKind.Completed"/>.</para>
    ///
    /// <para>Value equality is implemented so two replays of the same window sequence can be compared
    /// outcome-for-outcome; a run that went blind where another completed is then a difference, not a pair of
    /// matching zeros.</para>
    /// </summary>
    public readonly struct GuardCycleOutcome : IEquatable<GuardCycleOutcome>
    {
        private readonly GuardCycleResult _result;

        private GuardCycleOutcome(GuardCycleKind kind, in GuardCycleResult result)
        {
            Kind = kind;
            _result = result;
        }

        /// <summary>How the cycle ended.</summary>
        public GuardCycleKind Kind
        {
            get;
        }

        /// <summary>No pod reported anything the source could see, so nothing was evaluated.</summary>
        public static GuardCycleOutcome Blind => new(GuardCycleKind.Blind, default);

        /// <summary>The cycle threw and was skipped; it is logged and counted where it happened.</summary>
        public static GuardCycleOutcome Failed => new(GuardCycleKind.Failed, default);

        /// <summary>A cycle that evaluated a window, carrying what it decided.</summary>
        /// <param name="result">The cycle's counts.</param>
        public static GuardCycleOutcome Completed(in GuardCycleResult result)
        {
            return new GuardCycleOutcome(GuardCycleKind.Completed, result);
        }

        /// <summary>
        /// Hands back what the cycle decided, or <c>false</c> when it decided nothing.
        /// </summary>
        /// <param name="result">
        /// The cycle's counts when this returns <c>true</c>; <c>default</c> otherwise, which is a
        /// placeholder rather than a report of a quiet cluster.
        /// </param>
        /// <returns><c>true</c> only for <see cref="GuardCycleKind.Completed"/>.</returns>
        public bool TryGetResult(out GuardCycleResult result)
        {
            if (Kind != GuardCycleKind.Completed)
            {
                result = default;

                return false;
            }

            result = _result;

            return true;
        }

        /// <inheritdoc/>
        public bool Equals(GuardCycleOutcome other)
        {
            return Kind == other.Kind && _result.Equals(other._result);
        }

        /// <inheritdoc/>
        public override bool Equals(object? obj)
        {
            return obj is GuardCycleOutcome other && Equals(other);
        }

        /// <inheritdoc/>
        public override int GetHashCode()
        {
            return HashCode.Combine(Kind, _result);
        }

        /// <summary>Value equality.</summary>
        /// <param name="left">Left operand.</param>
        /// <param name="right">Right operand.</param>
        /// <returns>Whether both ended the same way with the same counts.</returns>
        public static bool operator ==(GuardCycleOutcome left, GuardCycleOutcome right)
        {
            return left.Equals(right);
        }

        /// <summary>Value inequality.</summary>
        /// <param name="left">Left operand.</param>
        /// <param name="right">Right operand.</param>
        /// <returns>Whether the two differ in kind or in counts.</returns>
        public static bool operator !=(GuardCycleOutcome left, GuardCycleOutcome right)
        {
            return !left.Equals(right);
        }

        /// <summary>A short form for a diagnostic line: the kind, plus the counts when there are any.</summary>
        /// <returns>Text naming the kind and, for a completed cycle, its counts.</returns>
        public override string ToString()
        {
            if (Kind != GuardCycleKind.Completed)
            {
                return Kind.ToString();
            }

            return $"{Kind} ({_result})";
        }
    }
}
