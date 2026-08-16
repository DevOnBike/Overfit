// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// How one evaluation cycle ended.
    ///
    /// <para><b>Three outcomes that used to be two values and are not the same finding.</b> A cycle that
    /// evaluated a window, a cycle whose cluster returned nothing at all, and a cycle that threw and was
    /// skipped were all reported to a caller as one <c>null</c>. They are already logged as three different
    /// events, so the loop knew the difference and only the return value did not — which mattered the moment
    /// something other than a log reader started collecting results. A replay whose cycles were blind and a
    /// replay whose cycles crashed can produce the identical incident-count sequence, and reading that
    /// sequence as "the cluster was quiet" is wrong in both cases.</para>
    /// </summary>
    public enum GuardCycleKind
    {
        /// <summary>
        /// No window came back: no pod reported anything the source could see.
        ///
        /// <para><b>Deliberately the zero value, so a default-constructed
        /// <see cref="GuardCycleOutcome"/> cannot read as a clean cycle that found nothing.</b> That
        /// substitution — an all-zero result standing in for "nothing ran" — is the exact ambiguity this
        /// enum exists to remove, and leaving <see cref="Completed"/> at zero would have reintroduced it
        /// through an uninitialised array.</para>
        /// </summary>
        Blind = 0,

        /// <summary>The cycle evaluated a window and decided something. The only kind that carries a result.</summary>
        Completed = 1,

        /// <summary>The cycle threw and was skipped. Logged and counted; the loop continues.</summary>
        Failed = 2,
    }
}
