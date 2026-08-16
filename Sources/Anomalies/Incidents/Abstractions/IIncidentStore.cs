// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Incidents.Abstractions
{
    /// <summary>
    /// Where the tracker's open incidents survive a restart.
    ///
    /// <para><b>Without one, a rolling update reopens every incident that was running.</b> The tracker is the
    /// piece that turns twelve notifications about one hour-long problem into one, and it holds that entirely
    /// in memory — so the first deploy of the guard itself undoes the work, and does it at the worst moment,
    /// when somebody is already looking at a change.</para>
    ///
    /// <para><b>A failure here must never fail a cycle.</b> Detection that stops because a disk is full has
    /// replaced the problem it was bought to detect. An unreadable or absent state is a cold start — correct,
    /// just noisier — and a failed save costs one restart's worth of identity, not the run.</para>
    /// </summary>
    public interface IIncidentStore
    {
        /// <summary>
        /// What went wrong on the last <see cref="Load"/> or <see cref="Save"/>, or <c>null</c> when it
        /// succeeded. Cleared on success, so it describes the current state rather than the run's history.
        ///
        /// <para><b>On the interface, not just on the implementation, because otherwise nobody can read
        /// it.</b> <c>FileIncidentStore</c> has recorded this since it was written; the guard holds the
        /// interface, so it could not see the field and no caller ever asked. A recorded reason nobody reads
        /// is not a report — the failure stayed exactly as silent as if it had never been captured, which is
        /// the whole condition these two methods promise not to create by swallowing their exceptions.</para>
        ///
        /// <para>Implementations that cannot fail return <c>null</c> always, which is honest: there is
        /// nothing to report.</para>
        /// </summary>
        string? LastError
        {
            get;
        }

        /// <summary>
        /// Reads the last saved state, or <c>null</c> when there is none or it cannot be read. Never throws
        /// for a missing or corrupt store: starting cold is a worse outcome than not starting at all only if
        /// the alternative were correct, and here it is not.
        ///
        /// <para>A <c>null</c> return is ambiguous by design — no state saved yet, or state that could not
        /// be read — and <see cref="LastError"/> is what separates the two. A cold start on an empty volume
        /// and a cold start on an unreadable one call for very different responses.</para>
        /// </summary>
        string? Load();

        /// <summary>
        /// Persists the state. Implementations should write atomically — a state file truncated by a kill
        /// mid-write is worse than none, because it looks readable.
        /// </summary>
        void Save(string state);
    }
}
