// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    /// <summary>
    /// The declared windows from configuration — the default calendar, and the one that needs nothing but a
    /// ConfigMap.
    ///
    /// <para>It keeps the promise the whole subsystem is sold on: the only dependency is an HTTP route to
    /// Prometheus. A calendar backed by a deployment pipeline is strictly better informed and strictly more
    /// to install, so it is an option rather than the default.</para>
    /// </summary>
    public sealed class StaticMaintenanceCalendar : IMaintenanceCalendar
    {
        private readonly IReadOnlyList<MaintenanceWindow> _windows;

        public StaticMaintenanceCalendar(IReadOnlyList<MaintenanceWindow> windows)
        {
            ArgumentNullException.ThrowIfNull(windows);

            _windows = windows;
        }

        /// <inheritdoc/>
        public bool IsDeclaredAbnormal(DateTimeOffset at, string workload, out string reason)
        {
            ArgumentNullException.ThrowIfNull(workload);

            for (var i = 0; i < _windows.Count; i++)
            {
                if (!_windows[i].Covers(at, workload))
                {
                    continue;
                }

                // Never empty on a true return — an unnamed window still has to say that one existed.
                reason = _windows[i].Reason.Length > 0
                    ? _windows[i].Reason
                    : "declared maintenance window";

                return true;
            }

            reason = string.Empty;

            return false;
        }
    }
}
