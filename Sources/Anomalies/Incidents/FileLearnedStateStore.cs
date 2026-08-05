// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;

namespace DevOnBike.Overfit.Anomalies.Incidents
{
    /// <summary>
    /// The learned state on disk, with the same atomic-write and never-throw behaviour as
    /// <see cref="FileIncidentStore"/>.
    ///
    /// <para>Composition rather than inheritance because <see cref="FileIncidentStore"/> is sealed and should
    /// stay so: the two stores hold different payloads and want identical <i>file</i> semantics, which is a
    /// reason to share an implementation and not a type.</para>
    ///
    /// <para><b>Worth putting on a volume that outlives the pod.</b> Losing it costs days of baseline, and the
    /// guard spends those days quieter than it should be — which is the failure mode that looks like success.</para>
    /// </summary>
    public sealed class FileLearnedStateStore : ILearnedStateStore
    {
        private readonly FileIncidentStore _file;

        public FileLearnedStateStore(string path)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(path);

            _file = new FileIncidentStore(path);
        }

        /// <inheritdoc/>
        public string? LastError => _file.LastError;

        /// <inheritdoc/>
        public string? Load() => _file.Load();

        /// <inheritdoc/>
        public void Save(string state) => _file.Save(state);
    }
}
