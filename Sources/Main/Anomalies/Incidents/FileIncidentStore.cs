// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.Anomalies.Incidents.Abstractions;

namespace DevOnBike.Overfit.Anomalies.Incidents
{
    /// <summary>
    /// The tracker's state in a file — a mounted volume, an <c>emptyDir</c>, anywhere the process can write.
    ///
    /// <para><b>Written atomically, because a half-written state file is worse than none.</b> A guard killed
    /// mid-write would otherwise leave a truncated file that still parses — some incidents present, others
    /// silently gone — and the failure would look like a tracker bug rather than a storage one. The write
    /// goes to a temporary file and is moved into place, so a reader sees either the previous state or the
    /// new one.</para>
    ///
    /// <para><b>Nothing here throws.</b> Detection that stops because a disk is full has replaced the problem
    /// it was bought to detect; a failed load is a cold start and a failed save costs one restart's worth of
    /// identity. Both are reported through <see cref="LastError"/> for a caller that wants to log them, and
    /// neither is allowed to reach the cycle.</para>
    /// </summary>
    public sealed class FileIncidentStore : IIncidentStore
    {
        private readonly string _path;

        public FileIncidentStore(string path)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(path);

            _path = path;
        }

        /// <summary>What went wrong on the last operation, or <c>null</c>. Cleared on success.</summary>
        public string? LastError
        {
            get; private set;
        }

        /// <inheritdoc/>
        public string? Load()
        {
            try
            {
                if (!File.Exists(_path))
                {
                    LastError = null;

                    return null;
                }

                LastError = null;

                return File.ReadAllText(_path, Encoding.UTF8);
            }
            catch (Exception ex)
            {
                LastError = $"could not read incident state from '{_path}': {ex.Message}";

                return null;
            }
        }

        /// <inheritdoc/>
        public void Save(string state)
        {
            ArgumentNullException.ThrowIfNull(state);

            try
            {
                var directory = Path.GetDirectoryName(_path);

                if (!string.IsNullOrEmpty(directory))
                {
                    Directory.CreateDirectory(directory);
                }

                // Same directory as the target: a move across volumes is a copy, and a copy is not atomic.
                var temporary = _path + ".tmp";

                File.WriteAllText(temporary, state, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
                File.Move(temporary, _path, overwrite: true);

                LastError = null;
            }
            catch (Exception ex)
            {
                LastError = $"could not write incident state to '{_path}': {ex.Message}";
            }
        }
    }
}
