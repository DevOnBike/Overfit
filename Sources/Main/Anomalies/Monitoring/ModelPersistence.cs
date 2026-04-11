// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.DeepLearning;

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    /// <summary>
    /// Saves and loads the complete anomaly detection model state — autoencoder
    /// weights plus scorer threshold — in a single binary file.
    ///
    /// File format (.ovfw):
    ///
    ///   [0..3]  Magic: 0x4F564657 ("OVFW")
    ///   [4..5]  Version: uint16
    ///   [6..9]  InputSize: int32
    ///   [10..13] Hidden1: int32
    ///   [14..17] Hidden2: int32
    ///   [18..21] BottleneckDim: int32
    ///   [22..25] Threshold: float32
    ///   [26..33] SavedAt: int64 (UTC ticks)
    ///   [34..N]  Label: length-prefixed UTF-8 string
    ///   [N+1..]  Autoencoder weights (AnomalyAutoencoder.Save format)
    ///   [..]     Scorer state (ReconstructionScorer.Save format)
    ///
    /// Usage:
    /// <code>
    ///   // After offline training:
    ///   ModelPersistence.Save("anomaly-detector.ovfw", autoencoder, scorer, label: "v1.2");
    ///
    ///   // On service startup:
    ///   var (autoencoder, scorer) = ModelPersistence.Load("anomaly-detector.ovfw");
    ///   autoencoder.Eval();
    ///   pipeline = AnomalyDetectionPipeline.Create(source, config, autoencoder, scorer, ...);
    /// </code>
    /// </summary>
    public static class ModelPersistence
    {
        // -------------------------------------------------------------------------
        // Save
        // -------------------------------------------------------------------------

        /// <summary>
        /// Saves the autoencoder and scorer to a single binary file.
        /// Creates or overwrites the file atomically using a temp-file swap
        /// to protect against partial writes on crash.
        /// </summary>
        /// <param name="path">Destination file path. Extension ".ovfw" recommended.</param>
        /// <param name="autoencoder">Trained autoencoder. Should be in Eval mode.</param>
        /// <param name="scorer">Calibrated scorer. Must have IsCalibrated == true.</param>
        /// <param name="label">
        ///   Human-readable label stored in the header (e.g. "prod-v2" or "pod-A-2026-04").
        ///   Useful for auditing which model is deployed. Truncated to 255 bytes if longer.
        /// </param>
        /// <exception cref="InvalidOperationException">When scorer is not calibrated.</exception>
        public static void Save(
            string path,
            AnomalyAutoencoder autoencoder,
            ReconstructionScorer scorer,
            string label = "")
        {
            ArgumentNullException.ThrowIfNull(path);
            ArgumentNullException.ThrowIfNull(autoencoder);
            ArgumentNullException.ThrowIfNull(scorer);

            if (!scorer.IsCalibrated)
            {
                throw new InvalidOperationException(
                    "Cannot save an uncalibrated scorer. " +
                    "Run OfflineTrainingJob.Run() or ReconstructionScorer.Calibrate() first.");
            }

            // Write to a temp file first — swap on success to protect against partial writes
            var dir = Path.GetDirectoryName(Path.GetFullPath(path)) ?? ".";
            var tempPath = Path.Combine(dir, $".{Path.GetFileName(path)}.tmp");

            try
            {
                using (var fs = new FileStream(tempPath, FileMode.Create, FileAccess.Write))
                using (var bw = new BinaryWriter(fs, System.Text.Encoding.UTF8))
                {
                    WriteHeader(bw, autoencoder, scorer, label);
                    autoencoder.Save(bw);
                    scorer.Save(bw);
                }

                File.Move(tempPath, path, overwrite: true);
            }
            catch
            {
                // Clean up temp file on failure so disk is not polluted
                if (File.Exists(tempPath)) { File.Delete(tempPath); }
                throw;
            }
        }

        /// <summary>
        /// Overload that writes to a pre-opened <see cref="BinaryWriter"/>.
        /// Used for embedding the bundle inside a larger file or network stream.
        /// Temp-file swap is not possible here — the caller must handle atomicity.
        /// </summary>
        public static void Save(
            BinaryWriter bw,
            AnomalyAutoencoder autoencoder,
            ReconstructionScorer scorer,
            string label = "")
        {
            ArgumentNullException.ThrowIfNull(bw);
            ArgumentNullException.ThrowIfNull(autoencoder);
            ArgumentNullException.ThrowIfNull(scorer);

            if (!scorer.IsCalibrated)
            {
                throw new InvalidOperationException("Cannot save uncalibrated scorer.");
            }

            WriteHeader(bw, autoencoder, scorer, label);
            autoencoder.Save(bw);
            scorer.Save(bw);
        }

        // -------------------------------------------------------------------------
        // Load
        // -------------------------------------------------------------------------

        /// <summary>
        /// Loads the autoencoder and scorer from a binary file written by <see cref="Save"/>.
        ///
        /// The returned autoencoder is left in Training mode.
        /// Call <c>autoencoder.Eval()</c> before inference.
        /// </summary>
        /// <returns>
        ///   A tuple of the loaded autoencoder (Training mode) and scorer (calibrated).
        /// </returns>
        /// <exception cref="InvalidDataException">
        ///   When the file has an invalid magic number or version.
        /// </exception>
        /// <exception cref="FileNotFoundException">When the file does not exist.</exception>
        public static (AnomalyAutoencoder Autoencoder, ReconstructionScorer Scorer) Load(
            string path)
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException($"Model bundle not found: {path}");
            }

            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
            using var br = new BinaryReader(fs, System.Text.Encoding.UTF8);

            return Load(br);
        }

        /// <summary>
        /// Overload that reads from a pre-opened <see cref="BinaryReader"/>.
        /// </summary>
        public static (AnomalyAutoencoder Autoencoder, ReconstructionScorer Scorer) Load(
            BinaryReader br)
        {
            ArgumentNullException.ThrowIfNull(br);

            var header = ReadHeader(br);

            var autoencoder = new AnomalyAutoencoder(
                header.InputSize,
                header.Hidden1,
                header.Hidden2,
                header.BottleneckDim);

            autoencoder.Load(br);

            var scorer = new ReconstructionScorer();
            scorer.Load(br);

            return (autoencoder, scorer);
        }

        // -------------------------------------------------------------------------
        // Header inspection
        // -------------------------------------------------------------------------

        /// <summary>
        /// Reads only the header of a bundle file without instantiating the model.
        /// Useful for deployment tooling that needs to inspect what is deployed.
        /// </summary>
        public static ModelBundleHeader ReadHeader(string path)
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException($"Model bundle not found: {path}");
            }

            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
            using var br = new BinaryReader(fs, System.Text.Encoding.UTF8);
            return ReadHeader(br);
        }

        // -------------------------------------------------------------------------
        // Private
        // -------------------------------------------------------------------------

        private static void WriteHeader(
            BinaryWriter bw,
            AnomalyAutoencoder autoencoder,
            ReconstructionScorer scorer,
            string label)
        {
            bw.Write(ModelBundleHeader.Magic);
            bw.Write(ModelBundleHeader.Version);
            bw.Write(autoencoder.InputSize);
            bw.Write(autoencoder.Hidden1);
            bw.Write(autoencoder.Hidden2);
            bw.Write(autoencoder.BottleneckDim);
            bw.Write(scorer.Threshold);
            bw.Write(DateTime.UtcNow.Ticks);

            // Label: truncate to 255 bytes to keep header bounded
            var safeLabel = label.Length > 255 ? label[..255] : label;
            bw.Write(safeLabel);
        }

        private static ModelBundleHeader ReadHeader(BinaryReader br)
        {
            var magic = br.ReadUInt32();
            if (magic != ModelBundleHeader.Magic)
            {
                throw new InvalidDataException(
                    $"Invalid model bundle: expected magic 0x{ModelBundleHeader.Magic:X8}, " +
                    $"got 0x{magic:X8}. Is this an .ovfw file?");
            }

            var version = br.ReadUInt16();
            if (version != ModelBundleHeader.Version)
            {
                throw new InvalidDataException(
                    $"Unsupported model bundle version {version}. " +
                    $"This library supports version {ModelBundleHeader.Version}.");
            }

            return new ModelBundleHeader
            {
                InputSize = br.ReadInt32(),
                Hidden1 = br.ReadInt32(),
                Hidden2 = br.ReadInt32(),
                BottleneckDim = br.ReadInt32(),
                Threshold = br.ReadSingle(),
                SavedAt = new DateTime(br.ReadInt64(), DateTimeKind.Utc),
                Label = br.ReadString()
            };
        }
    }
}