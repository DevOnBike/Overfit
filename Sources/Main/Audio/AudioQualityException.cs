// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio
{
    /// <summary>
    /// Thrown by <see cref="AudioQualityAssert"/> when a generated waveform fails an objective-similarity gate
    /// (SNR / correlation / mel distance vs. a reference). Framework-agnostic: any test runner treats a thrown
    /// exception as a failure, and the actionable detail (which metric, the measured value, the threshold) lives
    /// in the message.
    /// </summary>
    public sealed class AudioQualityException : OverfitException
    {
        public AudioQualityException(string message)
            : base(message)
        {
        }
    }
}
