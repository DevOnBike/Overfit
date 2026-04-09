// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Monitoring
{
    /// <summary>
    /// Metadata written at the start of every model bundle file.
    /// </summary>
    public sealed record ModelBundleHeader
    {
        public const uint Magic = 0x4F564657; // "OVFW" — Overfit
        public const ushort Version = 1;

        public int InputSize { get; init; }
        public int Hidden1 { get; init; }
        public int Hidden2 { get; init; }
        public int BottleneckDim { get; init; }
        public float Threshold { get; init; }
        public DateTime SavedAt { get; init; }
        public string Label { get; init; } = string.Empty;
    }
}