// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json.Serialization;

namespace DevOnBike.Overfit.Tests.Diagnostics.Tracing
{
    internal sealed class DiagnosticsTraceEntry
    {
        [JsonPropertyName("count")]
        public long Count { get; set; }

        [JsonPropertyName("durationMs")]
        public double DurationMs { get; set; }

        [JsonPropertyName("allocatedBytes")]
        public long AllocatedBytes { get; set; }
    }
}
