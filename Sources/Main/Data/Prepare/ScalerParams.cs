// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;

namespace DevOnBike.Overfit.Data.Prepare
{
    /// <summary>
    /// Fitted parameters of a RobustScalingLayer � median and IQR per column.
    /// Serialize to JSON and persist after Golden Window training.
    /// Load at inference time via RobustScalingLayer.ImportParams().
    /// </summary>
    public sealed class ScalerParams
    {
        /// <summary>Median per column. Length = MetricCount.</summary>
        public required float[] Medians { get; init; }

        /// <summary>IQR per column. Length = MetricCount.</summary>
        public required float[] Iqrs { get; init; }

        /// <summary>Saves params to a JSON file.</summary>
        public void SaveToFile(string path)
        {
            var json = JsonSerializer.Serialize(this, new JsonSerializerOptions
            {
                WriteIndented = true
            });

            File.WriteAllText(path, json);
        }

        /// <summary>Loads params from a JSON file.</summary>
        public static ScalerParams LoadFromFile(string path)
        {
            var json = File.ReadAllText(path);

            return JsonSerializer.Deserialize<ScalerParams>(json) ?? throw new InvalidOperationException($"Failed to deserialize ScalerParams from {path}.");
        }
    }
}