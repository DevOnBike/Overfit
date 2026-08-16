// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using DevOnBike.Overfit.Data.Serialization;

namespace DevOnBike.Overfit.Data.Prepare
{
    /// <summary>
    ///     Fitted parameters of a RobustScalingLayer � median and IQR per column.
    ///     Serialize to JSON and persist after Golden Window training.
    ///     Load at inference time via RobustScalingLayer.ImportParams().
    /// </summary>
    public sealed class ScalerParams
    {
        /// <summary>Median per column. Length = MetricCount.</summary>
        public required float[] Medians
        {
            get; init;
        }

        /// <summary>IQR per column. Length = MetricCount.</summary>
        public required float[] Iqrs
        {
            get; init;
        }

        // OVERFIT040 for the two persistence methods below.
        //
        // THE CONSTRAINT: these move one small JSON document — a median and an IQR per column — once, at the
        // boundary of a fitting run. `SaveToFile` runs after Golden-Window training has finished; `LoadFromFile`
        // runs once at inference-engine construction, before any request exists. Both run on the caller's own
        // thread; no pool thread is behind either and nothing is queued on them.
        //
        // WHAT IS GIVEN UP: both are public API of the shipped `DevOnBike.Overfit` package, and `LoadFromFile`
        // sits under `RobustScalingLayer.ImportParams()`, so a task-returning form would propagate async
        // through the preprocessing surface.
#pragma warning disable OVERFIT040

        /// <summary>Saves params to a JSON file.</summary>
        public void SaveToFile(string path)
        {
            var json = JsonSerializer.Serialize(this, OverfitJsonContext.Default.ScalerParams);

            File.WriteAllText(path, json);
        }

        /// <summary>Loads params from a JSON file.</summary>
        public static ScalerParams LoadFromFile(string path)
        {
            var json = File.ReadAllText(path);

            return JsonSerializer.Deserialize(json, OverfitJsonContext.Default.ScalerParams) ?? throw new OverfitRuntimeException($"Failed to deserialize ScalerParams from {path}.");
        }
#pragma warning restore OVERFIT040
    }
}