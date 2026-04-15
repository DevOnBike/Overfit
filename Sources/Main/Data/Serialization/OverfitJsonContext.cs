// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json.Serialization;
using DevOnBike.Overfit.Data.Prepare;
using DevOnBike.Overfit.Diagnostics.Contracts;
using DevOnBike.Overfit.Statistical;

namespace DevOnBike.Overfit.Data.Serialization
{
    [JsonSourceGenerationOptions(WriteIndented = false, GenerationMode = JsonSourceGenerationMode.Default)]
    [JsonSerializable(typeof(KernelDiagnosticEvent))]
    [JsonSerializable(typeof(ModuleDiagnosticEvent))]
    [JsonSerializable(typeof(GraphDiagnosticEvent))]
    [JsonSerializable(typeof(AllocationDiagnosticEvent))]
    [JsonSerializable(typeof(ScalerParams))]
    [JsonSerializable(typeof(HmmParams))]
    internal partial class OverfitJsonContext : JsonSerializerContext
    {

    }
}