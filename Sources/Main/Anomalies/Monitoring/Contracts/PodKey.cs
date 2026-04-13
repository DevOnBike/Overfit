// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.InteropServices;

namespace DevOnBike.Overfit.Anomalies.Monitoring.Contracts
{
    [StructLayout(LayoutKind.Sequential)]
    public readonly record struct PodKey
    {
        public DataCenter DC { get; init; }

        public string PodName { get; init; }

        public override string ToString()
        {
            return $"{DC}/{PodName}";
        }
    }

}