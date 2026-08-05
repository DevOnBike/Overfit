// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Incidents.Abstractions
{
    /// <summary>
    /// Where the seasonal baseline and the floor calibration survive a restart.
    ///
    /// <para><b>A distinct type purely so a container can tell the two stores apart.</b> Its shape is
    /// identical to <see cref="IIncidentStore"/> and it deliberately adds nothing: what differs is which
    /// payload goes where, and registering two implementations of one interface leaves that to resolution
    /// order — which is how a host ends up writing incidents over a week of learning.</para>
    ///
    /// <para><b>The payloads stay separate on purpose.</b> Open incidents are small, change every cycle and
    /// matter for hours; the learned state is large, changes slowly and matters for days. Sharing a file
    /// would mean a format change in one invalidating the other, and a restart that loses both.</para>
    /// </summary>
    public interface ILearnedStateStore : IIncidentStore
    {
    }
}
