// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio.Mp3
{
    /// <summary>MPEG audio channel mode (header bits 6–7).</summary>
    internal enum ChannelMode
    {
        Stereo = 0,
        JointStereo = 1,
        DualChannel = 2,
        Mono = 3,
    }
}
