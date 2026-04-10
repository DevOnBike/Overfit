// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Data.Contracts
{
    public enum LogMode
    {
        /// <summary>log(1 + x)</summary>
        Log1p,

        /// <summary>sign(x) * log(1 + |x|)</summary>
        SignedLog1p,

        /// <summary>log(x + epsilon)</summary>
        LogEps
    }
}