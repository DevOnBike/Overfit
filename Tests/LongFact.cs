// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Tests
{
    internal class LongFact : FactAttribute
    {
        public LongFact()
        {
            Skip = "This test is marked as long-running and is skipped by default. Remove the Skip property to run it.";
        }
    }
}
