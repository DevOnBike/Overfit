// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests
{
    /// <summary>
    /// A <see cref="FactAttribute"/> that runs only when the GPT-2 Small fixture
    /// (gpt2_small.bin + vocab.json + merges.txt) is present, and is skipped
    /// otherwise — e.g. on CI runners that do not carry the ~500 MB model.
    ///
    /// The fixture directory is resolved via <see cref="TestModelPaths.Gpt2Small"/>
    /// (override with the OVERFIT_GPT2_DIR environment variable). The presence
    /// check runs at test-discovery time, so the same build skips the test on a
    /// CI runner and runs it on a dev box that has the fixture in place.
    ///
    /// Unlike <see cref="LongFact"/> (always skipped), this skips conditionally —
    /// the GPT-2 parity / inference tests stay defended wherever the model exists.
    /// </summary>
    internal sealed class Gpt2ModelFact : FactAttribute
    {
        public Gpt2ModelFact()
        {
            if (!File.Exists(TestModelPaths.Gpt2Small.BinaryPath)
                || !File.Exists(TestModelPaths.Gpt2Small.VocabPath)
                || !File.Exists(TestModelPaths.Gpt2Small.MergesPath))
            {
                Skip = "GPT-2 Small fixture not present — run Scripts/convert_gpt2.py or set OVERFIT_GPT2_DIR. Skipped by default on CI.";
            }
        }
    }
}
