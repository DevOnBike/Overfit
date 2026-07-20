// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests
{
    /// <summary>
    /// A <see cref="FactAttribute"/> for the fast suite that runs only when the smallest real model fixture
    /// (Qwen2.5-0.5B Q4_K_M, ~490 MB, plus its tokenizer.json) is present, and skips otherwise — e.g. on CI
    /// runners that do not carry model files.
    ///
    /// <para><b>Why this exists.</b> Every test that touched a real model used to be <see cref="LongFact"/>,
    /// i.e. skipped by default. That left a structural hole: changes to the model-loading path — depth caps,
    /// argument guards, parser edits — could not be caught by a normal <c>dotnet test</c>, and the suite stayed
    /// green while the untrusted-input path went unexercised. This attribute is the middle ground: fast enough
    /// for the default run (the 0.5B model loads in ~1 s), real enough to exercise the actual parsers.</para>
    ///
    /// <para>Use it ONLY for cheap checks — metadata parsing, tokenizer round-trips, header validation. Anything
    /// that generates tokens, loads a multi-GB model, or takes seconds belongs in <see cref="LongFact"/>; the
    /// project is strict about default-suite runtime (see Tests/README.md).</para>
    /// </summary>
    internal sealed class SmallModelFact : FactAttribute
    {
        public SmallModelFact()
        {
            if (!File.Exists(TestModelPaths.Qwen05B.Q4KmGgufPath))
            {
                Skip = "Qwen2.5-0.5B fixture not present — set OVERFIT_QWEN3B_DIR to a directory containing "
                    + "qwen0.5b.q4km.gguf. Skipped by default on CI.";
            }
        }
    }
}
