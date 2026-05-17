// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace Benchmarks.Helpers
{
    /// <summary>
    /// Model-fixture path resolution for benchmarks — the benchmark-project
    /// counterpart of <c>TestModelPaths</c> (which lives in the test project
    /// and can't be referenced from here).
    ///
    /// <para>
    /// Resolution order for a GPT-2 fixture:
    /// </para>
    /// <list type="number">
    ///   <item><c>OVERFIT_GPT2_DIR</c> env var (same variable the test
    ///         project honors) — <c>{dir}/gpt2_small.bin</c>.</item>
    ///   <item>Conventional dev-box layout: <c>c:\gpt2\gpt2_small.bin</c>.</item>
    ///   <item>Repo-relative CI layout: <c>test_fixtures/gpt2_small.bin</c>
    ///         (resolved against the benchmark's working directory).</item>
    /// </list>
    ///
    /// <para>
    /// The first existing path wins. If none exist,
    /// <see cref="ResolveGpt2SmallBinary"/> throws a
    /// <see cref="FileNotFoundException"/> that names every probed location
    /// and the override env var — so a missing fixture fails loudly and
    /// actionably instead of silently producing <c>NA</c> benchmark rows.
    /// </para>
    /// </summary>
    internal static class BenchmarkModelPaths
    {
        private const string Gpt2DirEnvVar = "OVERFIT_GPT2_DIR";
        private const string Gpt2BinaryFileName = "gpt2_small.bin";

        /// <summary>
        /// Resolves the GPT-2 Small Overfit-binary checkpoint path. Throws
        /// with an actionable message if no candidate location has the file.
        /// </summary>
        public static string ResolveGpt2SmallBinary()
        {
            var candidates = new List<string>(3);

            var envDir = Environment.GetEnvironmentVariable(Gpt2DirEnvVar);
            if (!string.IsNullOrWhiteSpace(envDir))
            {
                candidates.Add(Path.Combine(envDir, Gpt2BinaryFileName));
            }

            candidates.Add(Path.Combine(@"c:\gpt2", Gpt2BinaryFileName));
            candidates.Add(Path.Combine("test_fixtures", Gpt2BinaryFileName));

            foreach (var candidate in candidates)
            {
                if (File.Exists(candidate))
                {
                    return candidate;
                }
            }

            throw new FileNotFoundException(
                $"GPT-2 Small checkpoint not found. Probed: {string.Join(", ", candidates)}. " +
                $"Set {Gpt2DirEnvVar} to the directory holding '{Gpt2BinaryFileName}', " +
                "or run: python3 Scripts/convert_gpt2.py --size small --out test_fixtures/");
        }
    }
}
