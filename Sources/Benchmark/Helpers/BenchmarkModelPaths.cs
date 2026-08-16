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

        /// <summary>Same variable the runtime and the demos honour (<c>OverfitEnvironment.ModelDir</c>).</summary>
        private const string ModelDirEnvVar = "OVERFIT_MODEL_DIR";

        /// <summary>No runtime counterpart exists; this is the benchmark project's own override.</summary>
        private const string BielikDirEnvVar = "OVERFIT_BIELIK_DIR";

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

        /// <summary>Qwen2.5-3B-Instruct, Q4_K_M — the model most of the runtime measurements use.</summary>
        public static string ResolveQwen3BQ4KM()
        {
            return Resolve("Qwen2.5-3B-Instruct Q4_K_M", "qwen.q4km.gguf", ModelDirEnvVar, @"c:\qwen3b");
        }

        /// <summary>Qwen2.5-0.5B, Q4_K_M — the DRAFT model in the speculative-decoding measurements.</summary>
        public static string ResolveQwen05BQ4KM()
        {
            return Resolve("Qwen2.5-0.5B Q4_K_M (speculative draft)", "qwen0.5b.q4km.gguf",
                ModelDirEnvVar, @"c:\qwen3b");
        }

        /// <summary>Bielik-4.5B-v3.0-Instruct, Q4_K_M — the TARGET in the Bielik speculative measurements.</summary>
        public static string ResolveBielikTarget()
        {
            return Resolve("Bielik-4.5B-v3.0-Instruct Q4_K_M",
                "Bielik-4.5B-v3.0-Instruct-Q4_K_M.gguf", BielikDirEnvVar, @"c:\bielik");
        }

        /// <summary>Bielik-1.5B-v3.0-Instruct, Q4_K_M imatrix — the Bielik DRAFT model. Same v3.0 vocabulary
        /// as the target, which is what makes it usable as a drafter at all.</summary>
        public static string ResolveBielikDraft()
        {
            return Resolve("Bielik-1.5B-v3.0-Instruct Q4_K_M (speculative draft)",
                "bielik-1.5b-v3.0-instruct-q4_k_m-imat.gguf", BielikDirEnvVar, @"c:\bielik");
        }

        /// <summary>
        /// The directory holding a model's sidecar files — tokenizer, config — rather than one weight file.
        /// Returns the first directory that contains <paramref name="mustContain"/>.
        /// </summary>
        public static string ResolveDirectoryContaining(string mustContain, params string[] directories)
        {
            var envDir = Environment.GetEnvironmentVariable(ModelDirEnvVar);
            var candidates = new List<string>(directories.Length + 1);

            if (!string.IsNullOrWhiteSpace(envDir))
            {
                candidates.Add(envDir);
            }

            candidates.AddRange(directories);

            foreach (var candidate in candidates)
            {
                if (File.Exists(Path.Combine(candidate, mustContain)))
                {
                    return candidate;
                }
            }

            throw new DirectoryNotFoundException(
                $"No directory containing '{mustContain}' found. Probed: {string.Join(", ", candidates)}. " +
                $"Set {ModelDirEnvVar} to the directory that holds it.");
        }

        /// <summary>
        /// The shared shape: environment override first, then the conventional dev-box location, then the
        /// repo-relative CI layout. The first existing path wins.
        ///
        /// <para><b>It throws rather than returning null, and that is deliberate</b> — the same choice
        /// <see cref="ResolveGpt2SmallBinary"/> already made. A benchmark that silently degrades to nothing
        /// produces an <c>NA</c> row, and an NA row in a results table reads as "measured, no difference"
        /// to everyone who did not run it. BenchmarkDotNet isolates each class in its own process, so a
        /// throw here fails the class whose fixture is missing and leaves the rest of the run intact.</para>
        /// </summary>
        private static string Resolve(string description, string fileName, string envVar, string devBoxDir)
        {
            var candidates = new List<string>(3);
            var envDir = Environment.GetEnvironmentVariable(envVar);

            if (!string.IsNullOrWhiteSpace(envDir))
            {
                candidates.Add(Path.Combine(envDir, fileName));
            }

            candidates.Add(Path.Combine(devBoxDir, fileName));
            candidates.Add(Path.Combine("test_fixtures", fileName));

            foreach (var candidate in candidates)
            {
                if (File.Exists(candidate))
                {
                    return candidate;
                }
            }

            throw new FileNotFoundException(
                $"{description} not found. Probed: {string.Join(", ", candidates)}. "
                + $"Set {envVar} to the directory holding '{fileName}'.");
        }
    }
}
