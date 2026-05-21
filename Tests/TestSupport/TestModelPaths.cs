// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Tests.TestSupport
{
    /// <summary>
    /// Centralized resolution of model fixture paths used by integration tests.
    ///
    /// Lookup order per family:
    ///   1. <c>OVERFIT_&lt;FAMILY&gt;_DIR</c> environment variable
    ///   2. Conventional fallback (the dev's local layout — works out of the box on this repo's primary dev box)
    ///
    /// Two access modes per file:
    ///   - <c>BinaryPath</c> / <c>GgufPath</c> / ... (properties) — return the resolved path;
    ///     the file may or may not exist. Use these when the test wants to peek before deciding.
    ///   - <c>RequireBinaryPath()</c> / <c>RequireGgufPath()</c> / ... (methods) — return the path
    ///     or throw <see cref="FileNotFoundException"/> with an actionable hint naming the
    ///     env var to override. Use these when the test cannot proceed without the file —
    ///     missing fixture should fail loudly, not silently skip.
    ///
    /// Skip-by-default for these tests is handled at the xUnit attribute level
    /// (<c>[LongFact]</c>); the file-existence check is a *secondary* safety net for
    /// cases where the dev has flipped to <c>[Fact]</c> locally.
    ///
    /// To run on a different machine, set the appropriate env var before invoking
    /// <c>dotnet test</c>:
    ///   PowerShell:  $env:OVERFIT_GPT2_DIR = "D:\models\gpt2"
    ///   Bash:        export OVERFIT_GPT2_DIR=/mnt/models/gpt2
    /// </summary>
    internal static class TestModelPaths
    {
        public static class Gpt2Small
        {
            private const string EnvVar = "OVERFIT_GPT2_DIR";
            public static string Dir => Resolve(EnvVar, @"c:\gpt2");
            public static string BinaryPath => Path.Combine(Dir, "gpt2_small.bin");
            public static string VocabPath => Path.Combine(Dir, "vocab.json");
            public static string MergesPath => Path.Combine(Dir, "merges.txt");

            /// <summary>
            /// PyTorch reference logits JSON. Small (~kilobytes), committed under
            /// <c>Tests/test_fixtures/</c> and copied to the test build output, so
            /// resolves from <c>AppContext.BaseDirectory</c> by default. If the
            /// user has a local override (e.g. regenerated via
            /// <c>Scripts/debug_gpt2_reference.py</c>), it takes precedence when
            /// present alongside the model.
            /// </summary>
            public static string ReferenceJsonPath
            {
                get
                {
                    var dirCopy = Path.Combine(Dir, "gpt2_reference_small.json");
                    if (File.Exists(dirCopy)) { return dirCopy; }
                    return Path.Combine(AppContext.BaseDirectory, "test_fixtures", "gpt2_reference_small.json");
                }
            }

            public static string RequireBinaryPath() => Require(BinaryPath, EnvVar, "GPT-2 binary weights");
            public static string RequireVocabPath() => Require(VocabPath, EnvVar, "GPT-2 BPE vocab.json");
            public static string RequireMergesPath() => Require(MergesPath, EnvVar, "GPT-2 BPE merges.txt");
            public static string RequireReferenceJsonPath() => Require(ReferenceJsonPath, EnvVar, "GPT-2 PyTorch reference JSON");
        }

        public static class Qwen3B
        {
            private const string EnvVar = "OVERFIT_QWEN3B_DIR";
            public static string Dir => Resolve(EnvVar, @"c:\qwen3b");
            public static string BinaryPath => Path.Combine(Dir, "qwen.bin");
            public static string GgufPath => Path.Combine(Dir, "qwen.gguf");
            public static string Q4KmGgufPath => Path.Combine(Dir, "qwen.q4km.gguf");
            public static string Q8GgufPath => Path.Combine(Dir, "qwen.q8_0.gguf");
            public static string VocabPath => Path.Combine(Dir, "vocab.json");
            public static string MergesPath => Path.Combine(Dir, "merges.txt");
            public static string TokenizerJsonPath => Path.Combine(Dir, "tokenizer.json");

            public static string RequireBinaryPath() => Require(BinaryPath, EnvVar, "Qwen3B Overfit binary weights");
            public static string RequireGgufPath() => Require(GgufPath, EnvVar, "Qwen3B FP16 GGUF");
            public static string RequireQ4KmGgufPath() => Require(Q4KmGgufPath, EnvVar, "Qwen3B Q4_K_M GGUF");
            public static string RequireQ8GgufPath() => Require(Q8GgufPath, EnvVar, "Qwen3B Q8_0 GGUF");
            public static string RequireTokenizerJsonPath() => Require(TokenizerJsonPath, EnvVar, "Qwen3B tokenizer.json");
            public static string RequireDir()
            {
                if (!Directory.Exists(Dir))
                {
                    throw new DirectoryNotFoundException(
                        $"Required Qwen3B fixture directory missing: '{Dir}'. " +
                        $"Set {EnvVar} env var to override the path.");
                }
                return Dir;
            }
        }

        public static class Mnist
        {
            private const string EnvVar = "OVERFIT_MNIST_DIR";
            public static string Dir => Resolve(EnvVar, @"d:\ml");
            public static string TrainImagesPath => Path.Combine(Dir, "train-images.idx3-ubyte");
            public static string TrainLabelsPath => Path.Combine(Dir, "train-labels.idx1-ubyte");
            public static string TestImagesPath => Path.Combine(Dir, "t10k-images.idx3-ubyte");
            public static string TestLabelsPath => Path.Combine(Dir, "t10k-labels.idx1-ubyte");

            public static string RequireTrainImagesPath() => Require(TrainImagesPath, EnvVar, "MNIST train images");
            public static string RequireTrainLabelsPath() => Require(TrainLabelsPath, EnvVar, "MNIST train labels");
            public static string RequireTestImagesPath() => Require(TestImagesPath, EnvVar, "MNIST test images");
            public static string RequireTestLabelsPath() => Require(TestLabelsPath, EnvVar, "MNIST test labels");
        }

        private static string Resolve(string envVar, string fallback)
        {
            var fromEnv = Environment.GetEnvironmentVariable(envVar);
            return string.IsNullOrWhiteSpace(fromEnv) ? fallback : fromEnv;
        }

        private static string Require(string path, string envVar, string what)
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException(
                    $"Required fixture missing — {what}: '{path}'. " +
                    $"Set {envVar} env var to override the directory.",
                    path);
            }
            return path;
        }
    }
}
