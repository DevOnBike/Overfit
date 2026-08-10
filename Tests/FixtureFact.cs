// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests
{
    /// <summary>
    /// A <see cref="LongFact"/> that skips — rather than passes — when a named <see cref="TestFixture"/> is
    /// not on the box.
    ///
    /// <para>The companion to <see cref="ModelFact"/>, which covers the cases where the path is a
    /// <c>const string</c> the attribute can take directly. This one covers the rest: paths built at
    /// runtime from <see cref="TestModelPaths"/> or found by walking the tree, which no attribute argument
    /// can express. Between them they replace the <c>if (missing) { return; }</c> pattern that made 66
    /// long tests report success without executing (measured 2026-08-07).</para>
    ///
    /// <para><b>Resolution lives here, not in the tests.</b> Eight call sites in
    /// <c>QwenTokenizerTests</c> all needed the same file; four in <c>MnistTrainingTests</c> the same one
    /// again. Each had its own private helper returning null, so each could drift independently — and the
    /// duplicated <c>LocateOverfitExe</c> in two unrelated files is that drift already happening.</para>
    /// </summary>
    internal sealed class FixtureFact : LongFact
    {
        public FixtureFact(TestFixture fixture, string runtime = null)
            : base(runtime)
        {
            Fixture = fixture;

            // Already skipped as long-running: leave that message, it is the more general reason.
            if (Skip is not null)
            {
                return;
            }

            var (path, what) = Locate(fixture);

            if (path is not null)
            {
                return;
            }

            Skip = $"{what} — this test is SKIPPED, not passed; it has checked nothing.";
        }

        /// <summary>Which prerequisite this test needs.</summary>
        public TestFixture Fixture { get; }

        /// <summary>
        /// The resolved path, or <see langword="null"/> with a sentence saying what is missing and what to
        /// do about it. The second half matters: "not found" without a remedy is not actionable, and these
        /// messages are read by somebody looking at a skipped test months later.
        /// </summary>
        internal static (string Path, string Explanation) Locate(TestFixture fixture)
        {
            switch (fixture)
            {
                case TestFixture.QwenSafetensors:
                    return Check(Path.Combine(TestModelPaths.Qwen3B.Dir, "model.safetensors"),
                        "Qwen safetensors weights not found beside the GGUF (set OVERFIT_QWEN3B_DIR)");

                case TestFixture.QwenTokenizerJson:
                    return Check(Path.Combine(TestModelPaths.Qwen3B.Dir, "tokenizer.json"),
                        "tokenizer.json not found in the Qwen model directory (set OVERFIT_QWEN3B_DIR)");

                case TestFixture.MnistTrainingImages:
                    return Check(TestModelPaths.Mnist.TrainImagesPath,
                        "MNIST training images not found (set OVERFIT_MNIST_DIR)");

                case TestFixture.MiniLmReferenceEmbeddings:
                    return Check(Path.Combine(TestModelPaths.MiniLm.Dir, "minilm_reference_embeddings.json"),
                        "MiniLM reference vectors not found — generate them with the sentence-transformers "
                        + "script before expecting parity");

                case TestFixture.BgeReferenceEmbeddings:
                    return Check(Path.Combine(TestModelPaths.Bge.Dir, "bge_reference_embeddings.json"),
                        "BGE reference vectors not found — generate them before expecting parity");

                case TestFixture.E5ReferenceEmbeddings:
                    return Check(Path.Combine(TestModelPaths.E5.Dir, "e5_reference_embeddings.json"),
                        "E5 reference vectors not found — generate them before expecting parity");

                case TestFixture.Qwen3BBinaryAndTokenizer:
                    return Check(
                        File.Exists(TestModelPaths.Qwen3B.BinaryPath)
                        && File.Exists(TestModelPaths.Qwen3B.TokenizerJsonPath)
                            ? TestModelPaths.Qwen3B.BinaryPath
                            : null,
                        "the Qwen-3B binary checkpoint and its tokenizer.json are not both present (set "
                        + "OVERFIT_QWEN3B_DIR)");

                case TestFixture.Gpt2SafetensorsAndBinary:
                    return Check(
                        File.Exists(Path.Combine(TestModelPaths.Gpt2Small.Dir, "model.safetensors"))
                        && File.Exists(TestModelPaths.Gpt2Small.BinaryPath)
                            ? TestModelPaths.Gpt2Small.BinaryPath
                            : null,
                        "GPT-2 model.safetensors and gpt2_small.bin are not both present, so the two "
                        + "loaders cannot be compared (set OVERFIT_GPT2_DIR, or run "
                        + "Scripts/convert_gpt2.py)");

                case TestFixture.QwenMoeGgufAndReferenceTokenizer:
                {
                    const string gguf = @"C:\qwen-moe\Qwen1.5-MoE-A2.7B-Chat.Q8_0.gguf";
                    var reference = File.Exists(Path.Combine(TestModelPaths.Qwen3B.Dir, "tokenizer.json"))
                                    || File.Exists(Path.Combine(TestModelPaths.Qwen3B.Dir, "vocab.json"));

                    return Check(File.Exists(gguf) && reference ? gguf : null,
                        "the Qwen1.5-MoE GGUF and a reference Qwen tokenizer (tokenizer.json or "
                        + "vocab.json) are not both available, so the GGUF-embedded vocab has nothing to "
                        + "be cross-checked against");
                }

                case TestFixture.MiniLmSafetensors:
                    return Check(TestModelPaths.MiniLm.SafetensorsPath,
                        "MiniLM weights not found (set OVERFIT_MINILM_DIR)");

                case TestFixture.Qwen3BQ4KmGguf:
                    return Check(TestModelPaths.Qwen3B.Q4KmGgufPath,
                        "the Qwen-3B Q4_K_M GGUF is not present (set OVERFIT_QWEN3B_DIR)");

                case TestFixture.Qwen3BTokenizerJson:
                    return Check(TestModelPaths.Qwen3B.TokenizerJsonPath,
                        "tokenizer.json is not present in the Qwen-3B directory (set OVERFIT_QWEN3B_DIR)");

                case TestFixture.Avx2AndFma:
                    // Not a path. Check() would ask the filesystem about a capability, so this answers
                    // directly — and returns a non-null sentinel rather than a path, because there is
                    // nothing on disk to point at.
                    return System.Runtime.Intrinsics.X86.Avx2.IsSupported
                           && System.Runtime.Intrinsics.X86.Fma.IsSupported
                        ? ("cpu", null)
                        : (null, "this CPU has no AVX2+FMA, so the x86 SIMD path under test does not exist "
                                 + "here — there is nothing to measure, and a pass would say otherwise");

                case TestFixture.OverfitCli:
                    return Check(LocateOverfitCli(),
                        "no built `overfit` CLI under Sources/Cli/bin — this is a BUILD output, not a "
                        + "downloaded fixture: `dotnet build -c Release Sources/Cli/Cli.csproj` first");

                default:
                    return (null, $"unknown fixture {fixture}");
            }
        }

        private static (string, string) Check(string path, string explanation)
        {
            return path is not null && File.Exists(path) ? (path, null) : (null, explanation);
        }

        /// <summary>
        /// Walks up to the repository root (the folder holding <c>Overfit.sln</c>) and looks for the built
        /// CLI beneath <c>Sources/Cli/bin</c>.
        ///
        /// <para>Consolidated from two byte-identical private copies in <c>XgboostScoreCliTests</c> and
        /// <c>RedactionGatewayE2ETests</c>. Two copies of a locator is one copy away from two different
        /// answers.</para>
        /// </summary>
        internal static string? LocateOverfitCli()
        {
            // Shared walk (XC-4), local policy: null means "no checkout, so no built CLI to find", which
            // callers turn into a skip. Only the walk moved; everything below is this locator's own job.
            var root = RepositoryPaths.TryFindRoot();

            if (root is null)
            {
                return null;
            }

            var binaries = Path.Combine(root, "Sources", "Cli", "bin");

            if (!Directory.Exists(binaries))
            {
                return null;
            }

            // Not hardcoded to "overfit.exe". The two copies this replaces both picked the name from the
            // platform, and dropping that would have made the fixture look permanently absent on Linux —
            // i.e. on CI, silently, which is the exact class of failure this whole change is about.
            var name = OperatingSystem.IsWindows() ? "overfit.exe" : "overfit";

            foreach (var candidate in Directory.EnumerateFiles(binaries, name, SearchOption.AllDirectories))
            {
                return candidate;
            }

            return null;
        }
    }
}
