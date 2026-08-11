// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests
{
    /// <summary>
    /// A <see cref="LongFact"/> that skips — rather than passes — when the model files it names are not on
    /// the box.
    ///
    /// <para><b>Why this exists.</b> Measured 2026-08-07, on the first run the <c>[LongFact]</c> suite has
    /// ever had: <b>66 of 256</b> long tests opened with a variation of</para>
    /// <code>
    ///   if (!File.Exists(Path)) { _out.WriteLine("missing model"); return; }   // <- a PASS
    /// </code>
    /// <para>which reports success without executing anything. On a machine with no fixtures — that is, on
    /// CI — a release gate built from these goes green having loaded no model at all. The tests affected
    /// include <c>Phi4_Loads_And_Generates_Coherent_English</c>,
    /// <c>Gemma2_Loads_And_Generates_Coherent_English</c>, <c>Bielik_Loads_And_Generates_Polish</c> and the
    /// whole <c>*_DecodeThroughput_BestOfN</c> family: precisely the checks somebody would point at to
    /// claim the loaders work.</para>
    ///
    /// <para><b>A skip and a pass are not the same result</b>, and the difference is the entire point. A
    /// skip appears in the TRX as <c>NotExecuted</c> and in any report as "not run"; a pass is
    /// indistinguishable from "checked and correct". The repository already had the right pattern —
    /// <see cref="SmallModelFact"/> and <see cref="Gpt2ModelFact"/> both set <c>Skip</c> from a file check
    /// at discovery time — but it predates most of these tests and never reached them.</para>
    ///
    /// <para><b>Why an attribute and not an assertion inside the test.</b> xUnit 2.9.3 has no dynamic skip
    /// (<c>Assert.Skip</c> arrived in v3; trying it here is a compile error, measured). Asserting instead
    /// would turn every missing fixture into a red gate on any developer machine without a full model
    /// collection, which is how a gate gets ignored.</para>
    ///
    /// <para><b>Paths must be compile-time constants</b>, because attribute arguments are. That is a real
    /// limitation and it is why this does not cover all 66: roughly half guard on a local variable
    /// returned by a helper (<c>tok</c>, <c>imgs</c>, <c>exe</c>) rather than on a constant, and those need
    /// their own predicate. Do not work around it by duplicating a path literal into the attribute — two
    /// copies of a path is how a test skips for one reason and fails for another.</para>
    ///
    /// <example>
    /// <code>
    ///   private const string Path = @"C:\phi\phi-4-Q4_K_M.gguf";
    ///
    ///   [ModelFact(Path, "45s")]
    ///   public void Phi4_Loads_And_Generates_Coherent_English() { ... }
    /// </code>
    /// </example>
    /// </summary>
    internal sealed class ModelFact : LongFact
    {
        /// <param name="requiredFile">Model file that must exist; a compile-time constant.</param>
        /// <param name="runtime">Measured wall-clock, as on <see cref="LongFact"/> — e.g. <c>"45s"</c>.</param>
        public ModelFact(
            string requiredFile,
            string runtime = null,
            [CallerFilePath] string sourceFilePath = null,
            [CallerLineNumber] int sourceLineNumber = -1)
            : this([requiredFile], runtime, sourceFilePath, sourceLineNumber)
        {
        }

        /// <param name="requiredFiles">
        /// Every file the test needs. All must exist. Listing them individually rather than checking a
        /// directory is deliberate: the skip message then names <b>which</b> one is missing, and a
        /// half-present fixture (a model without its tokenizer, say) is a common and confusing state.
        /// </param>
        /// <param name="runtime">Measured wall-clock, as on <see cref="LongFact"/>.</param>
        public ModelFact(
            string[] requiredFiles,
            string runtime = null,
            [CallerFilePath] string sourceFilePath = null,
            [CallerLineNumber] int sourceLineNumber = -1)
            : base(runtime, sourceFilePath, sourceLineNumber)
        {
            RequiredFiles = requiredFiles ?? [];

            // Already skipped as a long test. Leave that message: it is the more general reason, and
            // overwriting it would hide the fact that this never runs by default anyway.
            if (Skip is not null)
            {
                return;
            }

            // Directories count. Several of these fixtures are a folder (a safetensors model is a
            // directory of shards plus config), and `File.Exists` returns false for one — which would have
            // made the attribute report every such fixture as permanently missing.
            foreach (var file in RequiredFiles)
            {
                var resolved = Resolve(file);

                if (string.IsNullOrWhiteSpace(file) || File.Exists(resolved) || Directory.Exists(resolved))
                {
                    continue;
                }

                Skip = $"Fixture not present: {file}. This test is SKIPPED, not passed — it has checked "
                    + "nothing. Put the file in place, or point the test's path at one.";

                return;
            }
        }

        /// <summary>The files whose absence makes this skip.</summary>
        public string[] RequiredFiles
        {
            get;
        }

        /// <summary>
        /// Resolves a fixture path. An absolute one is returned unchanged — model fixtures live outside
        /// the tree (<c>C:\qwen3b</c>, <c>C:\bielik</c>) and are named absolutely on purpose.
        ///
        /// <para><b>A RELATIVE one is resolved against the repository root, not the working directory</b>,
        /// which is the whole point. An attribute argument must be a compile-time constant, so a fixture
        /// that lives inside the checkout can only be named relatively — and a bare name checked against
        /// the test process's working directory (<c>Tests/bin/Release/net10.0</c>) never exists, so the
        /// test would skip silently everywhere. That is how a hardcoded <c>D:\Overfit\…</c> came to be
        /// here in the first place; this removes the reason for it.</para>
        /// </summary>
        private static string Resolve(string file)
        {
            if (string.IsNullOrWhiteSpace(file) || Path.IsPathRooted(file))
            {
                return file;
            }

            var root = RepositoryPaths.TryFindRoot();

            // Shared walk (XC-4), local policy: no root found means the name is returned UNCHANGED, so the
            // caller reports "fixture not present" with what it was given rather than with a path assembled
            // from a guess. Neither throwing nor returning null would say that.
            return root is null ? file : Path.Combine(root, file);
        }

    }
}
