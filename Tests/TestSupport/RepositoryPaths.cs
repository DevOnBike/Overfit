// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Tests.TestSupport
{
    /// <summary>
    /// Paths inside this checkout, resolved at run time.
    ///
    /// <para><b>Why this exists.</b> Six sites hardcoded <c>D:\Overfit\…</c> — a developer's drive letter,
    /// in a public repository. On any other machine those defaults point at nothing, and the diagnostics
    /// that used them would have failed with "no guard configuration at D:/Overfit/…" on a clone. Found
    /// 2026-08-08 while auditing what the lab keeps outside git.</para>
    ///
    /// <para><b>Consolidated 2026-08-10 (`XC-4`), and it was not a mechanical replace.</b> Four call sites
    /// carried their own copy of the walk — <c>FixtureFact</c>, <c>ModelFact</c>,
    /// <c>TelemetryInstrumentWiringTests</c> and <c>HybridVsDenseOnDocsCorpusTests</c> — and they disagreed
    /// about what to do when the root is not found: two threw (with different messages), one returned
    /// <c>null</c>, and one returned its input unchanged. Each policy is deliberate and documented where it
    /// lives: a diagnostic that reads the source tree should fail loudly, and a corpus test that cannot find
    /// <c>docs/</c> should skip rather than turn a missing checkout into a red build. Replacing all four with
    /// <see cref="Root"/> would have converted two skips into failures.</para>
    ///
    /// <para>So <see cref="TryFindRoot"/> is the shared walk and each caller keeps its own answer to "and if
    /// it is not there?". That is the part worth sharing — one copy of the walk means one place to be wrong
    /// about it, which is the whole argument, and it is the argument <c>FixtureFact</c> already made when it
    /// folded two byte-identical CLI locators into one: <i>two copies of a locator is one copy away from two
    /// different answers.</i></para>
    /// </summary>
    internal static class RepositoryPaths
    {
        /// <summary>
        /// The folder holding <c>Overfit.sln</c>, or <c>null</c> when the walk reaches the drive root.
        ///
        /// <para>For callers whose correct response to "not in a checkout" is to skip rather than fail.</para>
        /// </summary>
        internal static string? TryFindRoot()
        {
            var directory = new DirectoryInfo(AppContext.BaseDirectory);

            // BOUND: `Parent` reaches null at the drive root, so the walk is bounded by the filesystem depth
            // of the test output directory.
            while (directory is not null)
            {
                if (File.Exists(Path.Combine(directory.FullName, "Overfit.sln")))
                {
                    return directory.FullName;
                }

                directory = directory.Parent;
            }

            return null;
        }

        /// <summary>The folder holding <c>Overfit.sln</c>, walking up from the test output directory.</summary>
        internal static string Root
        {
            get
            {
                return TryFindRoot()
                       ?? throw new InvalidOperationException(
                           $"could not find Overfit.sln above {AppContext.BaseDirectory}. This resolves "
                           + "paths inside the checkout, so it cannot fall back to a fixed location without "
                           + "reintroducing the hardcoded-drive-letter problem it exists to remove.");
            }
        }

        /// <summary>A path under <c>Tests/bin</c> — build output, gitignored, and where the lab
        /// diagnostics keep the guard configuration and learned state pulled out of the cluster.</summary>
        internal static string TestsBin(string fileName)
        {
            return Path.Combine(Root, "Tests", "bin", fileName);
        }

        /// <summary>A path relative to the repository root.</summary>
        internal static string FromRoot(params string[] parts)
        {
            return Path.Combine(new[] { Root }.Concat(parts).ToArray());
        }
    }
}
