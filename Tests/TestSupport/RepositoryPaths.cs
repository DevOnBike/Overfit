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
    /// <para>The walk itself is not new: <c>FixtureFact</c>, <c>TelemetryInstrumentWiringTests</c> and
    /// <c>HybridVsDenseOnDocsCorpusTests</c> each already carry a copy. This is a fourth implementation
    /// only in the sense that it is the one meant to be shared; consolidating the other three is separate
    /// work and is deliberately not done here.</para>
    /// </summary>
    internal static class RepositoryPaths
    {
        /// <summary>The folder holding <c>Overfit.sln</c>, walking up from the test output directory.</summary>
        internal static string Root
        {
            get
            {
                var directory = new DirectoryInfo(AppContext.BaseDirectory);

                while (directory is not null)
                {
                    if (File.Exists(Path.Combine(directory.FullName, "Overfit.sln")))
                    {
                        return directory.FullName;
                    }

                    directory = directory.Parent;
                }

                throw new InvalidOperationException(
                    $"could not find Overfit.sln above {AppContext.BaseDirectory}. This resolves paths "
                    + "inside the checkout, so it cannot fall back to a fixed location without "
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
