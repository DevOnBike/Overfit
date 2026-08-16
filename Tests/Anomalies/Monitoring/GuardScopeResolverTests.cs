// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;
using DevOnBike.Overfit.Anomalies.Monitoring;

namespace DevOnBike.Overfit.Tests.Anomalies.Monitoring
{
    /// <summary>
    /// Resolving a configuration file into the populations a guard process will watch.
    ///
    /// <para>Every case here is about the same hazard in a different disguise: <b>a guard watching fewer
    /// populations than its operator believes is indistinguishable from those populations being healthy</b>.
    /// So nothing is dropped quietly, ambiguity is refused rather than resolved by a rule nobody would
    /// remember, and a duplicate scope is rejected because two trackers over one population page twice for
    /// one problem.</para>
    /// </summary>
    public sealed class GuardScopeResolverTests
    {
        /// <summary>The migration guarantee: a file written before scopes existed still works, unchanged.</summary>
        [Fact]
        public void ASingleScopeFileResolvesToOneScope()
        {
            var problems = new List<string>();
            var file = new AnomalyGuardConfigFile
            {
                Namespace = "lab",
                PodRegex = "lab-workload-.*",
                Workload = "lab-workload",
                PeerGroupLabel = "role",
            };

            var scopes = GuardScopeResolver.Resolve(file, problems);

            Assert.Empty(problems);
            var only = Assert.Single(scopes);
            Assert.Equal("lab", only.Namespace);
            Assert.Equal("lab-workload-.*", only.PodRegex);
            Assert.Equal("lab-workload", only.Workload);
            Assert.Equal("role", only.PeerGroupLabel);
        }

        /// <summary>
        /// Two populations in one namespace — the case the design calls ordinary rather than exceptional,
        /// because comparing a worker replica against an API replica is the mistake peer grouping exists to
        /// prevent.
        /// </summary>
        [Fact]
        public void TwoScopesInOneNamespaceAreDistinct()
        {
            var problems = new List<string>();
            var file = new AnomalyGuardConfigFile
            {
                Scopes =
                [
                    new GuardScopeEntry { Namespace = "payments", PodRegex = "api-.*", Workload = "api" },
                    new GuardScopeEntry { Namespace = "payments", PodRegex = "worker-.*", Workload = "worker" },
                ],
            };

            var scopes = GuardScopeResolver.Resolve(file, problems);

            Assert.Empty(problems);
            Assert.Equal(2, scopes.Count);
            Assert.NotEqual(scopes[0].Name, scopes[1].Name);
            Assert.Equal("payments/api-.*", scopes[0].Name);
        }

        /// <summary>
        /// A repeated scope is refused, not deduplicated silently. Each scope gets its own tracker, so the
        /// same problem would open two incidents with two ids and page twice — the exact failure the tracker
        /// exists to prevent, reintroduced through configuration.
        /// </summary>
        [Fact]
        public void ARepeatedScopeIsRefusedAndReported()
        {
            var problems = new List<string>();
            var file = new AnomalyGuardConfigFile
            {
                Scopes =
                [
                    new GuardScopeEntry { Namespace = "search", PodRegex = "index-.*" },
                    new GuardScopeEntry { Namespace = "search", PodRegex = "index-.*" },
                ],
            };

            var scopes = GuardScopeResolver.Resolve(file, problems);

            Assert.Single(scopes);
            Assert.Contains(problems, p => p.Contains("search/index-.*", StringComparison.Ordinal));
        }

        /// <summary>
        /// Both forms at once is a contradiction, and resolving it either way silently watches a different
        /// set of pods than the file appears to describe.
        /// </summary>
        [Fact]
        public void DeclaringBothFormsIsRefused()
        {
            var problems = new List<string>();
            var file = new AnomalyGuardConfigFile
            {
                Namespace = "lab",
                PodRegex = "lab-.*",
                Scopes = [new GuardScopeEntry { Namespace = "search", PodRegex = "index-.*" }],
            };

            var scopes = GuardScopeResolver.Resolve(file, problems);

            Assert.Empty(scopes);
            Assert.Contains(problems, p => p.Contains("ambiguous", StringComparison.Ordinal));
        }

        /// <summary>
        /// A scope with no namespace is skipped and SAID so. The dangerous outcome is not the skip, it is a
        /// skip nobody hears about.
        /// </summary>
        [Fact]
        public void AScopeWithoutANamespaceIsSkippedAndReported()
        {
            var problems = new List<string>();
            var file = new AnomalyGuardConfigFile
            {
                Scopes =
                [
                    new GuardScopeEntry { Namespace = string.Empty, PodRegex = "orphan-.*" },
                    new GuardScopeEntry { Namespace = "search", PodRegex = "index-.*" },
                ],
            };

            var scopes = GuardScopeResolver.Resolve(file, problems);

            Assert.Single(scopes);
            Assert.Contains(problems, p => p.Contains("scopes[0]", StringComparison.Ordinal));
        }

        /// <summary>
        /// Nothing declared at all is a configuration failure, not an instruction to watch nothing. The
        /// second reading produces a process that starts, logs nothing and reports no incidents — which is
        /// what a perfectly healthy cluster also looks like.
        /// </summary>
        [Fact]
        public void AnEmptyFileIsAProblemRatherThanAnEmptyWatchList()
        {
            var problems = new List<string>();

            var scopes = GuardScopeResolver.Resolve(new AnomalyGuardConfigFile(), problems);

            Assert.Empty(scopes);
            Assert.NotEmpty(problems);
        }

        /// <summary>
        /// The scope's identity is derived from what defines the population, so it cannot be edited apart
        /// from it. A configured name would let somebody rename a scope and orphan its saved incidents and
        /// learned floors while appearing to edit a label.
        /// </summary>
        [Fact]
        public void TheNameIsDerivedFromTheNamespaceAndSelector()
        {
            Assert.Equal("payments/api-.*", new GuardScope("payments", "api-.*", "api", string.Empty).Name);
            Assert.Equal("payments", new GuardScope("payments", string.Empty, string.Empty, string.Empty).Name);
        }
    }
}
