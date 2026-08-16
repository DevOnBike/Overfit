// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Contracts;

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    /// <summary>
    /// Turns a configuration file into the list of populations the process will watch.
    ///
    /// <para><b>The existing single-scope file stays valid, and that is a requirement rather than a
    /// courtesy.</b> A file with <c>namespace</c> and <c>podRegex</c> at the top level resolves to a
    /// one-element list, so nothing deployed needs editing on the day multi-scope ships — which is also what
    /// makes the change testable, because the same measurement can be run before and after against the same
    /// manifest. A migration that requires every client to edit their config on upgrade is a migration whose
    /// before/after cannot be compared.</para>
    ///
    /// <para><b>Problems are reported, never guessed around.</b> Every rejection returns a line naming what
    /// was wrong; the alternative — dropping a malformed scope silently — produces a guard watching fewer
    /// populations than its operator believes, which is indistinguishable from those populations being
    /// healthy.</para>
    /// </summary>
    public static class GuardScopeResolver
    {
        /// <summary>
        /// Resolves <paramref name="file"/> into scopes.
        /// </summary>
        /// <param name="file">The parsed configuration.</param>
        /// <param name="problems">One line per entry that could not be used.</param>
        /// <returns>
        /// The scopes to watch, in file order. Empty when nothing usable was declared — a caller must treat
        /// that as a configuration failure and not as "watch nothing", which is the silent-blindness shape.
        /// </returns>
        public static IReadOnlyList<GuardScope> Resolve(AnomalyGuardConfigFile file, IList<string> problems)
        {
            ArgumentNullException.ThrowIfNull(file);
            ArgumentNullException.ThrowIfNull(problems);

            var scopes = new List<GuardScope>();

            // The single-scope form. Kept first so its behaviour is obviously unchanged.
            if (file.Scopes.Count == 0)
            {
                if (file.Namespace.Length == 0 && file.PodRegex.Length == 0)
                {
                    problems.Add(
                        "No scopes: the file declares neither a top-level 'namespace'/'podRegex' nor a "
                        + "'scopes' list, so there is nothing to watch.");

                    return scopes;
                }

                scopes.Add(new GuardScope(
                    file.Namespace, file.PodRegex, file.Workload, file.PeerGroupLabel));

                return scopes;
            }

            // Declaring both forms is not a merge, it is a contradiction: a reader cannot tell whether the
            // top-level fields are a scope of their own or defaults the list overrides, and guessing either
            // way silently watches a different set of pods than the file appears to describe.
            if (file.Namespace.Length > 0 || file.PodRegex.Length > 0)
            {
                problems.Add(
                    "Both a 'scopes' list and top-level 'namespace'/'podRegex' are set. Move the top-level "
                    + "pair into the list as its own entry; leaving both is ambiguous and is refused rather "
                    + "than resolved by a rule nobody would remember.");

                return scopes;
            }

            var seen = new HashSet<string>(StringComparer.Ordinal);

            for (var i = 0; i < file.Scopes.Count; i++)
            {
                var entry = file.Scopes[i];

                if (entry == null)
                {
                    problems.Add($"scopes[{i}] is null and was skipped.");

                    continue;
                }

                if (entry.Namespace.Length == 0)
                {
                    problems.Add($"scopes[{i}] has no namespace and was skipped.");

                    continue;
                }

                var scope = new GuardScope(
                    entry.Namespace, entry.PodRegex, entry.Workload, entry.PeerGroupLabel);

                // Two identical scopes are not harmless duplication: each gets its own tracker, so the same
                // problem opens two incidents with two ids and pages twice — the failure the tracker exists
                // to prevent, reintroduced through configuration.
                if (!seen.Add(scope.Name))
                {
                    problems.Add(
                        $"scopes[{i}] repeats '{scope.Name}', which would open two incidents for every "
                        + "problem in it. The duplicate was skipped.");

                    continue;
                }

                scopes.Add(scope);
            }

            if (scopes.Count == 0)
            {
                problems.Add("The 'scopes' list contained no usable entry, so there is nothing to watch.");
            }

            return scopes;
        }
    }
}
