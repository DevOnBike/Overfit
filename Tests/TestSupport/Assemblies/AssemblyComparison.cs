// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;

namespace DevOnBike.Overfit.Tests.TestSupport.Assemblies
{
    /// <summary>
    /// The result of comparing two assemblies. See <see cref="AssemblyComparer"/> for how to get one.
    ///
    /// <para><b>Read <see cref="HighestLevel"/> first.</b> Everything else on this object is the evidence
    /// behind it.</para>
    /// </summary>
    internal sealed class AssemblyComparison
    {
        internal AssemblyComparison(
            string leftName,
            string rightName,
            IReadOnlyList<MethodDifference> methodDifferences,
            IReadOnlyList<ApiDifference> apiDifferences,
            IReadOnlyList<ApiChange> changes,
            IReadOnlyList<InertDifference> inertDifferences,
            IReadOnlyList<string> warnings,
            IReadOnlyList<string> notes)
        {
            LeftName = leftName;
            RightName = rightName;
            MethodDifferences = methodDifferences;
            ApiDifferences = apiDifferences;
            Changes = changes;
            InertDifferences = inertDifferences;
            Warnings = warnings;
            Notes = notes;
        }

        internal string LeftName { get; }

        internal string RightName { get; }

        /// <summary>Methods added, removed, or whose normalised body moved.</summary>
        internal IReadOnlyList<MethodDifference> MethodDifferences { get; }

        /// <summary>Externally visible members added, removed or changed, before classification.</summary>
        internal IReadOnlyList<ApiDifference> ApiDifferences { get; }

        /// <summary>
        /// Every surface difference classified by severity, worst first — see
        /// <see cref="BreakingChangeClassifier"/>.
        /// </summary>
        internal IReadOnlyList<ApiChange> Changes { get; }

        /// <summary>Differences that are present on every rebuild and mean nothing.</summary>
        internal IReadOnlyList<InertDifference> InertDifferences { get; }

        /// <summary>
        /// Things the reader must not assume this comparison covered — precompiled native code above all.
        ///
        /// <para><b>Read these before acting on the level.</b> A warning here is the difference between "no
        /// logic changed" and "no <i>managed</i> logic changed, and there is a second payload I did not
        /// read".</para>
        /// </summary>
        internal IReadOnlyList<string> Warnings { get; }

        /// <summary>
        /// Facts that are neither a finding nor a blind spot — the assembly version, the referenced-assembly
        /// set. Reported, and deliberately outside <see cref="HighestLevel"/> and
        /// <see cref="BreaksNoConsumer"/>: every release moves the assembly version, so counting it would make
        /// both answers useless on exactly the comparisons this tool exists for.
        /// </summary>
        internal IReadOnlyList<string> Notes { get; }

        /// <summary>
        /// The single answer: the worst thing found.
        ///
        /// <para>Levels 3 and above come from the classifier. Level 2 is the case where compiled logic moved
        /// and the surface did not, and level 1 is the case where only build stamps differ — the result that
        /// says a version bump can be taken with no measurement owed.</para>
        /// </summary>
        internal ChangeLevel HighestLevel
        {
            get
            {
                var highest = ChangeLevel.None;

                // BOUND: one iteration per classified change.
                foreach (var change in Changes)
                {
                    if (change.Level > highest)
                    {
                        highest = change.Level;
                    }
                }

                if (highest > ChangeLevel.InternalOnly)
                {
                    return highest;
                }

                // An additive-only surface change still means the IL moved, so InternalOnly can never be an
                // upgrade here — but a surface that did not move at all leaves the IL as the deciding fact.
                if (MethodDifferences.Count > 0 && highest < ChangeLevel.InternalOnly)
                {
                    highest = ChangeLevel.InternalOnly;
                }

                if (highest == ChangeLevel.None && InertDifferences.Count > 0)
                {
                    highest = ChangeLevel.Inert;
                }

                return highest;
            }
        }

        /// <summary>Classified changes at or above <paramref name="level"/>.</summary>
        internal IReadOnlyList<ApiChange> AtOrAbove(ChangeLevel level)
        {
            var found = new List<ApiChange>();

            // BOUND: one iteration per classified change.
            foreach (var change in Changes)
            {
                if (change.Level >= level)
                {
                    found.Add(change);
                }
            }

            return found;
        }

        /// <summary>Classified changes at exactly <paramref name="level"/>.</summary>
        internal IReadOnlyList<ApiChange> At(ChangeLevel level)
        {
            var found = new List<ApiChange>();

            // BOUND: one iteration per classified change.
            foreach (var change in Changes)
            {
                if (change.Level == level)
                {
                    found.Add(change);
                }
            }

            return found;
        }

        /// <summary>
        /// Nothing here breaks a consumer: the worst finding is additive or below, and nothing was flagged as
        /// outside what this comparison covers.
        ///
        /// <para><b>Deliberately not called "safe to take".</b> It was, and the first test written against it
        /// asserted the opposite of what it returns, because "safe" runs two questions together: <i>does this
        /// break anyone</i> and <i>do I have to retest</i>. An internal-only change breaks nobody and still
        /// needs a retest. The two questions are now two properties — see <see cref="RequiresRetest"/> — and
        /// neither name can be read as the other.</para>
        /// </summary>
        internal bool BreaksNoConsumer
        {
            get { return HighestLevel <= ChangeLevel.Additive && Warnings.Count == 0; }
        }

        /// <summary>
        /// Compiled logic moved, so the behaviour is not proven unchanged.
        ///
        /// <para>False only at <see cref="ChangeLevel.Inert"/> and <see cref="ChangeLevel.None"/> — the case
        /// where identical IL on the same runtime means there is nothing to measure.</para>
        /// </summary>
        internal bool RequiresRetest
        {
            get { return HighestLevel >= ChangeLevel.InternalOnly || Warnings.Count > 0; }
        }

        /// <summary>No method was added, removed, or had its body change.</summary>
        internal bool IlIdentical
        {
            get { return MethodDifferences.Count == 0; }
        }

        /// <summary>
        /// No method that exists on <b>both</b> sides had its body change.
        ///
        /// <para>Separate from <see cref="IlIdentical"/> on purpose: adding a method is a real difference, and
        /// it is also the case where a comparator that reads raw token bytes falsely reports every neighbouring
        /// method as changed. This property is what pins that.</para>
        /// </summary>
        internal bool SharedMethodBodiesIdentical
        {
            get
            {
                // BOUND: one iteration per reported method difference.
                foreach (var difference in MethodDifferences)
                {
                    if (difference.Kind == DifferenceKind.Changed)
                    {
                        return false;
                    }
                }

                return true;
            }
        }

        internal bool PublicApiIdentical
        {
            get { return ApiDifferences.Count == 0; }
        }

        /// <summary>Method keys whose bodies changed — the "where did it change" answer.</summary>
        internal IReadOnlyList<string> ChangedMethods
        {
            get
            {
                var changed = new List<string>();

                // BOUND: one iteration per reported method difference.
                foreach (var difference in MethodDifferences)
                {
                    if (difference.Kind == DifferenceKind.Changed)
                    {
                        changed.Add(difference.Method);
                    }
                }

                return changed;
            }
        }

        /// <summary>A report a human can read, and the failure message when an assertion on this fails.</summary>
        internal string Report()
        {
            var report = new StringBuilder();

            report.Append(LeftName).Append("  ->  ").Append(RightName).Append('\n');
            report.Append("highest level: ").Append((int)HighestLevel).Append(' ').Append(HighestLevel)
                .Append('\n');

            AppendSection(report, "WARNINGS (what was NOT examined)", Warnings);
            AppendSection(report, "notes", Notes);

            // Worst first: the reader is deciding whether to ship, and the decision is made by the top line.
            // BOUND: one iteration per level, from the worst down to Additive.
            for (var level = ChangeLevel.SilentBehaviourChange; level >= ChangeLevel.Additive; level--)
            {
                var atLevel = At(level);

                AppendSection(report, "LEVEL " + (int)level + " " + level + " (" + atLevel.Count + ")",
                    Render(atLevel));
            }

            AppendSection(report, "methods (" + MethodDifferences.Count + ")", Render(MethodDifferences));
            AppendSection(report, "inert (" + InertDifferences.Count + ")", Render(InertDifferences));

            return report.ToString();
        }

        private static IReadOnlyList<string> Render<T>(IReadOnlyList<T> items)
        {
            var lines = new List<string>();

            // BOUND: one iteration per item.
            foreach (var item in items)
            {
                lines.Add(item.ToString());
            }

            return lines;
        }

        private static void AppendSection(StringBuilder report, string title, IReadOnlyList<string> lines)
        {
            if (lines.Count == 0)
            {
                return;
            }

            report.Append("--- ").Append(title).Append('\n');

            // BOUND: capped at 40 lines; a comparison of two real packages can differ in thousands of methods
            // and a failure message that long is not read, it is scrolled past.
            for (var index = 0; index < lines.Count && index < 40; index++)
            {
                report.Append("  ").Append(lines[index]).Append('\n');
            }

            if (lines.Count > 40)
            {
                report.Append("  … and ").Append(lines.Count - 40).Append(" more\n");
            }
        }
    }
}
