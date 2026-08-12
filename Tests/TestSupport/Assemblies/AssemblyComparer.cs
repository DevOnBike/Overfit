// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;

namespace DevOnBike.Overfit.Tests.TestSupport.Assemblies
{
    /// <summary>
    /// Answers "did the compiled logic actually change" for two builds of the same assembly, separating that
    /// from the differences that are present on every rebuild and mean nothing.
    ///
    /// <para><b>Where this came from.</b> On 2026-08-12 the question was whether System.Numerics.Tensors
    /// 10.0.10 could be swapped for 10.0.11 on a decode hot path. The honest answer needed no benchmark: the
    /// IL region differed by <b>zero bytes</b>, and the entire file-size delta was the Authenticode certificate
    /// table. That is a stronger result than a benchmark can produce — <b>a benchmark can only fail to detect a
    /// difference, whereas the artefact comparison shows there is none</b>. It was done by hand twice, and it
    /// is owed again on every servicing bump, so it is code now.</para>
    ///
    /// <para><b>Checked against that case rather than only against synthetic input</b> (2026-08-12, from the
    /// local package cache, as a throwaway probe — the committed tests take no fixture). For
    /// <c>lib/net10.0</c>: 3114 methods and 1253 visible members on both sides, <b>zero</b> method or API
    /// differences, metadata heap sizes byte-identical, <see cref="ChangeLevel.Inert"/>
    /// on six inert differences. The certificate table went 10,024 -> 10,064 against a file-size delta of
    /// exactly +40 bytes; <c>lib/net8.0</c> went 10,064 -> 10,024 against exactly -40. Not one IL operand token
    /// in either assembly failed to resolve to a stable name.</para>
    ///
    /// <para><b>One thing that run corrected.</b> <c>PE.TimeDateStamp</c> is listed as inert on the grounds of
    /// being a clock reading, and on a reproducible build it is not a clock at all — it is a content hash, and
    /// both these values are negative when read as a signed integer. It is still inert, for a better reason:
    /// it is derived from the build, so it cannot be evidence about the build.</para>
    ///
    /// <para><b>What it cannot answer, stated up front because this is where it will be misused.</b></para>
    /// <list type="bullet">
    ///   <item><b>Native code. THIS TOOL SAYS NOTHING ABOUT IT, AND THAT IS WHERE IT WILL BE MISUSED.</b>
    ///   It reads ECMA-335 and nothing else. The concrete case, examined the same evening as the one above:
    ///   <c>Microsoft.ML.OnnxRuntime</c> 1.28 -> 1.29 is a <b>+339 KB native binary</b> — reworked MLAS
    ///   kernels, new AVX2/VNNI weight paths, changed thread-pool defaults — sitting beside a managed wrapper
    ///   that barely moves. Point this comparator at that package and it will report a low level, entirely
    ///   correctly, because it is answering a narrower question than the reader is asking.
    ///   <b>"No managed change" is not "safe to take."</b> When the payload that does the work is native,
    ///   this result carries no information about it and a measurement is still owed.
    ///   <see cref="AssemblyFacts.HasPrecompiledNativeCode"/> catches only the <i>in-image</i> case
    ///   (ReadyToRun / NGen) and raises a warning; a sibling <c>.so</c> / <c>.dll</c> / <c>.dylib</c> shipped
    ///   in the same package is invisible to it, because it never opens the package.</item>
    ///   <item><b>The JIT.</b> Identical IL on the <i>same runtime</i> produces identical codegen. A runtime or
    ///   SDK bump changes codegen with the IL untouched, and that is a measurement question this cannot
    ///   substitute for.</item>
    ///   <item><b>Behaviour driven by data.</b> Embedded resources, satellite assemblies and config defaults
    ///   outside metadata are not read.</item>
    ///   <item><b>Metadata layout.</b> Token normalisation is deliberate (see <see cref="IlNormalizer"/>), so
    ///   a pure reordering with identical logic reports as identical. For the bit-identity question use
    ///   <see cref="AssemblyFacts.MetadataHeapSizes"/>.</item>
    /// </list>
    /// </summary>
    internal static class AssemblyComparer
    {
        /// <summary>Compares two assemblies on disk.</summary>
        internal static AssemblyComparison CompareFiles(string leftPath, string rightPath)
        {
            using var left = AssemblyFacts.FromFile(leftPath);
            using var right = AssemblyFacts.FromFile(rightPath);

            return Compare(left, right);
        }

        /// <summary>Compares two assemblies already in memory.</summary>
        internal static AssemblyComparison CompareImages(
            byte[] leftImage,
            byte[] rightImage,
            string leftName = "left",
            string rightName = "right")
        {
            using var left = AssemblyFacts.FromImage(leftImage, leftName);
            using var right = AssemblyFacts.FromImage(rightImage, rightName);

            return Compare(left, right);
        }

        /// <summary>Compares two already-read assemblies.</summary>
        internal static AssemblyComparison Compare(AssemblyFacts left, AssemblyFacts right)
        {
            ArgumentNullException.ThrowIfNull(left);
            ArgumentNullException.ThrowIfNull(right);

            var apiDifferences = CompareApi(left, right);

            return new AssemblyComparison(
                left.Name,
                right.Name,
                CompareMethods(left, right),
                apiDifferences,
                BreakingChangeClassifier.Classify(left, right, apiDifferences),
                CompareInert(left, right),
                CollectWarnings(left, right),
                CollectNotes(left, right));
        }

        private static IReadOnlyList<MethodDifference> CompareMethods(AssemblyFacts left, AssemblyFacts right)
        {
            var differences = new List<MethodDifference>();

            // BOUND: one iteration per method in the left assembly.
            foreach (var pair in left.MethodIl)
            {
                if (!right.MethodIl.TryGetValue(pair.Key, out var other))
                {
                    differences.Add(new MethodDifference(pair.Key, DifferenceKind.Removed));

                    continue;
                }

                if (!string.Equals(pair.Value, other, StringComparison.Ordinal))
                {
                    differences.Add(new MethodDifference(pair.Key, DifferenceKind.Changed));
                }
            }

            // BOUND: one iteration per method in the right assembly.
            foreach (var pair in right.MethodIl)
            {
                if (!left.MethodIl.ContainsKey(pair.Key))
                {
                    differences.Add(new MethodDifference(pair.Key, DifferenceKind.Added));
                }
            }

            differences.Sort((a, b) => string.CompareOrdinal(a.Method, b.Method));

            return differences;
        }

        /// <summary>
        /// Pairs members up by <see cref="ApiMember.MatchKey"/> so that a retype, a reorder, a rename or a
        /// changed default reports as one <see cref="DifferenceKind.Changed"/> rather than as an unremarkable
        /// removal-plus-addition.
        ///
        /// <para><b>The documented limit.</b> When several overloads share a match key — same name, same
        /// generic arity, same parameter count — an exact-descriptor match is done first, and only a leftover
        /// of exactly one on each side is called <c>Changed</c>. Two overloads retyped at once therefore report
        /// as two additions and two removals: correct, complete, and less pointed than it could be. Guessing
        /// which retyped overload corresponds to which would be inventing a pairing the metadata does not
        /// carry.</para>
        /// </summary>
        private static IReadOnlyList<ApiDifference> CompareApi(AssemblyFacts left, AssemblyFacts right)
        {
            var differences = new List<ApiDifference>();
            var leftGroups = GroupByMatchKey(left.PublicApi);
            var rightGroups = GroupByMatchKey(right.PublicApi);

            // BOUND: one iteration per distinct match key on the left.
            foreach (var group in leftGroups)
            {
                if (!rightGroups.TryGetValue(group.Key, out var counterparts))
                {
                    // BOUND: one iteration per member in this group.
                    foreach (var member in group.Value)
                    {
                        differences.Add(new ApiDifference(DifferenceKind.Removed, member, null));
                    }

                    continue;
                }

                MatchGroup(differences, group.Value, counterparts);
            }

            // BOUND: one iteration per distinct match key on the right.
            foreach (var group in rightGroups)
            {
                if (leftGroups.ContainsKey(group.Key))
                {
                    continue;
                }

                // BOUND: one iteration per member in this group.
                foreach (var member in group.Value)
                {
                    differences.Add(new ApiDifference(DifferenceKind.Added, null, member));
                }
            }

            differences.Sort((a, b) => string.CompareOrdinal(a.ToString(), b.ToString()));

            return differences;
        }

        private static void MatchGroup(
            List<ApiDifference> differences,
            List<ApiMember> left,
            List<ApiMember> right)
        {
            var unmatchedLeft = new List<ApiMember>(left);
            var unmatchedRight = new List<ApiMember>(right);

            // BOUND: one iteration per left member; each pass removes at most one entry from each list.
            for (var index = unmatchedLeft.Count - 1; index >= 0; index--)
            {
                var candidate = unmatchedRight.FindIndex(
                    member => string.Equals(member.Descriptor, unmatchedLeft[index].Descriptor,
                        StringComparison.Ordinal));

                if (candidate < 0)
                {
                    continue;
                }

                unmatchedRight.RemoveAt(candidate);
                unmatchedLeft.RemoveAt(index);
            }

            if (unmatchedLeft.Count == 1 && unmatchedRight.Count == 1)
            {
                differences.Add(new ApiDifference(
                    DifferenceKind.Changed, unmatchedLeft[0], unmatchedRight[0]));

                return;
            }

            // BOUND: one iteration per remaining left member.
            foreach (var member in unmatchedLeft)
            {
                differences.Add(new ApiDifference(DifferenceKind.Removed, member, null));
            }

            // BOUND: one iteration per remaining right member.
            foreach (var member in unmatchedRight)
            {
                differences.Add(new ApiDifference(DifferenceKind.Added, null, member));
            }
        }

        private static Dictionary<string, List<ApiMember>> GroupByMatchKey(IReadOnlyList<ApiMember> members)
        {
            var groups = new Dictionary<string, List<ApiMember>>(StringComparer.Ordinal);

            // BOUND: one iteration per member.
            foreach (var member in members)
            {
                if (!groups.TryGetValue(member.MatchKey, out var group))
                {
                    group = new List<ApiMember>();
                    groups[member.MatchKey] = group;
                }

                group.Add(member);
            }

            return groups;
        }

        private static IReadOnlyList<InertDifference> CompareInert(AssemblyFacts left, AssemblyFacts right)
        {
            var differences = new List<InertDifference>();

            Add(differences, "Mvid", left.Mvid.ToString(), right.Mvid.ToString());
            Add(differences, "AssemblyInformationalVersion",
                left.InformationalVersion, right.InformationalVersion);
            Add(differences, "PE.TimeDateStamp",
                left.PeTimeDateStamp.ToString(CultureInfo.InvariantCulture),
                right.PeTimeDateStamp.ToString(CultureInfo.InvariantCulture));
            Add(differences, "PE.CheckSum",
                left.PeCheckSum.ToString(CultureInfo.InvariantCulture),
                right.PeCheckSum.ToString(CultureInfo.InvariantCulture));
            Add(differences, "AuthenticodeCertificateTable.Size",
                left.CertificateTableSize.ToString(CultureInfo.InvariantCulture),
                right.CertificateTableSize.ToString(CultureInfo.InvariantCulture));
            Add(differences, "StrongNameSignature.Size",
                left.StrongNameSignatureSize.ToString(CultureInfo.InvariantCulture),
                right.StrongNameSignatureSize.ToString(CultureInfo.InvariantCulture));
            Add(differences, "DebugDirectory", left.DebugDirectory, right.DebugDirectory);

            return differences;
        }

        private static void Add(List<InertDifference> differences, string kind, string left, string right)
        {
            if (string.Equals(left, right, StringComparison.Ordinal))
            {
                return;
            }

            differences.Add(new InertDifference(kind, left ?? "<absent>", right ?? "<absent>"));
        }

        /// <summary>
        /// Only one thing: what this comparison could <b>not see</b>.
        ///
        /// <para><b>Narrowed on 2026-08-12 after running it on real packages, and the reason is worth keeping.</b>
        /// This list also carried the assembly version and the referenced-assembly set, which are facts the
        /// comparison sees perfectly well. The effect was that <see cref="AssemblyComparison.BreaksNoConsumer"/>
        /// came back <c>false</c> for System.Numerics.Tensors 10.0.10 -> 10.0.11 — a comparison with zero
        /// findings at every level — because a servicing release naturally moves the assembly version. A
        /// property that is false for every release is not a property, it is noise, and it would have been the
        /// first thing anyone stopped reading.</para>
        ///
        /// <para>So a warning now means exactly one thing: <b>a limit on what was examined</b>. Those facts
        /// moved to <see cref="AssemblyComparison.Notes"/>, where they are reported and do not pretend to be
        /// either a finding or a blind spot.</para>
        /// </summary>
        private static IReadOnlyList<string> CollectWarnings(AssemblyFacts left, AssemblyFacts right)
        {
            var warnings = new List<string>();

            if (left.HasPrecompiledNativeCode || right.HasPrecompiledNativeCode)
            {
                warnings.Add(
                    "at least one image carries precompiled native code (ReadyToRun/NGen): left="
                    + left.HasPrecompiledNativeCode + " right=" + right.HasPrecompiledNativeCode
                    + ". Identical IL does NOT imply identical execution here — the native payload is not read.");
            }

            if (!left.IsIlOnly || !right.IsIlOnly)
            {
                warnings.Add(
                    "at least one image is not IL-only: left.IsIlOnly=" + left.IsIlOnly
                    + " right.IsIlOnly=" + right.IsIlOnly + ".");
            }

            if (left.UnresolvedTokenKinds.Count > 0 || right.UnresolvedTokenKinds.Count > 0)
            {
                warnings.Add(
                    "some IL operand tokens could not be resolved to a stable name and fell back to a row "
                    + "number, which can produce a false difference: left=["
                    + string.Join(", ", left.UnresolvedTokenKinds) + "] right=["
                    + string.Join(", ", right.UnresolvedTokenKinds) + "].");
            }

            return warnings;
        }

        /// <summary>
        /// Facts that are neither a finding nor a blind spot: reported, and deliberately kept out of the level.
        /// </summary>
        private static IReadOnlyList<string> CollectNotes(AssemblyFacts left, AssemblyFacts right)
        {
            var notes = new List<string>();

            if (left.AssemblyVersion != right.AssemblyVersion)
            {
                notes.Add(
                    "assembly version " + left.AssemblyVersion + " -> " + right.AssemblyVersion
                    + ". Expected on any release and not a consumer break on .NET Core, where binding rolls "
                    + "forward automatically; it did require binding redirects on .NET Framework. A change to "
                    + "the assembly NAME or public key is a different matter and is reported at level 5.");
            }

            var before = string.Join(", ", left.ReferencedAssemblies);
            var after = string.Join(", ", right.ReferencedAssemblies);

            if (!string.Equals(before, after, StringComparison.Ordinal))
            {
                // Reference VERSIONS are deliberately excluded from the printed type names the IL comparison
                // uses (see SignatureTypeNameProvider), because including them makes a dependency rebind look
                // like every method changed. Excluded there means reported here, or dropped entirely.
                notes.Add("referenced assemblies differ: [" + before + "] -> [" + after + "].");
            }

            return notes;
        }
    }
}
