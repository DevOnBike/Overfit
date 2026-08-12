// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Tests.TestSupport.Assemblies
{
    /// <summary>
    /// Turns a list of API differences into an ordered classification: what breaks, how, and for whom.
    ///
    /// <para><b>The rules are not invented here.</b> They follow
    /// <c>dotnet/runtime/docs/coding-guidelines/breaking-change-rules.md</c> and
    /// <i>.NET API changes that affect compatibility</i>, and each finding carries the equivalent
    /// <c>Microsoft.DotNet.ApiCompat</c> diagnostic id where one exists, so a disagreement with the tool
    /// Microsoft ships can be looked up instead of argued about.</para>
    ///
    /// <para><b>Three rules are conditional, and getting the condition wrong is what makes a tool unusable.</b>
    /// Sealing a type, adding an <c>abstract</c> member, and restricting a <c>protected</c> member are all
    /// <b>allowed</b> when the type has no accessible constructor or is already <c>sealed</c> — nobody outside
    /// could have derived from it, so nobody outside can be broken. Likewise, adding a member to an interface
    /// is a binary break only when it has <i>no default implementation</i>. A classifier that skips these
    /// checks reports breaks on changes that cannot break anything, and a report full of false level-5s is a
    /// report nobody opens twice.</para>
    ///
    /// <para><b>It fails closed.</b> A difference that no rule matches is reported as
    /// <see cref="ChangeLevel.BinaryBreaking"/> under <c>AC-UNCLASSIFIED</c>, carrying both descriptors.
    /// Defaulting to <see cref="ChangeLevel.Additive"/> would be the comfortable choice and would silently
    /// downgrade every rule this classifier does not yet have.</para>
    ///
    /// <para><b>What it cannot decide, and says so instead of guessing.</b> Overload ambiguity depends on
    /// consumer call sites; whether an attribute is "observable" depends on who reads it; and behavioural
    /// changes inside a method body — a returned value, an exception, an event order — are not visible in
    /// metadata at all. Those are reported as risks with the reason, never as verdicts.</para>
    /// </summary>
    internal static class BreakingChangeClassifier
    {
        /// <summary>Classifies every difference, plus the assembly-identity changes that break everything.</summary>
        internal static IReadOnlyList<ApiChange> Classify(
            AssemblyFacts left,
            AssemblyFacts right,
            IReadOnlyList<ApiDifference> differences)
        {
            ArgumentNullException.ThrowIfNull(left);
            ArgumentNullException.ThrowIfNull(right);
            ArgumentNullException.ThrowIfNull(differences);

            var changes = new List<ApiChange>();

            ClassifyIdentity(changes, left, right);

            var leftTypes = TypesByName(left);
            var rightTypes = TypesByName(right);
            var removedTypes = new HashSet<string>(StringComparer.Ordinal);
            var addedTypes = new HashSet<string>(StringComparer.Ordinal);

            // BOUND: one iteration per difference.
            foreach (var difference in differences)
            {
                if (difference.Kind == DifferenceKind.Removed && difference.Left.Kind == ApiMemberKind.Type)
                {
                    removedTypes.Add(difference.Left.DeclaringType);
                }

                if (difference.Kind == DifferenceKind.Added && difference.Right.Kind == ApiMemberKind.Type)
                {
                    addedTypes.Add(difference.Right.DeclaringType);
                }
            }

            var handled = PairFieldsAndProperties(changes, differences);

            PairParameterCountChanges(changes, differences, handled);

            // BOUND: one iteration per difference.
            foreach (var difference in differences)
            {
                if (handled.Contains(difference))
                {
                    continue;
                }

                if (difference.Kind == DifferenceKind.Removed)
                {
                    ClassifyRemoval(changes, difference.Left, right, removedTypes);

                    continue;
                }

                if (difference.Kind == DifferenceKind.Added)
                {
                    ClassifyAddition(changes, difference.Right, rightTypes, addedTypes, left);

                    continue;
                }

                ClassifyChange(changes, difference.Left, difference.Right, rightTypes);
            }

            changes.Sort((a, b) =>
            {
                var byLevel = b.Level.CompareTo(a.Level);

                return byLevel != 0 ? byLevel : string.CompareOrdinal(a.Target, b.Target);
            });

            return changes;
        }

        private static void ClassifyIdentity(List<ApiChange> changes, AssemblyFacts left, AssemblyFacts right)
        {
            if (!string.Equals(left.AssemblyName, right.AssemblyName, StringComparison.Ordinal))
            {
                changes.Add(new ApiChange(
                    ChangeLevel.BinaryBreaking,
                    "AC-ASSEMBLY-NAME (CP0003)",
                    right.AssemblyName,
                    "the assembly's simple name changed from '" + left.AssemblyName + "' to '"
                    + right.AssemblyName + "'. This changes the assembly identity, so every compiled consumer "
                    + "fails to bind — it breaks the whole assembly, not one member."));
            }

            if (!string.Equals(left.PublicKey, right.PublicKey, StringComparison.Ordinal))
            {
                changes.Add(new ApiChange(
                    ChangeLevel.BinaryBreaking,
                    "AC-ASSEMBLY-KEY (CP0003)",
                    right.AssemblyName,
                    "the strong-name public key was added, removed or changed. Like a name change this moves "
                    + "the assembly identity and breaks every compiled consumer."));
            }
        }

        private static void ClassifyRemoval(
            List<ApiChange> changes,
            ApiMember member,
            AssemblyFacts right,
            HashSet<string> removedTypes)
        {
            if (member.Kind == ApiMemberKind.Type)
            {
                if (right.ForwardedTypes.Contains(member.DeclaringType))
                {
                    changes.Add(new ApiChange(
                        ChangeLevel.Additive,
                        "AC-TYPE-FORWARDED",
                        member.DeclaringType,
                        "the type is no longer defined in this assembly, but the assembly forwards it with "
                        + "TypeForwardedToAttribute. Moving a type between assemblies that way is allowed and "
                        + "breaks nobody."));

                    return;
                }

                changes.Add(new ApiChange(
                    ChangeLevel.BinaryBreaking,
                    "AC-TYPE-REMOVED (CP0001)",
                    member.DeclaringType,
                    "a visible type was removed or renamed. Every compiled consumer that names it fails with "
                    + "TypeLoadException, and every source consumer fails to compile."));

                return;
            }

            // The type's own removal already says everything; repeating it once per member is how a report on
            // a real package becomes a wall nobody reads.
            if (removedTypes.Contains(member.DeclaringType))
            {
                return;
            }

            changes.Add(new ApiChange(
                ChangeLevel.BinaryBreaking,
                "AC-MEMBER-REMOVED (CP0002)",
                Target(member),
                "a visible " + Describe(member.Kind) + " was removed, renamed, or had its accessibility "
                + "narrowed out of the visible surface. Compiled callers fail at run time with "
                + "MissingMethodException or MissingFieldException — the three causes are indistinguishable "
                + "from metadata and carry the same consequence."));
        }

        private static void ClassifyAddition(
            List<ApiChange> changes,
            ApiMember member,
            Dictionary<string, ApiMember> rightTypes,
            HashSet<string> addedTypes,
            AssemblyFacts left)
        {
            if (member.Kind == ApiMemberKind.Type)
            {
                changes.Add(new ApiChange(
                    ChangeLevel.Additive,
                    "AC-TYPE-ADDED",
                    member.DeclaringType,
                    "a new visible type. Safe, with one caveat this tool cannot check: a new type can collide "
                    + "with a same-named type a consumer already imports with `using`, which is a source break "
                    + "in their code, not in this assembly."));

                return;
            }

            // A member of a type that is itself new is covered by the type's own line.
            if (addedTypes.Contains(member.DeclaringType))
            {
                return;
            }

            rightTypes.TryGetValue(member.DeclaringType, out var owner);

            if (owner is not null && owner.IsInterface)
            {
                ClassifyInterfaceAddition(changes, member, owner);

                return;
            }

            if (member.IsAbstract)
            {
                // Allowed when nobody outside can have derived from the type — the condition that makes the
                // difference between a real break and crying wolf.
                if (owner is null || owner.IsSealed || !owner.HasAccessibleConstructor)
                {
                    changes.Add(new ApiChange(
                        ChangeLevel.Additive,
                        "AC-ABSTRACT-MEMBER-ADDED-SAFE",
                        Target(member),
                        "an abstract member was added to '" + member.DeclaringType + "', which is "
                        + (owner is not null && owner.IsSealed ? "sealed" : "without an accessible constructor")
                        + ". No consumer can have derived from it, so no consumer can be broken. Allowed."));

                    return;
                }

                changes.Add(new ApiChange(
                    ChangeLevel.BinaryBreaking,
                    "AC-ABSTRACT-MEMBER-ADDED (CP0005)",
                    Target(member),
                    "an abstract member was added to '" + member.DeclaringType + "', which is unsealed and has "
                    + "an accessible constructor. Every type outside this assembly that derives from it now "
                    + "fails to load with TypeLoadException, and fails to compile on rebuild."));

                return;
            }

            if (member.Kind == ApiMemberKind.Method && HasMemberNamed(left, member))
            {
                changes.Add(new ApiChange(
                    ChangeLevel.SourceBreaking,
                    "AC-OVERLOAD-ADDED",
                    Target(member),
                    "a new overload was added to an existing method group. Existing binaries keep calling the "
                    + "overload they were bound to, but a consumer that recompiles may bind to this one "
                    + "instead, silently changing behaviour if the two differ. This is a RISK, not a finding: "
                    + "whether it captures any real call site depends on consumer code this tool cannot see."));

                return;
            }

            changes.Add(new ApiChange(
                ChangeLevel.Additive,
                "AC-MEMBER-ADDED",
                Target(member),
                "a new visible " + Describe(member.Kind) + " on an existing type. Breaks nobody."));
        }

        private static void ClassifyInterfaceAddition(
            List<ApiChange> changes,
            ApiMember member,
            ApiMember owner)
        {
            if (member.IsStatic && !member.IsAbstract)
            {
                changes.Add(new ApiChange(
                    ChangeLevel.Additive,
                    "AC-INTERFACE-STATIC-MEMBER-ADDED",
                    Target(member),
                    "a static, non-abstract, non-virtual member was added to interface '"
                    + owner.DeclaringType + "'. Implementers do not have to provide it, so this is allowed."));

                return;
            }

            if (member.HasDefaultImplementation)
            {
                changes.Add(new ApiChange(
                    ChangeLevel.Additive,
                    "AC-INTERFACE-MEMBER-ADDED-WITH-DEFAULT",
                    Target(member),
                    "member '" + member.Name + "' was added to interface '" + owner.DeclaringType
                    + "' WITH a default implementation, which is the allowed form — existing implementers "
                    + "inherit the body and neither fail to load nor fail to compile. Two source-only caveats "
                    + "that do not make this a break by themselves: it requires C# 8 / .NET Core 3.0 of every "
                    + "consumer, and a `ref struct` implementer (C# 13+) cannot use a default implementation "
                    + "and must write its own."));

                return;
            }

            changes.Add(new ApiChange(
                ChangeLevel.BinaryBreaking,
                "AC-INTERFACE-MEMBER-ADDED (CP0006)",
                Target(member),
                "abstract member '" + member.Name + "' was added to interface '" + owner.DeclaringType
                + "' with NO default implementation. Every type outside this assembly that implements '"
                + owner.DeclaringType + "' stops compiling on rebuild, and — the part that is easy to miss — "
                + "the types already compiled fail at run time with TypeLoadException (\"does not have an "
                + "implementation\"). Adding a default implementation makes this allowed."));
        }

        private static void ClassifyChange(
            List<ApiChange> changes,
            ApiMember before,
            ApiMember after,
            Dictionary<string, ApiMember> rightTypes)
        {
            var start = changes.Count;

            if (after.Kind == ApiMemberKind.Type)
            {
                ClassifyTypeChange(changes, before, after);
            }

            if (after.Kind != ApiMemberKind.Type)
            {
                ClassifyMemberChange(changes, before, after, rightTypes);
            }

            ClassifyAccessibility(changes, before, after);
            ClassifyConstraints(changes, before, after);

            if (changes.Count != start)
            {
                return;
            }

            // Fail closed. The descriptors differ — that is why this pair is here at all — so something moved
            // that no rule above names. Calling it additive would be the comfortable answer and would silently
            // downgrade every rule this classifier does not yet have.
            changes.Add(new ApiChange(
                ChangeLevel.BinaryBreaking,
                "AC-UNCLASSIFIED",
                Target(after),
                "the member changed in a way no rule in this classifier recognises, so it is reported at the "
                + "breaking level rather than assumed safe. before: " + before.Descriptor + " | after: "
                + after.Descriptor));
        }

        private static void ClassifyTypeChange(List<ApiChange> changes, ApiMember before, ApiMember after)
        {
            if (before.IsValueType != after.IsValueType)
            {
                changes.Add(new ApiChange(
                    ChangeLevel.BinaryBreaking,
                    "AC-TYPE-KIND-CHANGED",
                    after.DeclaringType,
                    "the type changed between a class and a struct. Every compiled consumer is wrong about how "
                    + "to store, pass and copy it."));
            }

            if (after.IsEnum && !string.Equals(before.EnumUnderlyingType, after.EnumUnderlyingType,
                    StringComparison.Ordinal))
            {
                changes.Add(new ApiChange(
                    ChangeLevel.BinaryBreaking,
                    "AC-ENUM-UNDERLYING-TYPE (CP0010)",
                    after.DeclaringType,
                    "the enum's underlying type changed from " + before.EnumUnderlyingType + " to "
                    + after.EnumUnderlyingType + ". This is a compile-time, binary AND behavioural break, and "
                    + "it can make existing attribute arguments unparsable."));
            }

            if (!string.Equals(before.BaseType, after.BaseType, StringComparison.Ordinal)
                && before.IsValueType == after.IsValueType)
            {
                var wasObject = before.BaseType.EndsWith("System.Object", StringComparison.Ordinal);

                changes.Add(new ApiChange(
                    wasObject ? ChangeLevel.Additive : ChangeLevel.BinaryBreaking,
                    wasObject ? "AC-BASE-TYPE-INTRODUCED" : "AC-BASE-TYPE-REMOVED (CP0007)",
                    after.DeclaringType,
                    wasObject
                        ? "a base class was introduced between this type and System.Object. Allowed, provided "
                          + "it adds no abstract members and changes no behaviour — which this tool checks "
                          + "separately per member but cannot judge as a whole."
                        : "the base type changed from '" + before.BaseType + "' to '" + after.BaseType
                          + "'. Members inherited from the old base are gone from compiled consumers' view."));
            }

            ClassifyInterfaceSet(changes, before, after);

            if (!before.IsSealed && after.IsSealed)
            {
                changes.Add(Conditional(
                    after.HasAccessibleConstructor,
                    "AC-TYPE-SEALED (CP0009)",
                    after.DeclaringType,
                    "the type was sealed and has an accessible constructor, so consumers may have derived from "
                    + "it; those derived types now fail to load.",
                    "the type was sealed but has no accessible constructor, so nobody outside could have "
                    + "derived from it. Allowed."));
            }

            if (!before.IsAbstract && after.IsAbstract)
            {
                changes.Add(Conditional(
                    after.HasAccessibleConstructor,
                    "AC-TYPE-ABSTRACT",
                    after.DeclaringType,
                    "the type became abstract while having an accessible constructor. Consumers that "
                    + "instantiated it fail.",
                    "the type became abstract but has no accessible constructor, so nobody outside could "
                    + "instantiate it. Allowed."));
            }

            // Microsoft's rule is precise and narrower than "the layout changed": adding an instance field is
            // breaking only for a struct that had NO non-public instance field, because only then could a
            // caller skip initialisation (definite assignment, and [SkipLocalsInit] for the binary half).
            if (before.IsValueType && after.IsValueType
                && before.NonPublicInstanceFieldCount == 0
                && after.InstanceFieldCount > before.InstanceFieldCount)
            {
                changes.Add(new ApiChange(
                    ChangeLevel.BinaryBreaking,
                    "AC-STRUCT-FIELD-ADDED",
                    after.DeclaringType,
                    "an instance field was added to a struct that previously had no non-public instance field. "
                    + "Callers could declare a local of it without calling a constructor; they now fail "
                    + "definite-assignment on rebuild, and callers using [SkipLocalsInit] can already read "
                    + "uninitialised stack data."));
            }
        }

        private static void ClassifyInterfaceSet(List<ApiChange> changes, ApiMember before, ApiMember after)
        {
            var was = Split(before.Interfaces);
            var now = Split(after.Interfaces);

            // BOUND: one iteration per interface implemented before.
            foreach (var name in was)
            {
                if (now.Contains(name))
                {
                    continue;
                }

                changes.Add(new ApiChange(
                    ChangeLevel.BinaryBreaking,
                    "AC-INTERFACE-REMOVED (CP0008)",
                    after.DeclaringType,
                    "the type no longer implements '" + name + "'. Every consumer that casts or passes it as "
                    + "that interface fails. (Allowed only when the interface is still reached through a base "
                    + "type or a derived interface, which this tool does not resolve across assemblies.)"));
            }

            // BOUND: one iteration per interface implemented now.
            foreach (var name in now)
            {
                if (was.Contains(name))
                {
                    continue;
                }

                changes.Add(new ApiChange(
                    ChangeLevel.Additive,
                    "AC-INTERFACE-IMPLEMENTED",
                    after.DeclaringType,
                    "the type now also implements '" + name + "'. Generally allowed; Microsoft marks it "
                    + "'requires judgment' because a new interface can change what a designer or serializer "
                    + "emits."));
            }
        }

        private static void ClassifyMemberChange(
            List<ApiChange> changes,
            ApiMember before,
            ApiMember after,
            Dictionary<string, ApiMember> rightTypes)
        {
            ClassifySignature(changes, before, after);
            ClassifyDefaults(changes, before, after);
            ClassifyConstant(changes, before, after, rightTypes);

            if (!string.Equals(before.ParameterNames, after.ParameterNames, StringComparison.Ordinal)
                && string.Equals(before.Signature, after.Signature, StringComparison.Ordinal))
            {
                changes.Add(new ApiChange(
                    ChangeLevel.SourceBreaking,
                    "AC-PARAMETER-RENAMED (CP0017)",
                    Target(after),
                    "a parameter was renamed (" + before.ParameterNames + " -> " + after.ParameterNames
                    + "). The compiled signature is unchanged, so existing binaries keep working; consumers "
                    + "using named arguments, and late-bound callers such as C# `dynamic`, break on rebuild."));
            }

            if (!string.Equals(before.ParameterModifiers, after.ParameterModifiers, StringComparison.Ordinal)
                && string.Equals(before.Signature, after.Signature, StringComparison.Ordinal))
            {
                changes.Add(new ApiChange(
                    ChangeLevel.BinaryBreaking,
                    "AC-PARAMETER-MODIFIER-CHANGED",
                    Target(after),
                    "an `in`/`out`/`ref`/`params` modifier changed (" + before.ParameterModifiers + " -> "
                    + after.ParameterModifiers + "). These are part of the compiled signature."));
            }

            if (before.IsStatic != after.IsStatic)
            {
                changes.Add(new ApiChange(
                    ChangeLevel.BinaryBreaking,
                    "AC-STATIC-CHANGED",
                    Target(after),
                    "the `static` keyword was " + (after.IsStatic ? "added" : "removed")
                    + ". Instance and static calls use different IL, so every compiled caller is wrong."));
            }

            ClassifyVirtuality(changes, before, after);

            if (!before.IsFinal && after.IsFinal)
            {
                changes.Add(new ApiChange(
                    ChangeLevel.BinaryBreaking,
                    "AC-SEALED-MEMBER (CP0018)",
                    Target(after),
                    "the member became `sealed`. A derived type's override can no longer be called through "
                    + "it — on an interface default member this silently prevents an implementer's own "
                    + "version from running."));
            }

            if (before.Kind == ApiMemberKind.Field && !before.IsInitOnly && after.IsInitOnly)
            {
                changes.Add(new ApiChange(
                    ChangeLevel.BinaryBreaking,
                    "AC-FIELD-READONLY-ADDED",
                    Target(after),
                    "`readonly` was added to a field. Consumers that assigned it fail. (Removing `readonly` is "
                    + "allowed, unless the field's static type is a mutable value type.)"));
            }

            if (before.Kind == ApiMemberKind.Field && before.IsInitOnly && !after.IsInitOnly)
            {
                changes.Add(new ApiChange(
                    ChangeLevel.Additive,
                    "AC-FIELD-READONLY-REMOVED",
                    Target(after),
                    "`readonly` was removed from a field, which is allowed unless its static type is a mutable "
                    + "value type — a judgement this tool does not make."));
            }

            if (!before.HasDefaultImplementation && after.HasDefaultImplementation)
            {
                changes.Add(new ApiChange(
                    ChangeLevel.SourceBreaking,
                    "AC-INTERFACE-DEFAULT-ADDED-TO-EXISTING",
                    Target(after),
                    "a default implementation was added to an interface member that previously had none. This "
                    + "is disallowed separately from adding a new member with a default: an implementer that "
                    + "inherits two interfaces can now reach two candidate bodies (the diamond problem)."));
            }
        }

        private static void ClassifySignature(List<ApiChange> changes, ApiMember before, ApiMember after)
        {
            if (string.Equals(before.Signature, after.Signature, StringComparison.Ordinal))
            {
                return;
            }

            var wasParameters = ParametersOf(before.Signature);
            var nowParameters = ParametersOf(after.Signature);

            if (string.Equals(wasParameters, nowParameters, StringComparison.Ordinal))
            {
                changes.Add(new ApiChange(
                    ChangeLevel.BinaryBreaking,
                    "AC-RETURN-TYPE-CHANGED",
                    Target(after),
                    "the return type changed (" + before.Signature + " -> " + after.Signature
                    + "). The return type is part of the compiled signature, so every existing caller fails "
                    + "with MissingMethodException — including the common case of making a method async, where "
                    + "`void M()` becomes `Task M()`."));

                return;
            }

            changes.Add(new ApiChange(
                ChangeLevel.BinaryBreaking,
                "AC-SIGNATURE-CHANGED",
                Target(after),
                "the parameters were added, removed, reordered or retyped (" + before.Signature + " -> "
                + after.Signature + "). Adding a parameter WITH a default value is included: it looks "
                + "source-compatible to callers and is still a different method in metadata."));
        }

        private static void ClassifyDefaults(List<ApiChange> changes, ApiMember before, ApiMember after)
        {
            if (string.Equals(before.Defaults, after.Defaults, StringComparison.Ordinal))
            {
                return;
            }

            var was = ParseDefaults(before.Defaults);
            var now = ParseDefaults(after.Defaults);

            // BOUND: one iteration per parameter that had a default before.
            foreach (var pair in was)
            {
                if (!now.TryGetValue(pair.Key, out var value))
                {
                    changes.Add(new ApiChange(
                        ChangeLevel.SourceBreaking,
                        "AC-DEFAULT-REMOVED",
                        Target(after),
                        "the default value of parameter '" + pair.Key + "' was removed. Not a binary break — "
                        + "compiled callers already baked the old value in — but every source caller that "
                        + "omitted the argument stops compiling."));

                    continue;
                }

                if (string.Equals(pair.Value, value, StringComparison.Ordinal))
                {
                    continue;
                }

                changes.Add(new ApiChange(
                    ChangeLevel.SilentBehaviourChange,
                    "AC-DEFAULT-VALUE-CHANGED",
                    Target(after),
                    "the default value of parameter '" + pair.Key + "' changed from " + pair.Value + " to "
                    + value + ". An optional argument is copied into the CALLER's IL at the caller's compile "
                    + "time: consumers that do not rebuild keep passing " + pair.Value
                    + " forever and never see an error, and consumers that do rebuild silently start passing "
                    + value + ". Nothing anywhere reports either."));
            }

            // BOUND: one iteration per parameter that has a default now.
            foreach (var pair in now)
            {
                if (!was.ContainsKey(pair.Key))
                {
                    changes.Add(new ApiChange(
                        ChangeLevel.Additive,
                        "AC-DEFAULT-ADDED",
                        Target(after),
                        "parameter '" + pair.Key + "' gained a default value of " + pair.Value
                        + ", which lets new callers omit it and breaks none of the existing ones."));
                }
            }
        }

        private static void ClassifyConstant(
            List<ApiChange> changes,
            ApiMember before,
            ApiMember after,
            Dictionary<string, ApiMember> rightTypes)
        {
            if (string.Equals(before.ConstantValue, after.ConstantValue, StringComparison.Ordinal))
            {
                return;
            }

            if (!after.IsLiteral)
            {
                return;
            }

            rightTypes.TryGetValue(after.DeclaringType, out var owner);

            var isEnumMember = owner is not null && owner.IsEnum;

            changes.Add(new ApiChange(
                ChangeLevel.SilentBehaviourChange,
                isEnumMember ? "AC-ENUM-VALUE-CHANGED (CP0011)" : "AC-CONST-VALUE-CHANGED",
                Target(after),
                (isEnumMember ? "the value of enum member '" : "the value of public const '")
                + after.Name + "' changed from " + before.ConstantValue + " to " + after.ConstantValue
                + ". A literal is copied into every consumer's IL when THEY compile. They do not fail, they "
                + "do not need to rebuild, and they are already wrong — they keep using "
                + before.ConstantValue + " forever."
                + (isEnumMember
                    ? " Anything persisted, serialised or compared against the old number is now mismatched."
                    : string.Empty)));
        }

        private static void ClassifyVirtuality(List<ApiChange> changes, ApiMember before, ApiMember after)
        {
            if (before.IsVirtual && !after.IsVirtual)
            {
                changes.Add(new ApiChange(
                    ChangeLevel.BinaryBreaking,
                    "AC-VIRTUAL-REMOVED (CP0012)",
                    Target(after),
                    "`virtual` or `abstract` was removed. Every override in a consumer's derived type now "
                    + "overrides nothing and fails to load."));

                return;
            }

            if (!before.IsVirtual && after.IsVirtual)
            {
                changes.Add(new ApiChange(
                    ChangeLevel.BinaryBreaking,
                    "AC-VIRTUAL-ADDED (CP0013)",
                    Target(after),
                    "`virtual` was added to a member that was not virtual. Counter-intuitive but disallowed: "
                    + "the C# compiler increasingly optimises `callvirt` into a direct call when the target is "
                    + "non-virtual, so an existing consumer can keep calling the base body non-virtually — and "
                    + "other .NET languages never emitted `callvirt` in the first place."));

                return;
            }

            if (!before.IsAbstract && after.IsAbstract)
            {
                changes.Add(new ApiChange(
                    ChangeLevel.BinaryBreaking,
                    "AC-VIRTUAL-MADE-ABSTRACT",
                    Target(after),
                    "a virtual member became abstract. It no longer provides a body, so a derived type that "
                    + "relied on inheriting one must now supply its own."));

                return;
            }

            if (before.IsAbstract && !after.IsAbstract && after.IsVirtual)
            {
                changes.Add(new ApiChange(
                    ChangeLevel.Additive,
                    "AC-ABSTRACT-MADE-VIRTUAL",
                    Target(after),
                    "an abstract member became virtual: it now supplies a body that derived types may keep "
                    + "overriding. Allowed."));
            }
        }

        private static void ClassifyAccessibility(List<ApiChange> changes, ApiMember before, ApiMember after)
        {
            if (string.Equals(before.Accessibility, after.Accessibility, StringComparison.Ordinal))
            {
                return;
            }

            var was = AccessibilityRank(before.Accessibility);
            var now = AccessibilityRank(after.Accessibility);

            if (now < was)
            {
                changes.Add(new ApiChange(
                    ChangeLevel.BinaryBreaking,
                    "AC-VISIBILITY-REDUCED (CP0019)",
                    Target(after),
                    "visibility was reduced from " + before.Accessibility + " to " + after.Accessibility
                    + ". (Restricting a protected member IS allowed when the type has no accessible "
                    + "constructor or is sealed — a case this rule does not distinguish, so check the type "
                    + "before acting on it.)"));

                return;
            }

            changes.Add(new ApiChange(
                ChangeLevel.Additive,
                "AC-VISIBILITY-EXPANDED (CP0020)",
                Target(after),
                "visibility was expanded from " + before.Accessibility + " to " + after.Accessibility
                + ", which is allowed for a member that is not virtual."));
        }

        private static void ClassifyConstraints(List<ApiChange> changes, ApiMember before, ApiMember after)
        {
            if (string.Equals(before.Constraints, after.Constraints, StringComparison.Ordinal))
            {
                return;
            }

            changes.Add(new ApiChange(
                ChangeLevel.SourceBreaking,
                "AC-GENERIC-CONSTRAINT-CHANGED",
                Target(after),
                "generic constraints changed (" + Show(before.Constraints) + " -> " + Show(after.Constraints)
                + "). Tightening one stops consumers compiling with type arguments that used to be legal, and "
                + "an existing compiled instantiation can fail its constraint check at run time. Worth knowing: "
                + "Microsoft.DotNet.ApiCompat does not detect this at all (dotnet/sdk#39659)."));
        }

        /// <summary>
        /// Pairs a removed field with an added property of the same name, and the reverse.
        ///
        /// <para>These arrive as two unrelated differences because the member kind is part of the match key.
        /// Reported separately they read as an unremarkable removal plus an unremarkable addition; reported
        /// together they are one of the most common accidental binary breaks there is — a field access compiles
        /// to <c>ldfld</c> and a property access to <c>callvirt</c>, so no compiled consumer survives it.</para>
        /// </summary>
        private static HashSet<ApiDifference> PairFieldsAndProperties(
            List<ApiChange> changes,
            IReadOnlyList<ApiDifference> differences)
        {
            var handled = new HashSet<ApiDifference>();

            // BOUND: one iteration per difference.
            foreach (var removal in differences)
            {
                if (removal.Kind != DifferenceKind.Removed || handled.Contains(removal))
                {
                    continue;
                }

                if (removal.Left.Kind != ApiMemberKind.Field && removal.Left.Kind != ApiMemberKind.Property)
                {
                    continue;
                }

                // BOUND: one iteration per difference.
                foreach (var addition in differences)
                {
                    if (addition.Kind != DifferenceKind.Added || handled.Contains(addition))
                    {
                        continue;
                    }

                    var counterpart = addition.Right;
                    var flipped = removal.Left.Kind == ApiMemberKind.Field
                        ? ApiMemberKind.Property
                        : ApiMemberKind.Field;

                    if (counterpart.Kind != flipped
                        || !string.Equals(counterpart.Name, removal.Left.Name, StringComparison.Ordinal)
                        || !string.Equals(counterpart.DeclaringType, removal.Left.DeclaringType,
                            StringComparison.Ordinal))
                    {
                        continue;
                    }

                    handled.Add(removal);
                    handled.Add(addition);

                    changes.Add(new ApiChange(
                        ChangeLevel.BinaryBreaking,
                        "AC-FIELD-PROPERTY-SWAP",
                        Target(counterpart),
                        "'" + removal.Left.Name + "' changed from a " + Describe(removal.Left.Kind) + " to a "
                        + Describe(counterpart.Kind) + ". A field access compiles to ldfld and a property "
                        + "access to a call, so no compiled consumer survives the swap — even though the "
                        + "source of every consumer still reads identically."));

                    break;
                }
            }

            return handled;
        }

        /// <summary>
        /// Pairs a removed method with an added method of the same name whose parameter count differs.
        ///
        /// <para>Adding or removing a parameter changes the match key, so it arrives as an unrelated removal
        /// plus an unrelated addition. Read separately, the addition looks additive and the removal looks like
        /// a deletion; read together they are what they are — the signature changed. This is the shape that
        /// makes <b>adding a parameter with a default value</b> so easy to get wrong: it is source-compatible
        /// for every caller and still a different method in metadata.</para>
        ///
        /// <para>Pairs only when exactly one removal and one addition share the name. With more than one, the
        /// correspondence is genuinely not recoverable from metadata, and inventing a pairing would put a
        /// confident wrong sentence in front of the reader; those fall through and are reported separately.</para>
        /// </summary>
        private static void PairParameterCountChanges(
            List<ApiChange> changes,
            IReadOnlyList<ApiDifference> differences,
            HashSet<ApiDifference> handled)
        {
            var removals = new Dictionary<string, List<ApiDifference>>(StringComparer.Ordinal);
            var additions = new Dictionary<string, List<ApiDifference>>(StringComparer.Ordinal);

            // BOUND: one iteration per difference.
            foreach (var difference in differences)
            {
                if (handled.Contains(difference))
                {
                    continue;
                }

                if (difference.Kind == DifferenceKind.Removed
                    && difference.Left.Kind == ApiMemberKind.Method)
                {
                    Collect(removals, Target(difference.Left), difference);
                }

                if (difference.Kind == DifferenceKind.Added && difference.Right.Kind == ApiMemberKind.Method)
                {
                    Collect(additions, Target(difference.Right), difference);
                }
            }

            // BOUND: one iteration per distinct removed method name.
            foreach (var pair in removals)
            {
                if (pair.Value.Count != 1 || !additions.TryGetValue(pair.Key, out var added)
                    || added.Count != 1)
                {
                    continue;
                }

                var before = pair.Value[0].Left;
                var after = added[0].Right;

                handled.Add(pair.Value[0]);
                handled.Add(added[0]);

                changes.Add(new ApiChange(
                    ChangeLevel.BinaryBreaking,
                    "AC-SIGNATURE-CHANGED",
                    pair.Key,
                    "the parameter list changed (" + before.Signature + " -> " + after.Signature
                    + "). Every compiled caller fails with MissingMethodException. This includes the case that "
                    + "looks harmless — adding a parameter WITH a default value: source callers still compile "
                    + "unchanged, and the method they were compiled against no longer exists."));
            }
        }

        private static void Collect(
            Dictionary<string, List<ApiDifference>> map,
            string key,
            ApiDifference difference)
        {
            if (!map.TryGetValue(key, out var list))
            {
                list = new List<ApiDifference>();
                map[key] = list;
            }

            list.Add(difference);
        }

        private static ApiChange Conditional(
            bool breaking,
            string ruleId,
            string target,
            string breakingMessage,
            string safeMessage)
        {
            return new ApiChange(
                breaking ? ChangeLevel.BinaryBreaking : ChangeLevel.Additive,
                breaking ? ruleId : ruleId + "-SAFE",
                target,
                breaking ? breakingMessage : safeMessage);
        }

        private static Dictionary<string, ApiMember> TypesByName(AssemblyFacts facts)
        {
            var types = new Dictionary<string, ApiMember>(StringComparer.Ordinal);

            // BOUND: one iteration per visible member.
            foreach (var member in facts.PublicApi)
            {
                if (member.Kind == ApiMemberKind.Type)
                {
                    types[member.DeclaringType] = member;
                }
            }

            return types;
        }

        private static bool HasMemberNamed(AssemblyFacts facts, ApiMember member)
        {
            // BOUND: one iteration per visible member.
            foreach (var candidate in facts.PublicApi)
            {
                if (candidate.Kind == member.Kind
                    && string.Equals(candidate.Name, member.Name, StringComparison.Ordinal)
                    && string.Equals(candidate.DeclaringType, member.DeclaringType, StringComparison.Ordinal))
                {
                    return true;
                }
            }

            return false;
        }

        private static Dictionary<string, string> ParseDefaults(string defaults)
        {
            var map = new Dictionary<string, string>(StringComparer.Ordinal);

            if (string.IsNullOrEmpty(defaults))
            {
                return map;
            }

            // BOUND: one iteration per comma-separated entry.
            foreach (var entry in defaults.Split(", ", StringSplitOptions.RemoveEmptyEntries))
            {
                var split = entry.IndexOf(" = ", StringComparison.Ordinal);

                if (split > 0)
                {
                    map[entry[..split]] = entry[(split + 3)..];
                }
            }

            return map;
        }

        private static HashSet<string> Split(string list)
        {
            var set = new HashSet<string>(StringComparer.Ordinal);

            if (string.IsNullOrEmpty(list))
            {
                return set;
            }

            // BOUND: one iteration per comma-separated entry.
            foreach (var entry in list.Split(", ", StringSplitOptions.RemoveEmptyEntries))
            {
                set.Add(entry);
            }

            return set;
        }

        private static string ParametersOf(string signature)
        {
            var close = signature.LastIndexOf(')');

            return close < 0 ? signature : signature[..(close + 1)];
        }

        private static int AccessibilityRank(string accessibility)
        {
            return accessibility switch
            {
                "Public" => 4,
                "NestedPublic" => 4,
                "FamORAssem" => 3,
                "NestedFamORAssem" => 3,
                "Family" => 2,
                "NestedFamily" => 2,
                _ => 1,
            };
        }

        private static string Describe(ApiMemberKind kind)
        {
            return kind switch
            {
                ApiMemberKind.Type => "type",
                ApiMemberKind.Method => "method",
                ApiMemberKind.Field => "field",
                ApiMemberKind.Property => "property",
                _ => "event",
            };
        }

        private static string Show(string value)
        {
            return string.IsNullOrEmpty(value) ? "<none>" : value;
        }

        private static string Target(ApiMember member)
        {
            return member.Kind == ApiMemberKind.Type
                ? member.DeclaringType
                : member.DeclaringType + "." + member.Name;
        }
    }
}
