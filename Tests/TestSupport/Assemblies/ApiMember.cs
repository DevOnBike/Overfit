// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;

namespace DevOnBike.Overfit.Tests.TestSupport.Assemblies
{
    /// <summary>
    /// One externally visible member, described so that two builds can be compared without loading either.
    ///
    /// <para><b>Two keys, and the split is the whole point.</b> <see cref="MatchKey"/> is deliberately coarse —
    /// declaring type, kind, name, generic arity and parameter <i>count</i>. <see cref="Descriptor"/> is the
    /// full truth. A member whose <see cref="MatchKey"/> survives but whose <see cref="Descriptor"/> moved is
    /// reported as <b>changed</b> rather than as a removal plus an addition, and "changed" is the category that
    /// bites: a reordered or retyped parameter breaks every caller at compile time while an addition breaks
    /// nobody. Reporting a retype as "removed X, added Y" is technically true and hides the severity in two
    /// unremarkable lines.</para>
    ///
    /// <para><b>Every facet is a separate property, not a substring of <see cref="Descriptor"/>.</b>
    /// <see cref="BreakingChangeClassifier"/> has to answer questions like "was <c>virtual</c> removed" and
    /// "does this interface member carry a default implementation", and a classifier that re-parses a display
    /// string to find out is one formatting tweak away from silently classifying everything as additive. The
    /// descriptor is for humans and for equality; the properties are what the rules read.</para>
    /// </summary>
    internal sealed class ApiMember
    {
        internal required string DeclaringType
        {
            get; init;
        }

        internal required ApiMemberKind Kind
        {
            get; init;
        }

        internal required string Name
        {
            get; init;
        }

        internal int GenericArity
        {
            get; init;
        }

        internal int ParameterCount
        {
            get; init;
        }

        /// <summary>Parameter types and return type, with no names — the binary call shape.</summary>
        internal string Signature { get; init; } = string.Empty;

        /// <summary>Parameter names in order, comma separated. Renaming one breaks named-argument callers.</summary>
        internal string ParameterNames { get; init; } = string.Empty;

        /// <summary><c>in</c> / <c>out</c> / <c>ref</c> / <c>params</c> per parameter, in order.</summary>
        internal string ParameterModifiers { get; init; } = string.Empty;

        /// <summary>Optional-parameter defaults, as <c>name = value</c>, comma separated.</summary>
        internal string Defaults { get; init; } = string.Empty;

        /// <summary>Accessibility as ECMA-335 spells it: <c>Public</c>, <c>Family</c>, <c>FamORAssem</c>.</summary>
        internal string Accessibility { get; init; } = string.Empty;

        internal bool IsStatic
        {
            get; init;
        }

        internal bool IsAbstract
        {
            get; init;
        }

        internal bool IsVirtual
        {
            get; init;
        }

        /// <summary><c>final</c> in IL: the member cannot be overridden further.</summary>
        internal bool IsFinal
        {
            get; init;
        }

        /// <summary><c>initonly</c> on a field. Adding it is breaking; removing it is allowed.</summary>
        internal bool IsInitOnly
        {
            get; init;
        }

        /// <summary><c>literal</c> on a field: a <c>const</c>, or an enum member.</summary>
        internal bool IsLiteral
        {
            get; init;
        }

        /// <summary>The compile-time constant of a <c>const</c> field or enum member, copied into consumers.</summary>
        internal string ConstantValue
        {
            get; init;
        }

        /// <summary>
        /// An interface member that carries a body. Decides whether adding it to an interface is a binary break
        /// or an allowed addition, which is the difference between a usable tool and one that cries wolf on
        /// every modern library.
        /// </summary>
        internal bool HasDefaultImplementation
        {
            get; init;
        }

        /// <summary>Generic parameter constraints, rendered in order.</summary>
        internal string Constraints { get; init; } = string.Empty;

        internal bool IsInterface
        {
            get; init;
        }

        internal bool IsSealed
        {
            get; init;
        }

        internal bool IsValueType
        {
            get; init;
        }

        internal bool IsEnum
        {
            get; init;
        }

        internal string BaseType { get; init; } = string.Empty;

        /// <summary>Implemented interfaces, sorted.</summary>
        internal string Interfaces { get; init; } = string.Empty;

        /// <summary>
        /// A <c>public</c> or <c>protected</c> constructor exists.
        ///
        /// <para>Load-bearing, not decoration: sealing a type and adding an abstract member are both
        /// <b>allowed</b> when no accessible constructor exists, because nobody outside can have derived from
        /// it. A classifier that skips this check reports a break on types that cannot be broken.</para>
        /// </summary>
        internal bool HasAccessibleConstructor
        {
            get; init;
        }

        internal string EnumUnderlyingType { get; init; } = string.Empty;

        /// <summary>Instance fields of any accessibility — the input to the struct definite-assignment rule.</summary>
        internal int InstanceFieldCount
        {
            get; init;
        }

        /// <summary>Instance fields that are not public. Zero is the case where adding a field is breaking.</summary>
        internal int NonPublicInstanceFieldCount
        {
            get; init;
        }

        /// <summary>The coarse identity used to pair members up across two builds.</summary>
        internal string MatchKey
        {
            get
            {
                return DeclaringType + "|" + Kind + "|" + Name + "|`" + GenericArity + "|" + ParameterCount;
            }
        }

        /// <summary>Everything that is compared. Two members with equal descriptors are the same API.</summary>
        internal string Descriptor
        {
            get
            {
                var text = new StringBuilder();

                text.Append(Accessibility).Append(' ');
                Append(text, IsStatic, "static");
                Append(text, IsAbstract, "abstract");
                Append(text, IsVirtual, "virtual");
                Append(text, IsFinal, "final");
                Append(text, IsInitOnly, "initonly");
                Append(text, IsLiteral, "literal");
                Append(text, IsInterface, "interface");
                Append(text, IsSealed, "sealed");
                Append(text, IsValueType, "valuetype");
                Append(text, IsEnum, "enum");
                Append(text, HasDefaultImplementation, "default-impl");

                text.Append(Kind).Append(' ').Append(DeclaringType).Append('.').Append(Name);

                if (GenericArity > 0)
                {
                    text.Append('`').Append(GenericArity);
                }

                text.Append(' ').Append(Signature);

                AppendPart(text, "names", ParameterNames);
                AppendPart(text, "modifiers", ParameterModifiers);
                AppendPart(text, "defaults", Defaults);
                AppendPart(text, "const", ConstantValue);
                AppendPart(text, "where", Constraints);
                AppendPart(text, "base", BaseType);
                AppendPart(text, "implements", Interfaces);
                AppendPart(text, "underlying", EnumUnderlyingType);

                if (IsValueType)
                {
                    // Only meaningful on a value type, and including it unconditionally would make every
                    // class report a change whenever a private field was added — which is not observable.
                    text.Append(" fields(").Append(InstanceFieldCount).Append('/')
                        .Append(NonPublicInstanceFieldCount).Append(')');
                }

                return text.ToString();
            }
        }

        public override string ToString()
        {
            return Descriptor;
        }

        private static void Append(StringBuilder text, bool present, string word)
        {
            if (present)
            {
                text.Append(word).Append(' ');
            }
        }

        private static void AppendPart(StringBuilder text, string label, string value)
        {
            if (!string.IsNullOrEmpty(value))
            {
                text.Append(' ').Append(label).Append('(').Append(value).Append(')');
            }
        }
    }
}
