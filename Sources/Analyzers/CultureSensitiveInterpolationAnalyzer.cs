// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Immutable;
using System.Collections.Generic;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CodeAnalysis.Operations;

namespace DevOnBike.Overfit.Analyzers
{
    /// <summary>
    /// OVERFIT047 — an interpolated string whose value holes are formatted with the AMBIENT culture, so the
    /// same build emits different bytes on different machines. The durable half of the sweep that fixed 42
    /// such sites by hand and could only pin them with an enumerated test that cannot see site 43.
    ///
    /// <para><b>CA1305 is not this rule and that was measured, not assumed</b> (see
    /// <c>docs/measured-baselines.md</c>): it fires on <c>IFormatProvider</c> overload SELECTION, and a bare
    /// <c>$"...{v:P0}..."</c> selects no overload at all — the whole project yielded 2 diagnostics under
    /// <c>AnalysisMode=All</c> and neither was one of the 42. The two rules are complementary in both
    /// directions: CA1305 sees <c>d.ToString("F2")</c>, which this rule cannot (the hole's type is
    /// <c>string</c>), and this rule sees the bare hole, which CA1305 cannot.</para>
    ///
    /// <para><b>Two shapes, not one, and the second is half the sites.</b> Measured 2026-08-17 on Roslyn
    /// 5.0.0: <c>sb.Append($"{d:F2}")</c> and <c>string.Create(InvariantCulture, $"{d:F2}")</c> produce the
    /// IDENTICAL operation shape — the literal is converted to an interpolated-string HANDLER, so its parts
    /// are <see cref="IInterpolatedStringAppendOperation"/> rather than
    /// <see cref="IInterpolationOperation"/>. A rule that walks <c>Parts</c> and handles only the latter
    /// therefore gets the <c>string.Create</c> exemption BY ACCIDENT and is blind to every
    /// <c>StringBuilder.Append</c>/<c>AppendLine</c>. What separates the two is not the part kind but
    /// whether the ENCLOSING call takes an <c>IFormatProvider</c>, which is what this asks.</para>
    ///
    /// <para><b>The exception path is NOT exempt</b>, unlike OVERFIT006/OVERFIT014 which sit next to this
    /// file. Those are per-call ALLOCATION rules and a throw message should be informative; this is a
    /// DETERMINISM rule, and three of the 42 fixed sites were inside <c>throw new ArgumentException</c>.
    /// Copying their skeleton would have dropped exactly those three, so this reports through
    /// <see cref="OperationAnalysisContext.ReportDiagnostic"/> directly and lists no hot-path rule.</para>
    ///
    /// <para>Severity is per directory in <c>.editorconfig</c>: <c>none</c> globally, <c>error</c> where the
    /// backlog is zero and the strings are read by machines. <c>warning</c> is not available as an
    /// intermediate step — <c>aot-guard</c> publishes with <c>TreatWarningsAsErrors</c> across the graph, and
    /// an id placed in <c>WarningsNotAsErrors</c> to escape that also overrides a directory-scoped
    /// <c>error</c> (measured 2026-07-19).</para>
    /// </summary>
    [DiagnosticAnalyzer(LanguageNames.CSharp)]
    public sealed class CultureSensitiveInterpolationAnalyzer : DiagnosticAnalyzer
    {
        public const string DiagnosticId = "OVERFIT047";

        private static readonly DiagnosticDescriptor Rule = new(
            DiagnosticId,
            title: "Culture-sensitive interpolation",
            messageFormat: "Interpolated string formats {0} value(s) with the ambient culture (first: {1}) — the same build emits different bytes on different machines. Wrap it: string.Create(CultureInfo.InvariantCulture, $\"...\").",
            category: "Reliability",
            defaultSeverity: DiagnosticSeverity.Warning,
            isEnabledByDefault: true,
            description:
                "Fires on a float/double/decimal/Half/BigInteger hole whatever the format specifier; on a DateTime or " +
                "DateTimeOffset hole unless the specifier is one of the round-trip five (o O s u R r), which the BCL formats " +
                "against the invariant culture whatever provider is passed; on a TimeSpan hole only for ':g' and ':G', since " +
                "no specifier at all means the invariant constant format; and on an integral or enum hole only when the " +
                "specifier is culture-sensitive (c e f g n p r, or a custom specifier containing '.', ',', '%' or per-mille). " +
                "All of that is measured 2026-08-17 against invariant, pl-PL, ar-SA, sv-SE and fi-FI. Four things it " +
                "deliberately does NOT catch. " +
                "(1) A NEGATIVE integral hole with no specifier: sv-SE, lt-LT and fi-FI render the sign as U+2212 and ar-SA " +
                "prefixes U+061C (measured 2026-08-17, .NET 10 / ICU) — real, but flagging every integral hole to reach it is " +
                "the noise that teaches people the rule is wrong. (2) A value pre-rendered with ToString(\"F2\"), whose hole " +
                "type is string; CA1305 does catch that one and neither rule subsumes the other. (3) An unconstrained generic " +
                "hole, whose type the rule cannot know. (4) A provider the caller chose badly — passing CurrentCulture " +
                "explicitly is exempt, because the subject of this rule is 'nobody chose', not 'somebody chose badly'.");

        public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [Rule];

        /// <summary>Custom-format characters that move with the culture; the last is U+2030, per mille.</summary>
        private static readonly char[] CultureSensitiveCustomChars = ['.', ',', '%', '‰'];

        /// <summary>
        /// Standard numeric/date specifiers whose output moves with the culture. 'd', 'x' and 'b' are absent
        /// on purpose: measured 2026-08-17, ':D' and ':X' are byte-identical across seven cultures while
        /// ':N0' differs in every one of them.
        /// </summary>
        private const string CultureSensitiveStandardChars = "CcEeFfGgNnPpRr";

        /// <summary>
        /// The five round-trip date specifiers, which the BCL formats against
        /// <c>DateTimeFormatInfo.InvariantInfo</c> whatever provider is passed. Measured byte-identical
        /// across invariant, pl-PL, ar-SA, sv-SE and fi-FI, so firing on them is a false positive on correct
        /// code — see <see cref="IsCultureSensitiveHole"/>. Note 'R' sits here AND in the culture-sensitive
        /// numeric set above: round-trip means invariant for a date and means the culture's decimal separator
        /// for a double, which is why this decision is keyed on the hole's type and not on the character.
        /// </summary>
        private const string InvariantDateSpecifiers = "oOsuRr";

        /// <summary>
        /// The only two <c>TimeSpan</c> specifiers that move — they take the fractional-second separator from
        /// the culture. No specifier at all, 'c', 't' and 'T' are the invariant constant format.
        /// </summary>
        private const string CultureSensitiveTimeSpanSpecifiers = "gG";

        public override void Initialize(AnalysisContext context)
        {
            context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
            context.EnableConcurrentExecution();
            context.RegisterCompilationStartAction(OnCompilationStart);
        }

        /// <summary>
        /// The four non-<see cref="SpecialType"/> types and the three exemption types are resolved once per
        /// compilation rather than once per operation. A null result simply means that type is not in this
        /// compilation, and the rule then cannot fire for it there.
        /// </summary>
        private static void OnCompilationStart(CompilationStartAnalysisContext context)
        {
            var known = new KnownTypes(context.Compilation);

            context.RegisterOperationAction(
                operationContext => Analyze(operationContext, known),
                OperationKind.InterpolatedString);
        }

        private static void Analyze(OperationAnalysisContext context, KnownTypes known)
        {
            var interpolated = (IInterpolatedStringOperation)context.Operation;

            // One diagnostic per interpolated-string EXPRESSION, not per hole and not per concatenated
            // segment: the fix is to wrap the whole literal once, so one diagnostic should match one edit.
            // The same shape HotPathStringAnalyzer:57-61 uses for concatenation chains.
            var root = Outermost(interpolated);
            var segments = Segments(root);

            if (segments.Count == 0 || !ReferenceEquals(segments[0], interpolated))
            {
                return;
            }

            if (IsExempt(root, known))
            {
                return;
            }

            var count = 0;
            string? first = null;

            foreach (var segment in segments)
            {
                foreach (var part in segment.Parts)
                {
                    var hole = HoleType(part);

                    if (!IsCultureSensitiveHole(hole.Type, hole.Format, known))
                    {
                        continue;
                    }

                    count++;
                    first ??= hole.Type!.ToDisplayString();
                }
            }

            if (count == 0)
            {
                return;
            }

            context.ReportDiagnostic(Diagnostic.Create(Rule, root.Syntax.GetLocation(), count, first));
        }

        /// <summary>
        /// Walks out of a <c>$"a{x:F1}" + $"b{x:F2}"</c> chain. Both spellings the compiler can produce are
        /// handled: <see cref="IInterpolatedStringAdditionOperation"/> when the target is a handler, and a
        /// plain string-typed <see cref="IBinaryOperation"/> otherwise. Which one appears depends on the
        /// conversion, and a rule that knew only one of them would report twice for one edit in the other.
        /// </summary>
        private static IOperation Outermost(IOperation operation)
        {
            var current = operation;

            while (IsSegmentJoin(current.Parent))
            {
                current = current.Parent!;
            }

            return current;
        }

        private static bool IsSegmentJoin(IOperation? operation)
        {
            if (operation is IInterpolatedStringAdditionOperation)
            {
                return true;
            }

            return operation is IBinaryOperation
            {
                OperatorKind: BinaryOperatorKind.Add,
                Type.SpecialType: SpecialType.System_String
            };
        }

        /// <summary>
        /// Every interpolated-string segment under <paramref name="root"/>, in source order. Iterative rather
        /// than recursive so the depth of a concatenation chain cannot reach the stack.
        /// </summary>
        private static List<IInterpolatedStringOperation> Segments(IOperation root)
        {
            var found = new List<IInterpolatedStringOperation>();
            var pending = new Stack<IOperation>();

            pending.Push(root);

            while (pending.Count > 0)
            {
                var current = pending.Pop();

                if (current is IInterpolatedStringOperation segment)
                {
                    found.Add(segment);
                    continue;
                }

                if (current is IInterpolatedStringAdditionOperation addition)
                {
                    pending.Push(addition.Right);
                    pending.Push(addition.Left);
                    continue;
                }

                if (current is IBinaryOperation binary && IsSegmentJoin(binary))
                {
                    pending.Push(binary.RightOperand);
                    pending.Push(binary.LeftOperand);
                }
            }

            return found;
        }

        /// <summary>
        /// The caller chose a provider, or handed the literal to something that will. Choosing the WRONG
        /// provider is a different rule; the subject here is "nobody chose".
        /// </summary>
        private static bool IsExempt(IOperation root, KnownTypes known)
        {
            var parent = root.Parent;

            if (parent is IConversionOperation conversion && IsAny(conversion.Type, known.FormattableString, known.Formattable))
            {
                return true;
            }

            // The IFormatProvider exemption is gated on the literal actually being HANDLER-converted. Without
            // that gate `string.Format(CultureInfo.InvariantCulture, $"{d:F2}")` would be exempt, and it must
            // not be: there the literal is rendered to a string with the ambient culture BEFORE the provider
            // is ever consulted.
            if (parent is IInterpolatedStringHandlerCreationOperation creation)
            {
                return EnclosingCallTakesFormatProvider(creation, known.FormatProvider);
            }

            return false;
        }

        private static bool IsAny(ITypeSymbol? type, INamedTypeSymbol? first, INamedTypeSymbol? second)
        {
            if (type == null)
            {
                return false;
            }

            return SymbolEqualityComparer.Default.Equals(type, first)
                || SymbolEqualityComparer.Default.Equals(type, second);
        }

        private static bool EnclosingCallTakesFormatProvider(IOperation creation, INamedTypeSymbol? formatProvider)
        {
            if (formatProvider == null || creation.Parent is not IArgumentOperation argument)
            {
                return false;
            }

            var parameters = Parameters(argument.Parent);

            foreach (var parameter in parameters)
            {
                if (SymbolEqualityComparer.Default.Equals(parameter.Type, formatProvider))
                {
                    return true;
                }
            }

            return false;
        }

        private static ImmutableArray<IParameterSymbol> Parameters(IOperation? call)
        {
            if (call is IInvocationOperation invocation)
            {
                return invocation.TargetMethod.Parameters;
            }

            if (call is IObjectCreationOperation creation && creation.Constructor != null)
            {
                return creation.Constructor.Parameters;
            }

            return ImmutableArray<IParameterSymbol>.Empty;
        }

        /// <summary>
        /// The hole's type and format specifier, read from whichever of the two part shapes this is. The
        /// specifier comes from the CONSTANT value rather than the syntax text — the syntax reads <c>:F2</c>
        /// and the constant is <c>F2</c> — and on the handler side it is located by PARAMETER NAME, because
        /// the alignment overload <c>AppendFormatted&lt;T&gt;(T, int, string?)</c> shifts its position.
        /// </summary>
        private static (ITypeSymbol? Type, string? Format) HoleType(IInterpolatedStringContentOperation part)
        {
            if (part is IInterpolationOperation interpolation)
            {
                return (Unwrap(interpolation.Expression).Type, Constant(interpolation.FormatString));
            }

            if (part.Kind != OperationKind.InterpolatedStringAppendFormatted
                || part is not IInterpolatedStringAppendOperation append
                || append.AppendCall is not IInvocationOperation call
                || call.Arguments.Length == 0)
            {
                return (null, null);
            }

            string? format = null;

            foreach (var argument in call.Arguments)
            {
                if (argument.Parameter?.Name == "format")
                {
                    format = Constant(argument.Value);
                }
            }

            return (Unwrap(call.Arguments[0].Value).Type, format);
        }

        private static string? Constant(IOperation? operation)
        {
            if (operation == null)
            {
                return null;
            }

            return operation.ConstantValue.HasValue ? operation.ConstantValue.Value as string : null;
        }

        /// <summary>
        /// Strips compiler-inserted conversions (the boxing to <c>object</c> a handler overload can add)
        /// while leaving a cast written in the source alone — an explicit <c>(object)d</c> is a decision the
        /// author made and the rule should see it as written.
        /// </summary>
        private static IOperation Unwrap(IOperation operation)
        {
            var current = operation;

            while (current is IConversionOperation { IsImplicit: true } conversion)
            {
                current = conversion.Operand;
            }

            return current;
        }

        /// <summary>
        /// Three families, not one — and the split is measured (2026-08-17, .NET 10 / ICU, invariant against
        /// pl-PL, ar-SA, sv-SE, fi-FI):
        ///
        /// <list type="bullet">
        /// <item><b>Real numbers</b> move under every specifier tested (none, F2, N0, G, R, E2, P1), so they
        /// fire unconditionally;</item>
        /// <item><b>Dates</b> move under none, G, g, F, d, D and T — but <c>o O s u R r</c> are BYTE-IDENTICAL
        /// across all five cultures, because the BCL formats those five against
        /// <c>DateTimeFormatInfo.InvariantInfo</c> whatever provider is passed. Firing on them is a false
        /// positive on correct round-trip code, and the sweep this rule guards left exactly one such site
        /// (<c>Sources/Anomalies/Monitoring/SuppressionStore.cs:58</c>, <c>{until:u}</c>) which the first
        /// build of this rule reported;</item>
        /// <item><b><c>TimeSpan</c> is the reverse</b>: no specifier, <c>c</c>, <c>t</c> and <c>T</c> are all
        /// the invariant constant format, and ONLY <c>g</c>/<c>G</c> move (they take the fractional-second
        /// separator from the culture). So it fires only for those two.</item>
        /// </list>
        ///
        /// <para>The plan this rule was built from stated all three families as "fires regardless of format
        /// specifier"; the date and TimeSpan halves of that are refuted above and narrowed here.</para>
        /// </summary>
        private static bool IsCultureSensitiveHole(ITypeSymbol? type, string? format, KnownTypes known)
        {
            var effective = UnwrapNullable(type);

            if (effective == null)
            {
                return false;
            }

            if (IsRealNumber(effective, known))
            {
                return true;
            }

            if (IsDateLike(effective, known))
            {
                return !IsSingle(format, InvariantDateSpecifiers);
            }

            if (SymbolEqualityComparer.Default.Equals(effective, known.TimeSpan))
            {
                return IsSingle(format, CultureSensitiveTimeSpanSpecifiers);
            }

            return IsIntegralOrEnum(effective) && IsCultureSensitiveSpecifier(format);
        }

        /// <summary>
        /// <c>double?</c> reports <see cref="SpecialType.None"/> — measured. Without this unwrap the whole
        /// nullable family is missed silently, which is the worst kind of gap: the rule looks correct.
        /// </summary>
        private static ITypeSymbol? UnwrapNullable(ITypeSymbol? type)
        {
            if (type is INamedTypeSymbol { OriginalDefinition.SpecialType: SpecialType.System_Nullable_T } nullable
                && nullable.TypeArguments.Length == 1)
            {
                return nullable.TypeArguments[0];
            }

            return type;
        }

        /// <summary>
        /// Every one of these moves its decimal separator between the invariant culture and pl-PL, with or
        /// without a format specifier — which is why the three specifier-less sites a syntax-keyed rule would
        /// have missed are reached by a TYPE-keyed one.
        /// </summary>
        private static bool IsRealNumber(ITypeSymbol type, KnownTypes known)
        {
            switch (type.SpecialType)
            {
                case SpecialType.System_Single:
                case SpecialType.System_Double:
                case SpecialType.System_Decimal:
                    return true;
            }

            return SymbolEqualityComparer.Default.Equals(type, known.Half)
                || SymbolEqualityComparer.Default.Equals(type, known.BigInteger);
        }

        private static bool IsDateLike(ITypeSymbol type, KnownTypes known)
        {
            return type.SpecialType == SpecialType.System_DateTime
                || SymbolEqualityComparer.Default.Equals(type, known.DateTimeOffset);
        }

        /// <summary>A one-character specifier drawn from <paramref name="set"/>; anything else is not one.</summary>
        private static bool IsSingle(string? format, string set)
        {
            return format != null && format.Length == 1 && set.IndexOf(format[0]) >= 0;
        }

        private static bool IsIntegralOrEnum(ITypeSymbol type)
        {
            if (type.TypeKind == TypeKind.Enum)
            {
                return true;
            }

            switch (type.SpecialType)
            {
                case SpecialType.System_SByte:
                case SpecialType.System_Byte:
                case SpecialType.System_Int16:
                case SpecialType.System_UInt16:
                case SpecialType.System_Int32:
                case SpecialType.System_UInt32:
                case SpecialType.System_Int64:
                case SpecialType.System_UInt64:
                case SpecialType.System_IntPtr:
                case SpecialType.System_UIntPtr:
                case SpecialType.System_Char:
                    return true;
            }

            return false;
        }

        private static bool IsCultureSensitiveSpecifier(string? format)
        {
            if (string.IsNullOrEmpty(format))
            {
                return false;
            }

            if (IsStandardSpecifier(format!))
            {
                return CultureSensitiveStandardChars.IndexOf(format![0]) >= 0;
            }

            return format!.IndexOfAny(CultureSensitiveCustomChars) >= 0;
        }

        /// <summary>A standard specifier is one letter plus an optional precision; anything else is custom.</summary>
        private static bool IsStandardSpecifier(string format)
        {
            if (!char.IsLetter(format[0]))
            {
                return false;
            }

            for (var i = 1; i < format.Length; i++)
            {
                if (!char.IsDigit(format[i]))
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        /// The seven BCL symbols the rule needs, resolved once per compilation. Every one is a BCL type
        /// reached through the semantic model: <c>Sources/Analyzers</c> holds no project reference to
        /// repository code and must not gain one — it runs inside the compiler and ships nothing.
        ///
        /// <para>A null field means that type is not in the compilation being analyzed, and the rule then
        /// simply cannot fire for it there. That is the correct behaviour rather than a gap: a compilation
        /// without <c>System.Numerics.BigInteger</c> cannot contain a hole of that type.</para>
        /// </summary>
        private sealed class KnownTypes
        {
            public KnownTypes(Compilation compilation)
            {
                TimeSpan = compilation.GetTypeByMetadataName("System.TimeSpan");
                DateTimeOffset = compilation.GetTypeByMetadataName("System.DateTimeOffset");
                Half = compilation.GetTypeByMetadataName("System.Half");
                BigInteger = compilation.GetTypeByMetadataName("System.Numerics.BigInteger");
                FormatProvider = compilation.GetTypeByMetadataName("System.IFormatProvider");
                FormattableString = compilation.GetTypeByMetadataName("System.FormattableString");
                Formattable = compilation.GetTypeByMetadataName("System.IFormattable");
            }

            public INamedTypeSymbol? TimeSpan { get; }

            public INamedTypeSymbol? DateTimeOffset { get; }

            public INamedTypeSymbol? Half { get; }

            public INamedTypeSymbol? BigInteger { get; }

            public INamedTypeSymbol? FormatProvider { get; }

            public INamedTypeSymbol? FormattableString { get; }

            public INamedTypeSymbol? Formattable { get; }
        }
    }
}
