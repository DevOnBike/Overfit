// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Analyzers;

namespace DevOnBike.Overfit.Tests.Analyzers
{
    /// <summary>
    /// OVERFIT047 — the ban on interpolation that formats a value with the ambient culture.
    ///
    /// <para><b>The five tests that decide whether the rule is usable are the NEGATIVE ones</b>:
    /// <see cref="StringCreateWithAnInvariantProviderIsNotReported"/>,
    /// <see cref="StringBuilderAppendWithAnInvariantProviderIsNotReported"/>,
    /// <see cref="FormattableStringInvariantIsNotReported"/>,
    /// <see cref="AnAlreadyInvariantToStringIsNotReported"/> and
    /// <see cref="StringCreateAroundAConcatenationIsNotReported"/>. A false positive on any of them condemns
    /// all 42 sites the preceding sweep fixed, which is the shape every one of them was fixed into.</para>
    ///
    /// <para><b>The two StringBuilder tests are written first on purpose.</b> The handler-converted shape is
    /// half the remaining sites in the library and it is the one an obvious implementation misses — and it
    /// can only be tested if <see cref="AnalyzerHarness"/>'s reference set resolves
    /// <c>StringBuilder.AppendInterpolatedStringHandler</c>. If those two snippets stop compiling, every
    /// other row here is measuring a compilation that cannot see the shape under test.</para>
    /// </summary>
    public sealed class CultureSensitiveInterpolationAnalyzerTests
    {
        private static IReadOnlyList<string> Run(string body)
        {
            var source = """
                using System;
                using System.Globalization;
                using System.Numerics;
                using System.Text;

                namespace N
                {
                    public enum Colour
                    {
                        Red
                    }

                    public sealed class Box
                    {
                        public Box(IFormatProvider provider, ref System.Runtime.CompilerServices.DefaultInterpolatedStringHandler handler)
                        {
                            _ = provider;
                            _ = handler.ToStringAndClear();
                        }
                    }

                    public class C
                    {
                        public static string M<T>(T value)
                        {
                            return $"{value:F2}";
                        }

                        public void Body(double d, float f, decimal m, int i, long l, byte b, char c, bool flag,
                            string s, object o, Colour colour, double? n, TimeSpan ts, DateTime dt,
                            DateTimeOffset dto, Half h, BigInteger big, nint np, StringBuilder sb)
                        {
                            BODY
                        }

                        private static void Sink(object value)
                        {
                            _ = value;
                        }
                    }
                }
                """;

            return AnalyzerHarness.Run(new CultureSensitiveInterpolationAnalyzer(), source.Replace("BODY", body));
        }

        // ── Row 10 / Row 11 — the handler-converted shape, written first (see the class summary) ──────────

        /// <summary>
        /// Row 10. <c>sb.Append($"{d:F2}")</c> compiles to the SAME operation shape as
        /// <c>string.Create(InvariantCulture, $"{d:F2}")</c> — the literal becomes an interpolated-string
        /// handler — and uses the ambient culture. An implementation that walks <c>Parts</c> and handles only
        /// <c>IInterpolationOperation</c> is silent here while looking correct on every other row.
        /// </summary>
        [Fact]
        public void StringBuilderAppendOfADoubleIsReported()
        {
            Assert.Equal(["OVERFIT047"], Run("""sb.Append($"{d:F2}");"""));
        }

        /// <summary>
        /// Row 11. The same call with a provider — the caller chose, so the rule has nothing to say. Together
        /// with the row above this pins that the exemption comes from the <c>IFormatProvider</c> parameter and
        /// not from the part shape, which is what mutation M1 measures.
        /// </summary>
        [Fact]
        public void StringBuilderAppendWithAnInvariantProviderIsNotReported()
        {
            Assert.Empty(Run("""sb.Append(CultureInfo.InvariantCulture, $"{d:F2}");"""));
        }

        [Fact]
        public void StringBuilderAppendLineOfADoubleIsReported()
        {
            Assert.Equal(["OVERFIT047"], Run("""sb.AppendLine($"value {d}");"""));
        }

        // ── Rows 1-8 — the type predicate ────────────────────────────────────────────────────────────────

        /// <summary>Row 1.</summary>
        [Fact]
        public void ADoubleWithAFormatSpecifierIsReported()
        {
            Assert.Equal(["OVERFIT047"], Run("""Sink($"{d:F2}");"""));
        }

        /// <summary>
        /// Row 2, and the reason the predicate is keyed on the hole's TYPE rather than on the specifier's
        /// syntax: three of the 42 sites the sweep fixed had no specifier at all, so a syntax-keyed rule
        /// misses exactly the ones a human reviewer also misses.
        /// </summary>
        [Fact]
        public void ADoubleWithNoFormatSpecifierIsReported()
        {
            Assert.Equal(["OVERFIT047"], Run("""Sink($"{d}");"""));
        }

        /// <summary>Row 3.</summary>
        [Fact]
        public void FloatAndDecimalAreReported()
        {
            Assert.Equal(["OVERFIT047", "OVERFIT047"], Run("""Sink($"{f}"); Sink($"{m:F2}");"""));
        }

        /// <summary>Row 4 — the non-<c>SpecialType</c> types, resolved by metadata name.</summary>
        [Fact]
        public void TimeSpanDateTimeAndDateTimeOffsetAreReported()
        {
            Assert.Equal(
                ["OVERFIT047", "OVERFIT047", "OVERFIT047"],
                Run("""Sink($"{ts:g}"); Sink($"{dt}"); Sink($"{dto}");"""));
        }

        /// <summary>
        /// <b>The round-trip date specifiers are NOT culture-sensitive, and the first build of this rule said
        /// they were.</b> Measured 2026-08-17 (.NET 10 / ICU, invariant against pl-PL, ar-SA, sv-SE, fi-FI):
        /// <c>o O s u R r</c> are byte-identical in all five, because the BCL formats those five against
        /// <c>DateTimeFormatInfo.InvariantInfo</c> whatever provider is passed. The plan this rule was built
        /// from specified "fires regardless of format specifier" for dates; the first census run found the one
        /// site in <c>Sources/Anomalies</c> that shape produces — <c>SuppressionStore.cs:58</c>, a
        /// <c>{until:u}</c> that is already correct — which is a FALSE POSITIVE on the exact code the rule
        /// exists to protect.
        /// </summary>
        [Fact]
        public void RoundTripDateSpecifiersAreNotReported()
        {
            Assert.Empty(Run("""Sink($"{dt:o}"); Sink($"{dto:u}"); Sink($"{dt:s}"); Sink($"{dto:R}"); Sink($"{dt:r}"); Sink($"{dto:O}");"""));
        }

        /// <summary>
        /// The other side of the same boundary: <c>:T</c> is a round-trip-looking single letter and is NOT in
        /// the invariant set — measured, it differs under ar-SA and fi-FI. Without this the exemption above
        /// could be widened to "any single letter" and nothing would notice.
        /// </summary>
        [Fact]
        public void ACultureSensitiveDateSpecifierIsStillReported()
        {
            Assert.Equal(["OVERFIT047", "OVERFIT047"], Run("""Sink($"{dt:T}"); Sink($"{dto:g}");"""));
        }

        /// <summary>
        /// <c>TimeSpan</c> is the REVERSE of the date case and the plan had it the same way round: no
        /// specifier, <c>c</c>, <c>t</c> and <c>T</c> are all the invariant constant format (measured
        /// identical across the same five cultures), and only <c>g</c>/<c>G</c> move — they take the
        /// fractional-second separator from the culture.
        /// </summary>
        [Fact]
        public void ATimeSpanWithNoSpecifierOrTheConstantFormatIsNotReported()
        {
            Assert.Empty(Run("""Sink($"{ts}"); Sink($"{ts:c}"); Sink($"{ts:t}"); Sink($"{ts:T}");"""));
        }

        /// <summary>Row 4's <c>TimeSpan</c> case, and now the only <c>TimeSpan</c> shape that fires.</summary>
        [Fact]
        public void ATimeSpanWithAGeneralSpecifierIsReported()
        {
            Assert.Equal(["OVERFIT047", "OVERFIT047"], Run("""Sink($"{ts:g}"); Sink($"{ts:G}");"""));
        }

        /// <summary><c>Half</c> and <c>BigInteger</c>, the other two resolved by metadata name.</summary>
        [Fact]
        public void HalfAndBigIntegerAreReported()
        {
            Assert.Equal(["OVERFIT047", "OVERFIT047"], Run("""Sink($"{h}"); Sink($"{big:N0}");"""));
        }

        /// <summary>
        /// Row 5. <c>double?</c> reports <c>SpecialType.None</c>, so without an explicit <c>Nullable&lt;T&gt;</c>
        /// unwrap the whole nullable family is missed silently — the rule would look correct and cover half
        /// of what it claims.
        /// </summary>
        [Fact]
        public void ANullableDoubleIsReported()
        {
            Assert.Equal(["OVERFIT047"], Run("""Sink($"{n}");"""));
        }

        /// <summary>
        /// Row 6. Integral, character, boolean, enum, string and object holes with no specifier render
        /// identically under every culture this project runs on, and flagging them is the noise that teaches
        /// people the rule is wrong. The residual — a NEGATIVE integral under sv-SE, lt-LT or fi-FI, which
        /// renders U+2212 — is accepted by decision and is written into the rule's own description.
        /// </summary>
        [Fact]
        public void IntegralCharBoolEnumStringAndObjectHolesAreNotReported()
        {
            Assert.Empty(Run("""
                Sink($"{i}"); Sink($"{l}"); Sink($"{b}"); Sink($"{np}"); Sink($"{c}");
                Sink($"{flag}"); Sink($"{colour}"); Sink($"{s}"); Sink($"{o}");
                """));
        }

        /// <summary>Row 7 — an integral hole DOES fire once the specifier is culture-sensitive.</summary>
        [Fact]
        public void AnIntegerWithAGroupingSpecifierIsReported()
        {
            Assert.Equal(["OVERFIT047"], Run("""Sink($"{i:N0}");"""));
        }

        /// <summary>
        /// Row 8. Measured 2026-08-17: <c>:D</c> and <c>:X</c> are byte-identical across seven cultures while
        /// <c>:N0</c> differs in every one of them, which is where the split between the two lists comes from.
        /// </summary>
        [Fact]
        public void AnIntegerWithADecimalOrHexSpecifierIsNotReported()
        {
            Assert.Empty(Run("""Sink($"{i:D}"); Sink($"{i:X4}"); Sink($"{i:0000}");"""));
        }

        /// <summary>A custom specifier is judged by its separators, not by its first character.</summary>
        [Fact]
        public void AnIntegerWithACustomSeparatorSpecifierIsReported()
        {
            Assert.Equal(["OVERFIT047"], Run("""Sink($"{i:#,##0}");"""));
        }

        // ── Rows 9, 11-13, 15 — the exemptions that decide whether the rule is usable ─────────────────────

        /// <summary>Row 9 — the shape all 42 fixed sites were fixed INTO.</summary>
        [Fact]
        public void StringCreateWithAnInvariantProviderIsNotReported()
        {
            Assert.Empty(Run("""Sink(string.Create(CultureInfo.InvariantCulture, $"{d:F2}"));"""));
        }

        /// <summary>
        /// The exemption is the <c>IFormatProvider</c> PARAMETER, not the argument: passing
        /// <c>CurrentCulture</c> explicitly is exempt too. Deliberate — the subject of the rule is "nobody
        /// chose", not "somebody chose badly", which is also CA1305's semantics.
        /// </summary>
        [Fact]
        public void StringCreateWithAnExplicitCurrentCultureIsNotReported()
        {
            Assert.Empty(Run("""Sink(string.Create(CultureInfo.CurrentCulture, $"{d:F2}"));"""));
        }

        /// <summary>Row 12 — the consumer picks the culture and the analyzer cannot see where.</summary>
        [Fact]
        public void FormattableStringInvariantIsNotReported()
        {
            Assert.Empty(Run("""Sink(FormattableString.Invariant($"{d:F2}"));"""));
        }

        /// <summary>
        /// Row 13. The hole's type is <c>string</c> here, so the recommended per-hole fix is exempt for free —
        /// and by the same token this rule cannot see <c>d.ToString("F2")</c> without a provider. CA1305 does
        /// catch that one; the two rules are complementary and neither subsumes the other.
        /// </summary>
        [Fact]
        public void AnAlreadyInvariantToStringIsNotReported()
        {
            Assert.Empty(Run("""Sink($"{d.ToString(CultureInfo.InvariantCulture)}");"""));
        }

        /// <summary>The object-creation half of the exemption walk.</summary>
        [Fact]
        public void AConstructorTakingAFormatProviderIsNotReported()
        {
            Assert.Empty(Run("""Sink(new Box(CultureInfo.InvariantCulture, $"{d:F2}"));"""));
        }

        // ── Rows 14-19 — shape, the exception path, and the escape hatch ─────────────────────────────────

        /// <summary>
        /// Row 14. One diagnostic per interpolated-string EXPRESSION, not per segment: the fix wraps the whole
        /// expression once, so one diagnostic should match one edit.
        /// </summary>
        [Fact]
        public void AConcatenationOfTwoInterpolatedStringsIsReportedOnce()
        {
            Assert.Equal(["OVERFIT047"], Run("""Sink($"a{d:F1}" + $"b{d:F2}");"""));
        }

        /// <summary>Row 15 — the same concatenation, wrapped. Nothing is reported.</summary>
        [Fact]
        public void StringCreateAroundAConcatenationIsNotReported()
        {
            Assert.Empty(Run("""Sink(string.Create(CultureInfo.InvariantCulture, $"a{d:F1}" + $"b{d:F2}"));"""));
        }

        /// <summary>
        /// Row 16, and the one place this rule deliberately parts company with OVERFIT006/OVERFIT014 next to
        /// it. Those exempt the exception path because they are per-call ALLOCATION rules and a throw message
        /// should be informative. This is a DETERMINISM rule, and three of the 42 sites the sweep fixed were
        /// inside <c>throw new ArgumentException</c> — copying the neighbouring skeleton would have dropped
        /// exactly those three.
        /// </summary>
        [Fact]
        public void AnInterpolationInsideAThrowIsReported()
        {
            Assert.Equal(["OVERFIT047"], Run("""if (d > 0) { throw new ArgumentException($"{d}"); }"""));
        }

        /// <summary>
        /// Row 17. An unconstrained type parameter reports <c>SpecialType.None</c> and could be anything, so
        /// the rule cannot know. Stated as a test rather than left implicit because silence here is a
        /// LIMITATION and not a judgement that the site is safe.
        ///
        /// <para>The generic body being pinned is <c>M&lt;T&gt;</c> in the shared scaffold, which is compiled
        /// into EVERY test in this class — so a rule that fired on an unconstrained hole would redden all of
        /// them, not only this one. This test names the property; the scaffold is what enforces it.</para>
        /// </summary>
        [Fact]
        public void AnUnconstrainedGenericHoleIsNotReported()
        {
            Assert.Empty(Run("""Sink(M(d));"""));
        }

        /// <summary>Row 18 — a raw interpolated literal produces the same parts.</summary>
        [Fact]
        public void ARawInterpolatedLiteralIsReported()
        {
            Assert.Equal(["OVERFIT047"], Run("Sink($$\"\"\"value {{d:F2}} here\"\"\");"));
        }

        /// <summary>Row 19 — the standard escape hatch, as everywhere else here.</summary>
        [Fact]
        public void APragmaSuppressesTheDiagnostic()
        {
            Assert.Empty(Run("""
                #pragma warning disable OVERFIT047
                Sink($"{d:F2}");
                #pragma warning restore OVERFIT047
                """));
        }
    }
}
