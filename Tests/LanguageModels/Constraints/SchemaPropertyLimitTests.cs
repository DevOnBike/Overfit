// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.Exceptions;
using DevOnBike.Overfit.LanguageModels.Constraints.Schema;

namespace DevOnBike.Overfit.Tests.LanguageModels.Constraints
{
    /// <summary>
    /// The 64-property bound, which used to be enforced three times and announced zero times.
    ///
    /// <para><b>Three sites bounded themselves by 64 and each one changed behaviour silently.</b> The
    /// reachability mask in <see cref="JsonStringTrie"/> stopped at 64, so a property past that could not
    /// be generated — the tracker masked out the characters that spell it. The compiler dropped a
    /// <c>required</c> entry past 64 from its bitmask, so the tracker permitted closing an object missing
    /// it, which is precisely the guarantee constrained decoding exists to provide. And the emitted-property
    /// mask ignored anything past 64, so a duplicate key went undetected.</para>
    ///
    /// <para>Every one of those had a <c>bit &lt; 64</c> guard, so nothing ever crashed and nothing was
    /// ever reported. That is this repository's "truncation without a count" rule exactly: a capacity bound
    /// the caller cannot learn about.</para>
    /// </summary>
    public sealed class SchemaPropertyLimitTests
    {
        private static string SchemaWith(int properties, bool forbidAdditional, int requiredCount = 0)
        {
            var text = new StringBuilder();

            text.Append("{\"type\":\"object\",\"properties\":{");

            for (var i = 0; i < properties; i++)
            {
                text.Append(i == 0 ? "" : ",").Append($"\"p{i:D3}\":{{\"type\":\"string\"}}");
            }

            text.Append('}');

            if (requiredCount > 0)
            {
                text.Append(",\"required\":[");

                for (var i = 0; i < requiredCount; i++)
                {
                    text.Append(i == 0 ? "" : ",").Append($"\"p{i:D3}\"");
                }

                text.Append(']');
            }

            if (forbidAdditional)
            {
                text.Append(",\"additionalProperties\":false");
            }

            return text.Append('}').ToString();
        }

        /// <summary>Exactly at the bound still compiles — the guard must not cost the supported case.</summary>
        [Fact]
        public void SixtyFourPropertiesStillCompile()
        {
            var schema = JsonSchemaCompiler.Compile(
                SchemaWith(JsonStringTrie.MaxTrackedValues, forbidAdditional: true));

            Assert.NotNull(schema);
        }

        /// <summary>
        /// One past the bound is refused, with a message naming the limit rather than a schema that
        /// half-works.
        /// </summary>
        [Fact]
        public void SixtyFivePropertiesAreRefused()
        {
            var error = Assert.Throws<OverfitFormatException>(
                () => JsonSchemaCompiler.Compile(
                    SchemaWith(JsonStringTrie.MaxTrackedValues + 1, forbidAdditional: true)));

            Assert.Contains("64", error.Message, StringComparison.Ordinal);
        }

        /// <summary>
        /// The refusal does not depend on <c>additionalProperties</c>. Two of the three failures — the
        /// unenforced <c>required</c> entry and the untracked emitted property — do not need it, so
        /// refusing only in the forbidden case would leave the worst of the three in place.
        /// </summary>
        [Fact]
        public void TheRefusalDoesNotDependOnAdditionalProperties()
        {
            Assert.Throws<OverfitFormatException>(
                () => JsonSchemaCompiler.Compile(
                    SchemaWith(JsonStringTrie.MaxTrackedValues + 1, forbidAdditional: false)));
        }

        /// <summary>
        /// The case that motivated refusing rather than clamping: a required property past the bound was
        /// dropped from the mask, so the tracker would have allowed closing an object without it.
        /// </summary>
        [Fact]
        public void ARequiredPropertyPastTheBoundIsRefusedRatherThanIgnored()
        {
            Assert.Throws<OverfitFormatException>(
                () => JsonSchemaCompiler.Compile(
                    SchemaWith(70, forbidAdditional: true, requiredCount: 70)));
        }

        /// <summary>
        /// Enum values are a different case and must NOT be refused: their trie never uses the reachability
        /// mask, so the bound does not apply to them. Refusing here would break a legitimate schema for a
        /// limit that does not bind it.
        /// </summary>
        [Fact]
        public void ALargeStringEnumStillCompiles()
        {
            var text = new StringBuilder("{\"type\":\"string\",\"enum\":[");

            for (var i = 0; i < 200; i++)
            {
                text.Append(i == 0 ? "" : ",").Append($"\"v{i:D3}\"");
            }

            var schema = JsonSchemaCompiler.Compile(text.Append("]}").ToString());

            Assert.NotNull(schema);
        }
    }
}
