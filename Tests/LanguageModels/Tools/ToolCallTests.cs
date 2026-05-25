// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Tools;

namespace DevOnBike.Overfit.Tests.LanguageModels.Tools
{
    /// <summary>Tests <see cref="ToolCall.TryParse"/> — the parser for the constrained envelope.</summary>
    public sealed class ToolCallTests
    {
        [Fact]
        public void TryParse_ExtractsNameAndRawArguments()
        {
            Assert.True(ToolCall.TryParse(
                "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}", out var call));
            Assert.Equal("get_weather", call.Name);
            Assert.Contains("Paris", call.Arguments);
        }

        [Fact]
        public void TryParse_MissingArguments_DefaultsToEmptyObject()
        {
            Assert.True(ToolCall.TryParse("{\"name\": \"ping\"}", out var call));
            Assert.Equal("ping", call.Name);
            Assert.Equal("{}", call.Arguments);
        }

        [Theory]
        [InlineData("not json")]
        [InlineData("{\"arguments\": {}}")]      // no name
        [InlineData("{\"name\": 5}")]            // name not a string
        [InlineData("[1,2,3]")]                  // not an object
        [InlineData("")]
        public void TryParse_RejectsMalformedOrIncomplete(string json)
        {
            Assert.False(ToolCall.TryParse(json, out _));
        }
    }
}
