// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Linq;
using DevOnBike.Overfit.Redaction;
using DevOnBike.Overfit.Server;

namespace DevOnBike.Overfit.Tests.Redaction
{
    /// <summary>
    /// The JSON gateway config resolves into a built redactor (international + Polish + custom rules), policy
    /// (category → action with a default) and upstream — so a deployment reconfigures the gateway without a rebuild.
    /// </summary>
    public sealed class GatewayConfigTests
    {
        [Fact]
        public void Load_BuildsRedactorPolicyUpstream_FromJson()
        {
            const string json = """
            {
              "upstream": "https://api.example.com/v1",
              "rules": {
                "international": true,
                "polish": true,
                "custom": [ { "category": "EMPLOYEE_ID", "pattern": "\\bEMP\\d{4}\\b" } ]
              },
              "policy": {
                "default": "Block",
                "categories": { "EMAIL": "Allow", "EMPLOYEE_ID": "Redact" }
              }
            }
            """;

            var path = Path.Combine(Path.GetTempPath(), $"gwcfg_{Guid.NewGuid():N}.json");
            File.WriteAllText(path, json);
            try
            {
                var (redactor, policy, upstream, _, _) = GatewayConfig.Load(path);

                Assert.Equal("https://api.example.com/v1", upstream);

                // Policy: explicit mappings + the configured default for everything else.
                Assert.Equal(RedactionAction.Allow, policy.ActionFor("EMAIL"));
                Assert.Equal(RedactionAction.Redact, policy.ActionFor("EMPLOYEE_ID"));
                Assert.Equal(RedactionAction.Block, policy.ActionFor("SOMETHING_NEW"));

                // Rules: international (email) + Polish (valid PESEL) + the custom EMPLOYEE_ID all fire.
                var result = redactor.Redact("mail a@b.co, EMP1234, pesel 44051401359");
                var categories = result.Matches.Select(m => m.Category).ToHashSet();
                Assert.Contains("EMAIL", categories);
                Assert.Contains("EMPLOYEE_ID", categories);
                Assert.Contains("PESEL", categories);
            }
            finally
            {
                File.Delete(path);
            }
        }

        [Fact]
        public void Load_EmptyConfig_FallsBackToDefaults()
        {
            var path = Path.Combine(Path.GetTempPath(), $"gwcfg_{Guid.NewGuid():N}.json");
            File.WriteAllText(path, "{}");
            try
            {
                var (redactor, policy, upstream, clientKeys, scanResponses) = GatewayConfig.Load(path);

                Assert.Null(upstream);
                Assert.Empty(clientKeys);
                Assert.False(scanResponses);
                // No "rules" → international only; an email still redacts.
                Assert.True(redactor.Redact("write to a@b.co").HasRedactions);
                // No "policy" → the conservative default (secrets block).
                Assert.Equal(RedactionAction.Block, policy.ActionFor("API_KEY"));
            }
            finally
            {
                File.Delete(path);
            }
        }
    }
}
