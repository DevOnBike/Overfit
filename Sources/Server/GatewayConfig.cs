// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using DevOnBike.Overfit.Redaction;

namespace DevOnBike.Overfit.Server
{
    /// <summary>
    /// Loads a redaction-gateway config from JSON so a deployment can change detectors, policy and upstream without
    /// recompiling. Resolves the file into a built <see cref="Redactor"/>, <see cref="RedactionPolicy"/> and upstream
    /// URL. AOT-safe (System.Text.Json source-gen); custom regex rules are interpreted at runtime.
    ///
    /// <para>Shape:</para>
    /// <code>
    /// {
    ///   "upstream": "https://api.openai.com/v1",
    ///   "rules":  { "international": true, "polish": true,
    ///               "custom": [ { "category": "EMPLOYEE_ID", "pattern": "\\bEMP\\d{6}\\b" } ] },
    ///   "policy": { "default": "Redact",
    ///               "categories": { "API_KEY": "Block", "PESEL": "Redact", "IPV4": "Allow" } }
    /// }
    /// </code>
    /// </summary>
    public static class GatewayConfig
    {
        public static (Redactor Redactor, RedactionPolicy Policy, string? Upstream, IReadOnlyList<string> ClientKeys, bool ScanResponses) Load(string path)
        {
            ArgumentException.ThrowIfNullOrEmpty(path);

            using var stream = File.OpenRead(path);
            var file = JsonSerializer.Deserialize(stream, GatewayConfigJsonContext.Default.GatewayConfigFile)
                ?? new GatewayConfigFile();

            return (BuildRedactor(file.Rules), BuildPolicy(file.Policy), file.Upstream, file.ClientKeys ?? [], file.ScanResponses);
        }

        private static Redactor BuildRedactor(RulesConfig? rules)
        {
            // Default (no "rules" block): international detectors only — matches Redactor.CreateDefault().
            rules ??= new RulesConfig();

            var list = new List<RedactionRule>();

            if (rules.International)
            {
                list.AddRange(DefaultRedactionRules.All());
            }

            if (rules.Polish)
            {
                list.AddRange(PolishRedactionRules.All());
            }

            if (rules.Custom is not null)
            {
                foreach (var custom in rules.Custom)
                {
                    if (!string.IsNullOrEmpty(custom.Category) && !string.IsNullOrEmpty(custom.Pattern))
                    {
                        // Runtime regex (no source-gen) — AOT-compatible, just not the compiled fast path.
                        list.Add(new RedactionRule(custom.Category, new Regex(custom.Pattern)));
                    }
                }
            }

            return new Redactor(list.ToArray());
        }

        private static RedactionPolicy BuildPolicy(PolicyConfig? policy)
        {
            if (policy is null)
            {
                return RedactionPolicy.Default();
            }

            var map = new Dictionary<string, RedactionAction>(StringComparer.Ordinal);
            if (policy.Categories is not null)
            {
                foreach (var pair in policy.Categories)
                {
                    if (Enum.TryParse<RedactionAction>(pair.Value, ignoreCase: true, out var action))
                    {
                        map[pair.Key] = action;
                    }
                }
            }

            var defaultAction = RedactionAction.Redact;
            if (!string.IsNullOrEmpty(policy.Default)
                && Enum.TryParse<RedactionAction>(policy.Default, ignoreCase: true, out var parsed))
            {
                defaultAction = parsed;
            }

            return new RedactionPolicy(map, defaultAction);
        }
    }

    public sealed class GatewayConfigFile
    {
        [JsonPropertyName("upstream")] public string? Upstream { get; set; }
        [JsonPropertyName("rules")] public RulesConfig? Rules { get; set; }
        [JsonPropertyName("policy")] public PolicyConfig? Policy { get; set; }

        /// <summary>Gateway-issued client keys callers must present. Prefer the env var for real secrets.</summary>
        [JsonPropertyName("clientKeys")] public List<string>? ClientKeys { get; set; }

        /// <summary>Scan model responses and mask any model-generated secrets/PII (non-streaming). Default off.</summary>
        [JsonPropertyName("scanResponses")] public bool ScanResponses { get; set; }
    }

    public sealed class RulesConfig
    {
        [JsonPropertyName("international")] public bool International { get; set; } = true;
        [JsonPropertyName("polish")] public bool Polish { get; set; }
        [JsonPropertyName("custom")] public List<CustomRuleConfig>? Custom { get; set; }
    }

    public sealed class CustomRuleConfig
    {
        [JsonPropertyName("category")] public string Category { get; set; } = string.Empty;
        [JsonPropertyName("pattern")] public string Pattern { get; set; } = string.Empty;
    }

    public sealed class PolicyConfig
    {
        [JsonPropertyName("default")] public string? Default { get; set; }
        [JsonPropertyName("categories")] public Dictionary<string, string>? Categories { get; set; }
    }

    [JsonSerializable(typeof(GatewayConfigFile))]
    internal sealed partial class GatewayConfigJsonContext : JsonSerializerContext
    {
    }
}
