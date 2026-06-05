// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Cli
{
    /// <summary>Maps short model aliases to a HuggingFace GGUF repo + preferred quant pattern. A full
    /// <c>owner/repo</c> spec is accepted directly (the HF API validates it and lists the actual files).</summary>
    internal static class ModelAliases
    {
        private static readonly Dictionary<string, (string Repo, string Pattern)> Map =
            new(StringComparer.OrdinalIgnoreCase)
            {
                ["qwen"] = ("Qwen/Qwen2.5-3B-Instruct-GGUF", "q4_k_m"),
                ["qwen2.5-3b"] = ("Qwen/Qwen2.5-3B-Instruct-GGUF", "q4_k_m"),
                ["qwen2.5-0.5b"] = ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "q4_k_m"),
                ["qwen2.5-1.5b"] = ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "q4_k_m"),
                ["qwen2.5-7b"] = ("Qwen/Qwen2.5-7B-Instruct-GGUF", "q4_k_m"),
            };

        public static IReadOnlyCollection<string> Known => Map.Keys;

        /// <summary>Resolves a spec to a (repo, quant-pattern): a known alias, or a literal
        /// <c>owner/repo</c>. Returns null when the spec is neither.</summary>
        public static (string Repo, string Pattern)? Resolve(string spec)
        {
            if (Map.TryGetValue(spec, out var mapped))
            {
                return mapped;
            }
            // A literal HuggingFace repo id: exactly one '/', no whitespace.
            if (spec.Contains('/') && !spec.Contains(' ') && spec.IndexOf('/') == spec.LastIndexOf('/'))
            {
                return (spec, "q4_k_m");
            }
            return null;
        }
    }
}
