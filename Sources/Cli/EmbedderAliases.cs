// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Cli
{
    /// <summary>Maps short sentence-embedder aliases to a HuggingFace BERT repo. Unlike chat models (one GGUF),
    /// an embedder is a small directory; <see cref="Files"/> lists the files a <c>SentenceEmbedder</c> needs.</summary>
    internal static class EmbedderAliases
    {
        private static readonly Dictionary<string, string> Map =
            new(StringComparer.OrdinalIgnoreCase)
            {
                ["minilm"] = "sentence-transformers/all-MiniLM-L6-v2",
                ["all-minilm-l6-v2"] = "sentence-transformers/all-MiniLM-L6-v2",
                ["bge"] = "BAAI/bge-small-en-v1.5",
                ["bge-small"] = "BAAI/bge-small-en-v1.5",
                ["bge-small-en-v1.5"] = "BAAI/bge-small-en-v1.5",
                ["e5"] = "intfloat/e5-small-v2",
                ["e5-small"] = "intfloat/e5-small-v2",
                ["e5-small-v2"] = "intfloat/e5-small-v2",
            };

        /// <summary>The files pulled into the embedder's local directory — all three families ship exactly these.</summary>
        public static readonly string[] Files = ["config.json", "vocab.txt", "model.safetensors"];

        public static IReadOnlyCollection<string> Known => Map.Keys;

        /// <summary>Resolves an alias to its HuggingFace repo, or null when it isn't a known embedder alias.</summary>
        public static string? Resolve(string spec) => Map.TryGetValue(spec, out var repo) ? repo : null;
    }
}
