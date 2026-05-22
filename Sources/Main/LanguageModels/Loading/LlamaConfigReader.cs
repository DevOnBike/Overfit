// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.LanguageModels.Loading
{
    /// <summary>
    /// Reads a HuggingFace <c>config.json</c> (Llama / Mistral / Qwen family) into a
    /// <see cref="GPT1Config"/> — the C# port of <c>convert_llama.py</c>'s
    /// <c>detect_config</c>, with no Python and no reflection. Used by
    /// <see cref="SafetensorsLlamaLoader"/> to derive hyper-parameters before mapping
    /// the safetensors weights.
    ///
    /// The JSON is parsed with <see cref="Utf8JsonReader"/> (reflection-free, so the
    /// type stays Native-AOT clean — no <c>JsonSerializer.Deserialize</c>). Only the
    /// fields Overfit needs are read; everything else is skipped. Recognised keys:
    /// <c>num_hidden_layers</c>, <c>hidden_size</c>, <c>num_attention_heads</c>,
    /// <c>num_key_value_heads</c>, <c>intermediate_size</c>,
    /// <c>max_position_embeddings</c>, <c>rope_theta</c>, <c>vocab_size</c>,
    /// <c>head_dim</c>, <c>tie_word_embeddings</c>.
    ///
    /// SwiGLU + RoPE are assumed (the whole Llama/Qwen/Mistral family); the context
    /// length is capped at 8192 for memory sanity, matching <see cref="GgufLlamaLoader"/>.
    /// </summary>
    public static class LlamaConfigReader
    {
        private const int ContextCap = 8192;

        /// <summary>Reads <c>config.json</c> from a model directory.</summary>
        public static GPT1Config ReadFromDirectory(string modelDir)
        {
            if (string.IsNullOrEmpty(modelDir)) { throw new ArgumentException("Directory is empty.", nameof(modelDir)); }
            var path = Path.Combine(modelDir, "config.json");
            if (!File.Exists(path))
            {
                throw new FileNotFoundException($"No 'config.json' in directory '{modelDir}'.", path);
            }
            return Read(path);
        }

        /// <summary>Reads a <c>config.json</c> file by path.</summary>
        public static GPT1Config Read(string configJsonPath)
        {
            var bytes = File.ReadAllBytes(configJsonPath);
            return Parse(bytes);
        }

        /// <summary>Parses <c>config.json</c> bytes into a <see cref="GPT1Config"/>.</summary>
        public static GPT1Config Parse(ReadOnlySpan<byte> json)
        {
            // Sentinels: -1 = "not seen". Defaults applied after the scan.
            int layers = -1, dModel = -1, nHeads = -1, nKvHeads = -1, dFF = -1;
            int vocab = -1, maxPos = -1, headDim = -1;
            var ropeTheta = 10_000.0f;
            var tie = true;

            var reader = new Utf8JsonReader(json, isFinalBlock: true, state: default);
            if (!reader.Read() || reader.TokenType != JsonTokenType.StartObject)
            {
                throw new InvalidDataException("config.json is not a JSON object.");
            }

            // The root StartObject sits at CurrentDepth 0; its property names are at
            // depth 1. Only top-level scalar keys matter — nested objects/arrays
            // (rope_scaling, architectures, …) are skipped wholesale.
            var rootDepth = reader.CurrentDepth;
            var keyDepth = rootDepth + 1;
            while (reader.Read())
            {
                if (reader.TokenType == JsonTokenType.EndObject && reader.CurrentDepth == rootDepth)
                {
                    break;
                }
                if (reader.TokenType != JsonTokenType.PropertyName || reader.CurrentDepth != keyDepth)
                {
                    continue;
                }

                // Only top-level scalar keys matter; nested objects/arrays are skipped.
                if (reader.ValueTextEquals("num_hidden_layers")) { reader.Read(); layers = reader.GetInt32(); }
                else if (reader.ValueTextEquals("hidden_size")) { reader.Read(); dModel = reader.GetInt32(); }
                else if (reader.ValueTextEquals("num_attention_heads")) { reader.Read(); nHeads = reader.GetInt32(); }
                else if (reader.ValueTextEquals("num_key_value_heads")) { reader.Read(); nKvHeads = reader.GetInt32(); }
                else if (reader.ValueTextEquals("intermediate_size")) { reader.Read(); dFF = reader.GetInt32(); }
                else if (reader.ValueTextEquals("max_position_embeddings")) { reader.Read(); maxPos = reader.GetInt32(); }
                else if (reader.ValueTextEquals("vocab_size")) { reader.Read(); vocab = reader.GetInt32(); }
                else if (reader.ValueTextEquals("head_dim")) { reader.Read(); headDim = reader.GetInt32(); }
                else if (reader.ValueTextEquals("rope_theta")) { reader.Read(); ropeTheta = (float)reader.GetDouble(); }
                else if (reader.ValueTextEquals("tie_word_embeddings"))
                {
                    reader.Read();
                    tie = reader.TokenType == JsonTokenType.True;
                }
                else
                {
                    reader.Read();
                    if (reader.TokenType is JsonTokenType.StartObject or JsonTokenType.StartArray)
                    {
                        reader.Skip();
                    }
                }
            }

            if (layers <= 0 || dModel <= 0 || nHeads <= 0 || vocab <= 0)
            {
                throw new InvalidDataException(
                    "config.json missing one of num_hidden_layers / hidden_size / num_attention_heads / vocab_size.");
            }

            if (nKvHeads <= 0) { nKvHeads = nHeads; }
            if (dFF <= 0) { dFF = 4 * dModel; }
            if (maxPos <= 0) { maxPos = ContextCap; }

            // Overfit's runtime computes head_dim as DModel / NHeads everywhere
            // (CachedLlamaInferenceEngine), so a config.json head_dim that disagrees
            // is not representable — fail loudly rather than mis-map the weights.
            if (headDim > 0 && headDim * nHeads != dModel)
            {
                throw new NotSupportedException(
                    $"config.json head_dim ({headDim}) * num_attention_heads ({nHeads}) != hidden_size ({dModel}); " +
                    "Overfit requires head_dim == hidden_size / num_attention_heads.");
            }
            if (dModel % nHeads != 0)
            {
                throw new InvalidDataException(
                    $"hidden_size ({dModel}) is not divisible by num_attention_heads ({nHeads}).");
            }

            return new GPT1Config
            {
                NLayers = layers,
                DModel = dModel,
                NHeads = nHeads,
                NKvHeads = nKvHeads,
                VocabSize = vocab,
                ContextLength = maxPos > ContextCap ? ContextCap : maxPos,
                DFF = dFF,
                UseRoPE = true,
                RoPETheta = ropeTheta,
                FfnActivation = FeedForwardActivation.SwiGLU,
                TieWeights = tie,
            };
        }
    }
}
