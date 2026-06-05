// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;

namespace DevOnBike.Overfit.LanguageModels.Embeddings
{
    /// <summary>
    /// Reads a HuggingFace BERT-family <c>config.json</c> into a <see cref="BertConfig"/> — no Python,
    /// no reflection (parsed with <see cref="Utf8JsonReader"/> so it stays Native-AOT clean, mirroring
    /// <c>LlamaConfigReader</c>). Recognised keys: <c>hidden_size</c>, <c>num_hidden_layers</c>,
    /// <c>num_attention_heads</c>, <c>intermediate_size</c>, <c>max_position_embeddings</c>,
    /// <c>vocab_size</c>, <c>type_vocab_size</c>, <c>layer_norm_eps</c>.
    /// </summary>
    public static class BertConfigReader
    {
        public static BertConfig ReadFromDirectory(string modelDir)
        {
            ArgumentException.ThrowIfNullOrEmpty(modelDir);
            var path = Path.Combine(modelDir, "config.json");
            if (!File.Exists(path))
            {
                throw new FileNotFoundException($"No 'config.json' in directory '{modelDir}'.", path);
            }

            return Read(path);
        }

        public static BertConfig Read(string configJsonPath)
        {
            ArgumentException.ThrowIfNullOrEmpty(configJsonPath);
            return Parse(File.ReadAllBytes(configJsonPath));
        }

        public static BertConfig Parse(ReadOnlySpan<byte> json)
        {
            int hidden = -1, layers = -1, heads = -1, ffn = -1, maxPos = -1, vocab = -1, typeVocab = 2;
            var eps = 1e-12f;

            var reader = new Utf8JsonReader(json, isFinalBlock: true, state: default);
            if (!reader.Read() || reader.TokenType != JsonTokenType.StartObject)
            {
                throw new OverfitFormatException("config.json is not a JSON object.");
            }

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

                if (reader.ValueTextEquals("hidden_size")) { reader.Read(); hidden = reader.GetInt32(); }
                else if (reader.ValueTextEquals("num_hidden_layers")) { reader.Read(); layers = reader.GetInt32(); }
                else if (reader.ValueTextEquals("num_attention_heads")) { reader.Read(); heads = reader.GetInt32(); }
                else if (reader.ValueTextEquals("intermediate_size")) { reader.Read(); ffn = reader.GetInt32(); }
                else if (reader.ValueTextEquals("max_position_embeddings")) { reader.Read(); maxPos = reader.GetInt32(); }
                else if (reader.ValueTextEquals("vocab_size")) { reader.Read(); vocab = reader.GetInt32(); }
                else if (reader.ValueTextEquals("type_vocab_size")) { reader.Read(); typeVocab = reader.GetInt32(); }
                else if (reader.ValueTextEquals("layer_norm_eps")) { reader.Read(); eps = (float)reader.GetDouble(); }
            }

            if (hidden < 0 || layers < 0 || heads < 0 || ffn < 0 || maxPos < 0 || vocab < 0)
            {
                throw new OverfitFormatException(
                    "config.json is missing one of the required BERT keys " +
                    "(hidden_size, num_hidden_layers, num_attention_heads, intermediate_size, " +
                    "max_position_embeddings, vocab_size).");
            }

            return new BertConfig(hidden, layers, heads, ffn, maxPos, vocab, typeVocab, eps);
        }
    }
}
