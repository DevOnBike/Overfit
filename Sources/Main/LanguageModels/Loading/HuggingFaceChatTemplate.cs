// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using DevOnBike.Overfit.LanguageModels.Chat;

namespace DevOnBike.Overfit.LanguageModels.Loading
{
    /// <summary>
    /// Resolves a <see cref="ChatTemplate"/> from a HuggingFace repo's
    /// <c>tokenizer_config.json</c> — the safetensors-repo counterpart of reading
    /// <c>tokenizer.chat_template</c> from GGUF metadata. The Jinja template string is
    /// extracted (never executed) and fingerprinted by <see cref="ChatTemplate.Detect"/>.
    ///
    /// Handles both shapes HuggingFace emits for the <c>chat_template</c> field:
    /// a plain Jinja string, or the newer multi-template array
    /// <c>[{ "name": "default", "template": "…" }, …]</c> (the <c>default</c> entry,
    /// or the first, is used). When the field is absent the format falls back to
    /// <see cref="ChatTemplate.Detect"/>'s default. Parsed with the reflection-free
    /// <see cref="Utf8JsonReader"/> to stay Native-AOT clean.
    /// </summary>
    // OVERFIT040 for this type. THE CONSTRAINT: `FromDirectory` reads one `tokenizer_config.json` once, at
    // model-construction time, on the caller's own thread, to fingerprint a chat template — the safetensors
    // counterpart of reading the same string out of GGUF metadata, which is likewise synchronous. No pool
    // thread is behind it and nothing is queued on it. Its `Parse` takes a `ReadOnlySpan<byte>`, which cannot
    // cross an `await`.
    //
    // WHAT IS GIVEN UP: `FromDirectory` is public API of the shipped `DevOnBike.Overfit` package; a
    // task-returning form is a breaking change for every caller that resolves a chat template.
#pragma warning disable OVERFIT040
    public static class HuggingFaceChatTemplate
    {
        /// <summary>Reads <c>tokenizer_config.json</c> from a model directory.</summary>
        public static ChatTemplate FromDirectory(string modelDir)
        {
            if (string.IsNullOrEmpty(modelDir))
            {
                throw new ArgumentException("Directory is empty.", nameof(modelDir));
            }
            var path = Path.Combine(modelDir, "tokenizer_config.json");
            if (!File.Exists(path))
            {
                // No tokenizer_config.json → let Detect pick its default format.
                return ChatTemplate.Detect(null);
            }
            return Parse(File.ReadAllBytes(path));
        }

        /// <summary>Extracts the chat template from <c>tokenizer_config.json</c> bytes.</summary>
        public static ChatTemplate Parse(ReadOnlySpan<byte> tokenizerConfigJson)
        {
            return ChatTemplate.Detect(ExtractJinja(tokenizerConfigJson));
        }

        // Pulls the chat_template Jinja string from the top-level object, tolerating
        // both the string and the array-of-{name,template} shapes.
        private static string? ExtractJinja(ReadOnlySpan<byte> json)
        {
            var reader = new Utf8JsonReader(json, isFinalBlock: true, state: default);
            if (!reader.Read() || reader.TokenType != JsonTokenType.StartObject)
            {
                return null;
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

                if (!reader.ValueTextEquals("chat_template"))
                {
                    reader.Read();
                    if (reader.TokenType is JsonTokenType.StartObject or JsonTokenType.StartArray)
                    {
                        reader.Skip();
                    }
                    continue;
                }

                reader.Read();
                return reader.TokenType == JsonTokenType.StartArray
                    ? ReadTemplateArray(ref reader)
                    : reader.TokenType == JsonTokenType.String ? reader.GetString() : null;
            }

            return null;
        }

        // chat_template as [{ "name": ..., "template": ... }, …] — prefer "default".
        private static string? ReadTemplateArray(ref Utf8JsonReader reader)
        {
            string? first = null;
            string? defaultTemplate = null;

            // reader is positioned on StartArray.
            while (reader.Read() && reader.TokenType != JsonTokenType.EndArray)
            {
                if (reader.TokenType != JsonTokenType.StartObject)
                {
                    reader.Skip();
                    continue;
                }

                string? name = null;
                string? template = null;
                while (reader.Read() && reader.TokenType == JsonTokenType.PropertyName)
                {
                    var isName = reader.ValueTextEquals("name");
                    var isTemplate = reader.ValueTextEquals("template");
                    reader.Read();
                    if (isName && reader.TokenType == JsonTokenType.String)
                    {
                        name = reader.GetString();
                    }
                    if (isTemplate && reader.TokenType == JsonTokenType.String)
                    {
                        template = reader.GetString();
                    }

                    if (reader.TokenType is JsonTokenType.StartObject or JsonTokenType.StartArray)
                    {
                        reader.Skip();
                    }
                }

                first ??= template;
                if (string.Equals(name, "default", StringComparison.Ordinal))
                {
                    defaultTemplate = template;
                }
            }

            return defaultTemplate ?? first;
        }
    }
#pragma warning restore OVERFIT040
}
