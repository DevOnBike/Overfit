// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.LanguageModels.Chat;
using DevOnBike.Overfit.LanguageModels.Loading;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// Extracts the chat template from a HuggingFace <c>tokenizer_config.json</c>,
    /// covering the plain-string form, the multi-template array form
    /// (<c>[{name,template}]</c>), and a missing field (default fallback).
    /// </summary>
    public sealed class HuggingFaceChatTemplateTests
    {
        [Fact]
        public void Parse_StringTemplate_DetectsLlama3()
        {
            const string json =
                "{\"bos_token\":\"<s>\",\"chat_template\":" +
                "\"{{bos}}<|start_header_id|>{{role}}<|end_header_id|>\\n{{content}}\"}";
            var template = HuggingFaceChatTemplate.Parse(Encoding.UTF8.GetBytes(json));
            Assert.Equal(ChatTemplateFormat.Llama3, template.Format);
        }

        [Fact]
        public void Parse_ArrayTemplate_PrefersDefault_DetectsChatML()
        {
            // tool_use entry first, default second — Detect must fingerprint the default.
            const string json =
                "{\"chat_template\":[" +
                "{\"name\":\"tool_use\",\"template\":\"[INST] tools [/INST]\"}," +
                "{\"name\":\"default\",\"template\":\"<|im_start|>{{role}}\\n{{content}}<|im_end|>\"}]}";
            var template = HuggingFaceChatTemplate.Parse(Encoding.UTF8.GetBytes(json));
            Assert.Equal(ChatTemplateFormat.ChatML, template.Format);
        }

        [Fact]
        public void Parse_ArrayTemplate_NoDefault_UsesFirst()
        {
            const string json =
                "{\"chat_template\":[{\"name\":\"tool_use\",\"template\":\"[INST] x [/INST]\"}]}";
            var template = HuggingFaceChatTemplate.Parse(Encoding.UTF8.GetBytes(json));
            Assert.Equal(ChatTemplateFormat.Mistral, template.Format);
        }

        [Fact]
        public void Parse_NoChatTemplateField_FallsBackToDefault()
        {
            const string json = "{\"bos_token\":\"<s>\",\"model_max_length\":32768}";
            var template = HuggingFaceChatTemplate.Parse(Encoding.UTF8.GetBytes(json));
            Assert.Equal(ChatTemplateFormat.ChatML, template.Format);   // Detect's default
        }
    }
}
