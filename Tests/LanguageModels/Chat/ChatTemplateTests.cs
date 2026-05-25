// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Chat;

namespace DevOnBike.Overfit.Tests.LanguageModels.Chat
{
    public sealed class ChatTemplateTests
    {
        [Theory]
        [InlineData("{% for m in messages %}<|im_start|>{{m.role}}\n{{m.content}}<|im_end|>{% endfor %}", ChatTemplateFormat.ChatML)]
        [InlineData("<|start_header_id|>{{ role }}<|end_header_id|>", ChatTemplateFormat.Llama3)]
        [InlineData("{{ bos_token }}[INST] {{ message }} [/INST]", ChatTemplateFormat.Mistral)]
        [InlineData(null, ChatTemplateFormat.ChatML)]
        [InlineData("totally unknown template", ChatTemplateFormat.ChatML)]
        public void Detect_FingerprintsJinjaTemplate(string? jinja, ChatTemplateFormat expected)
        {
            Assert.Equal(expected, ChatTemplate.Detect(jinja).Format);
        }

        [Fact]
        public void ChatML_RendersMultiTurn_WithGenerationPrompt()
        {
            var t = new ChatTemplate(ChatTemplateFormat.ChatML);
            var prompt = t.Render(new[]
            {
                ChatMessage.System("You are helpful."),
                ChatMessage.User("Hi"),
            });

            Assert.Equal(
                "<|im_start|>system\nYou are helpful.<|im_end|>\n" +
                "<|im_start|>user\nHi<|im_end|>\n" +
                "<|im_start|>assistant\n",
                prompt);
        }

        [Fact]
        public void ChatML_NoGenerationPrompt_OmitsAssistantOpener()
        {
            var t = new ChatTemplate(ChatTemplateFormat.ChatML);
            var prompt = t.Render(new[] { ChatMessage.User("Hi") }, addGenerationPrompt: false);
            Assert.Equal("<|im_start|>user\nHi<|im_end|>\n", prompt);
        }

        [Fact]
        public void Llama3_RendersHeadersAndEot()
        {
            var t = new ChatTemplate(ChatTemplateFormat.Llama3);
            var prompt = t.Render(new[]
            {
                ChatMessage.System("S"),
                ChatMessage.User("U"),
            });

            Assert.Equal(
                "<|begin_of_text|>" +
                "<|start_header_id|>system<|end_header_id|>\n\nS<|eot_id|>" +
                "<|start_header_id|>user<|end_header_id|>\n\nU<|eot_id|>" +
                "<|start_header_id|>assistant<|end_header_id|>\n\n",
                prompt);
        }

        [Fact]
        public void Mistral_FoldsSystemIntoFirstUser_AndPairsTurns()
        {
            var t = new ChatTemplate(ChatTemplateFormat.Mistral);
            var prompt = t.Render(new[]
            {
                ChatMessage.System("S"),
                ChatMessage.User("U"),
                ChatMessage.Assistant("A"),
                ChatMessage.User("U2"),
            });

            Assert.Equal("[INST] S\n\nU [/INST] A</s>[INST] U2 [/INST]", prompt);
        }
    }
}
