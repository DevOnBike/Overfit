// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;

namespace DevOnBike.Overfit.LanguageModels.Chat
{
    /// <summary>
    /// Turnkey, zero-Python chat over ANY Llama-3 / Mistral / Qwen HuggingFace directory —
    /// the family-generic counterpart of <see cref="QwenChatModel"/>. One call wires the
    /// native safetensors loader (<see cref="SafetensorsLlamaLoader"/>), the file-driven BPE
    /// tokenizer (<see cref="HuggingFaceBpeTokenizer"/> — reads the pre-tokenizer from
    /// <c>tokenizer.json</c>, so no per-model hard-coding), and the chat template detected
    /// from <c>tokenizer_config.json</c> (<see cref="HuggingFaceChatTemplate"/>) into a ready
    /// <see cref="ChatSession"/>. Owns the engine + session and disposes both.
    ///
    /// <code>
    /// using var model = HuggingFaceChatModel.LoadFromDirectory(@"C:\llama3");
    /// model.Chat.AddSystem("You are concise.");
    /// var reply = model.Chat.Send("What is the capital of France?", options, Console.Write);
    /// </code>
    ///
    /// ByteLevel-BPE family only (Llama-3 / Mistral / Qwen); SentencePiece/Unigram models
    /// throw from the tokenizer. Single conversation — <see cref="ChatSession.ResetConversation"/>
    /// to start over.
    /// </summary>
    public sealed class HuggingFaceChatModel : IDisposable
    {
        private readonly CachedLlamaInferenceEngine _engine;
        private readonly CachedLlamaSession _session;
        private bool _disposed;

        private HuggingFaceChatModel(CachedLlamaInferenceEngine engine, CachedLlamaSession session, ChatSession chat)
        {
            _engine = engine;
            _session = session;
            Chat = chat;
        }

        /// <summary>The ready-to-use chat session over the loaded model.</summary>
        public ChatSession Chat { get; }

        /// <summary>The detected chat-prompt format (drives rendering + stop sequences).</summary>
        public ChatTemplateFormat Format { get; private set; }

        /// <summary>
        /// Loads a HuggingFace model directory (<c>model.safetensors</c> [+ shards] +
        /// <c>config.json</c> + <c>tokenizer.json</c> + <c>tokenizer_config.json</c>).
        /// </summary>
        public static HuggingFaceChatModel LoadFromDirectory(string modelDir, int? maxContextLength = null, bool quantize = true)
        {
            var engine = SafetensorsLlamaLoader.Load(modelDir, quantize);
            CachedLlamaSession? session = null;
            try
            {
                session = engine.CreateSession(maxContextLength);
                var tokenizer = HuggingFaceBpeTokenizer.Load(modelDir);
                var template = HuggingFaceChatTemplate.FromDirectory(modelDir);
                // String-level stops on the format's turn terminator (suppressed from the
                // reply) on top of ChatSession's EOS-token stop.
                var chat = new ChatSession(session, tokenizer, template, StopsFor(template.Format));
                return new HuggingFaceChatModel(engine, session, chat) { Format = template.Format };
            }
            catch
            {
                session?.Dispose();
                engine.Dispose();
                throw;
            }
        }

        private static string[] StopsFor(ChatTemplateFormat format) => format switch
        {
            ChatTemplateFormat.ChatML => ["<|im_end|>"],
            ChatTemplateFormat.Llama3 => ["<|eot_id|>"],
            ChatTemplateFormat.Mistral => ["</s>"],
            _ => [],
        };

        public void Dispose()
        {
            if (_disposed) { return; }
            _disposed = true;
            _session.Dispose();
            _engine.Dispose();
        }
    }
}
