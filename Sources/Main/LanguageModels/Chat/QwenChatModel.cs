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
    /// Turnkey, zero-Python chat over a Qwen2.5 HuggingFace directory: one call wires the
    /// native safetensors loader (<see cref="SafetensorsLlamaLoader"/>), the Qwen BPE
    /// tokenizer (<see cref="QwenChatTokenizer"/>), and the chat template read straight from
    /// <c>tokenizer_config.json</c> (<see cref="HuggingFaceChatTemplate"/>) into a ready
    /// <see cref="ChatSession"/>. Owns the engine + session and disposes both.
    ///
    /// <code>
    /// using var model = QwenChatModel.LoadFromDirectory(@"C:\qwen3b");
    /// model.Chat.AddSystem("You are a concise assistant.");
    /// var reply = model.Chat.Send("What is the capital of France?", options, Console.Write);
    /// </code>
    ///
    /// Scoped to Qwen because <see cref="QwenChatTokenizer"/> carries Qwen's cl100k-style
    /// pre-tokenizer; other families need their own tokenizer. The model is a single
    /// conversation — call <see cref="ChatSession.ResetConversation"/> to start over.
    /// </summary>
    public sealed class QwenChatModel : IDisposable
    {
        private readonly CachedLlamaInferenceEngine _engine;
        private readonly CachedLlamaSession _session;
        private bool _disposed;

        private QwenChatModel(CachedLlamaInferenceEngine engine, CachedLlamaSession session, ChatSession chat)
        {
            _engine = engine;
            _session = session;
            Chat = chat;
        }

        /// <summary>The ready-to-use chat session over the loaded model.</summary>
        public ChatSession Chat { get; }

        /// <summary>
        /// Loads a Qwen2.5 HF directory (<c>model.safetensors</c> [+ shards] + <c>config.json</c>
        /// + <c>tokenizer.json</c> + <c>tokenizer_config.json</c>) into a turnkey chat model.
        /// </summary>
        /// <param name="modelDir">Directory holding the HF model files.</param>
        /// <param name="maxContextLength">KV-cache context; defaults to the model's configured length.</param>
        /// <param name="quantize">Q8-resident weights when block-aligned (default), else F32.</param>
        public static QwenChatModel LoadFromDirectory(string modelDir, int? maxContextLength = null, bool quantize = true)
        {
            var engine = SafetensorsLlamaLoader.Load(modelDir, quantize);
            CachedLlamaSession? session = null;
            try
            {
                session = engine.CreateSession(maxContextLength);
                var tokenizer = new QwenChatTokenizer(QwenTokenizer.Load(modelDir));
                var template = HuggingFaceChatTemplate.FromDirectory(modelDir);
                // Stop the assistant turn on the ChatML terminators (string-level, so the
                // markers are suppressed from the reply) on top of the EOS-token stop.
                var chat = new ChatSession(session, tokenizer, template, ["<|im_end|>", "<|endoftext|>"]);
                return new QwenChatModel(engine, session, chat);
            }
            catch
            {
                session?.Dispose();
                engine.Dispose();
                throw;
            }
        }

        public void Dispose()
        {
            if (_disposed) { return; }
            _disposed = true;
            _session.Dispose();
            _engine.Dispose();
        }
    }
}
