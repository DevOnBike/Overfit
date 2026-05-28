// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Chat;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;

namespace DevOnBike.Overfit.LanguageModels
{
    /// <summary>
    /// Top-level turnkey facade for the in-process chat path — collapses the standard six-line wire-up
    /// (open GGUF reader, detect chat template, load tokenizer, load engine, create session, build
    /// ChatSession + stop sequences + sampling defaults) into a single static factory. Owns the
    /// engine + session + chat session and disposes them together.
    ///
    /// <code>
    /// using var client = OverfitClient.LoadGguf(@"C:\qwen3b\qwen.q4km.gguf");
    /// client.AddSystem("You are a concise assistant.");
    /// var reply = client.Send("What is the capital of France?");
    /// </code>
    ///
    /// For lower-level control (per-call constraints, streaming, custom samplers, tape inspection)
    /// the underlying <see cref="ChatSession"/> is exposed via <see cref="Chat"/>.
    /// </summary>
    public sealed class OverfitClient : IDisposable
    {
        private static readonly string[] _defaultStopSequences = ["<|im_end|>", "\n<|im_start|>"];

        private readonly CachedLlamaInferenceEngine _engine;
        private readonly CachedLlamaSession _session;
        private readonly ITokenizer _tokenizer;
        private readonly ChatSession _chat;
        private GenerationOptions _options;
        private bool _disposed;

        private OverfitClient(
            CachedLlamaInferenceEngine engine,
            CachedLlamaSession session,
            ITokenizer tokenizer,
            ChatSession chat,
            GenerationOptions options)
        {
            _engine = engine;
            _session = session;
            _tokenizer = tokenizer;
            _chat = chat;
            _options = options;
        }

        /// <summary>The underlying <see cref="ChatSession"/> — use for streaming callbacks, constrained outputs, custom samplers.</summary>
        public ChatSession Chat => _chat;

        /// <summary>The tokenizer (HuggingFace BPE) — auto-detected from the GGUF's sibling directory.</summary>
        public ITokenizer Tokenizer => _tokenizer;

        /// <summary>The inference engine — exposed for advanced usage (e.g. embedding extraction).</summary>
        public CachedLlamaInferenceEngine Engine => _engine;

        /// <summary>Generation options applied by <see cref="Send"/> / <see cref="SendAsync"/>. Mutate freely between calls.</summary>
        public GenerationOptions Options
        {
            get => _options;
            set => _options = value;
        }

        /// <summary>
        /// Loads a GGUF model (Qwen / Llama / Mistral / Mixtral / Qwen-MoE family) with auto-detected
        /// tokenizer (HuggingFace BPE from sibling <c>tokenizer.json</c>) and chat template
        /// (from GGUF's <c>tokenizer.chat_template</c> metadata). Default stop sequences cover the
        /// ChatML markers used by Qwen/most modern instruct models.
        /// </summary>
        /// <param name="ggufPath">Path to <c>*.gguf</c> file. The sibling directory must contain a tokenizer (vocab.json + merges.txt OR tokenizer.json).</param>
        /// <param name="maxContextLength">Session KV-cache size in tokens; clipped to the model's intrinsic max.</param>
        /// <param name="mmap">Memory-map verbatim K-quant weights (Q4_K / Q6_K) — default on, cuts working-set RAM.</param>
        /// <param name="quantize">Quantise FFN/LM-head/attention where supported (default Q8_0 fallback).</param>
        /// <param name="maxNewTokens">Default maximum response length in tokens (overridable via <see cref="Options"/>).</param>
        /// <param name="stopSequences">Override string stop sequences; defaults to ChatML markers.</param>
        public static OverfitClient LoadGguf(
            string ggufPath,
            int maxContextLength = 2048,
            bool mmap = true,
            bool quantize = true,
            int maxNewTokens = 256,
            IReadOnlyList<string>? stopSequences = null)
        {
            ArgumentException.ThrowIfNullOrEmpty(ggufPath);
            if (!File.Exists(ggufPath))
            {
                throw new FileNotFoundException($"GGUF model file not found: '{ggufPath}'.", ggufPath);
            }

            var modelDir = Path.GetDirectoryName(Path.GetFullPath(ggufPath))
                ?? throw new ArgumentException("Could not determine model directory from GGUF path.", nameof(ggufPath));

            // ── Chat template from GGUF metadata ──
            ChatTemplate template;
            using (var reader = new GgufReader(ggufPath))
            {
                template = ChatTemplate.Detect(reader.GetMeta("tokenizer.chat_template", string.Empty));
            }

            // ── Tokenizer from sibling directory ──
            // QwenChatTokenizer first when its files are present — it handles ChatML special tokens
            // (<|im_start|>, <|im_end|>) the way Qwen-family models expect. HuggingFaceBpeTokenizer
            // is the generic fallback for Llama/Mistral and Qwen base variants without ChatML.
            ITokenizer tokenizer;
            var hasQwenVocab = File.Exists(Path.Combine(modelDir, "vocab.json"))
                && File.Exists(Path.Combine(modelDir, "merges.txt"));
            if (hasQwenVocab)
            {
                try { tokenizer = new QwenChatTokenizer(QwenTokenizer.Load(modelDir)); }
                catch { tokenizer = HuggingFaceBpeTokenizer.Load(modelDir); }
            }
            else
            {
                tokenizer = HuggingFaceBpeTokenizer.Load(modelDir);
            }

            // ── Engine + session ──
            var engine = GgufLlamaLoader.Load(ggufPath, quantize: quantize, mmap: mmap);
            try
            {
                var session = engine.CreateSession(maxContextLength);
                try
                {
                    var chat = new ChatSession(session, tokenizer, template, stopSequences ?? _defaultStopSequences);

                    var options = new GenerationOptions(
                        maxNewTokens: maxNewTokens,
                        maxContextLength: maxContextLength,
                        sampling: SamplingOptions.Greedy,
                        stopOnEndOfTextToken: true,
                        endOfTextTokenId: tokenizer.EndOfTextTokenId);

                    return new OverfitClient(engine, session, tokenizer, chat, options);
                }
                catch
                {
                    session.Dispose();
                    throw;
                }
            }
            catch
            {
                engine.Dispose();
                throw;
            }
        }

        /// <summary>Appends a system message to the conversation. Call before the first <see cref="Send"/>.</summary>
        public void AddSystem(string content) => _chat.AddSystem(content);

        /// <summary>Clears the conversation history (keeps loaded weights / engine).</summary>
        public void Reset() => _chat.ResetConversation();

        /// <summary>
        /// Multi-turn chat send — appends the user message, generates the assistant reply, appends it,
        /// returns it. Optional <paramref name="onText"/> receives the reply incrementally as it streams.
        /// </summary>
        public string Send(
            string userMessage,
            Action<string>? onText = null,
            ITokenConstraint? constraint = null)
        {
            ThrowIfDisposed();
            var opts = _options;
            return _chat.Send(userMessage, in opts, onText, constraint);
        }

        /// <summary>
        /// Async version of <see cref="Send"/> — runs the (sync) generate loop on a thread-pool worker so
        /// the caller can <c>await</c>. Cancellation cooperates only at the boundary; in-flight token
        /// generation is not interruptible mid-step.
        /// </summary>
        public Task<string> SendAsync(
            string userMessage,
            Action<string>? onText = null,
            ITokenConstraint? constraint = null,
            CancellationToken cancellationToken = default)
        {
            return Task.Run(() => Send(userMessage, onText, constraint), cancellationToken);
        }

        public void Dispose()
        {
            if (_disposed) { return; }
            _disposed = true;
            _session.Dispose();
            _engine.Dispose();
        }

        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
        }
    }
}
