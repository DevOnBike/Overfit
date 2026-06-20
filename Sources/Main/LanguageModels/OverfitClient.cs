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

        private const int EmbedContextLength = 1024;

        private readonly CachedLlamaInferenceEngine _engine;
        private readonly CachedLlamaSession _session;
        private readonly ITokenizer _tokenizer;
        private readonly ChatSession _chat;
        private GenerationOptions _options;
        private CachedLlamaSession? _embedSession;   // lazy, dedicated — Embed() resets the cache, so it can't share _session
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
        /// <param name="sampling"></param>
        public static OverfitClient LoadGguf(
            string ggufPath,
            int maxContextLength = 2048,
            bool mmap = true,
            bool quantize = true,
            int maxNewTokens = 256,
            IReadOnlyList<string>? stopSequences = null,
            SamplingOptions? sampling = null)
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

            // ── Tokenizer: prefer sibling files, else fall back to the GGUF's embedded vocabulary ──
            // A bare *.gguf (e.g. downloaded from a GGUF-only HF repo) ships no tokenizer.json; in that
            // case reconstruct the tokenizer from the file's own tokenizer.ggml.* metadata.
            var tokenizer = HasSiblingTokenizer(modelDir)
                ? LoadTokenizer(modelDir)
                : new GgufEmbeddedTokenizer(GgufTokenizer.Load(ggufPath));

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
                        sampling: sampling ?? SamplingOptions.Greedy,
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

        /// <summary>
        /// Loads a model from a HuggingFace directory (<c>model.safetensors</c> [+ shards] + <c>config.json</c>
        /// + tokenizer files) — no GGUF conversion needed. Same turnkey wiring as <see cref="LoadGguf"/>:
        /// detects the chat template from <c>tokenizer_config.json</c>, loads the tokenizer, builds the engine
        /// (<see cref="DevOnBike.Overfit.LanguageModels.Loading.SafetensorsLlamaLoader"/>), session and
        /// ChatSession. Handy for small instruct models you already have unpacked (e.g. Qwen2.5-0.5B-Instruct).
        /// </summary>
        /// <param name="modelDir">Directory with <c>model.safetensors</c> (or sharded) + <c>config.json</c> + tokenizer.</param>
        /// <param name="maxContextLength">Maximum context length (in tokens) the session's KV cache is sized for.</param>
        /// <param name="quantize">When true, quantize weights at load (lower RAM); false keeps full precision.</param>
        /// <param name="maxNewTokens">Default maximum number of tokens to generate per chat turn.</param>
        /// <param name="stopSequences">Optional stop sequences that end generation, in addition to template defaults.</param>
        /// <param name="sampling">Optional sampling options (temperature, top-k/top-p); null uses defaults.</param>
        public static OverfitClient LoadPretrained(
            string modelDir,
            int maxContextLength = 2048,
            bool quantize = true,
            int maxNewTokens = 256,
            IReadOnlyList<string>? stopSequences = null,
            SamplingOptions? sampling = null)
        {
            ArgumentException.ThrowIfNullOrEmpty(modelDir);
            if (!Directory.Exists(modelDir))
            {
                throw new DirectoryNotFoundException($"Model directory not found: '{modelDir}'.");
            }

            var template = ChatTemplate.Detect(ReadChatTemplate(modelDir));
            var tokenizer = LoadTokenizer(modelDir);

            var engine = SafetensorsLlamaLoader.Load(modelDir, quantize: quantize);
            try
            {
                var session = engine.CreateSession(maxContextLength);
                try
                {
                    var chat = new ChatSession(session, tokenizer, template, stopSequences ?? _defaultStopSequences);
                    var options = new GenerationOptions(
                        maxNewTokens: maxNewTokens,
                        maxContextLength: maxContextLength,
                        sampling: sampling ?? SamplingOptions.Greedy,
                        stopOnEndOfTextToken: true,
                        endOfTextTokenId: tokenizer.EndOfTextTokenId);
                    return new OverfitClient(engine, session, tokenizer, chat, options);
                }
                catch { session.Dispose(); throw; }
            }
            catch { engine.Dispose(); throw; }
        }

        /// <summary>Reads the Jinja chat template from <c>tokenizer_config.json</c> (empty string if absent).</summary>
        private static string ReadChatTemplate(string modelDir)
        {
            var path = Path.Combine(modelDir, "tokenizer_config.json");
            if (!File.Exists(path))
            {
                return string.Empty;
            }
            try
            {
                using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(path));
                return doc.RootElement.TryGetProperty("chat_template", out var t) && t.ValueKind == System.Text.Json.JsonValueKind.String
                    ? t.GetString() ?? string.Empty
                    : string.Empty;
            }
            catch { return string.Empty; }
        }

        /// <summary>True when the directory ships tokenizer files (tokenizer.json, or vocab.json + merges.txt).</summary>
        private static bool HasSiblingTokenizer(string modelDir)
        {
            if (File.Exists(Path.Combine(modelDir, "tokenizer.json")))
            {
                return true;
            }
            return File.Exists(Path.Combine(modelDir, "vocab.json"))
                && File.Exists(Path.Combine(modelDir, "merges.txt"));
        }

        /// <summary>Tokenizer from a model directory: Qwen ChatML BPE when vocab.json+merges.txt present, else generic HF BPE.</summary>
        private static ITokenizer LoadTokenizer(string modelDir)
        {
            var hasQwenVocab = File.Exists(Path.Combine(modelDir, "vocab.json"))
                && File.Exists(Path.Combine(modelDir, "merges.txt"));
            if (hasQwenVocab)
            {
                try
                {
                    return new QwenChatTokenizer(QwenTokenizer.Load(modelDir));
                }
                catch { return HuggingFaceBpeTokenizer.Load(modelDir); }
            }
            return HuggingFaceBpeTokenizer.Load(modelDir);
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
        /// Stateless one-shot generation — does NOT read or modify the conversation history; renders the
        /// system message plus this single user turn, generates and returns the reply. Ideal for tool
        /// routing, JSON mode and RAG answers, which should not accumulate across requests (each call
        /// prefills only its own minimal prompt). Use <see cref="Send"/> for multi-turn conversation.
        /// </summary>
        public string Complete(
            string userMessage,
            Action<string>? onText = null,
            ITokenConstraint? constraint = null)
        {
            ThrowIfDisposed();
            var opts = _options;
            return _chat.Complete(userMessage, in opts, onText, constraint);
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

        /// <summary>Dimensionality of <see cref="Embed"/> vectors (the model's hidden size).</summary>
        public int EmbeddingDimension => EmbedSession.EmbeddingDimension;

        /// <summary>
        /// Embeds <paramref name="text"/> using the loaded model's own hidden states (mean-pooled, L2-normalised) —
        /// a multilingual, in-process embedding from the SAME GGUF you chat with, no separate sentence-embedder.
        /// Useful for RAG over non-English corpora where an English-only embedder (MiniLM) mis-retrieves. Uses a
        /// DEDICATED session because <see cref="CachedLlamaSession.Embed(System.ReadOnlySpan{int}, EmbeddingPooling, bool)"/>
        /// resets the KV cache and would otherwise wipe the chat conversation.
        /// </summary>
        public float[] Embed(string text, EmbeddingPooling pooling = EmbeddingPooling.Mean)
        {
            ThrowIfDisposed();
            ArgumentNullException.ThrowIfNull(text);

            // Byte-level BPE can emit up to ~4 tokens per char (multi-byte UTF-8), so size the scratch generously.
            var buffer = new int[text.Length * 4 + 16];
            var count = _tokenizer.Encode(text, buffer);
            if (count <= 0)
            {
                count = _tokenizer.Encode(" ", buffer);   // empty / whitespace → one space, avoid the empty-token throw
            }
            if (count > EmbedContextLength)
            {
                count = EmbedContextLength;                // truncate to the dedicated embed session's context
            }

            return EmbedSession.Embed(buffer.AsSpan(0, count), pooling);
        }

        private CachedLlamaSession EmbedSession
        {
            get
            {
                ThrowIfDisposed();
                return _embedSession ??= _engine.CreateSession(EmbedContextLength);
            }
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;
            _embedSession?.Dispose();
            _session.Dispose();
            _engine.Dispose();
        }

        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
        }
    }
}
