// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.InteropServices;
using System.Text;
using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.LanguageModels.Chat
{
    /// <summary>
    /// Turnkey multi-turn chat loop over any <see cref="ISlmSession"/> + <see cref="ITokenizer"/>.
    /// Owns the conversation history and, on each <see cref="Send"/>, renders the whole
    /// history with a <see cref="ChatTemplate"/>, tokenizes, generates, and assembles the
    /// reply — applying both the token-level end-of-text stop and any string stop
    /// sequences (via <see cref="StopSequenceDetector"/>) so callers don't hand-roll the
    /// decode/stop loop. The session and tokenizer are borrowed (not disposed here).
    ///
    /// <code>
    /// using var engine = GgufLlamaLoader.Load("qwen.q4km.gguf");
    /// var chat = new ChatSession(engine.CreateSession(), tokenizer, ChatTemplate.Detect(jinja));
    /// chat.AddSystem("You are concise.");
    /// var reply = chat.Send("What is 2+2?", options, Console.Write);
    /// </code>
    /// </summary>
    public sealed class ChatSession
    {
        private readonly ISlmSession _session;
        private readonly ITokenizer _tokenizer;
        private readonly ChatTemplate _template;
        private readonly string[] _stopSequences;
        private readonly List<ChatMessage> _history = [];

        public ChatSession(
            ISlmSession session,
            ITokenizer tokenizer,
            ChatTemplate template,
            IReadOnlyList<string>? stopSequences = null)
        {
            _session = session ?? throw new ArgumentNullException(nameof(session));
            _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
            _template = template ?? throw new ArgumentNullException(nameof(template));

            var stops = new List<string>();
            if (stopSequences is not null)
            {
                foreach (var s in stopSequences)
                {
                    if (!string.IsNullOrEmpty(s)) { stops.Add(s); }
                }
            }
            _stopSequences = stops.ToArray();
        }

        /// <summary>The conversation so far (system / user / assistant turns).</summary>
        public IReadOnlyList<ChatMessage> History => _history;

        /// <summary>
        /// Stats for the most recent <see cref="Send"/> — prompt/generated token counts and the
        /// decode time, exposing <see cref="GenerationStats.TokensPerSecond"/>. Timed over the
        /// decode loop only (prompt prefill excluded), so it reflects steady-state throughput.
        /// </summary>
        public GenerationStats LastStats { get; private set; }

        public void AddSystem(string content) => _history.Add(ChatMessage.System(content));

        /// <summary>Clears the conversation history.</summary>
        public void ResetConversation() => _history.Clear();

        /// <summary>
        /// Appends <paramref name="userMessage"/> to the history, generates the assistant
        /// reply, appends it to the history, and returns it. <paramref name="onText"/>, if
        /// supplied, receives the reply incrementally as it streams.
        /// </summary>
        public string Send(string userMessage, in GenerationOptions options, Action<string>? onText = null)
        {
            if (userMessage is null) { throw new ArgumentNullException(nameof(userMessage)); }

            _history.Add(ChatMessage.User(userMessage));

            // Render the whole conversation and prefill the session with it.
            var promptText = _template.Render(_history, addGenerationPrompt: true);
            var tokenCount = _tokenizer.CountTokens(promptText);
            var promptTokens = new int[tokenCount];
            var written = _tokenizer.Encode(promptText, promptTokens);
            _session.Reset(promptTokens.AsSpan(0, written));

            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            var reply = Generate(in options, onText, out var generatedTokens);
            stopwatch.Stop();
            LastStats = new GenerationStats(
                promptTokens: written,
                generatedTokens: generatedTokens,
                elapsedNanoseconds: stopwatch.Elapsed.Ticks * 100,   // 1 tick = 100 ns
                allocatedBytes: 0,
                usedKeyValueCache: true);

            _history.Add(ChatMessage.Assistant(reply));
            return reply;
        }

        private string Generate(in GenerationOptions options, Action<string>? onText, out int generatedTokens)
        {
            var stops = new StopSequenceDetector(_stopSequences);
            var generated = new List<int>();
            var reply = new StringBuilder();
            var prevText = string.Empty;
            var sampling = options.Sampling;
            var maxNew = options.MaxNewTokens > 0 ? options.MaxNewTokens : int.MaxValue;

            for (var i = 0; i < maxNew && _session.CurrentPosition < _session.MaxContextLength; i++)
            {
                var token = _session.GenerateNextToken(in sampling);

                if (token == _tokenizer.EndOfTextTokenId ||
                    (options.StopOnEndOfTextToken && options.EndOfTextTokenId >= 0 && token == options.EndOfTextTokenId))
                {
                    break;
                }

                generated.Add(token);

                // Incremental detokenize: decode the whole run and emit only the newly
                // stabilised suffix (byte-level BPE can leave a trailing partial codepoint
                // until the next token arrives — hold it back rather than emit garbage).
                var full = _tokenizer.DecodeToString(CollectionsMarshal.AsSpan(generated));
                if (full.Length <= prevText.Length || !full.StartsWith(prevText, StringComparison.Ordinal))
                {
                    continue;
                }
                var delta = full[prevText.Length..];
                prevText = full;

                var emit = stops.Append(delta);
                if (emit.Length > 0)
                {
                    reply.Append(emit);
                    onText?.Invoke(emit);
                }
                if (stops.Stopped)
                {
                    break;
                }
            }

            var tail = stops.Flush();
            if (tail.Length > 0)
            {
                reply.Append(tail);
                onText?.Invoke(tail);
            }
            generatedTokens = generated.Count;
            return reply.ToString();
        }
    }
}
