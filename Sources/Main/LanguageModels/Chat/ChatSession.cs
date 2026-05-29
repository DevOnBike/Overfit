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
        private readonly bool _slidingWindow;
        private readonly List<ChatMessage> _history = [];

        /// <param name="slidingWindow">
        /// When true, enables sliding-window KV eviction on the session so long conversations keep
        /// going past the model's context length (the oldest tokens roll off) instead of stopping at
        /// the limit. Requires a session that supports it (RoPE models — Qwen / Llama / Mistral);
        /// throws <see cref="NotSupportedException"/> otherwise.
        /// </param>
        public ChatSession(
            ISlmSession session,
            ITokenizer tokenizer,
            ChatTemplate template,
            IReadOnlyList<string>? stopSequences = null,
            bool slidingWindow = false)
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

            if (slidingWindow)
            {
                // Throws NotSupportedException for non-RoPE sessions — fail early, at construction.
                _session.EnableSlidingWindow();
                _slidingWindow = true;
            }
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

        /// <summary>
        /// Appends a user turn to the history without generating an assistant reply. Used to seed
        /// the conversation from a saved transcript, or to re-attach recent verbatim turns after a
        /// memory-compaction step.
        /// </summary>
        public void AddUser(string content) => _history.Add(ChatMessage.User(content));

        /// <summary>
        /// Appends an assistant turn to the history without invoking the model. Same use cases as
        /// <see cref="AddUser"/>: transcript seeding, post-compaction history rehydration.
        /// </summary>
        public void AddAssistant(string content) => _history.Add(ChatMessage.Assistant(content));

        /// <summary>Clears the conversation history.</summary>
        public void ResetConversation() => _history.Clear();

        /// <summary>
        /// Appends <paramref name="userMessage"/> to the history, generates the assistant
        /// reply, appends it to the history, and returns it. <paramref name="onText"/>, if
        /// supplied, receives the reply incrementally as it streams.
        /// </summary>
        /// <param name="userMessage">The user turn to append and respond to.</param>
        /// <param name="options">Generation options (sampling, max tokens, stop handling).</param>
        /// <param name="onText">Optional streaming callback receiving reply text incrementally.</param>
        /// <param name="constraint">
        /// Optional decode-time constraint (e.g. <c>JsonGrammarConstraint</c> for guaranteed
        /// well-formed JSON). When supplied, every generated token is masked to the constraint and
        /// the reply is structurally valid by construction. Create a fresh constraint per call —
        /// it is stateful. The underlying session must support constrained generation.
        /// </param>
        public string Send(
            string userMessage,
            in GenerationOptions options,
            Action<string>? onText = null,
            ITokenConstraint? constraint = null)
        {
            if (userMessage is null) { throw new ArgumentNullException(nameof(userMessage)); }

            _history.Add(ChatMessage.User(userMessage));
            var reply = GenerateFor(_history, in options, onText, constraint);
            _history.Add(ChatMessage.Assistant(reply));
            return reply;
        }

        /// <summary>
        /// One-shot generation that does NOT touch the conversation history: renders the current
        /// system turn(s) plus <paramref name="userMessage"/> as a single exchange, generates a reply
        /// and returns it — recording neither the user turn nor the reply. Use for stateless task
        /// calls (tool routing, JSON mode, retrieval answers) so they neither inherit earlier turns nor
        /// accumulate across calls: each one prefills only its own (minimal) prompt. <see cref="Send"/>
        /// remains the multi-turn conversational path.
        /// </summary>
        public string Complete(
            string userMessage,
            in GenerationOptions options,
            Action<string>? onText = null,
            ITokenConstraint? constraint = null)
        {
            if (userMessage is null) { throw new ArgumentNullException(nameof(userMessage)); }

            // [system turns] + this single user turn — no prior user/assistant turns, nothing retained.
            var oneShot = new List<ChatMessage>();
            foreach (var message in _history)
            {
                if (string.Equals(message.Role, "system", StringComparison.Ordinal)) { oneShot.Add(message); }
            }
            oneShot.Add(ChatMessage.User(userMessage));

            return GenerateFor(oneShot, in options, onText, constraint);
        }

        // Renders the given turns, prefills the session, runs the decode loop and records LastStats.
        private string GenerateFor(
            IReadOnlyList<ChatMessage> messages,
            in GenerationOptions options,
            Action<string>? onText,
            ITokenConstraint? constraint)
        {
            var promptText = _template.Render(messages, addGenerationPrompt: true);
            var tokenCount = _tokenizer.CountTokens(promptText);
            var promptTokens = new int[tokenCount];
            var written = _tokenizer.Encode(promptText, promptTokens);
            _session.Reset(promptTokens.AsSpan(0, written));

            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            var reply = Generate(in options, onText, constraint, out var generatedTokens);
            stopwatch.Stop();
            LastStats = new GenerationStats(
                promptTokens: written,
                generatedTokens: generatedTokens,
                elapsedNanoseconds: stopwatch.Elapsed.Ticks * 100,   // 1 tick = 100 ns
                allocatedBytes: 0,
                usedKeyValueCache: true);

            return reply;
        }

        private string Generate(
            in GenerationOptions options,
            Action<string>? onText,
            ITokenConstraint? constraint,
            out int generatedTokens)
        {
            var stops = new StopSequenceDetector(_stopSequences);
            var generated = new List<int>();
            var reply = new StringBuilder();
            var prevText = string.Empty;
            var sampling = options.Sampling;
            var maxNew = options.MaxNewTokens > 0 ? options.MaxNewTokens : int.MaxValue;

            // With sliding-window enabled the cache never overflows (oldest tokens roll off), so we
            // bound generation by MaxNewTokens only; otherwise we stop when the context fills.
            for (var i = 0; i < maxNew && (_slidingWindow || _session.CurrentPosition < _session.MaxContextLength); i++)
            {
                var token = _session.GenerateNextToken(in sampling, constraint);

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

                // A structural constraint (tool call / JSON grammar) reports IsComplete the instant a
                // complete root value closes. Stop there: the model would otherwise keep sampling the
                // only tokens still left unmasked — trailing whitespace — up to MaxNewTokens, which both
                // wastes ~MaxNewTokens decode steps and appends a junk whitespace tail to the output.
                if (constraint is { IsComplete: true })
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
