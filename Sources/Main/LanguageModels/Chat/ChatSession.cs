// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.InteropServices;
using System.Text;
using DevOnBike.Overfit.Diagnostics;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Runtime;

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

        /// <summary>Test hook: restore the pre-early-emit ordering (emit after the forward pass, not before).
        /// Defaults from <see cref="OverfitEnvironment.DisableEarlyEmit"/> so both orderings can be served by
        /// two processes and compared inside a single interleaved run.</summary>
        internal static bool DisableEarlyEmit =
            Environment.GetEnvironmentVariable(OverfitEnvironment.DisableEarlyEmit) == "1";

        /// <summary>Test hook: force the exact single-token decode loop instead of the speculative path.
        /// Defaults from <see cref="OverfitEnvironment.DisableSpeculative"/> so both can be served side by
        /// side and measured in one interleaved run.</summary>
        internal static bool DisableSpeculative =
            Environment.GetEnvironmentVariable(OverfitEnvironment.DisableSpeculative) == "1";

        /// <param name="session">Underlying SLM session that runs prefill/decode and owns the KV cache.</param>
        /// <param name="tokenizer">Tokenizer used to encode prompts and decode generated tokens.</param>
        /// <param name="template">Chat template that formats messages into the model's prompt format.</param>
        /// <param name="stopSequences">Optional extra stop sequences that end generation; merged with template defaults.</param>
        /// <param name="slidingWindow">
        /// When true, enables sliding-window KV eviction on the session so long conversations keep
        /// going past the model's context length (the oldest tokens roll off) instead of stopping at
        /// the limit. Requires a session that supports it (RoPE models — Qwen / Llama / Mistral);
        /// throws <see cref="OverfitRuntimeException"/> otherwise.
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
                    if (!string.IsNullOrEmpty(s))
                    {
                        stops.Add(s);
                    }
                }
            }
            _stopSequences = stops.ToArray();

            if (slidingWindow)
            {
                // Throws OverfitRuntimeException for non-RoPE sessions — fail early, at construction.
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
        public GenerationStats LastStats
        {
            get; private set;
        }

        /// <summary>
        /// How many prompt tokens the most recent turn took from the KV cache instead of re-encoding —
        /// 0 on the first turn of a conversation, and typically the whole preceding conversation
        /// afterwards. <c>LastStats.PromptTokens</c> minus this is what was actually forwarded.
        /// </summary>
        public int CachedPromptTokens
        {
            get; private set;
        }

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
            if (userMessage is null)
            {
                throw new ArgumentNullException(nameof(userMessage));
            }

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
            if (userMessage is null)
            {
                throw new ArgumentNullException(nameof(userMessage));
            }

            // [system turns] + this single user turn — no prior user/assistant turns, nothing retained.
            var oneShot = new List<ChatMessage>();
            foreach (var message in _history)
            {
                if (string.Equals(message.Role, "system", StringComparison.Ordinal))
                {
                    oneShot.Add(message);
                }
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

            // Reuse the KV already built for the shared prefix of the previous turn. Every turn re-sends the
            // whole conversation, so the tokens up to the end of the last assistant reply are byte-identical
            // to what this session just encoded — re-prefilling them is pure duplicate work. Falls back to a
            // full prefill on its own when the session is fresh or the conversation diverged.
            CachedPromptTokens = _session.PrefillReusingCache(promptTokens.AsSpan(0, written));

            var stopwatch = ValueStopwatch.StartNew();
            var reply = Generate(promptTokens.AsSpan(0, written), in options, onText, constraint, out var generatedTokens);
            LastStats = new GenerationStats(
                promptTokens: written,
                generatedTokens: generatedTokens,
                elapsedNanoseconds: stopwatch.GetElapsedTime().Ticks * 100,   // 1 tick = 100 ns
                allocatedBytes: 0,
                usedKeyValueCache: true);

            return reply;
        }

        private string Generate(
            ReadOnlySpan<int> promptTokens,
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
            var stopOnEot = options.StopOnEndOfTextToken;          // hoisted: `in` params can't be captured by a local fn
            var eotTokenId = options.EndOfTextTokenId;

            // Per-token handling shared by the single-token and speculative paths. Returns true when
            // generation should stop (end-of-text, a completed stop sequence, or a closed constraint).
            bool EmitToken(int token)
            {
                if (token == _tokenizer.EndOfTextTokenId ||
                    (stopOnEot && eotTokenId >= 0 && token == eotTokenId))
                {
                    return true;
                }

                generated.Add(token);

                // Incremental detokenize: decode the whole run and emit only the newly stabilised
                // suffix (byte-level BPE can leave a trailing partial codepoint until the next token
                // arrives — hold it back rather than emit garbage).
                var full = _tokenizer.DecodeToString(CollectionsMarshal.AsSpan(generated));
                if (full.Length <= prevText.Length || !full.StartsWith(prevText, StringComparison.Ordinal))
                {
                    return false;
                }
                var delta = full[prevText.Length..];
                prevText = full;

                var emit = stops.Append(delta);
                if (emit.Length > 0)
                {
                    reply.Append(emit);
                    onText?.Invoke(emit);
                }

                // A structural constraint (tool call / JSON grammar) reports IsComplete the instant a
                // complete root value closes — stop there rather than sample a junk whitespace tail up
                // to MaxNewTokens. (Never set in the speculative path, which only runs unconstrained.)
                return stops.Stopped || constraint is { IsComplete: true };
            }

            // Emit the token the instant it is sampled, before the forward pass that prepares the NEXT
            // logits — otherwise every token, the first one included, arrives one whole weight-pass late.
            // Returning the stop decision straight back lets the session skip that pass when the answer is
            // over. Allocated once per generation, not per token.
            var stopped = false;
            var onSampled = new Func<int, bool>(token =>
            {
                stopped = EmitToken(token);
                return stopped;
            });

            // Test hook: emit after the step instead of during it, i.e. the pre-early-emit ordering.
            if (DisableEarlyEmit)
            {
                onSampled = null!;
            }

            // Speculative fast path (prompt-lookup, adaptively gated): commits ≥1 token per batched
            // verify, and ~free when drafts don't fire (dn == 0 takes a plain single-token step, so the
            // only cost is the drafter call) — but it can't mask the draft against a per-token
            // constraint, so it only runs unconstrained on a speculation-capable session. Everything
            // else falls back to the exact single-token loop.
            //
            // "Sampling-correct" used to be claimed here and it needs qualifying, because THIS is the
            // caller-facing surface. The rejection sampling is exact with respect to the distribution the
            // verify forward computes — but that forward is batched and quantized, and its logits differ
            // from the single-token path's by 0.47-1.02 on Qwen2.5-3B Q4_K_M (measured 2026-08-07), which
            // is more than the usual gap between the top two tokens. So a caller passing
            // SamplingOptions.Greedy can get DIFFERENT TEXT depending on whether speculation engaged,
            // and whether it engaged depends on the adaptive gate and on the drafter finding an n-gram —
            // neither of which the caller can see. OVERFIT_DISABLE_SPECULATIVE forces the exact
            // single-token loop for the whole process. T11 in docs/test-gate-backlog.md carries the
            // measurements and the open decision on whether greedy should opt in rather than out.
            // Hoisted out of the condition: the speculative session is needed inside the branch, and a
            // second (negated) test could not re-introduce a pattern variable in the same scope.
            var spec = _session as CachedLlamaSession;
            var useSpeculative = constraint is null && spec is not null && spec.CanSpeculate && !DisableSpeculative;

            if (useSpeculative)
            {
                const int maxDraft = 8;
                var history = new List<int>(promptTokens.Length + Math.Min(maxNew, 4096));
                foreach (var t in promptTokens)
                {
                    history.Add(t);
                }
                var committed = new int[maxDraft + 2];

                while (generated.Count < maxNew &&
                       (_slidingWindow || _session.CurrentPosition < _session.MaxContextLength))
                {
                    var n = spec!.GenerateSpeculative(
                        CollectionsMarshal.AsSpan(history), committed, in sampling, maxDraft, onSampled);
                    var stop = false;
                    for (var c = 0; c < n; c++)
                    {
                        var token = committed[c];
                        history.Add(token);

                        // committed[0] is the token the hook already emitted before the verify forward ran;
                        // re-emitting it would duplicate it in the stream.
                        if (c == 0)
                        {
                            if (DisableEarlyEmit)
                            {
                                stopped = EmitToken(token);
                            }

                            if (stopped || generated.Count >= maxNew)
                            {
                                stop = true;
                                break;
                            }

                            continue;
                        }

                        if (EmitToken(token) || generated.Count >= maxNew)
                        {
                            stop = true;
                            break;
                        }
                    }
                    if (stop)
                    {
                        break;
                    }
                }
            }

            if (!useSpeculative)
            {
                // With sliding-window enabled the cache never overflows (oldest tokens roll off), so we
                // bound generation by MaxNewTokens only; otherwise we stop when the context fills.
                for (var i = 0; i < maxNew &&
                     (_slidingWindow || _session.CurrentPosition < _session.MaxContextLength); i++)
                {
                    // The hook emits; `stopped` carries its verdict back out. Sessions without early-emit
                    // support still invoke it exactly once per token, just after their forward.
                    var produced = _session.GenerateNextToken(in sampling, constraint, onSampled);
                    if (DisableEarlyEmit)
                    {
                        stopped = EmitToken(produced);
                    }

                    if (stopped)
                    {
                        break;
                    }
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
