// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Constraints;
using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.LanguageModels.Tools
{
    /// <summary>
    /// An <see cref="ITokenConstraint"/> that forces the model to emit a single, valid tool call as
    /// the canonical envelope <c>{"name": "&lt;tool&gt;", "arguments": &lt;json&gt;}</c>:
    /// <list type="bullet">
    /// <item>the structural punctuation is fixed,</item>
    /// <item>the <c>name</c> value is constrained to be exactly one of the registered tool names
    /// (an enum DFA over their characters),</item>
    /// <item>the <c>arguments</c> value is constrained to be well-formed JSON (delegated to
    /// <see cref="JsonStateMachine"/>).</item>
    /// </list>
    /// The reply is therefore always parseable by <see cref="ToolCall.TryParse"/> with a valid tool
    /// name — the model only supplies the choice and the argument values, never the structure.
    /// Argument <i>typing</i> (per-tool schema) is the JSON-Schema follow-on; the handler validates
    /// arguments for now.
    ///
    /// Up to 64 tools (the name-viability bit-mask). Stateful — one per generation.
    /// </summary>
    public sealed class ToolCallConstraint : ITokenConstraint
    {
        private readonly string[] _tokenText;
        private readonly string[] _names;
        private readonly int _eosTokenId;
        private Envelope _state;

        public ToolCallConstraint(IReadOnlyList<ToolDefinition> tools, ITokenizer tokenizer)
        {
            ArgumentNullException.ThrowIfNull(tools);
            ArgumentNullException.ThrowIfNull(tokenizer);
            if (tools.Count == 0) { throw new ArgumentException("At least one tool is required.", nameof(tools)); }
            if (tools.Count > 64) { throw new ArgumentException("At most 64 tools are supported.", nameof(tools)); }

            _names = new string[tools.Count];
            for (var i = 0; i < tools.Count; i++) { _names[i] = tools[i].Name; }

            _eosTokenId = tokenizer.EndOfTextTokenId;

            var vocab = tokenizer.VocabularySize;
            _tokenText = new string[vocab];
            Span<int> one = stackalloc int[1];
            for (var t = 0; t < vocab; t++)
            {
                one[0] = t;
                _tokenText[t] = tokenizer.DecodeToString(one);
            }

            _state = Envelope.Initial(tools.Count);
        }

        public bool IsComplete => _state.IsComplete;

        public void ApplyMask(Span<float> logits)
        {
            // Model logit vectors can be longer than the tokenizer vocab (GGUF pads it); those
            // trailing padding slots have no text and are always masked.
            if (logits.Length < _tokenText.Length)
            {
                throw new ArgumentException(
                    $"Logits length ({logits.Length}) is smaller than the tokenizer vocabulary ({_tokenText.Length}).",
                    nameof(logits));
            }

            for (var t = 0; t < logits.Length; t++)
            {
                if (t == _eosTokenId)
                {
                    if (!_state.IsComplete) { logits[t] = float.NegativeInfinity; }
                    continue;
                }

                var text = t < _tokenText.Length ? _tokenText[t] : string.Empty;
                if (text.Length == 0 || !Accepts(text))
                {
                    logits[t] = float.NegativeInfinity;
                }
            }
        }

        public void Accept(int token)
        {
            if (token == _eosTokenId) { return; }
            if ((uint)token >= (uint)_tokenText.Length) { return; }

            var text = _tokenText[token];
            for (var i = 0; i < text.Length; i++)
            {
                _state.TryAdvance(text[i], _names);
            }
        }

        private bool Accepts(string text)
        {
            var probe = _state;   // value-type copy (struct, incl. the inner JSON machine)
            for (var i = 0; i < text.Length; i++)
            {
                if (!probe.TryAdvance(text[i], _names)) { return false; }
            }
            return true;
        }

        // The envelope acceptor, as a value type so candidate tokens can be tested on a copy.
        private struct Envelope
        {
            // Canonical, whitespace-free structure — constrained decoding drives it exactly.
            private const string Open = "{\"name\": \"";
            private const string Mid = "\", \"arguments\": ";

            private Stage _stage;
            private int _openIndex;
            private int _midIndex;
            private ulong _nameMask;   // viable tool indices given the name characters seen so far
            private int _nameLen;
            private JsonStateMachine _args;

            public static Envelope Initial(int toolCount) => new()
            {
                _stage = Stage.Open,
                _nameMask = toolCount >= 64 ? ulong.MaxValue : (1UL << toolCount) - 1,
            };

            public readonly bool IsComplete => _stage == Stage.Done;

            public bool TryAdvance(char c, string[] names)
            {
                switch (_stage)
                {
                    case Stage.Open:
                        if (c != Open[_openIndex]) { return false; }
                        if (++_openIndex == Open.Length) { _stage = Stage.Name; }
                        return true;

                    case Stage.Name:
                        if (c == '"')
                        {
                            // Close the name only if it exactly equals some registered tool.
                            if (!AnyCompleteName(names)) { return false; }
                            _stage = Stage.Mid;
                            _midIndex = 1;   // the '"' just consumed is Mid[0]
                            return true;
                        }
                        return ExtendName(c, names);

                    case Stage.Mid:
                        if (c != Mid[_midIndex]) { return false; }
                        if (++_midIndex == Mid.Length) { _stage = Stage.Args; _args = default; }
                        return true;

                    case Stage.Args:
                        var probe = _args;
                        if (probe.TryAdvance(c)) { _args = probe; return true; }
                        if (_args.IsComplete && c == '}') { _stage = Stage.Done; return true; }
                        return false;

                    case Stage.Done:
                        return c is ' ' or '\t' or '\n' or '\r';

                    default:
                        return false;
                }
            }

            private bool ExtendName(char c, string[] names)
            {
                ulong next = 0;
                for (var i = 0; i < names.Length; i++)
                {
                    if ((_nameMask & (1UL << i)) == 0) { continue; }
                    var name = names[i];
                    if (name.Length > _nameLen && name[_nameLen] == c) { next |= 1UL << i; }
                }
                if (next == 0) { return false; }
                _nameMask = next;
                _nameLen++;
                return true;
            }

            private readonly bool AnyCompleteName(string[] names)
            {
                for (var i = 0; i < names.Length; i++)
                {
                    if ((_nameMask & (1UL << i)) != 0 && names[i].Length == _nameLen) { return true; }
                }
                return false;
            }

            private enum Stage : byte
            {
                Open = 0,   // matching {"name": "
                Name,       // matching one of the tool names
                Mid,        // matching ", "arguments":
                Args,       // a well-formed JSON value (the arguments)
                Done,       // envelope closed
            }
        }
    }
}
