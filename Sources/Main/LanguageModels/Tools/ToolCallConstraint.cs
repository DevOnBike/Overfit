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
    /// <item>the <c>arguments</c> value is constrained to the chosen tool's parameter schema when it
    /// declares one (exact keys, in order, each value of the declared type), otherwise to any
    /// well-formed JSON value (delegated to <see cref="JsonStateMachine"/>).</item>
    /// </list>
    /// The reply is therefore always parseable by <see cref="ToolCall.TryParse"/> with a valid tool
    /// name; for a schema'd tool the arguments are guaranteed to carry exactly the right keys with the
    /// right value types, so the C# handler never sees an invented or missing argument.
    ///
    /// Up to 64 tools (the name-viability bit-mask). Stateful — one per generation.
    /// </summary>
    public sealed class ToolCallConstraint : ITokenConstraint
    {
        private readonly string[] _tokenText;
        private readonly string[] _names;
        private readonly CompiledSchema?[] _schemas;
        private readonly int _eosTokenId;
        private Envelope _state;

        public ToolCallConstraint(IReadOnlyList<ToolDefinition> tools, ITokenizer tokenizer)
        {
            ArgumentNullException.ThrowIfNull(tools);
            ArgumentNullException.ThrowIfNull(tokenizer);
            if (tools.Count == 0) { throw new ArgumentException("At least one tool is required.", nameof(tools)); }
            if (tools.Count > 64) { throw new ArgumentException("At most 64 tools are supported.", nameof(tools)); }

            _names = new string[tools.Count];
            _schemas = new CompiledSchema?[tools.Count];
            for (var i = 0; i < tools.Count; i++)
            {
                _names[i] = tools[i].Name;
                _schemas[i] = CompiledSchema.From(tools[i].Parameters);
            }

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
                _state.TryAdvance(text[i], _names, _schemas);
            }
        }

        private bool Accepts(string text)
        {
            var probe = _state;   // value-type copy (struct, incl. the inner JSON machine)
            for (var i = 0; i < text.Length; i++)
            {
                if (!probe.TryAdvance(text[i], _names, _schemas)) { return false; }
            }
            return true;
        }

        // A tool's argument object compiled to a DFA: fixed literal segments interleaved with typed
        // value slots. For params [a:String, b:Integer] the literals are ["{\"a\": ", ", \"b\": ", "}"]
        // and the kinds are [String, Integer] — Literals.Length == Kinds.Length + 1. Null when the tool
        // declares no parameters (the arguments then accept any well-formed JSON value).
        private sealed class CompiledSchema
        {
            public string[] Literals = [];
            public ToolParameterKind[] Kinds = [];

            public static CompiledSchema? From(IReadOnlyList<ToolParameter> parameters)
            {
                if (parameters.Count == 0) { return null; }

                var literals = new string[parameters.Count + 1];
                var kinds = new ToolParameterKind[parameters.Count];
                for (var i = 0; i < parameters.Count; i++)
                {
                    var prefix = i == 0 ? "{\"" : ", \"";
                    literals[i] = prefix + parameters[i].Name + "\": ";
                    kinds[i] = parameters[i].Kind;
                }
                literals[parameters.Count] = "}";
                return new CompiledSchema { Literals = literals, Kinds = kinds };
            }
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
            private int _toolIndex;     // resolved when the name closes (-1 until then)
            private JsonStateMachine _args;

            // Schema-mode (a tool that declares parameters) cursor.
            private int _segIndex;      // which literal segment we are matching
            private int _segPos;        // position within that literal
            private bool _inValue;      // currently inside a typed value slot
            private bool _valueStarted; // the value slot has consumed its first character
            private ToolParameterKind _valueKind;

            public static Envelope Initial(int toolCount) => new()
            {
                _stage = Stage.Open,
                _nameMask = toolCount >= 64 ? ulong.MaxValue : (1UL << toolCount) - 1,
                _toolIndex = -1,
            };

            public readonly bool IsComplete => _stage == Stage.Done;

            public bool TryAdvance(char c, string[] names, CompiledSchema?[] schemas)
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
                            _toolIndex = ResolveName(names);
                            if (_toolIndex < 0) { return false; }
                            _stage = Stage.Mid;
                            _midIndex = 1;   // the '"' just consumed is Mid[0]
                            return true;
                        }
                        return ExtendName(c, names);

                    case Stage.Mid:
                        if (c != Mid[_midIndex]) { return false; }
                        if (++_midIndex == Mid.Length)
                        {
                            // Switch to schema-driven arguments when the chosen tool declares them.
                            if (schemas[_toolIndex] is null)
                            {
                                _stage = Stage.Args;
                                _args = default;
                            }
                            else
                            {
                                _stage = Stage.ArgsSchema;
                                _segIndex = 0;
                                _segPos = 0;
                                _inValue = false;
                            }
                        }
                        return true;

                    case Stage.Args:
                        var probe = _args;
                        if (probe.TryAdvance(c)) { _args = probe; return true; }
                        if (_args.IsComplete && c == '}') { _stage = Stage.Done; return true; }
                        return false;

                    case Stage.ArgsSchema:
                        return TryAdvanceSchema(c, schemas[_toolIndex]!);

                    case Stage.EnvelopeClose:
                        // The schema closed the arguments object; the envelope needs its own '}'.
                        if (c == '}') { _stage = Stage.Done; return true; }
                        return false;

                    case Stage.Done:
                        return c is ' ' or '\t' or '\n' or '\r';

                    default:
                        return false;
                }
            }

            // Walks the fixed literal segments and typed value slots of the chosen tool's schema.
            private bool TryAdvanceSchema(char c, CompiledSchema schema)
            {
                if (_inValue)
                {
                    if (!_valueStarted)
                    {
                        // The value's first character pins its JSON type — reject a wrong-typed start.
                        if (!ValueFirstCharOk(_valueKind, c)) { return false; }
                        _valueStarted = true;
                        _args = default;
                        return _args.TryAdvance(c);
                    }

                    var probe = _args;
                    if (probe.TryAdvance(c)) { _args = probe; return true; }

                    // The value can't extend. If it is a complete JSON value, this character must
                    // begin the next literal segment (a ',' between pairs, or the closing '}').
                    if (!_args.IsComplete) { return false; }
                    _inValue = false;
                    _segIndex++;
                    _segPos = 0;
                    // fall through to literal matching of c
                }

                var lit = schema.Literals[_segIndex];
                if (_segPos >= lit.Length || c != lit[_segPos]) { return false; }
                if (++_segPos == lit.Length)
                {
                    if (_segIndex < schema.Kinds.Length)
                    {
                        _inValue = true;
                        _valueStarted = false;
                        _valueKind = schema.Kinds[_segIndex];
                    }
                    else
                    {
                        // Matched the final "}" that closes the arguments object; one more "}" (the
                        // envelope's own close) follows.
                        _stage = Stage.EnvelopeClose;
                    }
                }
                return true;
            }

            private static bool ValueFirstCharOk(ToolParameterKind kind, char c) => kind switch
            {
                ToolParameterKind.String => c == '"',
                ToolParameterKind.Boolean => c is 't' or 'f',
                _ => c == '-' || c is >= '0' and <= '9',   // Number / Integer
            };

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

            // The (single) viable tool whose full name has been matched, or -1 if none.
            private readonly int ResolveName(string[] names)
            {
                for (var i = 0; i < names.Length; i++)
                {
                    if ((_nameMask & (1UL << i)) != 0 && names[i].Length == _nameLen) { return i; }
                }
                return -1;
            }

            private enum Stage : byte
            {
                Open = 0,   // matching {"name": "
                Name,       // matching one of the tool names
                Mid,        // matching ", "arguments":
                Args,           // a well-formed JSON value (schema-less tool)
                ArgsSchema,     // the chosen tool's typed argument object
                EnvelopeClose,  // schema closed the args object; expect the envelope's own '}'
                Done,           // envelope closed
            }
        }
    }
}
