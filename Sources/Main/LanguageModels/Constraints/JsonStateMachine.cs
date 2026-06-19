// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Constraints
{
    /// <summary>
    /// A character-level acceptor for a single well-formed JSON document (RFC 8259): one value,
    /// optionally surrounded by whitespace. It is a value type — copy it to test a candidate token
    /// speculatively, then keep the copy only if every character was accepted — so masking the
    /// vocabulary allocates nothing.
    ///
    /// Nesting (object/array) is tracked in a 64-bit bit-stack (one bit per level: object vs array)
    /// plus a depth counter; numbers and string escapes are sub-DFAs. <see cref="TryAdvance"/> returns
    /// false the moment a character would make the document un-completable; <see cref="IsComplete"/>
    /// is true exactly when the stream could validly end here (used to gate the end-of-text token).
    ///
    /// Scope: enforces structural well-formedness (balanced brackets/quotes, commas/colons in the
    /// right places, RFC-valid numbers and <c>true/false/null</c> literals). It does not enforce a
    /// specific schema — that is the JSON-Schema follow-on.
    /// </summary>
    public struct JsonStateMachine
    {
        // Bit i = nesting level i: 0 ⇒ object, 1 ⇒ array. Depth ⇒ number of open containers.
        private ulong _stack;
        private int _depth;
        private Phase _phase;
        private NumberState _number;

        // Literal matching (true/false/null): the target text and how far we've matched.
        private byte _literalKind;   // 0 none, 1 true, 2 false, 3 null
        private int _literalIndex;

        // String unicode escape (\uXXXX): hex digits seen so far.
        private int _unicodeDigits;

        // After a string closes, where to go: a key returns to "expect colon", a value completes.
        private bool _stringIsKey;

        /// <summary>Maximum object/array nesting depth (bounded by the 64-bit bit-stack).</summary>
        public const int MaxDepth = 64;

        /// <summary>True when the stream could validly terminate now (a complete root value).</summary>
        public readonly bool IsComplete =>
            _depth == 0 && (_phase == Phase.Done || (_phase == Phase.InNumber && IsTerminalNumber));

        // ── Structural accessors for the JSON-Schema overlay (JsonSchemaTracker) ──
        // The schema tracker advances in lockstep with this machine and needs to observe its structural
        // position (which phase, nesting depth, whether the current string is a key, array vs object top).
        internal Phase CurrentPhase => _phase;
        internal int Depth => _depth;
        internal bool CurrentStringIsKey => _stringIsKey;
        internal bool TopIsArray => _depth > 0 && (_stack & (1UL << (_depth - 1))) != 0;

        /// <summary>
        /// Feeds one character, advancing the acceptor. Returns false if the character cannot extend
        /// any well-formed JSON document from the current state (the caller discards the mutated copy).
        /// </summary>
        public bool TryAdvance(char c)
        {
            switch (_phase)
            {
                case Phase.ExpectValue:
                case Phase.ExpectValueOrClose:
                    if (IsWhitespace(c))
                    {
                        return true;
                    }
                    if (c == ']' && _phase == Phase.ExpectValueOrClose)
                    {
                        return CloseContainer(isArray: true);
                    }
                    return BeginValue(c);

                case Phase.ObjectStart:
                    if (IsWhitespace(c))
                    {
                        return true;
                    }
                    if (c == '"')
                    {
                        _phase = Phase.InString;
                        _stringIsKey = true;
                        return true;
                    }
                    if (c == '}')
                    {
                        return CloseContainer(isArray: false);
                    }
                    return false;

                case Phase.ObjectExpectKey:
                    if (IsWhitespace(c))
                    {
                        return true;
                    }
                    if (c == '"')
                    {
                        _phase = Phase.InString;
                        _stringIsKey = true;
                        return true;
                    }
                    return false;

                case Phase.ObjectAfterKey:
                    if (IsWhitespace(c))
                    {
                        return true;
                    }
                    if (c == ':')
                    {
                        _phase = Phase.ExpectValue;
                        return true;
                    }
                    return false;

                case Phase.ObjectAfterPair:
                    if (IsWhitespace(c))
                    {
                        return true;
                    }
                    if (c == ',')
                    {
                        _phase = Phase.ObjectExpectKey;
                        return true;
                    }
                    if (c == '}')
                    {
                        return CloseContainer(isArray: false);
                    }
                    return false;

                case Phase.ArrayAfterElement:
                    if (IsWhitespace(c))
                    {
                        return true;
                    }
                    if (c == ',')
                    {
                        _phase = Phase.ExpectValue;
                        return true;
                    }
                    if (c == ']')
                    {
                        return CloseContainer(isArray: true);
                    }
                    return false;

                case Phase.Done:
                    return IsWhitespace(c);   // only trailing whitespace after a complete root value

                case Phase.InString:
                    return AdvanceString(c);

                case Phase.StringEscape:
                    return AdvanceStringEscape(c);

                case Phase.StringUnicode:
                    if (!IsHex(c))
                    {
                        return false;
                    }
                    if (++_unicodeDigits == 4)
                    {
                        _phase = Phase.InString;
                    }
                    return true;

                case Phase.InNumber:
                    return AdvanceNumber(c);

                case Phase.InLiteral:
                    return AdvanceLiteral(c);

                default:
                    return false;
            }
        }

        // ── Value dispatch ───────────────────────────────────────────────────

        private bool BeginValue(char c)
        {
            switch (c)
            {
                case '{':
                    if (!Push(isArray: false))
                    {
                        return false;
                    }
                    _phase = Phase.ObjectStart;
                    return true;
                case '[':
                    if (!Push(isArray: true))
                    {
                        return false;
                    }
                    _phase = Phase.ExpectValueOrClose;
                    return true;
                case '"':
                    _phase = Phase.InString;
                    _stringIsKey = false;
                    return true;
                case '-':
                    _phase = Phase.InNumber;
                    _number = NumberState.AfterSign;
                    return true;
                case 't':
                    return BeginLiteral(kind: 1);
                case 'f':
                    return BeginLiteral(kind: 2);
                case 'n':
                    return BeginLiteral(kind: 3);
                default:
                    if (c == '0')
                    {
                        _phase = Phase.InNumber;
                        _number = NumberState.AfterZero;
                        return true;
                    }
                    if (c is >= '1' and <= '9')
                    {
                        _phase = Phase.InNumber;
                        _number = NumberState.Int;
                        return true;
                    }
                    return false;
            }
        }

        // ── Strings ──────────────────────────────────────────────────────────

        private bool AdvanceString(char c)
        {
            if (c == '"')
            {
                _phase = _stringIsKey ? Phase.ObjectAfterKey : PhaseAfterValue();
                return true;
            }
            if (c == '\\')
            {
                _phase = Phase.StringEscape;
                return true;
            }
            return c >= 0x20;   // control characters must be escaped
        }

        private bool AdvanceStringEscape(char c)
        {
            switch (c)
            {
                case '"':
                case '\\':
                case '/':
                case 'b':
                case 'f':
                case 'n':
                case 'r':
                case 't':
                    _phase = Phase.InString;
                    return true;
                case 'u':
                    _phase = Phase.StringUnicode;
                    _unicodeDigits = 0;
                    return true;
                default:
                    return false;
            }
        }

        // ── Numbers ──────────────────────────────────────────────────────────

        private readonly bool IsTerminalNumber =>
            _number is NumberState.AfterZero or NumberState.Int or NumberState.Frac or NumberState.Exp;

        private bool AdvanceNumber(char c)
        {
            switch (_number)
            {
                case NumberState.AfterSign:
                    if (c == '0')
                    {
                        _number = NumberState.AfterZero;
                        return true;
                    }
                    if (c is >= '1' and <= '9')
                    {
                        _number = NumberState.Int;
                        return true;
                    }
                    return false;

                case NumberState.AfterZero:
                    if (c == '.')
                    {
                        _number = NumberState.AfterDot;
                        return true;
                    }
                    if (c is 'e' or 'E')
                    {
                        _number = NumberState.AfterExp;
                        return true;
                    }
                    if (c is >= '0' and <= '9')
                    {
                        return false;
                    }   // no leading zeros
                    return TerminateNumberAndReprocess(c);

                case NumberState.Int:
                    if (c is >= '0' and <= '9')
                    {
                        return true;
                    }
                    if (c == '.')
                    {
                        _number = NumberState.AfterDot;
                        return true;
                    }
                    if (c is 'e' or 'E')
                    {
                        _number = NumberState.AfterExp;
                        return true;
                    }
                    return TerminateNumberAndReprocess(c);

                case NumberState.AfterDot:
                    if (c is >= '0' and <= '9')
                    {
                        _number = NumberState.Frac;
                        return true;
                    }
                    return false;

                case NumberState.Frac:
                    if (c is >= '0' and <= '9')
                    {
                        return true;
                    }
                    if (c is 'e' or 'E')
                    {
                        _number = NumberState.AfterExp;
                        return true;
                    }
                    return TerminateNumberAndReprocess(c);

                case NumberState.AfterExp:
                    if (c is '+' or '-')
                    {
                        _number = NumberState.AfterExpSign;
                        return true;
                    }
                    if (c is >= '0' and <= '9')
                    {
                        _number = NumberState.Exp;
                        return true;
                    }
                    return false;

                case NumberState.AfterExpSign:
                    if (c is >= '0' and <= '9')
                    {
                        _number = NumberState.Exp;
                        return true;
                    }
                    return false;

                case NumberState.Exp:
                    if (c is >= '0' and <= '9')
                    {
                        return true;
                    }
                    return TerminateNumberAndReprocess(c);

                default:
                    return false;
            }
        }

        // A number ends only when a non-number character arrives; that character then has to be
        // valid in the post-value context, so finalize the number and re-dispatch the character.
        private bool TerminateNumberAndReprocess(char c)
        {
            if (!IsTerminalNumber)
            {
                return false;
            }
            _phase = PhaseAfterValue();
            return TryAdvance(c);
        }

        // ── Literals (true / false / null) ────────────────────────────────────

        private bool BeginLiteral(byte kind)
        {
            _phase = Phase.InLiteral;
            _literalKind = kind;
            _literalIndex = 1;   // the dispatching character was the literal's first char
            return true;
        }

        private bool AdvanceLiteral(char c)
        {
            var text = _literalKind switch
            {
                1 => "true",
                2 => "false",
                _ => "null"
            };
            if (_literalIndex >= text.Length || c != text[_literalIndex])
            {
                return false;
            }
            if (++_literalIndex == text.Length)
            {
                _phase = PhaseAfterValue();
            }
            return true;
        }

        // ── Container stack ────────────────────────────────────────────────────

        private bool Push(bool isArray)
        {
            if (_depth >= MaxDepth)
            {
                return false;
            }
            if (isArray)
            {
                _stack |= 1UL << _depth;
            }
            else
            {
                _stack &= ~(1UL << _depth);
            }
            _depth++;
            return true;
        }

        private bool CloseContainer(bool isArray)
        {
            if (_depth == 0)
            {
                return false;
            }
            var topIsArray = (_stack & (1UL << (_depth - 1))) != 0;
            if (topIsArray != isArray)
            {
                return false;
            }
            _depth--;
            _phase = PhaseAfterValue();   // the closed container is itself a completed value
            return true;
        }

        // Where to go after a value completes: depends on the now-current container (or root).
        private readonly Phase PhaseAfterValue()
        {
            if (_depth == 0)
            {
                return Phase.Done;
            }
            var topIsArray = (_stack & (1UL << (_depth - 1))) != 0;
            return topIsArray ? Phase.ArrayAfterElement : Phase.ObjectAfterPair;
        }

        private static bool IsWhitespace(char c) => c is ' ' or '\t' or '\n' or '\r';

        private static bool IsHex(char c) =>
            c is (>= '0' and <= '9') or (>= 'a' and <= 'f') or (>= 'A' and <= 'F');

        internal enum Phase : byte
        {
            ExpectValue = 0,        // root start, after ':' , or after ',' in an array
            ExpectValueOrClose,     // right after '[' (a value or ']')
            ObjectStart,            // right after '{' (a key or '}')
            ObjectExpectKey,        // after ',' in an object
            ObjectAfterKey,         // after a key string (expect ':')
            ObjectAfterPair,        // after a pair value (expect ',' or '}')
            ArrayAfterElement,      // after an element (expect ',' or ']')
            Done,                   // a complete root value (only trailing whitespace allowed)
            InString,
            StringEscape,
            StringUnicode,
            InNumber,
            InLiteral,
        }

        private enum NumberState : byte
        {
            AfterSign = 0,   // saw '-', need a digit
            AfterZero,       // saw a leading '0'
            Int,             // integer digits (1-9 …)
            AfterDot,        // saw '.', need a digit
            Frac,            // fractional digits
            AfterExp,        // saw 'e'/'E'
            AfterExpSign,    // saw 'e'/'E' then '+'/'-'
            Exp,             // exponent digits
        }
    }
}
