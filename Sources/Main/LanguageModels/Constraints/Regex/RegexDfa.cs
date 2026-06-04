// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Constraints.Regex
{
    /// <summary>
    /// A compiled deterministic finite automaton over the ASCII alphabet (0–127) for constrained decoding —
    /// the runtime form of a regular expression. The whole generated string must match (anchored full match);
    /// the DFA state is a single <see cref="int"/>, so a constraint can copy it freely for speculative token
    /// probing. (Approach ported from dotLLM's regex constraint — parse → Thompson NFA → subset construction.)
    ///
    /// Supported syntax (MVP): literals, <c>.</c>, character classes <c>[...]</c> with ranges / negation,
    /// the predefined classes <c>\d \w \s \D \W \S</c>, quantifiers <c>* + ? {n} {n,m} {n,}</c>, alternation
    /// <c>|</c>, and groups <c>(...)</c>. Non-ASCII characters and unsupported escapes are not matched.
    /// </summary>
    public sealed class RegexDfa
    {
        private const int Alphabet = 128;

        private readonly int[] _transitions;   // [state * 128 + c] → next state, or -1 (dead)
        private readonly bool[] _accepting;

        private RegexDfa(int[] transitions, bool[] accepting)
        {
            _transitions = transitions;
            _accepting = accepting;
        }

        /// <summary>Number of DFA states; state 0 is the start.</summary>
        public int StateCount => _accepting.Length;

        /// <summary>The start state.</summary>
        public int Start => 0;

        /// <summary>Whether <paramref name="state"/> is an accepting (full-match) state.</summary>
        public bool IsAccepting(int state) => _accepting[state];

        /// <summary>The state reached from <paramref name="state"/> on <paramref name="c"/>, or -1 if none
        /// (a dead end: <paramref name="c"/> cannot extend any match from here).</summary>
        public int Next(int state, char c)
        {
            if (c >= Alphabet) { return -1; }
            return _transitions[state * Alphabet + (int)c];
        }

        /// <summary>Compiles <paramref name="pattern"/> into a DFA. Throws <see cref="ArgumentException"/>
        /// on a malformed or too-large pattern.</summary>
        public static RegexDfa Compile(string pattern)
        {
            ArgumentNullException.ThrowIfNull(pattern);

            var nfa = new Nfa();
            var frag = nfa.Parse(pattern);
            return Subset(nfa, frag.Start, frag.End);
        }

        // ── Subset construction: NFA → DFA over the ASCII alphabet ───────────────
        private static RegexDfa Subset(Nfa nfa, int nfaStart, int nfaAccept)
        {
            var startSet = nfa.Closure(new HashSet<int> { nfaStart });
            var dfa = new List<HashSet<int>> { startSet };
            var index = new Dictionary<string, int> { [Key(startSet)] = 0 };
            var transitions = new List<int[]>();
            var accepting = new List<bool>();

            for (var s = 0; s < dfa.Count; s++)
            {
                var set = dfa[s];
                var row = new int[Alphabet];
                for (var c = 0; c < Alphabet; c++)
                {
                    var move = nfa.Closure(nfa.Move(set, c));
                    if (move.Count == 0) { row[c] = -1; continue; }

                    var key = Key(move);
                    if (!index.TryGetValue(key, out var target))
                    {
                        target = dfa.Count;
                        index[key] = target;
                        dfa.Add(move);
                    }
                    row[c] = target;
                }
                transitions.Add(row);
                accepting.Add(set.Contains(nfaAccept));

                if (dfa.Count > 4096)
                {
                    throw new ArgumentException("Regex compiles to too many states (pattern too complex).", nameof(nfa));
                }
            }

            var flat = new int[transitions.Count * Alphabet];
            for (var s = 0; s < transitions.Count; s++)
            {
                transitions[s].AsSpan().CopyTo(flat.AsSpan(s * Alphabet, Alphabet));
            }
            return new RegexDfa(flat, accepting.ToArray());
        }

        private static string Key(HashSet<int> set)
        {
            var ids = new int[set.Count];
            var i = 0;
            foreach (var v in set) { ids[i++] = v; }
            System.Array.Sort(ids);
            return string.Join(',', ids);
        }

        /// <summary>Thompson-construction NFA built by a recursive-descent parse of the pattern.</summary>
        private sealed class Nfa
        {
            private readonly List<List<int>> _eps = [];                       // epsilon transitions per state
            private readonly List<List<(ulong Lo, ulong Hi, int Target)>> _moves = [];   // labeled transitions
            private string _pattern = string.Empty;
            private int _pos;

            public Fragment Parse(string pattern)
            {
                _pattern = pattern;
                _pos = 0;
                var frag = ParseAlternation();
                if (_pos != _pattern.Length)
                {
                    throw new ArgumentException($"Unexpected '{_pattern[_pos]}' at position {_pos} in regex.", nameof(pattern));
                }
                return frag;
            }

            public HashSet<int> Closure(HashSet<int> states)
            {
                var stack = new Stack<int>(states);
                var result = new HashSet<int>(states);
                while (stack.Count > 0)
                {
                    var s = stack.Pop();
                    foreach (var t in _eps[s])
                    {
                        if (result.Add(t)) { stack.Push(t); }
                    }
                }
                return result;
            }

            public HashSet<int> Move(HashSet<int> states, int c)
            {
                var result = new HashSet<int>();
                var lo = c < 64 ? 1UL << c : 0UL;
                var hi = c >= 64 ? 1UL << (c - 64) : 0UL;
                foreach (var s in states)
                {
                    foreach (var (mLo, mHi, target) in _moves[s])
                    {
                        if ((mLo & lo) != 0 || (mHi & hi) != 0) { result.Add(target); }
                    }
                }
                return result;
            }

            // ── Recursive descent (each returns an NFA fragment start→end) ────────
            private Fragment ParseAlternation()
            {
                var left = ParseConcat();
                while (Peek() == '|')
                {
                    _pos++;
                    var right = ParseConcat();
                    var start = NewState();
                    var end = NewState();
                    _eps[start].Add(left.Start);
                    _eps[start].Add(right.Start);
                    _eps[left.End].Add(end);
                    _eps[right.End].Add(end);
                    left = new Fragment(start, end);
                }
                return left;
            }

            private Fragment ParseConcat()
            {
                // Empty concat (e.g. an empty alternative) → an epsilon fragment.
                if (Peek() is '\0' or '|' or ')')
                {
                    var s = NewState();
                    return new Fragment(s, s);
                }

                var frag = ParseRepeat();
                while (Peek() is not ('\0' or '|' or ')'))
                {
                    var next = ParseRepeat();
                    _eps[frag.End].Add(next.Start);
                    frag = new Fragment(frag.Start, next.End);
                }
                return frag;
            }

            private Fragment ParseRepeat()
            {
                var atom = ParseAtom();
                var c = Peek();
                switch (c)
                {
                    case '*': _pos++; return Star(atom, optional: true);
                    case '+': _pos++; return Star(atom, optional: false);
                    case '?': _pos++; return Optional(atom);
                    case '{': return ParseCounted(atom);
                    default: return atom;
                }
            }

            private Fragment Star(Fragment atom, bool optional)
            {
                var start = NewState();
                var end = NewState();
                _eps[start].Add(atom.Start);
                _eps[atom.End].Add(atom.Start);   // loop back
                _eps[atom.End].Add(end);
                if (optional) { _eps[start].Add(end); }   // '*' may match zero
                return new Fragment(start, end);
            }

            private Fragment Optional(Fragment atom)
            {
                var start = NewState();
                var end = NewState();
                _eps[start].Add(atom.Start);
                _eps[start].Add(end);
                _eps[atom.End].Add(end);
                return new Fragment(start, end);
            }

            private Fragment ParseCounted(Fragment first)
            {
                // '{' already at Peek; parse {n}, {n,m}, {n,}.
                _pos++;   // consume '{'
                var n = ParseInt();
                var hasMax = false;
                var m = n;
                if (Peek() == ',')
                {
                    _pos++;
                    hasMax = true;
                    if (Peek() == '}') { m = -1; }   // {n,} = n then star
                    else { m = ParseInt(); }
                }
                if (Peek() != '}') { throw new ArgumentException("Unterminated '{' quantifier in regex.", nameof(first)); }
                _pos++;   // consume '}'

                // Pre-clone every needed copy of the atom FROM THE PRISTINE fragment first: chaining (Append)
                // mutates a fragment's end-state epsilon list, so a later Clone would otherwise pull in the
                // already-chained earlier copies and the count would be wrong.
                var parts = new List<Fragment>();
                for (var i = 0; i < n; i++) { parts.Add(i == 0 ? first : Clone(first)); }
                if (hasMax && m < 0)
                {
                    parts.Add(Star(n == 0 ? first : Clone(first), optional: true));   // {n,} → n then star
                }
                else if (hasMax)
                {
                    for (var i = n; i < m; i++) { parts.Add(Optional(Clone(first))); }   // {n,m} → (m-n) optional
                }

                // Assemble by chaining (now that all cloning is done).
                Fragment? chain = null;
                foreach (var p in parts) { chain = Append(chain, p); }
                if (chain is null)
                {
                    var s = NewState();   // {0} → epsilon
                    return new Fragment(s, s);
                }
                return chain.Value;
            }

            private Fragment Append(Fragment? chain, Fragment next)
            {
                if (chain is null) { return next; }
                _eps[chain.Value.End].Add(next.Start);
                return new Fragment(chain.Value.Start, next.End);
            }

            private Fragment ParseAtom()
            {
                var c = Peek();
                switch (c)
                {
                    case '(':
                        _pos++;
                        var inner = ParseAlternation();
                        if (Peek() != ')') { throw new ArgumentException("Unbalanced '(' in regex.", nameof(inner)); }
                        _pos++;
                        return inner;
                    case '[':
                        return ParseClass();
                    case '.':
                        _pos++;
                        return CharFragment(AnyMask());
                    case '\\':
                        _pos++;
                        return EscapeFragment();
                    case '\0':
                        throw new ArgumentException("Unexpected end of regex.", nameof(c));
                    default:
                        if (c is '*' or '+' or '?' or ')' or '|')
                        {
                            throw new ArgumentException($"Unexpected '{c}' in regex at position {_pos}.", nameof(c));
                        }
                        _pos++;
                        return CharFragment(SingleMask(c));
                }
            }

            private Fragment EscapeFragment()
            {
                var c = Peek();
                _pos++;
                var (lo, hi) = c switch
                {
                    'd' => DigitMask(),
                    'w' => WordMask(),
                    's' => SpaceMask(),
                    'D' => Negate(DigitMask()),
                    'W' => Negate(WordMask()),
                    'S' => Negate(SpaceMask()),
                    'n' => SingleMask('\n'),
                    't' => SingleMask('\t'),
                    'r' => SingleMask('\r'),
                    _ => SingleMask(c),   // escaped literal (\. \[ \\ etc.)
                };
                return CharFragment((lo, hi));
            }

            private Fragment ParseClass()
            {
                _pos++;   // consume '['
                var negate = false;
                if (Peek() == '^') { negate = true; _pos++; }
                ulong lo = 0, hi = 0;

                while (Peek() is not ('\0' or ']'))
                {
                    var c = Next();
                    if (c == '\\')
                    {
                        var e = Next();
                        var (elo, ehi) = e switch
                        {
                            'd' => DigitMask(),
                            'w' => WordMask(),
                            's' => SpaceMask(),
                            'n' => SingleMask('\n'),
                            't' => SingleMask('\t'),
                            'r' => SingleMask('\r'),
                            _ => SingleMask(e),
                        };
                        lo |= elo; hi |= ehi;
                        continue;
                    }
                    if (Peek() == '-' && Peek(1) is not (']' or '\0'))
                    {
                        _pos++;   // consume '-'
                        var to = Next();
                        SetRange(ref lo, ref hi, c, to);
                    }
                    else
                    {
                        SetBit(ref lo, ref hi, c);
                    }
                }
                if (Peek() != ']') { throw new ArgumentException("Unterminated '[' character class in regex.", nameof(negate)); }
                _pos++;   // consume ']'

                return CharFragment(negate ? Negate((lo, hi)) : (lo, hi));
            }

            // ── State + mask helpers ─────────────────────────────────────────────
            private int NewState()
            {
                _eps.Add([]);
                _moves.Add([]);
                return _eps.Count - 1;
            }

            private Fragment CharFragment((ulong Lo, ulong Hi) mask)
            {
                var s = NewState();
                var e = NewState();
                _moves[s].Add((mask.Lo, mask.Hi, e));
                return new Fragment(s, e);
            }

            // Clones the sub-NFA reachable from `frag.Start` up to `frag.End`, returning a fresh fragment.
            private Fragment Clone(Fragment frag)
            {
                var map = new Dictionary<int, int>();
                var stack = new Stack<int>();
                stack.Push(frag.Start);
                map[frag.Start] = NewState();
                while (stack.Count > 0)
                {
                    var old = stack.Pop();
                    foreach (var t in _eps[old])
                    {
                        if (!map.TryGetValue(t, out _)) { map[t] = NewState(); stack.Push(t); }
                    }
                    foreach (var (_, _, t) in _moves[old])
                    {
                        if (!map.TryGetValue(t, out _)) { map[t] = NewState(); stack.Push(t); }
                    }
                }
                foreach (var (old, fresh) in map)
                {
                    foreach (var t in _eps[old]) { _eps[fresh].Add(map[t]); }
                    foreach (var (l, h, t) in _moves[old]) { _moves[fresh].Add((l, h, map[t])); }
                }
                return new Fragment(map[frag.Start], map[frag.End]);
            }

            private char Peek(int ahead = 0)
                => _pos + ahead < _pattern.Length ? _pattern[_pos + ahead] : '\0';

            private char Next() => _pattern[_pos++];

            private int ParseInt()
            {
                var start = _pos;
                while (Peek() is >= '0' and <= '9') { _pos++; }
                if (_pos == start) { throw new ArgumentException("Expected a number in '{...}' quantifier.", nameof(start)); }
                return int.Parse(_pattern.AsSpan(start, _pos - start));
            }

            private static (ulong, ulong) SingleMask(char c)
            {
                ulong lo = 0, hi = 0;
                SetBit(ref lo, ref hi, c);
                return (lo, hi);
            }

            private static (ulong, ulong) AnyMask()
            {
                ulong lo = ~0UL, hi = ~0UL;
                ClearBit(ref lo, ref hi, '\n');   // '.' excludes newline, as usual
                return (lo, hi);
            }

            private static (ulong, ulong) DigitMask()
            {
                ulong lo = 0, hi = 0;
                SetRange(ref lo, ref hi, '0', '9');
                return (lo, hi);
            }

            private static (ulong, ulong) WordMask()
            {
                ulong lo = 0, hi = 0;
                SetRange(ref lo, ref hi, 'A', 'Z');
                SetRange(ref lo, ref hi, 'a', 'z');
                SetRange(ref lo, ref hi, '0', '9');
                SetBit(ref lo, ref hi, '_');
                return (lo, hi);
            }

            private static (ulong, ulong) SpaceMask()
            {
                ulong lo = 0, hi = 0;
                foreach (var c in " \t\n\r\f\v") { SetBit(ref lo, ref hi, c); }
                return (lo, hi);
            }

            private static (ulong, ulong) Negate((ulong Lo, ulong Hi) m)
            {
                // Complement within the printable+control ASCII alphabet (0–127).
                return (~m.Lo, ~m.Hi);
            }

            private static void SetRange(ref ulong lo, ref ulong hi, char from, char to)
            {
                for (var c = from; c <= to; c++) { SetBit(ref lo, ref hi, c); }
            }

            private static void SetBit(ref ulong lo, ref ulong hi, char c)
            {
                if (c < 64) { lo |= 1UL << c; }
                else if (c < 128) { hi |= 1UL << (c - 64); }
            }

            private static void ClearBit(ref ulong lo, ref ulong hi, char c)
            {
                if (c < 64) { lo &= ~(1UL << c); }
                else if (c < 128) { hi &= ~(1UL << (c - 64)); }
            }
        }

        private readonly record struct Fragment(int Start, int End);
    }
}
