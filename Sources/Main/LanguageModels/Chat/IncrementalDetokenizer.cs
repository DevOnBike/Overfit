// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Exceptions;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.LanguageModels.Chat
{
    /// <summary>
    /// Turns a growing token run into the text that has newly stabilised, one step per generated token.
    ///
    /// <para><b>Why the whole run is decoded every time, and not just the new token.</b> Byte-level BPE
    /// splits a codepoint across tokens, so decoding the newest id alone yields a replacement character
    /// where the original had a letter. Worse, a later token can change how an earlier one renders. The
    /// only safe reading is to decode everything and emit the part that has stopped changing — which is
    /// what this does, and why the cost is quadratic in the reply length by construction.</para>
    ///
    /// <para><b>What changed on 2026-08-11.</b> The quadratic term used to be quadratic in ALLOCATIONS as
    /// well: this lived inside a closure in <c>ChatSession.Generate</c> and called
    /// <c>ITokenizer.DecodeToString</c>, so every token allocated a string holding the entire reply so far,
    /// plus whatever the tokenizer allocated underneath. Two rolling pooled buffers replace it. The work is
    /// unchanged — only a different algorithm removes that, and the hazard above is why nobody has.</para>
    ///
    /// <para><b>The delta is still a string</b>, because <c>StopSequenceDetector.Append</c> and the
    /// <c>onText</c> callback both take one, and those are public contracts. That allocation is linear in
    /// the reply, not quadratic, so it was never the problem.</para>
    ///
    /// <para>Extracted from the closure so it can be tested at all: the partial-codepoint rule is the
    /// subtlest thing in the streaming path and it had no test of its own.</para>
    /// </summary>
    internal sealed class IncrementalDetokenizer : IDisposable
    {
        private PooledBuffer<char> _current;
        private PooledBuffer<char> _previous;
        private int _previousLength;
        private bool _disposed;

        /// <param name="initialCapacity">Characters. Grows on demand; this only avoids the first few grows.</param>
        public IncrementalDetokenizer(int initialCapacity = 1024)
        {
            _current = new PooledBuffer<char>(initialCapacity, clearMemory: false);
            _previous = new PooledBuffer<char>(initialCapacity, clearMemory: false);
        }

        /// <summary>Characters emitted so far, i.e. the length of the stabilised prefix.</summary>
        public int StableLength => _previousLength;

        /// <summary>
        /// Decodes <paramref name="tokens"/> and reports the text that appeared since the last call.
        ///
        /// <para>Returns <see langword="false"/> — with an empty <paramref name="delta"/> — when the decode
        /// did not grow, or when it no longer starts with what was already emitted. The second case is the
        /// one worth naming: a byte-level tokenizer can RE-RENDER earlier text once a following byte
        /// arrives, and emitting a delta against a prefix that has changed underneath would send the reader
        /// text the model never produced. Holding back until it agrees again is correct and is what the
        /// previous implementation did.</para>
        /// </summary>
        public bool TryAdvance(ITokenizer tokenizer, ReadOnlySpan<int> tokens, out ReadOnlySpan<char> delta)
        {
            ArgumentNullException.ThrowIfNull(tokenizer);
            ObjectDisposedException.ThrowIf(_disposed, this);

            var written = Decode(tokenizer, tokens);
            var text = _current.Span.Slice(0, written);

            if (written <= _previousLength
                || !text.Slice(0, _previousLength).SequenceEqual(_previous.Span.Slice(0, _previousLength)))
            {
                delta = default;

                return false;
            }

            delta = text.Slice(_previousLength);

            // Swap rather than copy: the buffer just decoded into becomes the reference for the next token,
            // and the old reference becomes the scratch. One assignment instead of copying the whole reply
            // on every token, which would have put the quadratic term straight back.
            (_previous, _current) = (_current, _previous);
            _previousLength = written;

            return true;
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            _current.Dispose();
            _previous.Dispose();
        }

        /// <summary>
        /// Decodes into <see cref="_current"/>, growing it until it fits.
        ///
        /// <para>Tokenizers without a zero-allocation decode go through <c>DecodeToString</c> and a copy —
        /// the same result, one string per token, which is what they cost anyway. Keeping one code path
        /// rather than branching in the caller is deliberate: two paths through the partial-codepoint rule
        /// is exactly how they drift.</para>
        /// </summary>
        private int Decode(ITokenizer tokenizer, ReadOnlySpan<int> tokens)
        {
            if (!tokenizer.SupportsZeroAllocationDecode)
            {
                var text = tokenizer.DecodeToString(tokens);

                EnsureCapacity(text.Length);
                text.AsSpan().CopyTo(_current.Span);

                return text.Length;
            }

            // The tokenizer reports the required length by refusing, so grow and retry rather than ask for
            // a bound it does not expose.
            //
            // BOUND: each attempt doubles the buffer, so `MaxGrowthAttempts` covers 2^24 times the initial
            // capacity — over 16 billion characters from the 1024-char default, which no reply reaches
            // before the context window ends it. The bound exists because an unbounded retry around a
            // failure the tokenizer decides is a hang, not a slow path (OVERFIT023 / NASA rule 2): a
            // tokenizer that refuses for a reason OTHER than size would spin here forever, growing until
            // the machine dies, and that is precisely the shape this rule exists to forbid.
            const int MaxGrowthAttempts = 24;

            for (var attempt = 0; attempt < MaxGrowthAttempts; attempt++)
            {
                try
                {
                    return tokenizer.Decode(tokens, _current.Span);
                }
                catch (ArgumentException)
                {
                    EnsureCapacity(_current.Length * 2);
                }
            }

            // No `tokenizer.GetType().Name` in the message, however useful it would be: GetType is
            // System.Reflection, which is RS0030-banned here because it breaks Native AOT.
            throw new OverfitRuntimeException(
                $"The tokenizer kept refusing a decode destination after {MaxGrowthAttempts} doublings, up "
                + $"to {_current.Length} chars, for {tokens.Length} token(s). It is rejecting the buffer for "
                + "some reason other than its size.");
        }

        private void EnsureCapacity(int required)
        {
            if (_current.Length >= required)
            {
                return;
            }

            var replacement = new PooledBuffer<char>(Math.Max(required, _current.Length * 2), clearMemory: false);
            var previous = _current;

            _current = replacement;
            previous.Dispose();

            // `_previous` is deliberately NOT grown here, and the invariant is worth stating because the
            // first version of this method did grow it "for safety". It cannot be too small: `_previous` is
            // whatever `_current` was at the last successful advance, and `_previousLength` is the length
            // that was written into it then — so `_previous.Length >= _previousLength` always. A mutation
            // deleting that growth on 2026-08-11 left every test green, which is what sent me back to check
            // the reasoning rather than the test; the branch was unreachable and its comment was wrong.

            // `_previous` holds the stabilised prefix and must be able to hold it after the swap, or the
            // next token's comparison would run against a buffer that is too small. Grown together for
            // that reason, not for symmetry.

        }
    }
}
