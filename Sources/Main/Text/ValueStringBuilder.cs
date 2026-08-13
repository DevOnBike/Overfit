// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Text
{
    /// <summary>
    /// Builds text into a caller-supplied stack buffer, growing into pooled memory only when it has to.
    ///
    /// <para><b>Where it came from and what changed.</b> This is the shape of
    /// <c>System.Text.ValueStringBuilder</c>, an <c>internal</c> type shared by source across
    /// <c>dotnet/runtime</c>. It cannot be referenced — there is no package that exposes it — so using it
    /// means rewriting it, and rewriting it here means two deliberate changes: the runtime rents from
    /// <c>ArrayPool&lt;char&gt;.Shared</c>, which <c>BannedSymbols.txt</c> forbids (RS0030, error in Main)
    /// because this project keeps <b>one</b> audit point over the pool, and it copies with
    /// <c>Array.Copy</c>, also banned. Both go through <see cref="PooledBuffer{T}"/> and
    /// <c>Span.CopyTo</c> instead. The precedent is <c>ValueStopwatch</c>: an allocation-free value type
    /// standing in for a BCL type that allocates.</para>
    ///
    /// <para><b>When to use it, and it is narrower than it looks.</b> Only where a <see cref="string"/>
    /// must actually be produced and the building takes several growth steps. Where the result is consumed
    /// as characters, the house pattern — a caller-owned <c>Span&lt;char&gt;</c> destination, as on
    /// <c>ITokenizer.Decode</c> — is strictly better, because this type still allocates the final string
    /// and that one does not. Swapping a span-based path onto this would be a regression dressed as an
    /// optimisation.</para>
    ///
    /// <para><b>CORRECTION 2026-08-13 — this paragraph used to say "small builds can be slower than
    /// <c>StringBuilder</c>", and a measurement refuted it.</b> At 64 characters this type is <b>2.4x
    /// faster</b> than a default <c>StringBuilder</c> and 1.5x faster than a pre-sized one (15.01 ns
    /// against 35.85 / 22.46), and it is ahead at every length measured — 64 to 16384, ratios 0.42 to 0.67
    /// — including six growth steps. Bytes are roughly halved throughout; the span-destination form
    /// allocates <b>zero</b>. Numbers and their caveats: <c>docs/measured-baselines.md</c>.</para>
    ///
    /// <para><b>What the reasoning behind the old claim got right, and where it stops.</b> A stack buffer
    /// cannot be register-allocated and <see cref="ToString"/> does pay a pool return — both true, and both
    /// smaller than what is saved. What the measurement does NOT license is extrapolation: it covers
    /// repeated <c>Append</c> into one build, on this box, at these lengths. A different call shape can
    /// still lose. <b>Treat any use as a hypothesis until measured with <c>MemoryDiagnoser</c> on both
    /// shapes</b> — see <c>docs/performance-discipline.md</c>. The rule survives; only the direction it
    /// predicted did not.</para>
    ///
    /// <example>
    /// <code>
    ///   Span&lt;char&gt; scratch = stackalloc char[256];
    ///   var text = new ValueStringBuilder(scratch);
    ///   text.Append("id=");
    ///   text.Append(id);
    ///   return text.ToString();     // disposes; see the warning on ToString
    /// </code>
    /// </example>
    /// </summary>
    internal ref struct ValueStringBuilder
    {
        private PooledBuffer<char> _pooled;
        private Span<char> _chars;
        private int _position;

        /// <param name="initialBuffer">
        /// Usually a <c>stackalloc</c>. Not owned and never returned to a pool — growth moves off it and
        /// leaves it untouched, so passing a stack span here is safe even if the build outgrows it.
        /// </param>
        public ValueStringBuilder(Span<char> initialBuffer)
        {
            _pooled = default;
            _chars = initialBuffer;
            _position = 0;
        }

        /// <summary>
        /// Starts from a pooled buffer of at least <paramref name="capacity"/> characters, with no stack
        /// involvement at all.
        ///
        /// <para><b>Prefer this over a large <c>stackalloc</c> when the size is known or boundable.</b>
        /// <c>OVERFIT025</c> caps a single <c>stackalloc</c> at 512 B here and <c>OVERFIT026</c> refuses a
        /// variable element count outright, both as build errors — and they are right: the stack cost of a
        /// variable-length reservation cannot be read off the line, and this repository decodes sequences
        /// whose length is set by whatever the model generated. A rental costs nothing once the pool is warm
        /// and cannot overflow.</para>
        /// </summary>
        public ValueStringBuilder(int capacity)
        {
            _pooled = new PooledBuffer<char>(capacity, clearMemory: false);
            _chars = _pooled.Span;
            _position = 0;
        }

        /// <summary>Characters written so far.</summary>
        public readonly int Length => _position;

        /// <summary>
        /// What has been written, without copying. Valid until the next append — a growth reallocates and
        /// the returned span then points at memory that has gone back to the pool, which is the classic way
        /// to read another request's characters. Take the span, use it, do not store it.
        /// </summary>
        public readonly ReadOnlySpan<char> AsSpan() => _chars.Slice(0, _position);

        public void Append(char value)
        {
            if (_position >= _chars.Length)
            {
                Grow(1);
            }

            _chars[_position++] = value;
        }

        public void Append(ReadOnlySpan<char> value)
        {
            if (value.Length > _chars.Length - _position)
            {
                Grow(value.Length);
            }

            value.CopyTo(_chars.Slice(_position));
            
            _position += value.Length;
        }

        public void Append(string? value)
        {
            if (value != null)
            {
                Append(value.AsSpan());
            }
        }

        /// <summary>
        /// Copies out without allocating and without disposing — the counterpart to <see cref="ToString"/>
        /// for callers that own their destination. Returns <see langword="false"/> and writes nothing when
        /// the destination is too small, rather than throwing or writing a truncated prefix: a partially
        /// filled buffer that reports success is the failure this signature exists to make impossible.
        /// </summary>
        /// <remarks>
        /// Checked against Cysharp's ZString on 2026-08-11, since it is the best-known third-party take on
        /// this type. Its two builders disagree with each other: <c>Utf16ValueStringBuilder.TryCopyTo</c>
        /// does not dispose, while its vendored <c>Number/ValueStringBuilder.TryCopyTo</c> disposes on BOTH
        /// the success and the failure branch — so a caller that got <c>false</c> for a too-small
        /// destination has also silently lost its buffer and cannot retry with a bigger one. Not disposing
        /// here is deliberate for exactly that reason, and the retry it enables is what
        /// <c>IncrementalDetokenizer</c> relies on.
        /// </remarks>
        public readonly bool TryCopyTo(Span<char> destination, out int written)
        {
            if (destination.Length < _position)
            {
                written = 0;

                return false;
            }

            _chars.Slice(0, _position).CopyTo(destination);
            written = _position;

            return true;
        }

        /// <summary>
        /// Materialises the string <b>and disposes</b> — after this the builder is <c>default</c> and any
        /// further append throws.
        ///
        /// <para><b>This is a footgun and it is kept on purpose.</b> It is what the runtime's type does, so
        /// anyone who has met that one is not surprised here; the alternative — a ToString that leaves the
        /// pooled buffer outstanding — leaks a rented array on every call that forgets a <c>using</c>, which
        /// is the worse of the two and silent. <c>ValueStringBuilderTests</c> pins the behaviour so it
        /// cannot drift into the other shape by accident.</para>
        /// </summary>
        public override string ToString()
        {
            var result = _chars.Slice(0, _position).ToString();

            Dispose();

            return result;
        }

        /// <summary>
        /// Returns the pooled buffer, if one was ever rented, and blanks the struct.
        ///
        /// <para><c>this = default</c> rather than merely nulling a field: it makes use-after-dispose fail
        /// immediately on an empty span instead of writing into an array somebody else is now renting.</para>
        /// </summary>
        public void Dispose()
        {
            var pooled = _pooled;

            this = default;

            pooled.Dispose();
        }

        /// <summary>
        /// Moves to a larger pooled buffer. Doubling, or exactly what was asked for if that is larger, so a
        /// single huge append does not need a run of doublings to accommodate it.
        /// </summary>
        private void Grow(int additional)
        {
            // The old buffer must be released only AFTER its contents have been copied out. Disposing first
            // and copying from a returned array is the defect this ordering exists to prevent, and it is
            // invisible in a single-threaded test: the array is still readable, just no longer owned.
            var required = _position + additional;
            var doubled = _chars.Length * 2;
            var capacity = required > doubled ? required : doubled;

            var replacement = new PooledBuffer<char>(capacity, clearMemory: false);

            _chars.Slice(0, _position).CopyTo(replacement.Span);

            var previous = _pooled;

            _pooled = replacement;
            _chars = replacement.Span;

            previous.Dispose();
        }
    }
}
