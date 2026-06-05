// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Loading
{
    /// <summary>
    /// A forward-only read stream that serves bytes from a lazily produced sequence of
    /// <c>byte[]</c> chunks. Used to feed a consumer that wants a <see cref="Stream"/> /
    /// <see cref="BinaryReader"/> (e.g. <c>GPT1Model.Load</c>) without first
    /// materialising the whole byte sequence in memory — only the current chunk is
    /// alive, so peak RAM stays at <c>model + one chunk</c> instead of
    /// <c>model + full serialized copy</c>.
    ///
    /// <para>Limited seek support:</para> the stream reports <see cref="CanSeek"/> = true
    /// and allows seeking <b>backward within the current chunk only</b> — enough for the
    /// 4-byte peek-then-rewind that <c>MultiHeadAttentionLayer.Load</c> uses to detect
    /// legacy checkpoints (the peek always lands at a chunk start and never spans a
    /// chunk, since every emitted parameter block is ≥ 8 bytes). Any other seek throws
    /// <see cref="OverfitRuntimeException"/> — past chunks are gone and future chunks are
    /// not yet produced.
    /// </summary>
    internal sealed class SequentialChunkReadStream : Stream
    {
        private readonly IEnumerator<byte[]> _chunks;
        private byte[] _current = [];
        private int _offset;
        private long _chunkStart;   // absolute position at which _current begins
        private long _position;
        private bool _exhausted;

        public SequentialChunkReadStream(IEnumerable<byte[]> chunks)
        {
            if (chunks is null) { throw new ArgumentNullException(nameof(chunks)); }
            _chunks = chunks.GetEnumerator();
        }

        public override bool CanRead => true;
        public override bool CanSeek => true;
        public override bool CanWrite => false;
        public override long Length => throw new OverfitRuntimeException();

        public override long Position
        {
            get => _position;
            set => Seek(value, SeekOrigin.Begin);
        }

        public override int Read(byte[] buffer, int offset, int count)
        {
            return Read(buffer.AsSpan(offset, count));
        }

        public override int Read(Span<byte> buffer)
        {
            var total = 0;
            while (total < buffer.Length)
            {
                if (_offset >= _current.Length)
                {
                    if (!Advance()) { break; }
                    continue;
                }

                var available = _current.Length - _offset;
                var want = buffer.Length - total;
                var take = available < want ? available : want;

                _current.AsSpan(_offset, take).CopyTo(buffer.Slice(total, take));
                _offset += take;
                _position += take;
                total += take;
            }

            return total;
        }

        // Pulls the next non-empty chunk; returns false when the producer is done.
        private bool Advance()
        {
            while (!_exhausted)
            {
                if (!_chunks.MoveNext())
                {
                    _exhausted = true;
                    _current = [];
                    return false;
                }
                _chunkStart = _position;
                _current = _chunks.Current ?? [];
                _offset = 0;
                if (_current.Length > 0) { return true; }
            }
            return false;
        }

        public override long Seek(long offset, SeekOrigin origin)
        {
            var target = origin switch
            {
                SeekOrigin.Begin => offset,
                SeekOrigin.Current => _position + offset,
                SeekOrigin.End => throw new OverfitRuntimeException("Cannot seek from the end of a streamed source."),
                _ => throw new ArgumentOutOfRangeException(nameof(origin)),
            };

            // Only backward (or no-op) seeks within the chunk already in hand are
            // supported — the peek-then-rewind pattern. Anything else would touch a
            // discarded or not-yet-produced chunk.
            if (target < _chunkStart || target > _chunkStart + _current.Length)
            {
                throw new OverfitRuntimeException(
                    $"Seek to {target} is outside the current chunk [{_chunkStart}, {_chunkStart + _current.Length}].");
            }

            _offset = (int)(target - _chunkStart);
            _position = target;
            return _position;
        }

        public override void Flush() { }
        public override void SetLength(long value) => throw new OverfitRuntimeException();
        public override void Write(byte[] buffer, int offset, int count) => throw new OverfitRuntimeException();

        protected override void Dispose(bool disposing)
        {
            if (disposing) { _chunks.Dispose(); }
            base.Dispose(disposing);
        }
    }
}
