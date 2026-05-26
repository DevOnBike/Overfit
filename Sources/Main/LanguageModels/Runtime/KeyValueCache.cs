// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Single-sequence FP32 key-value cache for autoregressive SLM decoding.
    ///
    /// Layout per cache:
    ///
    /// [layer, head, position, headDim]
    ///
    /// flattened as:
    ///
    /// (((layer * headCount) + head) * maxSequenceLength + position) * headDimension
    ///
    /// This implementation is intentionally simple:
    ///
    /// - batch size = 1,
    /// - FP32 only,
    /// - one contiguous key buffer,
    /// - one contiguous value buffer,
    /// - caller writes K/V for each generated position,
    /// - caller advances CurrentLength explicitly.
    ///
    /// It is a runtime building block. It does not compute attention by itself.
    /// </summary>
    public sealed class KeyValueCache : IKeyValueCache
    {
        private float[] _keys;
        private float[] _values;
        private bool _disposed;

        public KeyValueCache(KeyValueCacheShape shape)
        {
            ValidateShape(shape);

            if (shape.ElementsPerCache > int.MaxValue)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(shape),
                    $"Key/value cache is too large for a single managed array: {shape.ElementsPerCache} elements per cache.");
            }

            Shape = shape;
            MaxLength = shape.MaxSequenceLength;

            _keys = new float[(int)shape.ElementsPerCache];
            _values = new float[(int)shape.ElementsPerCache];
        }

        public KeyValueCacheShape Shape { get; }

        public int CurrentLength { get; private set; }

        public int MaxLength { get; }

        public bool IsFull => CurrentLength >= MaxLength;

        /// <summary>
        /// Number of tokens evicted from the front so far (sliding window). The live
        /// slot <c>s</c> holds the token whose absolute sequence position is
        /// <c>BasePosition + s</c>. Used by RoPE to rotate at the true absolute
        /// position even after the physical slots have shifted down (cached K stay
        /// rotated at their original positions, so attention scores — which depend
        /// only on the relative offset — remain correct). 0 until the first
        /// <see cref="Evict"/>.
        /// </summary>
        public int BasePosition { get; private set; }

        /// <summary>
        /// Creates a KV cache.
        /// For GQA: pass the KV head count (smaller than Q head count).
        /// For MHA: pass the full head count.
        /// </summary>
        public static KeyValueCache Create(
            int layerCount,
            int kvHeadCount,
            int maxSequenceLength,
            int headDimension)
        {
            return new KeyValueCache(
                new KeyValueCacheShape(
                    layerCount,
                    kvHeadCount,
                    maxSequenceLength,
                    headDimension));
        }

        public void Reset()
        {
            ThrowIfDisposed();

            CurrentLength = 0;
            BasePosition = 0;

            // Clear for deterministic tests and to avoid exposing stale values
            // if a caller accidentally reads after reset. A future performance
            // path may expose Reset(clear: false) if profiling shows this matters.
            Array.Clear(_keys);
            Array.Clear(_values);
        }

        /// <summary>
        /// Sliding-window eviction: drops the oldest <paramref name="count"/> tokens by
        /// shifting every (layer, head) K/V block down by <paramref name="count"/> slots,
        /// so the live window stays contiguous in slots <c>[0, CurrentLength)</c> and the
        /// existing contiguous read path is unchanged. <see cref="CurrentLength"/> shrinks
        /// and <see cref="BasePosition"/> grows by <paramref name="count"/>. Retained K/V
        /// are NOT re-rotated — RoPE scores depend only on the relative offset, which the
        /// climbing <see cref="BasePosition"/> preserves. Only valid for RoPE models
        /// (learned absolute-position models can't slide without re-embedding).
        /// </summary>
        public void Evict(int count)
        {
            ThrowIfDisposed();

            if (count <= 0 || count > CurrentLength)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(count),
                    $"Evict count {count} must be in [1, CurrentLength={CurrentLength}].");
            }

            var keep = CurrentLength - count;
            var hd = Shape.HeadDimension;
            var shift = count * hd;
            var keepElems = keep * hd;

            if (keepElems > 0)
            {
                for (var layer = 0; layer < Shape.LayerCount; layer++)
                {
                    for (var head = 0; head < Shape.KvHeadCount; head++)
                    {
                        var baseOffset = GetOffset(layer, head, 0);
                        Array.Copy(_keys, baseOffset + shift, _keys, baseOffset, keepElems);
                        Array.Copy(_values, baseOffset + shift, _values, baseOffset, keepElems);
                    }
                }
            }

            CurrentLength = keep;
            BasePosition += count;
        }

        public void Advance(int tokenCount = 1)
        {
            ThrowIfDisposed();

            ArgumentOutOfRangeException.ThrowIfNegative(tokenCount);

            if (CurrentLength + tokenCount > MaxLength)
            {
                throw new InvalidOperationException(
                    $"Cannot advance KV cache by {tokenCount} tokens. CurrentLength={CurrentLength}, MaxLength={MaxLength}.");
            }

            CurrentLength += tokenCount;
        }

        /// <summary>
        /// Rolls <see cref="CurrentLength"/> back to <paramref name="length"/> — used by speculative
        /// decoding to discard rejected draft tokens after a batched verify wrote their K/V. The stale
        /// K/V left in slots <c>[length, oldLength)</c> are NOT cleared: reads only ever cover
        /// <c>[0, CurrentLength)</c>, and the next decode overwrites slot <paramref name="length"/>
        /// before it is read. Cannot grow the cache (use <see cref="Advance"/>).
        /// </summary>
        public void TruncateTo(int length)
        {
            ThrowIfDisposed();

            if (length < 0 || length > CurrentLength)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(length), $"TruncateTo {length} must be in [0, CurrentLength={CurrentLength}].");
            }

            CurrentLength = length;
        }

        public Span<float> GetKeyWriteSpan(
            int layerIndex,
            int headIndex,
            int position)
        {
            ThrowIfDisposed();
            ValidateLayerHeadPosition(layerIndex, headIndex, position);

            return _keys.AsSpan(
                GetOffset(layerIndex, headIndex, position),
                Shape.HeadDimension);
        }

        public Span<float> GetValueWriteSpan(
            int layerIndex,
            int headIndex,
            int position)
        {
            ThrowIfDisposed();
            ValidateLayerHeadPosition(layerIndex, headIndex, position);

            return _values.AsSpan(
                GetOffset(layerIndex, headIndex, position),
                Shape.HeadDimension);
        }

        public ReadOnlySpan<float> GetKeyReadSpan(
            int layerIndex,
            int headIndex,
            int fromPosition,
            int length)
        {
            ThrowIfDisposed();
            ValidateLayerHeadRange(layerIndex, headIndex, fromPosition, length);

            return _keys.AsSpan(
                GetOffset(layerIndex, headIndex, fromPosition),
                length * Shape.HeadDimension);
        }

        public ReadOnlySpan<float> GetValueReadSpan(
            int layerIndex,
            int headIndex,
            int fromPosition,
            int length)
        {
            ThrowIfDisposed();
            ValidateLayerHeadRange(layerIndex, headIndex, fromPosition, length);

            return _values.AsSpan(
                GetOffset(layerIndex, headIndex, fromPosition),
                length * Shape.HeadDimension);
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            CurrentLength = 0;
            _keys = [];
            _values = [];
        }

        private static void ValidateShape(KeyValueCacheShape shape)
        {
            if (shape.LayerCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(shape), "LayerCount must be positive.");
            }

            if (shape.KvHeadCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(shape), "KvHeadCount must be positive.");
            }

            if (shape.MaxSequenceLength <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(shape), "MaxSequenceLength must be positive.");
            }

            if (shape.HeadDimension <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(shape), "HeadDimension must be positive.");
            }
        }

        private void ValidateLayerHeadPosition(
            int layerIndex,
            int headIndex,
            int position)
        {
            if ((uint)layerIndex >= (uint)Shape.LayerCount)
            {
                throw new ArgumentOutOfRangeException(nameof(layerIndex));
            }

            if ((uint)headIndex >= (uint)Shape.KvHeadCount)
            {
                throw new ArgumentOutOfRangeException(nameof(headIndex));
            }

            if ((uint)position >= (uint)Shape.MaxSequenceLength)
            {
                throw new ArgumentOutOfRangeException(nameof(position));
            }
        }

        private void ValidateLayerHeadRange(
            int layerIndex,
            int headIndex,
            int fromPosition,
            int length)
        {
            if ((uint)layerIndex >= (uint)Shape.LayerCount)
            {
                throw new ArgumentOutOfRangeException(nameof(layerIndex));
            }

            if ((uint)headIndex >= (uint)Shape.KvHeadCount)
            {
                throw new ArgumentOutOfRangeException(nameof(headIndex));
            }

            ArgumentOutOfRangeException.ThrowIfNegative(fromPosition);

            ArgumentOutOfRangeException.ThrowIfNegative(length);

            if (fromPosition + length > CurrentLength)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(length),
                    $"Requested range [{fromPosition}, {fromPosition + length}) exceeds CurrentLength={CurrentLength}.");
            }

            if (fromPosition + length > Shape.MaxSequenceLength)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(length),
                    $"Requested range [{fromPosition}, {fromPosition + length}) exceeds MaxSequenceLength={Shape.MaxSequenceLength}.");
            }
        }

        private int GetOffset(
            int layerIndex,
            int headIndex,
            int position)
        {
            return (((layerIndex * Shape.KvHeadCount) + headIndex) *
                    Shape.MaxSequenceLength +
                    position) *
                   Shape.HeadDimension;
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(KeyValueCache));
            }
        }
    }
}
