// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Single-sequence key-value cache for autoregressive SLM decoding.
    ///
    /// Layout per cache:
    ///
    /// [layer, head, position, headDim]
    ///
    /// flattened as:
    ///
    /// (((layer * headCount) + head) * maxSequenceLength + position) * headDimension
    ///
    /// Two element types (<see cref="KvCacheDType"/>): the default <see cref="KvCacheDType.F32"/> keeps K/V
    /// in full precision; <see cref="KvCacheDType.Q8"/> stores each cached K/V vector as <c>headDim</c>
    /// symmetric int8 plus one F32 scale (<see cref="Q8KvQuant"/>) — ~4× less KV RAM and attention read
    /// traffic for long context. The lifecycle (Reset / Advance / Evict / TruncateTo) is identical for both;
    /// only the storage and the read/write surface differ:
    ///
    /// - F32: write via <see cref="GetKeyWriteSpan"/> / read via <see cref="GetKeyReadSpan"/>.
    /// - Q8: write via <see cref="WriteKey"/> (quantizes), read int8 via <see cref="GetKeyQuants"/> +
    ///   <see cref="GetKeyScales"/> (decode), or dequantize a range to F32 via
    ///   <see cref="DequantizeKeyRange"/> (batched prefill / verify, which reuses the F32 attention kernel).
    ///   <see cref="WriteKey"/> / <see cref="WriteValue"/> and <see cref="DequantizeKeyRange"/> /
    ///   <see cref="DequantizeValueRange"/> work in BOTH modes (F32 = plain copy), so call sites stay uniform.
    ///
    /// batch size = 1, one contiguous key buffer, one contiguous value buffer, caller writes K/V per
    /// generated position and advances <see cref="CurrentLength"/> explicitly. It does not compute attention.
    /// </summary>
    public sealed class KeyValueCache : IKeyValueCache
    {
        // F32 storage (allocated only for KvCacheDType.F32).
        private float[] _keys;
        private float[] _values;

        // Q8 storage (allocated only for KvCacheDType.Q8): per-vector symmetric int8 + one F32 scale/vector.
        private sbyte[] _keysQ;
        private sbyte[] _valuesQ;
        private float[] _keyScales;
        private float[] _valueScales;

        private bool _disposed;

        public KeyValueCache(KeyValueCacheShape shape)
            : this(shape, KvCacheDType.F32)
        {
        }

        public KeyValueCache(KeyValueCacheShape shape, KvCacheDType dtype)
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
            Dtype = dtype;

            var elems = (int)shape.ElementsPerCache;
            if (dtype == KvCacheDType.Q8)
            {
                var vectors = shape.LayerCount * shape.KvHeadCount * shape.MaxSequenceLength;
                _keysQ = new sbyte[elems];
                _valuesQ = new sbyte[elems];
                _keyScales = new float[vectors];
                _valueScales = new float[vectors];
                _keys = [];
                _values = [];
            }
            else
            {
                _keys = new float[elems];
                _values = new float[elems];
                _keysQ = [];
                _valuesQ = [];
                _keyScales = [];
                _valueScales = [];
            }
        }

        public KeyValueCacheShape Shape
        {
            get;
        }

        public KvCacheDType Dtype
        {
            get;
        }

        /// <summary>True when K/V are stored as int8 (<see cref="KvCacheDType.Q8"/>) — attend via
        /// <see cref="CachedAttentionKernel.ComputeSingleHeadQ8"/> / dequantize for the batched path.</summary>
        public bool IsQuantized => Dtype == KvCacheDType.Q8;

        public int CurrentLength
        {
            get; private set;
        }

        public int MaxLength
        {
            get;
        }

        public bool IsFull => CurrentLength >= MaxLength;

        /// <summary>
        /// Number of tokens evicted from the front so far (sliding window). The live
        /// slot <c>s</c> holds the token whose absolute sequence position is
        /// <c>BasePosition + s</c>. Used by RoPE to rotate at the true absolute
        /// position even after the physical slots have shifted down. 0 until the first
        /// <see cref="Evict"/>.
        /// </summary>
        public int BasePosition
        {
            get; private set;
        }

        /// <summary>
        /// Creates a KV cache. For GQA: pass the KV head count (smaller than Q head count).
        /// For MHA: pass the full head count. <paramref name="dtype"/> selects F32 (default) or Q8 storage.
        /// </summary>
        public static KeyValueCache Create(
            int layerCount,
            int kvHeadCount,
            int maxSequenceLength,
            int headDimension,
            KvCacheDType dtype = KvCacheDType.F32)
        {
            return new KeyValueCache(
                new KeyValueCacheShape(layerCount, kvHeadCount, maxSequenceLength, headDimension),
                dtype);
        }

        public void Reset()
        {
            ThrowIfDisposed();

            CurrentLength = 0;
            BasePosition = 0;

            // Clear for deterministic tests and to avoid exposing stale values if a caller accidentally
            // reads after reset. A future path may expose Reset(clear: false) if profiling shows it matters.
            if (IsQuantized)
            {
                Array.Clear(_keysQ);
                Array.Clear(_valuesQ);
                Array.Clear(_keyScales);
                Array.Clear(_valueScales);
            }
            else
            {
                Array.Clear(_keys);
                Array.Clear(_values);
            }
        }

        /// <summary>
        /// Sliding-window eviction: drops the oldest <paramref name="count"/> tokens by shifting every
        /// (layer, head) K/V block down by <paramref name="count"/> slots, so the live window stays
        /// contiguous in slots <c>[0, CurrentLength)</c> and the existing contiguous read path is unchanged.
        /// <see cref="CurrentLength"/> shrinks and <see cref="BasePosition"/> grows by <paramref name="count"/>.
        /// Retained K/V are NOT re-rotated — RoPE scores depend only on the relative offset, which the
        /// climbing <see cref="BasePosition"/> preserves. Only valid for RoPE models.
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
                        if (IsQuantized)
                        {
                            // Overlapping in-place shift — Span.CopyTo has memmove semantics.
                            _keysQ.AsSpan(baseOffset + shift, keepElems).CopyTo(_keysQ.AsSpan(baseOffset, keepElems));
                            _valuesQ.AsSpan(baseOffset + shift, keepElems).CopyTo(_valuesQ.AsSpan(baseOffset, keepElems));

                            var scaleBase = ScaleOffset(layer, head, 0);
                            _keyScales.AsSpan(scaleBase + count, keep).CopyTo(_keyScales.AsSpan(scaleBase, keep));
                            _valueScales.AsSpan(scaleBase + count, keep).CopyTo(_valueScales.AsSpan(scaleBase, keep));
                        }
                        else
                        {
                            _keys.AsSpan(baseOffset + shift, keepElems).CopyTo(_keys.AsSpan(baseOffset, keepElems));
                            _values.AsSpan(baseOffset + shift, keepElems).CopyTo(_values.AsSpan(baseOffset, keepElems));
                        }
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
                throw new OverfitRuntimeException(
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

        /// <summary>
        /// Captures the live region <c>[0, CurrentLength)</c> across every (layer, head) into a compact
        /// <see cref="KvCacheSnapshot"/> — for prefix / system-prompt KV reuse. F32 only for now.
        /// </summary>
        public KvCacheSnapshot Snapshot()
        {
            ThrowIfDisposed();
            ThrowIfQuantized(nameof(Snapshot));

            var lhCount = Shape.LayerCount * Shape.KvHeadCount;
            var headDim = Shape.HeadDimension;
            var maxSeq = MaxLength;
            var run = CurrentLength * headDim;          // contiguous run per (layer, head)

#pragma warning disable OVERFIT001 // snapshot contract: returns caller-owned arrays (prefix-KV save/restore feature, not a hot path)
            var keys = new float[(long)lhCount * run];
            var values = new float[(long)lhCount * run];
#pragma warning restore OVERFIT001
            for (var lh = 0; lh < lhCount; lh++)
            {
                var src = (long)lh * maxSeq * headDim;
                var dst = (long)lh * run;
                _keys.AsSpan((int)src, run).CopyTo(keys.AsSpan((int)dst, run));
                _values.AsSpan((int)src, run).CopyTo(values.AsSpan((int)dst, run));
            }

            return new KvCacheSnapshot(
                keys, values, CurrentLength, BasePosition,
                Shape.LayerCount, Shape.KvHeadCount, headDim);
        }

        /// <summary>
        /// Restores a <see cref="KvCacheSnapshot"/> (a prefix) into this cache, overwriting positions
        /// <c>[0, snapshot.Length)</c> and setting <see cref="CurrentLength"/> / <see cref="BasePosition"/>
        /// accordingly. F32 only for now.
        /// </summary>
        public void RestoreFrom(KvCacheSnapshot snapshot)
        {
            ThrowIfDisposed();
            ThrowIfQuantized(nameof(RestoreFrom));
            ArgumentNullException.ThrowIfNull(snapshot);

            if (!snapshot.MatchesShape(Shape))
            {
                throw new ArgumentException("Snapshot shape does not match this cache (layers / KV heads / head dim).", nameof(snapshot));
            }
            if (snapshot.Length > MaxLength)
            {
                throw new ArgumentException($"Snapshot length {snapshot.Length} exceeds cache MaxLength {MaxLength}.", nameof(snapshot));
            }

            var lhCount = Shape.LayerCount * Shape.KvHeadCount;
            var headDim = Shape.HeadDimension;
            var maxSeq = MaxLength;
            var run = snapshot.Length * headDim;
            for (var lh = 0; lh < lhCount; lh++)
            {
                var src = (long)lh * run;
                var dst = (long)lh * maxSeq * headDim;
                snapshot.Keys.AsSpan((int)src, run).CopyTo(_keys.AsSpan((int)dst, run));
                snapshot.Values.AsSpan((int)src, run).CopyTo(_values.AsSpan((int)dst, run));
            }

            CurrentLength = snapshot.Length;
            BasePosition = snapshot.BasePosition;
        }

        // ── F32 write/read surface (throws in Q8 mode) ──

        public Span<float> GetKeyWriteSpan(int layerIndex, int headIndex, int position)
        {
            ThrowIfDisposed();
            ThrowIfQuantized(nameof(GetKeyWriteSpan));
            ValidateLayerHeadPosition(layerIndex, headIndex, position);

            return _keys.AsSpan(GetOffset(layerIndex, headIndex, position), Shape.HeadDimension);
        }

        public Span<float> GetValueWriteSpan(int layerIndex, int headIndex, int position)
        {
            ThrowIfDisposed();
            ThrowIfQuantized(nameof(GetValueWriteSpan));
            ValidateLayerHeadPosition(layerIndex, headIndex, position);

            return _values.AsSpan(GetOffset(layerIndex, headIndex, position), Shape.HeadDimension);
        }

        public ReadOnlySpan<float> GetKeyReadSpan(int layerIndex, int headIndex, int fromPosition, int length)
        {
            ThrowIfDisposed();
            ThrowIfQuantized(nameof(GetKeyReadSpan));
            ValidateLayerHeadRange(layerIndex, headIndex, fromPosition, length);

            return _keys.AsSpan(GetOffset(layerIndex, headIndex, fromPosition), length * Shape.HeadDimension);
        }

        public ReadOnlySpan<float> GetValueReadSpan(int layerIndex, int headIndex, int fromPosition, int length)
        {
            ThrowIfDisposed();
            ThrowIfQuantized(nameof(GetValueReadSpan));
            ValidateLayerHeadRange(layerIndex, headIndex, fromPosition, length);

            return _values.AsSpan(GetOffset(layerIndex, headIndex, fromPosition), length * Shape.HeadDimension);
        }

        // ── Mode-agnostic write: F32 copies, Q8 quantizes (so call sites are uniform) ──

        /// <summary>Writes one F32 key vector into slot (layer, head, position) — copies in F32 mode,
        /// quantizes to int8 + scale in Q8 mode.</summary>
        public void WriteKey(int layerIndex, int headIndex, int position, ReadOnlySpan<float> keyVector)
        {
            ThrowIfDisposed();
            ValidateLayerHeadPosition(layerIndex, headIndex, position);
            WriteVector(_keys, _keysQ, _keyScales, layerIndex, headIndex, position, keyVector);
        }

        /// <summary>Writes one F32 value vector into slot (layer, head, position) — copies in F32 mode,
        /// quantizes to int8 + scale in Q8 mode.</summary>
        public void WriteValue(int layerIndex, int headIndex, int position, ReadOnlySpan<float> valueVector)
        {
            ThrowIfDisposed();
            ValidateLayerHeadPosition(layerIndex, headIndex, position);
            WriteVector(_values, _valuesQ, _valueScales, layerIndex, headIndex, position, valueVector);
        }

        // ── Q8 int8 read surface (decode attend; throws in F32 mode) ──

        public ReadOnlySpan<sbyte> GetKeyQuants(int layerIndex, int headIndex, int fromPosition, int length)
        {
            ThrowIfDisposed();
            ThrowIfNotQuantized(nameof(GetKeyQuants));
            ValidateLayerHeadRange(layerIndex, headIndex, fromPosition, length);
            return _keysQ.AsSpan(GetOffset(layerIndex, headIndex, fromPosition), length * Shape.HeadDimension);
        }

        public ReadOnlySpan<sbyte> GetValueQuants(int layerIndex, int headIndex, int fromPosition, int length)
        {
            ThrowIfDisposed();
            ThrowIfNotQuantized(nameof(GetValueQuants));
            ValidateLayerHeadRange(layerIndex, headIndex, fromPosition, length);
            return _valuesQ.AsSpan(GetOffset(layerIndex, headIndex, fromPosition), length * Shape.HeadDimension);
        }

        public ReadOnlySpan<float> GetKeyScales(int layerIndex, int headIndex, int fromPosition, int length)
        {
            ThrowIfDisposed();
            ThrowIfNotQuantized(nameof(GetKeyScales));
            ValidateLayerHeadRange(layerIndex, headIndex, fromPosition, length);
            return _keyScales.AsSpan(ScaleOffset(layerIndex, headIndex, fromPosition), length);
        }

        public ReadOnlySpan<float> GetValueScales(int layerIndex, int headIndex, int fromPosition, int length)
        {
            ThrowIfDisposed();
            ThrowIfNotQuantized(nameof(GetValueScales));
            ValidateLayerHeadRange(layerIndex, headIndex, fromPosition, length);
            return _valueScales.AsSpan(ScaleOffset(layerIndex, headIndex, fromPosition), length);
        }

        // ── Mode-agnostic dequantize-to-F32 for the batched prefill / verify path ──

        /// <summary>Dequantizes the key range <c>[fromPosition, fromPosition+length)</c> into
        /// <paramref name="destination"/> (<c>length * headDim</c> F32). In F32 mode it is a plain copy of
        /// the cached span — so the batched attention reads F32 uniformly in both modes.</summary>
        public void DequantizeKeyRange(int layerIndex, int headIndex, int fromPosition, int length, Span<float> destination)
            => DequantizeRange(_keys, _keysQ, _keyScales, layerIndex, headIndex, fromPosition, length, destination);

        /// <summary>Dequantizes the value range into <paramref name="destination"/> — see
        /// <see cref="DequantizeKeyRange"/>.</summary>
        public void DequantizeValueRange(int layerIndex, int headIndex, int fromPosition, int length, Span<float> destination)
            => DequantizeRange(_values, _valuesQ, _valueScales, layerIndex, headIndex, fromPosition, length, destination);

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
            _keysQ = [];
            _valuesQ = [];
            _keyScales = [];
            _valueScales = [];
        }

        private void WriteVector(
            float[] f32,
            sbyte[] q8,
            Span<float> scales,
            int layerIndex,
            int headIndex,
            int position,
            ReadOnlySpan<float> vector)
        {
            var hd = Shape.HeadDimension;
            if (vector.Length < hd)
            {
                throw new ArgumentException($"Vector span ({vector.Length}) is smaller than headDimension ({hd}).", nameof(vector));
            }

            var offset = GetOffset(layerIndex, headIndex, position);
            if (IsQuantized)
            {
                scales[ScaleOffset(layerIndex, headIndex, position)] = Q8KvQuant.Quantize(vector.Slice(0, hd), q8.AsSpan(offset, hd));
            }
            else
            {
                vector.Slice(0, hd).CopyTo(f32.AsSpan(offset, hd));
            }
        }

        private void DequantizeRange(
            float[] f32,
            ReadOnlySpan<sbyte> q8,
            ReadOnlySpan<float> scales,
            int layerIndex,
            int headIndex,
            int fromPosition,
            int length,
            Span<float> destination)
        {
            ThrowIfDisposed();
            ValidateLayerHeadRange(layerIndex, headIndex, fromPosition, length);

            var hd = Shape.HeadDimension;
            var elems = length * hd;
            if (destination.Length < elems)
            {
                throw new ArgumentException($"Destination span ({destination.Length}) is smaller than length*headDim ({elems}).", nameof(destination));
            }

            var byteOffset = GetOffset(layerIndex, headIndex, fromPosition);
            if (!IsQuantized)
            {
                f32.AsSpan(byteOffset, elems).CopyTo(destination);
                return;
            }

            var scaleBase = ScaleOffset(layerIndex, headIndex, fromPosition);
            for (var t = 0; t < length; t++)
            {
                var scale = scales[scaleBase + t];
                var src = byteOffset + t * hd;
                var dstBase = t * hd;
                for (var d = 0; d < hd; d++)
                {
                    destination[dstBase + d] = scale * q8[src + d];
                }
            }
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

        private int ScaleOffset(int layerIndex, int headIndex, int position)
            => ((layerIndex * Shape.KvHeadCount) + headIndex) * Shape.MaxSequenceLength + position;

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(KeyValueCache));
            }
        }

        private void ThrowIfQuantized(string op)
        {
            if (IsQuantized)
            {
                throw new OverfitRuntimeException($"{op} is F32-only; this cache is Q8. Use the Q8 read surface (GetKeyQuants/GetKeyScales) or DequantizeKeyRange.");
            }
        }

        private void ThrowIfNotQuantized(string op)
        {
            if (!IsQuantized)
            {
                throw new OverfitRuntimeException($"{op} is Q8-only; this cache is F32. Use GetKeyReadSpan / GetKeyWriteSpan.");
            }
        }
    }
}
