// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Data.Abstractions;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Data.Normalizers
{
    /// <summary>
    /// Log1p → Z-Score normalizer for metrics with a log-normal distribution.
    ///
    /// Pipeline:
    ///   1. ReLU:    x = max(0, x)              — protects the log from negative values
    ///   2. Log1p:   x = log(1 + x)             — compresses the long tail of the distribution
    ///   3. Z-Score: x = (x - mean) / stdDev    — centers and scales
    ///
    /// Intended for (from our 12 metrics):
    ///   MemoryWorkingSetBytes, LatencyP50/95/99Ms, RequestsPerSecond,
    ///   GcGen2HeapBytes, ThreadPoolQueueLength.
    ///
    /// Usage:
    ///   // Offline — Golden Window:
    ///   var norm = new Log1pNormalizer();
    ///   norm.FitBatch(data);         // can be called multiple times
    ///   norm.Freeze();               // freezes the parameters
    ///   norm.Save(bw);
    ///
    ///   // Online — production:
    ///   var norm = new Log1pNormalizer();
    ///   norm.Load(br);               // loads frozen parameters
    ///   norm.TransformInPlace(data); // transform only, no fit
    /// </summary>
    public sealed class Log1pNormalizer : IFeatureNormalizer
    {
        private readonly ZScoreNormalizer _zscore = new();

        // Frozen parameters — active after Freeze() or Load()
        private float _frozenMean;
        private float _frozenInvStdDev;
        private bool _frozen;

        public bool IsFitted => _zscore.Count > 0 || _frozen;
        public bool IsFrozen => _frozen;
        public float Mean => _frozen ? _frozenMean : _zscore.Mean;

        // Safe — returns 0 when not frozen and not yet fitted
        public float StdDev => _frozen ? _frozenInvStdDev > 0f ? 1f / _frozenInvStdDev : 0f : _zscore.StandardDeviation;

        public long Count => _zscore.Count;

        // ---------------------------------------------------------------------------
        // Fit — tylko offline (Golden Window)
        // ---------------------------------------------------------------------------

        /// <summary>
        /// Batch fitting: ReLU → Log1p → accumulation in ZScoreNormalizer.
        /// Can be called multiple times — state is accumulated via Chan's algorithm.
        /// </summary>
        public void FitBatch(ReadOnlySpan<float> data)
        {
            ThrowIfFrozen();

            if (data.Length == 0)
            {
                return;
            }

            using var buffer = new PooledBuffer<float>(data.Length);

            var tmp = buffer.Span;

            ApplyReLuAndLog1P(data, tmp);

            _zscore.FitBatch(tmp);
        }

        /// <summary>Inkrementalne fitowanie — Welford po transformacji log1p.</summary>
        public void FitIncremental(float value)
        {
            ThrowIfFrozen();

            _zscore.FitIncremental(MathF.Log(1f + MathF.Max(0f, value)));
        }

        /// <summary>
        /// Freezes the Z-Score parameters after fitting on the Golden Window.
        /// After Freeze(), FitBatch/FitIncremental throw an exception.
        /// </summary>
        public void Freeze()
        {
            if (_zscore.Count == 0)
            {
                throw new OverfitRuntimeException("Cannot freeze before fitting. Call FitBatch or FitIncremental first.");
            }

            var std = _zscore.StandardDeviation;

            _frozenMean = _zscore.Mean;
            _frozenInvStdDev = 1f / (std < 1e-8f ? 1e-8f : std);
            _frozen = true;
        }

        // ---------------------------------------------------------------------------
        // Transform — online (produkcja)
        // ---------------------------------------------------------------------------

        /// <summary>
        /// Applies the full pipeline in-place: ReLU → Log1p → Z-Score.
        /// Requires a prior call to Freeze() or Load().
        /// </summary>
        public void TransformInPlace(Span<float> data)
        {
            ThrowIfNotFrozen();

            if (data.Length == 0)
            {
                return;
            }

            ApplyReLuAndLog1P(data, data);

            TensorPrimitives.Subtract(data, _frozenMean, data);
            TensorPrimitives.Multiply(data, _frozenInvStdDev, data);
        }

        // ---------------------------------------------------------------------------
        // Persistence
        // ---------------------------------------------------------------------------

        public void Save(BinaryWriter bw)
        {
            ThrowIfNotFrozen();

            bw.Write(_frozenMean);
            bw.Write(_frozenInvStdDev);
        }

        public void Load(BinaryReader br)
        {
            _frozenMean = br.ReadSingle();
            _frozenInvStdDev = br.ReadSingle();
            _frozen = true;
        }

        public void Reset()
        {
            _zscore.Reset();
            _frozenMean = 0f;
            _frozenInvStdDev = 0f;
            _frozen = false;
        }

        // ---------------------------------------------------------------------------
        // Private
        // ---------------------------------------------------------------------------

        private static void ApplyReLuAndLog1P(ReadOnlySpan<float> src, Span<float> dst)
        {
            // ReLU — SIMD: x = max(0, x)
            TensorPrimitives.Max(src, 0f, dst);

            // Log1p via two SIMD passes — TensorPrimitives has no Log1P for float span in .NET 10
            TensorPrimitives.Add(dst, 1f, dst);   // dst = x + 1
            TensorPrimitives.Log(dst, dst);        // dst = log(x + 1)
        }

        private void ThrowIfFrozen()
        {
            if (_frozen)
            {
                throw new OverfitRuntimeException("Normalizer is frozen. Call Reset() before fitting again.");
            }
        }

        private void ThrowIfNotFrozen()
        {
            if (!_frozen)
            {
                throw new OverfitRuntimeException("Normalizer is not frozen. Call Freeze() after fitting, or Load() to restore.");
            }
        }
    }
}