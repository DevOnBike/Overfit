// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers;
using System.Numerics.Tensors;

namespace DevOnBike.Overfit.Statistical
{
    /// <summary>
    /// Normalizator Log1p → Z-Score dla metryk z rozkładem log-normalnym.
    ///
    /// Pipeline:
    ///   1. ReLU:    x = max(0, x)              — chroni log przed wartościami ujemnymi
    ///   2. Log1p:   x = log(1 + x)             — spłaszcza długi ogon rozkładu
    ///   3. Z-Score: x = (x - mean) / stdDev    — centruje i skaluje
    ///
    /// Przeznaczenie (z naszych 12 metryk):
    ///   MemoryWorkingSetBytes, LatencyP50/95/99Ms, RequestsPerSecond,
    ///   GcGen2HeapBytes, ThreadPoolQueueLength.
    ///
    /// Użycie:
    ///   // Offline — Golden Window:
    ///   var norm = new Log1pNormalizer();
    ///   norm.FitBatch(data);         // można wołać wielokrotnie
    ///   norm.Freeze();               // zamraża parametry
    ///   norm.Save(bw);
    ///
    ///   // Online — produkcja:
    ///   var norm = new Log1pNormalizer();
    ///   norm.Load(br);               // wczytuje zamrożone parametry
    ///   norm.TransformInPlace(data); // tylko transform, bez fit
    /// </summary>
    public sealed class Log1pNormalizer
    {
        private readonly ZScoreNormalizer _zscore = new();

        // Zamrożone parametry — aktywne po Freeze() lub Load()
        private float _frozenMean;
        private float _frozenInvStdDev;
        private bool _frozen;

        public bool IsFitted => _zscore.Count > 0 || _frozen;
        public bool IsFrozen => _frozen;
        public float Mean => _frozen ? _frozenMean : _zscore.Mean;
        public float StdDev => _frozen ? 1f / _frozenInvStdDev : _zscore.StandardDeviation;
        public long Count => _zscore.Count;

        // ---------------------------------------------------------------------------
        // Fit — tylko offline (Golden Window)
        // ---------------------------------------------------------------------------

        /// <summary>
        /// Wsadowe fitowanie: ReLU → Log1p → akumulacja w ZScoreNormalizer.
        /// Można wołać wielokrotnie — stan akumulowany przez algorytm Chana.
        /// </summary>
        public void FitBatch(ReadOnlySpan<float> data)
        {
            ThrowIfFrozen();
            
            if (data.Length == 0)
            {
                return;
            }

            var buffer = ArrayPool<float>.Shared.Rent(data.Length);

            try
            {
                var tmp = buffer.AsSpan(0, data.Length);
                ApplyReLuAndLog1P(data, tmp);
                _zscore.FitBatch(tmp);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(buffer);
            }
        }

        /// <summary>Inkrementalne fitowanie — Welford po transformacji log1p.</summary>
        public void FitIncremental(float value)
        {
            ThrowIfFrozen();
            _zscore.FitIncremental(MathF.Log(1f + MathF.Max(0f, value)));
        }

        /// <summary>
        /// Zamraża parametry Z-Score po zakończeniu fitu na Golden Window.
        /// Po Freeze() FitBatch/FitIncremental rzucają wyjątek.
        /// TransformInPlace używa zamrożonych parametrów.
        /// </summary>
        public void Freeze()
        {
            if (_zscore.Count == 0)
            {
                throw new InvalidOperationException(
                    "Cannot freeze before fitting. Call FitBatch or FitIncremental first.");
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
        /// Aplikuje pełny pipeline in-place: ReLU → Log1p → Z-Score.
        /// Wymaga wcześniejszego Freeze() lub Load().
        /// </summary>
        public void TransformInPlace(Span<float> data)
        {
            ThrowIfNotFrozen();
            
            if (data.Length == 0)
            {
                return;
            }

            ApplyReLuAndLog1P(data, data);

            // Z-Score z zamrożonymi parametrami — bez alokacji
            TensorPrimitives.Subtract(data, _frozenMean, data);
            TensorPrimitives.Multiply(data, _frozenInvStdDev, data);
        }

        // ---------------------------------------------------------------------------
        // Persistence
        // ---------------------------------------------------------------------------

        /// <summary>Zapisuje zamrożone parametry. Wywołaj po Freeze().</summary>
        public void Save(BinaryWriter bw)
        {
            ThrowIfNotFrozen();
            
            bw.Write(_frozenMean);
            bw.Write(_frozenInvStdDev);
        }

        /// <summary>Wczytuje zamrożone parametry — gotowy do TransformInPlace.</summary>
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
 
            // Log1p — TensorPrimitives nie ma Log1P dla float span w .NET 10.
            // Implementujemy przez Add(1) + Log — dwa przejścia SIMD.
            // Alternatywa: pętla z MathF.Log1P (scalar, brak SIMD) — wolniejsza.
            TensorPrimitives.Add(dst, 1f, dst);    // dst = x + 1
            TensorPrimitives.Log(dst, dst);         // dst = log(x + 1)
        }

        private void ThrowIfFrozen()
        {
            if (_frozen)
            {
                throw new InvalidOperationException(
                    "Normalizer is frozen. Call Reset() before fitting again.");
            }
        }

        private void ThrowIfNotFrozen()
        {
            if (!_frozen)
            {
                throw new InvalidOperationException(
                    "Normalizer is not frozen. Call Freeze() after fitting, or Load() to restore.");
            }
        }
    }
}