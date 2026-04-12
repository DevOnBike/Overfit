// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;

namespace DevOnBike.Overfit.Statistical
{
    /// <summary>
    /// Normalizator Min-Max dla sparse metryk i zdarzeń rzadkich.
    ///
    /// Pipeline:
    ///   x' = (x - min) / (max - min)    → zakres [0, 1]
    ///
    ///   Opcjonalnie z ClipMax:
    ///   x  = min(x, ClipMax)            — obcinanie wartości ekstremalnych
    ///   x' = x / ClipMax                → zakres [0, 1]
    ///
    /// Przeznaczenie:
    ///   OomEventsRate     — ClipMax=5 (>5 OOM w oknie to anomalia)
    ///   ContainerRestarts — ClipMax=5
    ///   IsThrottled       — binary 0/1, ClipMax=1
    ///
    /// Użycie:
    ///   // Tryb empiryczny — min/max z danych:
    ///   var norm = new MinMaxNormalizer();
    ///   norm.FitBatch(data);
    ///   norm.Freeze();
    ///
    ///   // Tryb z ustalonym ClipMax (zalecany dla sparse):
    ///   var norm = MinMaxNormalizer.WithClipMax(clipMax: 5f);
    ///   // od razu gotowy, nie wymaga Fit/Freeze
    /// </summary>
    public sealed class MinMaxNormalizer
    {
        private float _observedMin = float.MaxValue;
        private float _observedMax = float.MinValue;

        // Zamrożone parametry
        private float _frozenMin;
        private float _frozenMax;
        private float _frozenScale;   // = 1f / (max - min)
        private bool _frozen;
        private bool _hasClipMax;
        private float _clipMax;

        /// <summary>
        /// When true, values outside [min, max] are clipped to [0, 1] during Transform.
        /// When false (default for empirical mode), values above max produce output > 1.0 —
        /// useful for anomaly detection where out-of-range values are the signal.
        /// Always true when using WithClipMax().
        /// </summary>
        public bool ClipToRange { get; init; }

        public bool IsFrozen => _frozen;
        public float FrozenMin => _frozenMin;
        public float FrozenMax => _frozenMax;
        public float ObservedMin => _observedMin;
        public float ObservedMax => _observedMax;

        // ---------------------------------------------------------------------------
        // Factory
        // ---------------------------------------------------------------------------

        /// <summary>
        /// Tworzy normalizator z ustalonym ClipMax — nie wymaga Fit/Freeze.
        /// Min = 0, Max = clipMax. Idealne dla OomEventsRate, ContainerRestarts.
        /// </summary>
        public static MinMaxNormalizer WithClipMax(float clipMax)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(clipMax);

            var norm = new MinMaxNormalizer
            {
                _hasClipMax = true,
                _clipMax = clipMax,
                _frozenMin = 0f,
                _frozenMax = clipMax,
                _frozenScale = 1f / clipMax,
                _frozen = true
            };
            
            return norm;
        }

        /// <summary>
        /// Tworzy normalizator dla binary feature (0/1) — TransformInPlace jest no-op.
        /// </summary>
        public static MinMaxNormalizer Binary() => WithClipMax(1f);

        // ---------------------------------------------------------------------------
        // Fit
        // ---------------------------------------------------------------------------

        /// <summary>
        /// Aktualizuje obserwowane min/max z batcha danych.
        /// Można wołać wielokrotnie — akumuluje globalny min/max.
        /// </summary>
        public void FitBatch(ReadOnlySpan<float> data)
        {
            ThrowIfFrozen();

            if (data.Length == 0)
            {
                return;
            }

            var batchMin = TensorPrimitives.Min(data);
            var batchMax = TensorPrimitives.Max(data);

            if (batchMin < _observedMin)
            {
                _observedMin = batchMin;
            }

            if (batchMax > _observedMax)
            {
                _observedMax = batchMax;
            }
        }

        public void FitIncremental(float value)
        {
            ThrowIfFrozen();
            
            if (value < _observedMin)
            {
                _observedMin = value;
            }

            if (value > _observedMax)
            {
                _observedMax = value;
            }
        }

        public void Freeze()
        {
            if (_frozen)
            {
                return;
            }

            if (_observedMin == float.MaxValue)
            {
                throw new InvalidOperationException("Cannot freeze before fitting. Call FitBatch or FitIncremental first, or use MinMaxNormalizer.WithClipMax() for a fixed range.");
            }

            var range = _observedMax - _observedMin;

            _frozenMin = _observedMin;
            _frozenMax = _observedMax;
            _frozenScale = range < 1e-8f ? 1f : 1f / range;
            _frozen = true;
        }

        // ---------------------------------------------------------------------------
        // Transform
        // ---------------------------------------------------------------------------

        /// <summary>
        /// Aplikuje Min-Max in-place: x' = (x - min) / (max - min).
        /// </summary>
        public void TransformInPlace(Span<float> data)
        {
            ThrowIfNotFrozen();

            if (data.Length == 0)
            {
                return;
            }

            if (_hasClipMax)
            {
                // Clip do [0, clipMax] — zawsze dla WithClipMax()
                TensorPrimitives.Max(data, 0f, data);
                TensorPrimitives.Min(data, _clipMax, data);
            }
            else if (ClipToRange)
            {
                // Clip do [frozenMin, frozenMax] — opt-in dla trybu empirycznego
                TensorPrimitives.Max(data, _frozenMin, data);
                TensorPrimitives.Min(data, _frozenMax, data);
            }
            else
            {
                // Tylko dolny clip — wartości > max mogą przekroczyć 1.0
                // Pożądane dla anomaly detection: ekstremalny wynik = sygnał anomalii
                TensorPrimitives.Max(data, _frozenMin, data);
            }

            TensorPrimitives.Subtract(data, _frozenMin, data);
            TensorPrimitives.Multiply(data, _frozenScale, data);
        }

        // ---------------------------------------------------------------------------
        // Persistence
        // ---------------------------------------------------------------------------

        public void Save(BinaryWriter bw)
        {
            ThrowIfNotFrozen();
            bw.Write(_frozenMin);
            bw.Write(_frozenMax);
            bw.Write(_frozenScale);
            bw.Write(_hasClipMax);
            if (_hasClipMax)
            {
                bw.Write(_clipMax);
            }
        }

        public void Load(BinaryReader br)
        {
            _frozenMin = br.ReadSingle();
            _frozenMax = br.ReadSingle();
            _frozenScale = br.ReadSingle();
            _hasClipMax = br.ReadBoolean();

            if (_hasClipMax)
            {
                _clipMax = br.ReadSingle();
            }

            _frozen = true;
        }

        public void Reset()
        {
            _observedMin = float.MaxValue;
            _observedMax = float.MinValue;
            _frozenMin = 0f;
            _frozenMax = 0f;
            _frozenScale = 0f;
            _hasClipMax = false;
            _clipMax = 0f;
            _frozen = false;
        }

        // ---------------------------------------------------------------------------
        // Private
        // ---------------------------------------------------------------------------

        private void ThrowIfFrozen()
        {
            if (_frozen)
            {
                throw new InvalidOperationException("Normalizer is frozen. Call Reset() before fitting again.");
            }
        }

        private void ThrowIfNotFrozen()
        {
            if (!_frozen)
            {
                throw new InvalidOperationException("Normalizer is not frozen. Call Freeze() after fitting, or use MinMaxNormalizer.WithClipMax().");
            }
        }
    }
}