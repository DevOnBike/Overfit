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
    /// Uwaga: Dla binary feature (0/1) użyj ClipMax=1 i Min=0 —
    ///   TransformInPlace nic nie zmienia, wartości zostają 0/1.
    ///
    /// Użycie:
    ///   // Tryb empiryczny — min/max z danych:
    ///   var norm = new MinMaxNormalizer();
    ///   norm.FitBatch(data);
    ///   norm.Freeze();
    ///
    ///   // Tryb z ustalonym ClipMax (zalecany dla sparse):
    ///   var norm = MinMaxNormalizer.WithClipMax(clipMax: 5f);
    ///   norm.Freeze(); // od razu gotowy, nie wymaga fit
    /// </summary>
    public sealed class MinMaxNormalizer
    {
        private float _observedMin = float.MaxValue;
        private float _observedMax = float.MinValue;

        // Zamrożone parametry
        private float _frozenMin;
        private float _frozenScale;   // = 1f / (max - min)
        private bool _frozen;
        private bool _hasClipMax;
        private float _clipMax;

        public bool IsFrozen => _frozen;
        public float FrozenMin => _frozenMin;
        public float FrozenMax => _frozen ? (_frozenMin + 1f / _frozenScale) : _observedMax;
        public float ObservedMin => _observedMin;
        public float ObservedMax => _observedMax;

        // ---------------------------------------------------------------------------
        // Factory
        // ---------------------------------------------------------------------------

        /// <summary>
        /// Tworzy normalizator z ustalonym ClipMax — nie wymaga Fit.
        /// Min = 0, Max = clipMax. Idealne dla OomEventsRate, ContainerRestarts.
        /// </summary>
        public static MinMaxNormalizer WithClipMax(float clipMax)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(clipMax);

            return new MinMaxNormalizer
            {
                _hasClipMax = true,
                _clipMax = clipMax,
                _frozenMin = 0f,
                _frozenScale = 1f / clipMax,
                _frozen = true
            };
        }

        /// <summary>
        /// Tworzy normalizator dla binary feature (0/1) — TransformInPlace jest no-op.
        /// </summary>
        public static MinMaxNormalizer Binary()
        {
            return WithClipMax(1f);
        }

        // ---------------------------------------------------------------------------
        // Fit — offline (Golden Window) — dla trybu empirycznego
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

        /// <summary>Inkrementalne fitowanie.</summary>
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

        /// <summary>
        /// Zamraża parametry po zakończeniu fitu.
        /// </summary>
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
            _frozenScale = range < 1e-8f ? 1f : 1f / range;
            _frozen = true;
        }

        // ---------------------------------------------------------------------------
        // Transform — online (produkcja)
        // ---------------------------------------------------------------------------

        /// <summary>
        /// Aplikuje Min-Max in-place: x' = (x - min) / (max - min).
        /// Jeśli ClipMax — najpierw obcina wartości powyżej clipMax.
        /// Wymaga Freeze() lub WithClipMax().
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
                // Clip do [0, clipMax] potem skaluj
                TensorPrimitives.Max(data, 0f, data);
                TensorPrimitives.Min(data, _clipMax, data);
            }
            else
            {
                // Clip do obserwowanego zakresu
                TensorPrimitives.Max(data, _frozenMin, data);
            }

            // (x - min) * scale
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