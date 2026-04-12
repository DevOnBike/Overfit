// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Data.Abstractions;

namespace DevOnBike.Overfit.Data.Normalizers
{
    public sealed class MinMaxNormalizer : IFeatureNormalizer
    {
        private float _observedMin = float.MaxValue;
        private float _observedMax = float.MinValue;
        private bool _hasSeenData; // <-- JAWNA FLAGA STANOWA

        private float _frozenMin;
        private float _frozenMax;
        private float _frozenScale;
        private bool _frozen;
        private bool _hasClipMax;
        private float _clipMax;

        public bool ClipToRange { get; init; }

        public bool IsFrozen => _frozen;
        public float FrozenMin => _frozenMin;
        public float FrozenMax => _frozenMax;
        public float ObservedMin => _observedMin;
        public float ObservedMax => _observedMax;

        public static MinMaxNormalizer WithClipMax(float clipMax)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(clipMax);

            return new MinMaxNormalizer
            {
                _hasClipMax = true,
                _clipMax = clipMax,
                _frozenMin = 0f,
                _frozenMax = clipMax,
                _frozenScale = 1f / clipMax,
                _frozen = true
            };
        }

        public static MinMaxNormalizer Binary() => WithClipMax(1f);

        public void FitBatch(ReadOnlySpan<float> data)
        {
            ThrowIfFrozen();

            if (data.Length == 0)
            {
                return;
            }

            _hasSeenData = true; // <-- AKTUALIZACJA FLAGI

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

            _hasSeenData = true; // <-- AKTUALIZACJA FLAGI

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

            if (!_hasSeenData && !_hasClipMax) // <-- BEZPIECZNE SPRAWDZENIE
            {
                throw new InvalidOperationException("Cannot freeze before fitting. Call FitBatch or FitIncremental first.");
            }

            var range = _observedMax - _observedMin;

            _frozenMin = _observedMin;
            _frozenMax = _observedMax;
            _frozenScale = range < 1e-8f ? 1f : 1f / range;
            _frozen = true;
        }

        public void TransformInPlace(Span<float> data)
        {
            ThrowIfNotFrozen();

            if (data.Length == 0)
            {
                return;
            }

            if (_hasClipMax)
            {
                TensorPrimitives.Max(data, 0f, data);
                TensorPrimitives.Min(data, _clipMax, data);
            }
            else if (ClipToRange)
            {
                TensorPrimitives.Max(data, _frozenMin, data);
                TensorPrimitives.Min(data, _frozenMax, data);
            }
            else
            {
                TensorPrimitives.Max(data, _frozenMin, data);
            }

            TensorPrimitives.Subtract(data, _frozenMin, data);
            TensorPrimitives.Multiply(data, _frozenScale, data);
        }

        // Metody Load, Save, Reset bez zmian...
        public void Reset()
        {
            _observedMin = float.MaxValue;
            _observedMax = float.MinValue;
            _hasSeenData = false;
            _frozenMin = 0f;
            _frozenMax = 0f;
            _frozenScale = 0f;
            _hasClipMax = false;
            _clipMax = 0f;
            _frozen = false;
        }

        private void ThrowIfFrozen()
        {
            if (_frozen)
            {
                throw new InvalidOperationException("Normalizer is frozen.");
            }
        }

        private void ThrowIfNotFrozen()
        {
            if (!_frozen)
            {
                throw new InvalidOperationException("Normalizer is not frozen.");
            }
        }

        public void Save(BinaryWriter bw)
        {
            if (!_frozen)
            {
                throw new InvalidOperationException("Cannot save unfrozen normalizer.");
            }

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
    }
}