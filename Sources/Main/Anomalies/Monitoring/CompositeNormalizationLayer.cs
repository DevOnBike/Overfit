// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.Data.Abstractions;
using DevOnBike.Overfit.Data.Normalizers;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    /// <summary>
    /// Aplikuje dedykowane algorytmy normalizacji (Log1p, ZScore, MinMax) 
    /// </summary>
    public sealed class CompositeNormalizationLayer
    {
        private readonly IFeatureNormalizer[] _normalizers;
        private bool _isFrozen;

        public CompositeNormalizationLayer()
        {
            _normalizers = new IFeatureNormalizer[(int)MetricIndex.Count];

            // 1. Kategoria: Bounded Ratios [0,1] -> Z-Score
            _normalizers[(int)MetricIndex.CpuUsageRatio] = new ZScoreNormalizer();
            _normalizers[(int)MetricIndex.CpuThrottleRatio] = new ZScoreNormalizer();

            // 2. Kategoria: Zdarzenia rzadkie (Sufit) -> MinMax z twardym odcięciem
            _normalizers[(int)MetricIndex.OomEventsRate] = MinMaxNormalizer.WithClipMax(5f);

            // 3. Kategoria: Long-tail / Rozkłady prawoskośne -> Log1p
            _normalizers[(int)MetricIndex.MemoryWorkingSetBytes] = new Log1pNormalizer();
            _normalizers[(int)MetricIndex.LatencyP50Ms] = new Log1pNormalizer();
            _normalizers[(int)MetricIndex.LatencyP95Ms] = new Log1pNormalizer();
            _normalizers[(int)MetricIndex.LatencyP99Ms] = new Log1pNormalizer();
            _normalizers[(int)MetricIndex.RequestsPerSecond] = new Log1pNormalizer();
            _normalizers[(int)MetricIndex.ErrorRate] = new Log1pNormalizer();
            _normalizers[(int)MetricIndex.GcGen2HeapBytes] = new Log1pNormalizer();
            _normalizers[(int)MetricIndex.GcPauseRatio] = new Log1pNormalizer();
            _normalizers[(int)MetricIndex.ThreadPoolQueueLength] = new Log1pNormalizer();
        }

        public void ProcessInPlace(Span<float> data, int windowSize, int metricCount, bool isTraining)
        {
            using var buffer = new PooledBuffer<float>(windowSize);

            var metricSpan = buffer.Span;

            for (var m = 0; m < metricCount; m++)
            {
                var normalizer = _normalizers[m];

                for (var t = 0; t < windowSize; t++)
                {
                    metricSpan[t] = data[t * metricCount + m];
                }

                if (isTraining)
                {
                    if (!normalizer.IsFrozen)
                    {
                        normalizer.FitBatch(metricSpan);
                    }
                }
                else
                {
                    normalizer.TransformInPlace(metricSpan);

                    for (var t = 0; t < windowSize; t++)
                    {
                        data[t * metricCount + m] = metricSpan[t];
                    }
                }
            }
        }

        public void FreezeAll()
        {
            if (_isFrozen)
            {
                return;
            }

            foreach (var norm in _normalizers)
            {
                if (!norm.IsFrozen)
                {
                    norm.Freeze();
                }
            }

            _isFrozen = true;
        }

        public void ResetAll()
        {
            foreach (var norm in _normalizers)
            {
                norm.Reset();
            }

            _isFrozen = false;
        }

        public void Save(BinaryWriter bw)
        {
            if (!_isFrozen)
            {
                throw new InvalidOperationException("Cannot save before freezing. Call FreezeAll() first.");
            }

            foreach (var norm in _normalizers)
            {
                bw.Write((byte)GetNormalizerType(norm));

                norm.Save(bw);
            }
        }

        public void Load(BinaryReader br)
        {
            for (var m = 0; m < _normalizers.Length; m++)
            {
                var type = (NormalizerType)br.ReadByte();

                _normalizers[m] = CreateNormalizer(type);
                _normalizers[m].Load(br);
            }

            _isFrozen = true;
        }

        private enum NormalizerType : byte
        {
            ZScore = 0,
            Log1p = 1,
            MinMax = 2
        }

        private static NormalizerType GetNormalizerType(IFeatureNormalizer norm)
        {
            return norm switch
            {
                ZScoreNormalizer => NormalizerType.ZScore,
                Log1pNormalizer => NormalizerType.Log1p,
                MinMaxNormalizer => NormalizerType.MinMax,

                _ => throw new InvalidOperationException($"Unknown normalizer type: {norm.GetType().Name}")
            };
        }

        private static IFeatureNormalizer CreateNormalizer(NormalizerType type)
        {
            return type switch
            {
                NormalizerType.ZScore => new ZScoreNormalizer(),
                NormalizerType.Log1p => new Log1pNormalizer(),
                NormalizerType.MinMax => new MinMaxNormalizer(),

                _ => throw new InvalidDataException($"Unknown normalizer type: {(byte)type}")
            };
        }
    }
}