// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.Data.Abstractions;
using DevOnBike.Overfit.Data.Contracts;

namespace DevOnBike.Overfit.Data.Prepare
{
    /// <summary>
    /// Implements outlier-robust scaling using the median and Interquartile Range (IQR).
    /// Formula: x' = (x - median) / IQR
    /// </summary>
    public sealed class RobustScalingLayer : IDataLayer
    {
        private readonly HashSet<int> _columnIndices;
        private readonly HashSet<int> _excludedColumns;
        private readonly float _fallbackIqr;
        private readonly bool _centerByMedian;

        // Persisted statistics from the Fit phase - reused during Transform (inference)
        private float[] _medians;
        private float[] _iqrs;
        private bool _fitted;

        /// <param name="columnIndices">Indices of columns to scale. Null = scale all (excluding specifically excluded columns).</param>
        /// <param name="excludedColumns">Columns to skip (e.g., binary or one-hot encoded features). Scaling 0/1 by IQR is undefined.</param>
        /// <param name="fallbackIqr">Default IQR to use if the calculated IQR is 0 (near-constant column). Defaults to 1.0.</param>
        /// <param name="centerByMedian">Whether to subtract the median (centering). False = divide by IQR only.</param>
        public RobustScalingLayer(
            HashSet<int> columnIndices = null,
            HashSet<int> excludedColumns = null,
            float fallbackIqr = 1f,
            bool centerByMedian = true)
        {
            if (fallbackIqr <= 0f)
            {
                throw new ArgumentOutOfRangeException(nameof(fallbackIqr), "Fallback IQR must be positive.");
            }

            _columnIndices = columnIndices;
            _excludedColumns = excludedColumns ?? new HashSet<int>();
            _fallbackIqr = fallbackIqr;
            _centerByMedian = centerByMedian;
        }

        public PipelineContext Process(PipelineContext context)
        {
            var rows = context.Features.GetDim(0);
            var cols = context.Features.GetDim(1);

            if (rows == 0 || cols == 0)
            {
                return context;
            }

            var span = context.Features.AsSpan();

            if (!_fitted)
            {
                if (rows < 2)
                {
                    return context;
                }

                Fit(span, rows, cols);
            }

            Transform(span, rows, cols);

            return context;
        }

        /// <summary>
        /// Calculates the median and IQR per column and persists them for inference.
        /// </summary>
        private void Fit(ReadOnlySpan<float> span, int rows, int cols)
        {
            _medians = new float[cols];
            _iqrs = new float[cols];

            Array.Fill(_iqrs, _fallbackIqr);

            using var sortBuffer = new FastBuffer<float>(rows);
            var bufferSpan = sortBuffer.AsSpan();

            for (var c = 0; c < cols; c++)
            {
                if (!ShouldScaleColumn(c, cols))
                {
                    continue;
                }

                for (var r = 0; r < rows; r++)
                {
                    bufferSpan[r] = span[r * cols + c];
                }

                bufferSpan.Sort();

                _medians[c] = InterpolatePercentile(bufferSpan, rows, 0.5f);

                var q1 = InterpolatePercentile(bufferSpan, rows, 0.25f);
                var q3 = InterpolatePercentile(bufferSpan, rows, 0.75f);
                var iqr = q3 - q1;

                _iqrs[c] = iqr > 0f ? iqr : _fallbackIqr;
            }

            _fitted = true;
        }

        /// <summary>
        /// Applies the persisted statistics to data in-place.
        /// </summary>
        private void Transform(Span<float> span, int rows, int cols)
        {
            for (var c = 0; c < cols; c++)
            {
                if (!ShouldScaleColumn(c, cols))
                {
                    continue;
                }

                var median = _medians[c];
                var iqr = _iqrs[c];
                var invIqr = 1f / iqr;

                if (_centerByMedian)
                {
                    for (var r = 0; r < rows; r++)
                    {
                        ref var val = ref span[r * cols + c];
                        val = (val - median) * invIqr;
                    }
                }
                else
                {
                    for (var r = 0; r < rows; r++)
                    {
                        ref var val = ref span[r * cols + c];
                        val *= invIqr;
                    }
                }
            }
        }

        private bool ShouldScaleColumn(int colIndex, int totalCols)
        {
            if (_excludedColumns.Contains(colIndex))
            {
                return false;
            }

            if (_columnIndices == null)
            {
                return true;
            }

            return _columnIndices.Contains(colIndex);
        }

        /// <summary>
        /// Linear interpolation of percentiles.
        /// Compatible with numpy.percentile(interpolation='linear').
        /// Formula: rank = p * (N - 1). Result is lerp between adjacent samples.
        /// </summary>
        private static float InterpolatePercentile(ReadOnlySpan<float> sorted, int count, float percentile)
        {
            var rank = percentile * (count - 1);
            var lowerIdx = (int)rank;
            var upperIdx = lowerIdx + 1;

            if (upperIdx >= count)
            {
                return sorted[lowerIdx];
            }

            var fraction = rank - lowerIdx;

            return sorted[lowerIdx] * (1f - fraction) + sorted[upperIdx] * fraction;
        }

        /// <summary>
        /// Exports fitted median and IQR parameters for persistence.
        /// Call after processing the Golden Window data.
        /// Throws if the scaler has not been fitted yet.
        /// </summary>
        public ScalerParams ExportParams()
        {
            if (!_fitted)
            {
                throw new InvalidOperationException("Scaler has not been fitted yet. Call Process() on training data first.");
            }

            return new ScalerParams
            {
                Medians = (float[])_medians.Clone(),
                Iqrs = (float[])_iqrs.Clone()
            };
        }

        /// <summary>
        /// Imports previously saved parameters, bypassing the Fit phase.
        /// Use at inference time to restore a scaler fitted on the Golden Window.
        /// </summary>
        public void ImportParams(ScalerParams scalerParams)
        {
            ArgumentNullException.ThrowIfNull(scalerParams);

            if (scalerParams.Medians.Length != scalerParams.Iqrs.Length)
            {
                throw new ArgumentException("Medians and Iqrs must have the same length.");
            }

            _medians = (float[])scalerParams.Medians.Clone();
            _iqrs = (float[])scalerParams.Iqrs.Clone();
            _fitted = true;
        }

        /// <summary>
        /// Resets the persisted statistics, forcing a re-Fit on the next Process call.
        /// </summary>
        public void Reset()
        {
            _medians = null;
            _iqrs = null;
            _fitted = false;
        }
    }

}