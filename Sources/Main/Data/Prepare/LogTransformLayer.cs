// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Data.Abstractions;
using DevOnBike.Overfit.Data.Contracts;

namespace DevOnBike.Overfit.Data.Prepare
{
    /// <summary>
    ///     Logarithmic transformation for columns with highly skewed distributions.
    ///     Used to stabilize variance, reduce outlier impact, and improve gradient convergence
    ///     by shifting feature distributions closer to a normal distribution.
    /// </summary>
    public sealed class LogTransformLayer : IDataLayer
    {
        private readonly List<int> _columnIndices;
        private readonly float _epsilon;
        private readonly LogMode _mode;

        public LogTransformLayer(
            List<int> columnIndices,
            LogMode mode = LogMode.Log1p,
            float epsilon = 1e-7f)
        {
            if (columnIndices == null || columnIndices.Count == 0)
            {
                throw new ArgumentException("The list of columns for transformation cannot be empty.", nameof(columnIndices));
            }

            if (epsilon <= 0f)
            {
                throw new ArgumentOutOfRangeException(nameof(epsilon), "Epsilon must be positive.");
            }

            _columnIndices = columnIndices;
            _mode = mode;
            _epsilon = epsilon;
        }

        public PipelineContext Process(PipelineContext context)
        {
            var rows = context.Features.GetDim(0);
            var cols = context.Features.GetDim(1);

            if (rows == 0 || cols == 0)
            {
                return context;
            }

            foreach (var c in _columnIndices)
            {
                if (c < 0 || c >= cols)
                {
                    throw new InvalidOperationException($"Column index {c} is out of tensor range (0–{cols - 1}).");
                }
            }

            var span = context.Features.AsSpan();

            switch (_mode)
            {
                case LogMode.Log1p:
                    ApplyLog1p(span, rows, cols);
                    break;

                case LogMode.SignedLog1p:
                    ApplySignedLog1p(span, rows, cols);
                    break;

                case LogMode.LogEps:
                    ApplyLogEps(span, rows, cols);
                    break;

                default:
                    throw new InvalidOperationException($"Unsupported LogMode: {_mode}");
            }

            return context;
        }

        /// <summary>
        ///     Applies log(1 + x). Ideal for non-negative data (e.g., price, area).
        ///     Safe for x = 0 (results in 0). Negative values are clamped to 0.
        /// </summary>
        private void ApplyLog1p(Span<float> span, int rows, int cols)
        {
            foreach (var c in _columnIndices)
            {
                for (var r = 0; r < rows; r++)
                {
                    ref var val = ref span[r * cols + c];

                    if (val < 0f)
                    {
                        val = 0f;
                        continue;
                    }

                    val = MathF.Log(1f + val);
                }
            }
        }

        /// <summary>
        ///     Applies sign(x) * log(1 + |x|).
        ///     Used for symmetric distributions containing negative values (e.g., price changes or residuals).
        /// </summary>
        private void ApplySignedLog1p(Span<float> span, int rows, int cols)
        {
            foreach (var c in _columnIndices)
            {
                for (var r = 0; r < rows; r++)
                {
                    ref var val = ref span[r * cols + c];

                    if (val == 0f)
                    {
                        continue;
                    }

                    val = MathF.Sign(val) * MathF.Log(1f + MathF.Abs(val));
                }
            }
        }

        /// <summary>
        ///     Applies log(x + epsilon).
        ///     Use this for strictly positive data when log1p flattening of small values is undesirable.
        ///     Epsilon protects against log(0) resulting in negative infinity.
        /// </summary>
        private void ApplyLogEps(Span<float> span, int rows, int cols)
        {
            foreach (var c in _columnIndices)
            {
                for (var r = 0; r < rows; r++)
                {
                    ref var val = ref span[r * cols + c];

                    if (val < 0f)
                    {
                        val = MathF.Log(_epsilon);
                        continue;
                    }

                    val = MathF.Log(val + _epsilon);
                }
            }
        }
    }
}