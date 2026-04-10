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
    /// Filters out columns that are constant or have a low unique value ratio.
    /// This layer follows the Fit-Transform pattern, identifying columns during the first pass.
    /// </summary>
    public sealed class ConstantColumnFilterLayer : IDataLayer
    {
        private readonly float _epsilon;
        private readonly float _minUniqueRatio;

        private int[] _keptIndices;
        private bool _fitted;

        /// <param name="epsilon">The threshold for considering values as identical. Use 0 for exact match.</param>
        /// <param name="minUniqueRatio">Minimum ratio of unique values [0, 1]. Columns below this threshold are dropped.</param>
        public ConstantColumnFilterLayer(float epsilon = 0f, float minUniqueRatio = 0f)
        {
            if (epsilon < 0f)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(epsilon), "Epsilon cannot be negative.");
            }

            if (minUniqueRatio is < 0f or > 1f)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(minUniqueRatio), "Unique value ratio must be in the range [0, 1].");
            }

            _epsilon = epsilon;
            _minUniqueRatio = minUniqueRatio;
        }

        public PipelineContext Process(PipelineContext context)
        {
            var rows = context.Features.GetDim(0);
            var cols = context.Features.GetDim(1);

            if (cols == 0)
            {
                return context;
            }

            if (!_fitted)
            {
                if (rows <= 1)
                {
                    return context;
                }

                var span = context.Features.AsReadOnlySpan();
                var keptList = new List<int>(cols);

                if (_minUniqueRatio > 0f)
                {
                    IdentifyByUniqueRatio(span, rows, cols, keptList);
                }
                else
                {
                    IdentifyByVariance(span, rows, cols, keptList);
                }

                _keptIndices = keptList.Count == cols || keptList.Count == 0 ? null : keptList.ToArray();

                _fitted = true;
            }

            if (_keptIndices == null)
            {
                return context;
            }

            var filtered = ExtractColumns(context.Features, _keptIndices, context.Features.GetDim(0));

            context.Features.Dispose();
            context.Features = filtered;

            return context;
        }

        /// <summary>
        /// Creates a new tensor containing only the selected columns.
        /// </summary>
        private FastTensor<float> ExtractColumns(FastTensor<float> src, int[] indices, int rows)
        {
            var oldCols = src.GetDim(1);
            var newCols = indices.Length;

            var result = new FastTensor<float>(rows, newCols);
            var srcSpan = src.AsReadOnlySpan();
            var dstSpan = result.AsSpan();

            for (var r = 0; r < rows; r++)
            {
                var srcOffset = r * oldCols;
                var dstOffset = r * newCols;

                for (var c = 0; c < newCols; c++)
                {
                    dstSpan[dstOffset + c] = srcSpan[srcOffset + indices[c]];
                }
            }

            return result;
        }

        /// <summary>
        /// Identifies non-constant columns based on a variance threshold (epsilon).
        /// </summary>
        private void IdentifyByVariance(ReadOnlySpan<float> span, int rows, int cols, List<int> keptIndices)
        {
            for (var c = 0; c < cols; c++)
            {
                var firstVal = span[c];
                var isConstant = true;

                if (_epsilon == 0f)
                {
                    for (var r = 1; r < rows; r++)
                    {
                        if (span[r * cols + c] != firstVal)
                        {
                            isConstant = false;
                            break;
                        }
                    }
                }
                else
                {
                    for (var r = 1; r < rows; r++)
                    {
                        if (MathF.Abs(span[r * cols + c] - firstVal) > _epsilon)
                        {
                            isConstant = false;
                            break;
                        }
                    }
                }

                if (!isConstant)
                {
                    keptIndices.Add(c);
                }
            }
        }

        /// <summary>
        /// Identifies columns that meet the minimum unique value ratio requirement.
        /// </summary>
        private void IdentifyByUniqueRatio(ReadOnlySpan<float> span, int rows, int cols, List<int> keptIndices)
        {
            var minUnique = (int)(rows * _minUniqueRatio);
            var uniqueValues = new HashSet<float>(rows / 4);

            for (var c = 0; c < cols; c++)
            {
                uniqueValues.Clear();
                var earlyPass = false;

                for (var r = 0; r < rows; r++)
                {
                    uniqueValues.Add(span[r * cols + c]);

                    if (uniqueValues.Count > minUnique)
                    {
                        earlyPass = true;
                        break;
                    }
                }

                if (earlyPass || uniqueValues.Count > minUnique)
                {
                    keptIndices.Add(c);
                }
            }
        }

        /// <summary>
        /// Resets the fitted state, allowing the layer to learn from a new dataset.
        /// </summary>
        public void Reset()
        {
            _keptIndices = null;
            _fitted = false;
        }
    }
}

