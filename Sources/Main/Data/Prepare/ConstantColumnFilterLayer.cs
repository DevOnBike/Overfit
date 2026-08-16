// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Data.Abstractions;
using DevOnBike.Overfit.Data.Contracts;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Data.Prepare
{
    public sealed class ConstantColumnFilterLayer : IDataLayer
    {
        private readonly float _epsilon;
        private readonly float _minUniqueRatio;
        private bool _fitted;
        private int[]? _keptIndices;

        public ConstantColumnFilterLayer(float epsilon = 0f, float minUniqueRatio = 0f)
        {
            if (epsilon < 0f)
            {
                throw new ArgumentOutOfRangeException(nameof(epsilon), "Epsilon cannot be negative.");
            }
            if (minUniqueRatio is < 0f or > 1f)
            {
                throw new ArgumentOutOfRangeException(nameof(minUniqueRatio), "Unique value ratio must be in the range [0, 1].");
            }

            _epsilon = epsilon;
            _minUniqueRatio = minUniqueRatio;
        }

        public PipelineContext Process(PipelineContext context)
        {
            var rows = context.Features.GetView().GetDim(0);
            var cols = context.Features.GetView().GetDim(1);

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

                var span = context.Features.GetView().AsReadOnlySpan();
                var keptList = new List<int>(cols);

                if (_minUniqueRatio > 0f)
                {
                    IdentifyByUniqueRatio(span, rows, cols, keptList);
                }

                if (_minUniqueRatio <= 0f)
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

            var filtered = ExtractColumns(context.Features, _keptIndices, rows);

            context.Features.Dispose();
            context.Features = filtered;

            return context;
        }

        private FastTensor<float> ExtractColumns(FastTensor<float> src, ReadOnlySpan<int> indices, int rows)
        {
            var oldCols = src.GetView().GetDim(1);
            var newCols = indices.Length;

            var result = new FastTensor<float>(rows, newCols, clearMemory: false);
            var srcSpan = src.GetView().AsReadOnlySpan();
            var dstSpan = result.GetView().AsSpan();

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

                if (_epsilon != 0f)
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
        /// Keeps a column when at least <c>rows * minUniqueRatio</c> of its values are distinct.
        ///
        /// <para><b>The comparison was strict, which made the strictest available setting a no-op.</b> At
        /// <c>minUniqueRatio = 1.0</c> - a value the constructor validates and accepts - the bar became
        /// "more than <c>rows</c> distinct values in <c>rows</c> values", which no column can clear. Every
        /// column was then dropped, the "kept nothing, so keep everything" fallback fired, and the filter
        /// turned itself off. Worse than an error: the constant columns this layer exists to remove are
        /// exactly what makes a later scaler divide by a zero range.</para>
        /// </summary>
        private void IdentifyByUniqueRatio(ReadOnlySpan<float> span, int rows, int cols, List<int> keptIndices)
        {
            // At least one distinct value is not a filter, so a ratio that rounds down to zero still asks for
            // one - otherwise a tiny ratio on a small frame would keep a genuinely constant column.
            var minUnique = Math.Max(1, (int)(rows * _minUniqueRatio));
            var uniqueValues = new HashSet<float>(rows / 4);

            for (var c = 0; c < cols; c++)
            {
                uniqueValues.Clear();
                var earlyPass = false;

                for (var r = 0; r < rows; r++)
                {
                    uniqueValues.Add(span[r * cols + c]);
                    if (uniqueValues.Count >= minUnique)
                    {
                        earlyPass = true;
                        break;
                    }
                }

                if (earlyPass || uniqueValues.Count >= minUnique)
                {
                    keptIndices.Add(c);
                }
            }
        }

        public void Reset()
        {
            _keptIndices = null;
            _fitted = false;
        }

        /// <summary>
        /// State that only exists after <c>Fit</c>. Reading it earlier used to dereference null; this turns
        /// "transform before fit" into a named error instead of a NullReferenceException from inside a loop.
        /// </summary>
        private T RequireFitted<T>(T? value, string field)
            where T : class
        {
            return value ?? throw new OverfitRuntimeException(
                $"{nameof(ConstantColumnFilterLayer)}.{field} is not available — call Fit before Transform.");
        }

    }
}