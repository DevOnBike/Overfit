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
    ///     The initial pipeline layer responsible for sanitizing technical artifacts in the data.
    ///     Cleans NaN, ±Infinity, and subnormal values, with an option to discard rows exceeding a corruption threshold.
    /// </summary>
    public sealed class TechnicalSanityLayer : IDataLayer
    {
        private readonly float _maxCorruptedRatio;
        private readonly float _replacementValue;

        /// <param name="maxCorruptedRatio">
        ///     Maximum allowed ratio of corrupted values per row (0.0–1.0).
        ///     Rows exceeding this threshold will be discarded.
        ///     A value of 1.0 disables filtering and only performs in-place cleaning.
        /// </param>
        /// <param name="replacementValue">The value used to replace NaN/Inf/Subnormal entries (default is 0).</param>
        public TechnicalSanityLayer(float maxCorruptedRatio = 1.0f, float replacementValue = 0f)
        {
            if (maxCorruptedRatio is < 0f or > 1f)
            {
                throw new ArgumentOutOfRangeException(nameof(maxCorruptedRatio), "Corrupted value threshold must be in the range [0, 1].");
            }

            _maxCorruptedRatio = maxCorruptedRatio;
            _replacementValue = replacementValue;
        }

        public PipelineContext Process(PipelineContext context)
        {
            var rows = context.Features.GetView().GetDim(0);
            var cols = context.Features.GetView().GetDim(1);

            if (rows == 0 || cols == 0)
            {
                return context;
            }

            CleanSpan(context.Targets.GetView().AsSpan());

            if (_maxCorruptedRatio >= 1.0f)
            {
                CleanSpan(context.Features.GetView().AsSpan());
                return context;
            }

            var featureSpan = context.Features.GetView().AsSpan();
            var maxCorruptedPerRow = (int)(cols * _maxCorruptedRatio);

            using var corruptedCounts = new PooledBuffer<int>(rows);
            var countSpan = corruptedCounts.Span;

            for (var r = 0; r < rows; r++)
            {
                var offset = r * cols;
                var corrupted = 0;

                for (var c = 0; c < cols; c++)
                {
                    if (IsCorrupted(featureSpan[offset + c]))
                    {
                        corrupted++;
                    }
                }

                countSpan[r] = corrupted;
            }

            var keptIndices = new List<int>(rows);
            for (var r = 0; r < rows; r++)
            {
                if (countSpan[r] <= maxCorruptedPerRow)
                {
                    keptIndices.Add(r);
                }
            }

            if (keptIndices.Count == rows)
            {
                CleanSpan(featureSpan);
                return context;
            }

            if (keptIndices.Count == 0)
            {
                CleanSpan(featureSpan);
                return context;
            }

            var newRows = keptIndices.Count;
            var newFeatures = new FastTensor<float>(newRows, cols, clearMemory: false);
            var newTargets = new FastTensor<float>(newRows, 1, clearMemory: false);

            var srcFeatures = context.Features.GetView().AsReadOnlySpan();
            var srcTargets = context.Targets.GetView().AsReadOnlySpan();
            var dstFeatures = newFeatures.GetView().AsSpan();
            var dstTargets = newTargets.GetView().AsSpan();

            for (var i = 0; i < newRows; i++)
            {
                var srcRow = keptIndices[i];

                srcFeatures.Slice(srcRow * cols, cols).CopyTo(dstFeatures.Slice(i * cols, cols));

                dstTargets[i] = srcTargets[srcRow];
            }

            CleanSpan(dstFeatures);
            CleanSpan(dstTargets);

            context.Features.Dispose();
            context.Targets.Dispose();

            return new PipelineContext(newFeatures, newTargets);
        }

        private void CleanSpan(Span<float> span)
        {
            for (var i = 0; i < span.Length; i++)
            {
                if (IsCorrupted(span[i]))
                {
                    span[i] = _replacementValue;
                }
            }
        }

        /// <summary>
        ///     Checks if a value is technically corrupted: NaN, ±Infinity, or subnormal (denormalized).
        /// </summary>
        /// <remarks>
        ///     Subnormals can cause significant performance degradation (up to 100x slowdown)
        ///     in floating-point operations on certain CPUs.
        /// </remarks>
        private static bool IsCorrupted(float value)
        {
            return float.IsNaN(value) || float.IsInfinity(value) || float.IsSubnormal(value);
        }
    }
}