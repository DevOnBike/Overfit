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
    /// Detects and removes duplicate rows based on feature values.
    /// Employs a two-stage comparison strategy: fast hashing followed by full Span verification 
    /// to eliminate false positive collisions.
    /// </summary>
    public sealed class DuplicateRowFilterLayer : IDataLayer
    {
        private readonly bool _includeTargetInComparison;

        /// <param name="includeTargetInComparison">
        /// Whether to include the Target column when identifying duplicates.
        /// <list type="bullet">
        /// <item><description><c>true</c>: Rows with identical features but different targets are NOT considered duplicates.</description></item>
        /// <item><description><c>false</c>: Only features are compared (default; safer for regression tasks).</description></item>
        /// </list>
        /// </param>
        public DuplicateRowFilterLayer(bool includeTargetInComparison = false)
        {
            _includeTargetInComparison = includeTargetInComparison;
        }

        public PipelineContext Process(PipelineContext context)
        {
            var rows = context.Features.GetDim(0);
            var cols = context.Features.GetDim(1);

            if (rows <= 1)
            {
                return context;
            }

            var featureSpan = context.Features.AsReadOnlySpan();
            var targetSpan = context.Targets.AsReadOnlySpan();

            using var rowHashes = new FastBuffer<int>(rows);
            var hashSpan = rowHashes.AsSpan();

            for (var r = 0; r < rows; r++)
            {
                hashSpan[r] = ComputeRowHash(featureSpan, targetSpan, r, cols);
            }

            var keptIndices = new List<int>(rows);
            var hashBuckets = new Dictionary<int, List<int>>(rows);

            for (var r = 0; r < rows; r++)
            {
                var hash = hashSpan[r];

                if (!hashBuckets.TryGetValue(hash, out var bucket))
                {
                    bucket = new List<int>(1);
                    hashBuckets[hash] = bucket;
                    bucket.Add(r);
                    keptIndices.Add(r);
                    continue;
                }

                var isDuplicate = false;
                foreach (var existingRow in bucket)
                {
                    if (RowsEqual(featureSpan, targetSpan, existingRow, r, cols))
                    {
                        isDuplicate = true;
                        break;
                    }
                }

                if (!isDuplicate)
                {
                    bucket.Add(r);
                    keptIndices.Add(r);
                }
            }

            if (keptIndices.Count == rows)
            {
                return context;
            }

            var newRows = keptIndices.Count;
            var newFeatures = new FastTensor<float>(newRows, cols);
            var newTargets = new FastTensor<float>(newRows, 1);

            var dstFeatures = newFeatures.AsSpan();
            var dstTargets = newTargets.AsSpan();

            for (var i = 0; i < newRows; i++)
            {
                var srcRow = keptIndices[i];

                featureSpan.Slice(srcRow * cols, cols).CopyTo(dstFeatures.Slice(i * cols, cols));
                dstTargets[i] = targetSpan[srcRow];
            }

            context.Features.Dispose();
            context.Targets.Dispose();

            return new PipelineContext(newFeatures, newTargets);
        }

        private int ComputeRowHash(
            ReadOnlySpan<float> features,
            ReadOnlySpan<float> targets,
            int row,
            int cols)
        {
            var hash = new HashCode();
            var offset = row * cols;

            var c = 0;
            var limit = cols - 3;
            for (; c < limit; c += 4)
            {
                hash.Add(features[offset + c]);
                hash.Add(features[offset + c + 1]);
                hash.Add(features[offset + c + 2]);
                hash.Add(features[offset + c + 3]);
            }

            for (; c < cols; c++)
            {
                hash.Add(features[offset + c]);
            }

            if (_includeTargetInComparison)
            {
                hash.Add(targets[row]);
            }

            return hash.ToHashCode();
        }

        private bool RowsEqual(
            ReadOnlySpan<float> features,
            ReadOnlySpan<float> targets,
            int rowA,
            int rowB,
            int cols)
        {
            var offsetA = rowA * cols;
            var offsetB = rowB * cols;

            if (!features.Slice(offsetA, cols).SequenceEqual(features.Slice(offsetB, cols)))
            {
                return false;
            }

            if (_includeTargetInComparison)
            {
                return targets[rowA] == targets[rowB];
            }

            return true;
        }
    }
}