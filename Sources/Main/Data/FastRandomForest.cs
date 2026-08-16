// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Data.Contracts;
using DevOnBike.Overfit.Runtime;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Data
{
    /// <summary>
    /// A randomised regression forest, used here to score feature importance rather than to predict.
    ///
    /// <para><b>It did not split its data, and every consequence of that was silent.</b> Until 2026-08-02
    /// <c>BuildRecursive</c> picked a feature and a threshold, stored them on the node, and then handed the
    /// <i>identical, unpartitioned</i> rows to both children. Three things followed, none of which raises
    /// anything:</para>
    ///
    /// <list type="bullet">
    /// <item>Every leaf averaged the whole target column, so every tree returned the same constant and
    /// <see cref="Predict"/> could not distinguish two rows.</item>
    /// <item>The importance scores were a tally of which features were drawn at random - noise wearing the
    /// clothes of statistics, and consumed as statistics by <c>BorutaSelectionLayer</c> and
    /// <c>ShapSelectionLayer</c>.</item>
    /// <item>The row count never shrank, so the <c>rows &lt; 2</c> stop could never fire and the only exit was
    /// the depth limit: always a <b>complete</b> binary tree. At the default depth of 10 that is ~2 000 nodes
    /// and merely wrong; the constructor accepts 30, which is over two billion nodes per tree no matter how
    /// small the dataset.</item>
    /// </list>
    ///
    /// <para>Rows are now carried as an index array partitioned in place, so a node sees exactly its own
    /// subset and both stopping conditions become reachable. A split that separates nothing - every row on
    /// one side, which a random threshold on a constant column produces every time - becomes a leaf rather
    /// than two identical children.</para>
    ///
    /// <para><b>What this still is not:</b> there is no bootstrap resampling, so trees differ only by their
    /// random feature and threshold draws. That is closer to extremely-randomised trees than to Breiman's
    /// forest, and it is a limit of scope rather than an oversight.</para>
    /// </summary>
    public sealed class FastRandomForest : IDisposable
    {
        private readonly int _maxDepth;
        private readonly int _numTrees;
        private readonly List<FastTreeNode[]> _forest = [];

        /// <summary>Hard ceiling on <c>maxDepth</c>. Tree building recurses once per level, so an
        /// unvalidated depth is an unbounded stack; 64 is already far past the point where a tree is
        /// useful (2^64 leaves) and keeps the recursion provably shallow.</summary>
        private const int MaxAllowedDepth = 64;

        public FastRandomForest(int numTrees = 50, int maxDepth = 10)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numTrees);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxDepth);
            ArgumentOutOfRangeException.ThrowIfGreaterThan(maxDepth, MaxAllowedDepth);

            _numTrees = numTrees;
            _maxDepth = maxDepth;
        }

        public float[] TrainAndGetImportance(FastTensor<float> x, FastTensor<float> y)
        {
            var view = x.GetView();
            var rowCount = view.GetDim(0);
            var cols = view.GetDim(1);
            var totalImportance = new float[cols];
            var lockObj = new object();

            _forest.Clear();

            var trees = new FastTreeNode[_numTrees][];

            OverfitParallel.For(0, _numTrees, t =>
            {
                var localImportance = new float[cols];
                var nodes = new List<FastTreeNode>();

                // One index array per tree, permuted in place as the tree is built. Per tree rather than
                // shared: the trees are built concurrently, and one permutation between them would have each
                // tree partitioning the others' rows.
                var rows = new int[rowCount];

                for (var i = 0; i < rowCount; i++)
                {
                    rows[i] = i;
                }

                BuildRecursive(x, y, rows, 0, rowCount, 0, nodes, localImportance);

                trees[t] = nodes.ToArray();

                lock (lockObj)
                {
                    for (var i = 0; i < cols; i++)
                    {
                        totalImportance[i] += localImportance[i];
                    }
                }
            });

            _forest.AddRange(trees);

            return totalImportance;
        }

        public void Train(FastTensor<float> x, FastTensor<float> y)
        {
            TrainAndGetImportance(x, y);
        }

        public float Predict(ReadOnlySpan<float> features)
        {
            if (_forest.Count == 0)
            {
                return 0f;
            }

            double sum = 0;
            for (var i = 0; i < _forest.Count; i++)
            {
                sum += Traverse(_forest[i], features);
            }

            return (float)(sum / _forest.Count);
        }

        private float Traverse(FastTreeNode[] tree, ReadOnlySpan<float> features)
        {
            var currentIdx = 0;

            // BOUND: MaxAllowedDepth steps. Walking to a leaf takes at most tree-depth hops, but the hop
            // target comes from node data — a corrupt or cyclic tree would otherwise spin here forever with
            // no exception and no log line. The counter turns that into a reportable error.
            for (var step = 0; step <= MaxAllowedDepth; step++)
            {
                ref readonly var node = ref tree[currentIdx];

                if (node.IsLeaf)
                {
                    return node.Value;
                }

                currentIdx = features[node.FeatureIndex] <= node.Threshold ? node.LeftChildIndex : node.RightChildIndex;

                if (currentIdx == -1)
                {
                    return node.Value;
                }
            }

            throw new OverfitRuntimeException(
                $"Tree traversal exceeded {MaxAllowedDepth} levels without reaching a leaf — the tree is "
                + "malformed or contains a cycle.");
        }

        /// <param name="rows">Row indices, permuted in place so that <c>[start, end)</c> is this node's subset.</param>
        /// <param name="start">First index of this node's slice of <paramref name="rows"/>.</param>
        /// <param name="end">One past the last.</param>
        /// <param name="x">Feature matrix, rows by features.</param>
        /// <param name="y">Target column, one entry per row of <paramref name="x"/>.</param>
        /// <param name="depth">Depth of this node, compared against the configured maximum to decide whether to split.</param>
        /// <param name="nodes">Flat node list being built; this method appends its own node and returns that index.</param>
        /// <param name="importance">Per-feature importance accumulator, added to as splits are chosen.</param>
        private int BuildRecursive(
            FastTensor<float> x,
            FastTensor<float> y,
            int[] rows,
            int start,
            int end,
            int depth,
            List<FastTreeNode> nodes,
            float[] importance)
        {
            var nodeIdx = nodes.Count;
            nodes.Add(default);

            var cols = x.GetView().GetDim(1);
            var count = end - start;

            if (depth >= _maxDepth || count < 2)
            {
                nodes[nodeIdx] = new FastTreeNode { IsLeaf = true, Value = CalculateMean(x, y, rows, start, end) };

                return nodeIdx;
            }

            var featureIdx = Random.Shared.Next(cols);
            var threshold = GetRandomThreshold(x, rows, start, end, featureIdx, cols);
            var split = Partition(x, rows, start, end, featureIdx, threshold, cols);

            // A split that leaves every row on one side separates nothing, and recursing on it would build
            // two identical children — which is exactly what this class used to do at every node. A constant
            // column produces it every time, so it is the common case rather than a corner one.
            if (split == start || split == end)
            {
                nodes[nodeIdx] = new FastTreeNode { IsLeaf = true, Value = CalculateMean(x, y, rows, start, end) };

                return nodeIdx;
            }

            importance[featureIdx] += 1.0f / (depth + 1);

#pragma warning disable OVERFIT022 // Bounded: depth >= _maxDepth returns above, and _maxDepth <= MaxAllowedDepth (64) is enforced in the constructor.
            var leftIdx = BuildRecursive(x, y, rows, start, split, depth + 1, nodes, importance);
            var rightIdx = BuildRecursive(x, y, rows, split, end, depth + 1, nodes, importance);
#pragma warning restore OVERFIT022

            nodes[nodeIdx] = new FastTreeNode
            {
                IsLeaf = false,
                FeatureIndex = featureIdx,
                Threshold = threshold,
                LeftChildIndex = leftIdx,
                RightChildIndex = rightIdx
            };

            return nodeIdx;
        }

        /// <summary>
        /// Moves every row at or below the threshold to the front of the slice and returns where the two
        /// halves meet. The same comparison <see cref="Traverse"/> uses, or a row would train on one side and
        /// be predicted from the other.
        /// </summary>
        private static int Partition(
            FastTensor<float> x, int[] rows, int start, int end, int feature, float threshold, int cols)
        {
            var span = x.GetView().AsReadOnlySpan();
            var lo = start;

            for (var i = start; i < end; i++)
            {
                if (span[(rows[i] * cols) + feature] > threshold)
                {
                    continue;
                }

                (rows[lo], rows[i]) = (rows[i], rows[lo]);
                lo++;
            }

            return lo;
        }

        /// <summary>The threshold is drawn from this node's own rows, not from the whole column.</summary>
        private static float GetRandomThreshold(
            FastTensor<float> x, int[] rows, int start, int end, int col, int cols)
        {
            var span = x.GetView().AsReadOnlySpan();
            var min = span[(rows[start] * cols) + col];
            var max = min;

            for (var i = start + 1; i < end; i++)
            {
                var v = span[(rows[i] * cols) + col];

                if (v < min)
                {
                    min = v;
                }

                if (v > max)
                {
                    max = v;
                }
            }

            return min + ((max - min) * Random.Shared.NextSingle());
        }

        /// <summary>
        /// The mean of this node's rows. It used to be the mean of the entire target column, which is why
        /// every leaf of every tree returned the same number.
        /// </summary>
        private static float CalculateMean(
            FastTensor<float> x, FastTensor<float> y, int[] rows, int start, int end)
        {
            var span = y.GetView().AsReadOnlySpan();
            var rowCount = x.GetView().GetDim(0);

            if (span.Length == 0 || rowCount == 0 || end <= start)
            {
                return 0f;
            }

            // Targets arrive as [n] or [n, 1]; anything wider is read by its first column, which is what the
            // callers in Prepare/ produce.
            var stride = span.Length / rowCount;

            if (stride == 0)
            {
                return 0f;
            }

            double sum = 0;

            for (var i = start; i < end; i++)
            {
                sum += span[rows[i] * stride];
            }

            return (float)(sum / (end - start));
        }

        public void Dispose() => _forest.Clear();
    }
}
