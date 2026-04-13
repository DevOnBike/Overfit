// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.Data.Contracts;

namespace DevOnBike.Overfit.Data
{
    public sealed class FastRandomForest : IDisposable
    {
        private readonly int _maxDepth;
        private readonly int _numTrees;
        private readonly List<FastTreeNode[]> _forest = [];

        public FastRandomForest(int numTrees = 50, int maxDepth = 10)
        {
            _numTrees = numTrees;
            _maxDepth = maxDepth;
        }

        public float[] TrainAndGetImportance(FastTensor<float> x, FastTensor<float> y)
        {
            var cols = x.GetView().GetDim(1);
            var totalImportance = new float[cols];
            var lockObj = new object();

            _forest.Clear();

            var trees = new FastTreeNode[_numTrees][];

            Parallel.For(0, _numTrees, t => {
                var localImportance = new float[cols];
                var nodes = new List<FastTreeNode>();

                BuildRecursive(x, y, 0, nodes, localImportance);

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

            while (true)
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
        }

        private int BuildRecursive(FastTensor<float> x, FastTensor<float> y, int depth, List<FastTreeNode> nodes, float[] importance)
        {
            var nodeIdx = nodes.Count;
            nodes.Add(default);

            var rows = x.GetView().GetDim(0);
            var cols = x.GetView().GetDim(1);

            if (depth >= _maxDepth || rows < 2)
            {
                nodes[nodeIdx] = new FastTreeNode { IsLeaf = true, Value = CalculateMean(y) };
                return nodeIdx;
            }

            var featureIdx = Random.Shared.Next(cols);
            var threshold = GetRandomThreshold(x, featureIdx);

            importance[featureIdx] += 1.0f / (depth + 1);

            var leftIdx = BuildRecursive(x, y, depth + 1, nodes, importance);
            var rightIdx = BuildRecursive(x, y, depth + 1, nodes, importance);

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

        private float GetRandomThreshold(FastTensor<float> x, int col)
        {
            var rows = x.GetView().GetDim(0);
            var cols = x.GetView().GetDim(1);
            var span = x.GetView().AsReadOnlySpan();
            float min = span[col], max = span[col];

            for (var r = 1; r < rows; r++)
            {
                var v = span[r * cols + col];
                if (v < min)
                {
                    min = v;
                }
                if (v > max)
                {
                    max = v;
                }
            }
            return min + (max - min) * Random.Shared.NextSingle();
        }

        private float CalculateMean(FastTensor<float> y)
        {
            var span = y.GetView().AsReadOnlySpan();
            if (span.Length == 0)
            {
                return 0f;
            }

            double sum = 0;
            for (var i = 0; i < span.Length; i++)
            {
                sum += span[i];
            }

            return (float)(sum / span.Length);
        }

        public void Dispose() => _forest.Clear();
    }
}