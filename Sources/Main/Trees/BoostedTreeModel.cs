// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Runtime;

namespace DevOnBike.Overfit.Trees
{
    /// <summary>
    /// A gradient-boosted decision-tree ensemble, ready for zero-allocation inference. All trees are
    /// concatenated into flat, cache-friendly arrays (one allocation per column for the whole ensemble);
    /// child links are pre-resolved to global node indices so traversal is a tight pointer-chase with no
    /// per-tree indirection. Built by <see cref="XgboostModelLoader"/>; prediction allocates nothing.
    ///
    /// Node layout (parallel arrays, indexed by global node id):
    ///   • leaf node      ⇒ <c>Left[i] &lt; 0</c>; its value is <c>Threshold[i]</c>.
    ///   • internal node  ⇒ go left when <c>feature[FeatureIndex[i]] &lt; Threshold[i]</c>, else right;
    ///                       a missing (NaN) feature follows <c>DefaultLeft[i]</c>.
    /// </summary>
    public sealed class BoostedTreeModel
    {
        private readonly int[] _featureIndex;
        private readonly float[] _threshold;
        private readonly int[] _left;
        private readonly int[] _right;
        private readonly byte[] _defaultLeft;
        private readonly int[] _treeRoot;
        private readonly int[] _treeGroup;
        private readonly float[] _baseMargin;

        public BoostedTreeModel(
            int numFeatures,
            int numGroups,
            TreeObjective objective,
            float[] baseMargin,
            int[] featureIndex,
            float[] threshold,
            int[] left,
            int[] right,
            byte[] defaultLeft,
            int[] treeRoot,
            int[] treeGroup)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numFeatures);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numGroups);
            ArgumentNullException.ThrowIfNull(baseMargin);
            ArgumentNullException.ThrowIfNull(featureIndex);
            ArgumentNullException.ThrowIfNull(threshold);
            ArgumentNullException.ThrowIfNull(left);
            ArgumentNullException.ThrowIfNull(right);
            ArgumentNullException.ThrowIfNull(defaultLeft);
            ArgumentNullException.ThrowIfNull(treeRoot);
            ArgumentNullException.ThrowIfNull(treeGroup);

            if (baseMargin.Length != numGroups)
            {
                throw new ArgumentException(
                    $"baseMargin length {baseMargin.Length} must equal numGroups {numGroups}.",
                    nameof(baseMargin));
            }

            if (treeGroup.Length != treeRoot.Length)
            {
                throw new ArgumentException(
                    $"treeGroup length {treeGroup.Length} must equal treeRoot length {treeRoot.Length}.",
                    nameof(treeGroup));
            }

            NumFeatures = numFeatures;
            NumGroups = numGroups;
            Objective = objective;
            _baseMargin = baseMargin;
            _featureIndex = featureIndex;
            _threshold = threshold;
            _left = left;
            _right = right;
            _defaultLeft = defaultLeft;
            _treeRoot = treeRoot;
            _treeGroup = treeGroup;
        }

        /// <summary>Number of input features each row must supply.</summary>
        public int NumFeatures
        {
            get;
        }

        /// <summary>Number of output groups: 1 for regression / binary, <c>num_class</c> for multiclass.</summary>
        public int NumGroups
        {
            get;
        }

        /// <summary>Number of trees in the ensemble.</summary>
        public int NumTrees => _treeRoot.Length;

        /// <summary>The output transform applied to the summed raw margins.</summary>
        public TreeObjective Objective
        {
            get;
        }

        /// <summary>
        /// Single-output convenience for regression / binary models (<see cref="NumGroups"/> == 1):
        /// returns the transformed score (probability for <see cref="TreeObjective.Logistic"/>,
        /// raw value for <see cref="TreeObjective.Identity"/>).
        /// </summary>
        public float Predict(ReadOnlySpan<float> features)
        {
            if (NumGroups != 1)
            {
                throw new OverfitRuntimeException(
                    $"Predict(features) returns a scalar but this model has {NumGroups} output groups. " +
                    "Use Predict(features, output) with a span of length NumGroups.");
            }

            Span<float> one = stackalloc float[1];
            Predict(features, one);
            return one[0];
        }

        /// <summary>
        /// Fills <paramref name="output"/> (length <see cref="NumGroups"/>) with the transformed scores:
        /// probabilities for logistic / softmax models, raw values for regression.
        /// </summary>
        public void Predict(ReadOnlySpan<float> features, Span<float> output)
        {
            PredictRawMargins(features, output);

            switch (Objective)
            {
                case TreeObjective.Logistic:
                    output[0] = Sigmoid(output[0]);
                    break;

                case TreeObjective.Softmax:
                    Softmax(output);
                    break;

                case TreeObjective.Identity:
                default:
                    break;
            }
        }

        /// <summary>
        /// Fills <paramref name="margins"/> (length <see cref="NumGroups"/>) with the raw, pre-transform
        /// margins (base score + summed leaf weights per group) — the equivalent of XGBoost's
        /// <c>output_margin=True</c>.
        /// </summary>
        public void PredictRawMargins(ReadOnlySpan<float> features, Span<float> margins)
        {
            if (features.Length < NumFeatures)
            {
                throw new ArgumentException(
                    $"features length {features.Length} is smaller than NumFeatures {NumFeatures}.",
                    nameof(features));
            }

            if (margins.Length < NumGroups)
            {
                throw new ArgumentException(
                    $"margins length {margins.Length} is smaller than NumGroups {NumGroups}.",
                    nameof(margins));
            }

            for (var g = 0; g < NumGroups; g++)
            {
                margins[g] = _baseMargin[g];
            }

            var treeCount = _treeRoot.Length;

            for (var t = 0; t < treeCount; t++)
            {
                var node = _treeRoot[t];

                // Internal node ⇒ Left[node] >= 0. Walk until we hit a leaf.
                while (_left[node] >= 0)
                {
                    var value = features[_featureIndex[node]];
                    bool goLeft;

                    if (float.IsNaN(value))
                    {
                        goLeft = _defaultLeft[node] != 0;
                    }
                    else
                    {
                        goLeft = value < _threshold[node];
                    }

                    node = goLeft ? _left[node] : _right[node];
                }

                margins[_treeGroup[t]] += _threshold[node];
            }
        }

        /// <summary>
        /// Scores a contiguous batch: <paramref name="featuresFlat"/> is <paramref name="rowCount"/> rows of
        /// <see cref="NumFeatures"/> each (row-major); <paramref name="outputFlat"/> receives
        /// <paramref name="rowCount"/> × <see cref="NumGroups"/> transformed scores. Sequential and
        /// zero-allocation — parallelism is a separate (measured) concern.
        /// </summary>
        public void PredictBatch(ReadOnlySpan<float> featuresFlat, int rowCount, Span<float> outputFlat)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(rowCount);

            if (featuresFlat.Length < (long)rowCount * NumFeatures)
            {
                throw new ArgumentException(
                    $"featuresFlat needs {(long)rowCount * NumFeatures} elements for {rowCount} rows.",
                    nameof(featuresFlat));
            }

            if (outputFlat.Length < (long)rowCount * NumGroups)
            {
                throw new ArgumentException(
                    $"outputFlat needs {(long)rowCount * NumGroups} elements for {rowCount} rows.",
                    nameof(outputFlat));
            }

            for (var r = 0; r < rowCount; r++)
            {
                var row = featuresFlat.Slice(r * NumFeatures, NumFeatures);
                var dst = outputFlat.Slice(r * NumGroups, NumGroups);
                Predict(row, dst);
            }
        }

        /// <summary>
        /// Parallel <see cref="PredictBatch"/> — rows are independent, so the batch fans out across the
        /// <see cref="OverfitParallel"/> worker pool with no reassociation: the output is bit-identical to
        /// the sequential path (its parity guard). Zero managed allocation per call (the flat model arrays
        /// and the caller's buffers are pinned; the body runs over raw pointers).
        /// </summary>
        public void PredictBatchParallel(ReadOnlySpan<float> featuresFlat, int rowCount, Span<float> outputFlat)
            => PredictBatchParallel(featuresFlat, rowCount, outputFlat, branchless: true);

        /// <summary>
        /// A/B hook for the perf measurement: <paramref name="branchless"/> selects the branchless
        /// traversal body (production default) vs the original branchy one. Both produce bit-identical
        /// output — the lever is purely how the per-node go-left decision and child select compile.
        /// </summary>
        internal unsafe void PredictBatchParallel(
            ReadOnlySpan<float> featuresFlat,
            int rowCount,
            Span<float> outputFlat,
            bool branchless)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(rowCount);

            if (featuresFlat.Length < (long)rowCount * NumFeatures)
            {
                throw new ArgumentException(
                    $"featuresFlat needs {(long)rowCount * NumFeatures} elements for {rowCount} rows.",
                    nameof(featuresFlat));
            }

            if (outputFlat.Length < (long)rowCount * NumGroups)
            {
                throw new ArgumentException(
                    $"outputFlat needs {(long)rowCount * NumGroups} elements for {rowCount} rows.",
                    nameof(outputFlat));
            }

            if (rowCount == 0)
            {
                return;
            }

            fixed (float* features = featuresFlat, output = outputFlat, threshold = _threshold, baseMargin = _baseMargin)
            fixed (int* featureIndex = _featureIndex, left = _left, right = _right, treeRoot = _treeRoot, treeGroup = _treeGroup)
            fixed (byte* defaultLeft = _defaultLeft)
            {
                var context = new BatchContext
                {
                    Features = features,
                    Output = output,
                    FeatureIndex = featureIndex,
                    Threshold = threshold,
                    Left = left,
                    Right = right,
                    DefaultLeft = defaultLeft,
                    TreeRoot = treeRoot,
                    TreeGroup = treeGroup,
                    BaseMargin = baseMargin,
                    NumFeatures = NumFeatures,
                    NumGroups = NumGroups,
                    NumTrees = _treeRoot.Length,
                    Objective = Objective
                };

                // Grain of 256 rows/worker — below 512 rows it runs inline (no dispatch).
                if (branchless)
                {
                    OverfitParallel.For(0, rowCount, 256, &ScoreRowsBranchless, &context);
                }
                else
                {
                    OverfitParallel.For(0, rowCount, 256, &ScoreRows, &context);
                }
            }
        }

        private static unsafe void ScoreRows(int rowStart, int rowEnd, void* context)
        {
            ref var ctx = ref Unsafe.AsRef<BatchContext>(context);

            for (var r = rowStart; r < rowEnd; r++)
            {
                var features = ctx.Features + (long)r * ctx.NumFeatures;
                var output = ctx.Output + (long)r * ctx.NumGroups;

                for (var g = 0; g < ctx.NumGroups; g++)
                {
                    output[g] = ctx.BaseMargin[g];
                }

                for (var t = 0; t < ctx.NumTrees; t++)
                {
                    var node = ctx.TreeRoot[t];

                    while (ctx.Left[node] >= 0)
                    {
                        var value = features[ctx.FeatureIndex[node]];
                        bool goLeft;

                        if (float.IsNaN(value))
                        {
                            goLeft = ctx.DefaultLeft[node] != 0;
                        }
                        else
                        {
                            goLeft = value < ctx.Threshold[node];
                        }

                        node = goLeft ? ctx.Left[node] : ctx.Right[node];
                    }

                    output[ctx.TreeGroup[t]] += ctx.Threshold[node];
                }

                Transform(output, ctx.NumGroups, ctx.Objective);
            }
        }

        // Branchless variant of ScoreRows: the per-node go-left decision is a bitwise bool expression
        // (no NaN if-branch) and the child select is arithmetic (no data-dependent branch on the ~50/50
        // split outcome — the dominant misprediction source in tree traversal). Bit-identical to ScoreRows.
        private static unsafe void ScoreRowsBranchless(int rowStart, int rowEnd, void* context)
        {
            ref var ctx = ref Unsafe.AsRef<BatchContext>(context);

            for (var r = rowStart; r < rowEnd; r++)
            {
                var features = ctx.Features + (long)r * ctx.NumFeatures;
                var output = ctx.Output + (long)r * ctx.NumGroups;

                for (var g = 0; g < ctx.NumGroups; g++)
                {
                    output[g] = ctx.BaseMargin[g];
                }

                for (var t = 0; t < ctx.NumTrees; t++)
                {
                    var node = ctx.TreeRoot[t];
                    int leftChild;

                    while ((leftChild = ctx.Left[node]) >= 0)
                    {
                        var value = features[ctx.FeatureIndex[node]];

                        // value < threshold is false for NaN (any compare with NaN is false), so a NaN
                        // falls through to the default-direction term. Non-short-circuit | / & ⇒ no branch.
                        var goLeft = (value < ctx.Threshold[node]) | (float.IsNaN(value) & (ctx.DefaultLeft[node] != 0));
                        var goLeftBit = Unsafe.As<bool, byte>(ref goLeft);
                        var rightChild = ctx.Right[node];

                        // goLeftBit == 1 ⇒ leftChild, == 0 ⇒ rightChild. Arithmetic select, no branch.
                        node = rightChild + ((leftChild - rightChild) * goLeftBit);
                    }

                    output[ctx.TreeGroup[t]] += ctx.Threshold[node];
                }

                Transform(output, ctx.NumGroups, ctx.Objective);
            }
        }

        private static unsafe void Transform(float* output, int numGroups, TreeObjective objective)
        {
            switch (objective)
            {
                case TreeObjective.Logistic:
                    output[0] = Sigmoid(output[0]);
                    break;

                case TreeObjective.Softmax:
                    var max = output[0];

                    for (var i = 1; i < numGroups; i++)
                    {
                        if (output[i] > max)
                        {
                            max = output[i];
                        }
                    }

                    var sum = 0f;

                    for (var i = 0; i < numGroups; i++)
                    {
                        var e = MathF.Exp(output[i] - max);
                        output[i] = e;
                        sum += e;
                    }

                    var inv = 1f / sum;

                    for (var i = 0; i < numGroups; i++)
                    {
                        output[i] *= inv;
                    }

                    break;

                case TreeObjective.Identity:
                default:
                    break;
            }
        }

        private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

        private static void Softmax(Span<float> v)
        {
            var max = v[0];

            for (var i = 1; i < v.Length; i++)
            {
                if (v[i] > max)
                {
                    max = v[i];
                }
            }

            var sum = 0f;

            for (var i = 0; i < v.Length; i++)
            {
                var e = MathF.Exp(v[i] - max);
                v[i] = e;
                sum += e;
            }

            var inv = 1f / sum;

            for (var i = 0; i < v.Length; i++)
            {
                v[i] *= inv;
            }
        }

        // Raw-pointer view of the model + caller buffers, passed by address to the parallel body.
        // Unmanaged (pointers + scalars only) so it can be addressed with &context and pinned cheaply.
        private unsafe struct BatchContext
        {
            public float* Features;
            public float* Output;
            public int* FeatureIndex;
            public float* Threshold;
            public int* Left;
            public int* Right;
            public byte* DefaultLeft;
            public int* TreeRoot;
            public int* TreeGroup;
            public float* BaseMargin;
            public int NumFeatures;
            public int NumGroups;
            public int NumTrees;
            public TreeObjective Objective;
        }
    }
}
