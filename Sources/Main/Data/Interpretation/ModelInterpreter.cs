// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Data.Contracts;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Data.Interpretation
{
    public class ModelInterpreter
    {
        private readonly List<string> _featureNames;
        private readonly TableSchema _schema;

        public ModelInterpreter(TableSchema schema, List<string> expandedFeatureNames)
        {
            _schema = schema;
            _featureNames = expandedFeatureNames;
        }

        public void PrintFeatureImportance(
            AutogradNode w1,
            AutogradNode b1,
            AutogradNode w2,
            AutogradNode b2,
            FastTensor<float> x,
            FastTensor<float> y)
        {
            var baselineLoss = CalculateLoss(w1, b1, w2, b2, x, y);
            var scores = new float[x.GetView().GetDim(1)];

            for (var c = 0; c < x.GetView().GetDim(1); c++)
            {
                using var shuffledX = CloneAndShuffleColumn(x, c);
                var shuffledLoss = CalculateLoss(w1, b1, w2, b2, shuffledX, y);

                scores[c] = MathF.Max(0, shuffledLoss - baselineLoss);
            }

            var total = 0f;
            var combined = new (float Score, string Name)[scores.Length];

            for (var i = 0; i < scores.Length; i++)
            {
                total += scores[i];
                combined[i] = (scores[i], _featureNames[i]);
            }

            Array.Sort(combined);

            Console.WriteLine("\n=== FEATURE IMPORTANCE (PFI) ===");

            for (var i = combined.Length - 1; i >= 0; i--)
            {
                var item = combined[i];
                var pct = total > 0 ? item.Score / total * 100 : 0;

                Console.WriteLine($"{item.Name,-25} : {pct,6:F1}% (score: {item.Score:F4})");
            }
        }

        public void PrintCorrelations(FastTensor<float> features)
        {
            var cols = features.GetView().GetDim(1);

            Console.WriteLine("\n=== STRONG FEATURE CORRELATIONS (>30%) ===");

            for (var i = 0; i < cols; i++)
            {
                for (var j = i + 1; j < cols; j++)
                {
                    var r = CalculatePearson(features, i, j);
                    if (MathF.Abs(r) > 0.3f)
                    {
                        Console.WriteLine($"{_featureNames[i],-20} <-> {_featureNames[j],-20} : {r * 100,6:F1}%");
                    }
                }
            }
        }

        private float CalculateLoss(
            AutogradNode w1,
            AutogradNode b1,
            AutogradNode w2,
            AutogradNode b2,
            FastTensor<float> x,
            FastTensor<float> y)
        {
            var input = new AutogradNode(x, false);
            var target = new AutogradNode(y, false);

            using var l1 = TensorMath.ReLU(null, TensorMath.AddBias(null, TensorMath.MatMul(null, input, w1), b1));
            using var pred = TensorMath.AddBias(null, TensorMath.MatMul(null, l1, w2), b2);
            using var loss = TensorMath.MSELoss(null, pred, target);

            return loss.DataView.AsReadOnlySpan()[0];
        }

        private float CalculatePearson(FastTensor<float> t, int colA, int colB)
        {
            var view = t.GetView();
            var rows = view.GetDim(0);
            var cols = view.GetDim(1);
            var s = view.AsReadOnlySpan();

            float sumA = 0, sumB = 0, sumAB = 0, sumA2 = 0, sumB2 = 0;

            for (var r = 0; r < rows; r++)
            {
                var a = s[r * cols + colA];
                var b = s[r * cols + colB];
                sumA += a;
                sumB += b;
                sumAB += a * b;
                sumA2 += a * a;
                sumB2 += b * b;
            }

            var num = rows * sumAB - sumA * sumB;
            var den = MathF.Sqrt((rows * sumA2 - sumA * sumA) * (rows * sumB2 - sumB * sumB));

            return den == 0 ? 0 : num / den;
        }

        private FastTensor<float> CloneAndShuffleColumn(FastTensor<float> src, int colIdx)
        {
            var res = FastTensor<float>.SameShape(src, clearMemory: false);
            var srcView = src.GetView();
            var resView = res.GetView();

            srcView.AsReadOnlySpan().CopyTo(resView.AsSpan());

            var rows = resView.GetDim(0);
            var cols = resView.GetDim(1);
            var span = resView.AsSpan();

            for (var i = rows - 1; i > 0; i--)
            {
                var j = Random.Shared.Next(i + 1);
                (span[i * cols + colIdx], span[j * cols + colIdx]) = (span[j * cols + colIdx], span[i * cols + colIdx]);
            }
            return res;
        }
    }
}