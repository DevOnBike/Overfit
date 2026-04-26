// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Data.Contracts;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors.Core; // Dodano Core

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

        // Zmiana z FastTensor na AutogradNode
        public void PrintFeatureImportance(
            AutogradNode w1,
            AutogradNode b1,
            AutogradNode w2,
            AutogradNode b2,
            AutogradNode x,
            AutogradNode y)
        {
            var baselineLoss = CalculateLoss(w1, b1, w2, b2, x, y);
            var scores = new float[x.Shape.D1];

            for (var c = 0; c < x.Shape.D1; c++)
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

        // Zmiana z FastTensor na AutogradNode
        public void PrintCorrelations(AutogradNode features)
        {
            var cols = features.Shape.D1;

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

        // Zmiana z FastTensor na AutogradNode
        private float CalculateLoss(
            AutogradNode w1,
            AutogradNode b1,
            AutogradNode w2,
            AutogradNode b2,
            AutogradNode input,
            AutogradNode target)
        {
            // Ponieważ input to teraz AutogradNode, wchodzimy tu bezpośrednio, zero alokacji "using"!
            using var l1 = TensorMath.ReLU(null, TensorMath.AddBias(null, TensorMath.MatMul(null, input, w1), b1));
            using var pred = TensorMath.AddBias(null, TensorMath.MatMul(null, l1, w2), b2);
            using var loss = TensorMath.MSELoss(null, pred, target);

            return loss.DataView.AsReadOnlySpan()[0];
        }

        // Zmiana z FastTensor na AutogradNode
        private float CalculatePearson(AutogradNode t, int colA, int colB)
        {
            var rows = t.Shape.D0;
            var cols = t.Shape.D1;
            var s = t.DataView.AsReadOnlySpan();

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

        // Zmiana na zero-alloc TensorStorage
        private AutogradNode CloneAndShuffleColumn(AutogradNode src, int colIdx)
        {
            var resStorage = new TensorStorage<float>(src.Shape.Size, clearMemory: false);
            src.DataView.AsReadOnlySpan().CopyTo(resStorage.AsSpan());

            var rows = src.Shape.D0;
            var cols = src.Shape.D1;
            var span = resStorage.AsSpan();

            for (var i = rows - 1; i > 0; i--)
            {
                var j = Random.Shared.Next(i + 1);
                (span[i * cols + colIdx], span[j * cols + colIdx]) = (span[j * cols + colIdx], span[i * cols + colIdx]);
            }

            return new AutogradNode(resStorage, src.Shape, false);
        }
    }
}