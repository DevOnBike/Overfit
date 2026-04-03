using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.Data.Contracts;

namespace DevOnBike.Overfit.Data
{
    public class ModelInterpreter
    {
        private readonly TableSchema _schema;
        private readonly List<string> _featureNames;

        public ModelInterpreter(TableSchema schema, List<string> expandedFeatureNames)
        {
            _schema = schema;
            _featureNames = expandedFeatureNames;
        }

        // --- ANALIZA ISTOTNOŚCI (PFI) ---
        public void PrintFeatureImportance(
            AutogradNode w1,
            AutogradNode b1,
            AutogradNode w2,
            AutogradNode b2,
            FastTensor<float> x,
            FastTensor<float> y)
        {
            ComputationGraph.Active.IsRecording = false; // Tryb inferencji

            // 1. Strata bazowa
            var baselineLoss = CalculateLoss(w1, b1, w2, b2, x, y);
            var scores = new float[x.GetDim(1)];

            // 2. Permutacja każdej kolumny
            for (var c = 0; c < x.GetDim(1); c++)
            {
                using var shuffledX = CloneAndShuffleColumn(x, c);
                var shuffledLoss = CalculateLoss(w1, b1, w2, b2, shuffledX, y);
                
                scores[c] = MathF.Max(0, shuffledLoss - baselineLoss);
            }

            // 3. Obliczanie procentowe
            var total = scores.Sum();
            Console.WriteLine("\n=== ISTOTNOŚĆ CECH (PFI) ===");
            var sorted = scores.Select((s, i) => new
                {
                    Name = _featureNames[i],
                    Val = s
                })
                .OrderByDescending(x => x.Val);

            foreach (var item in sorted)
            {
                var pct = total > 0 ? (item.Val / total) * 100 : 0;
                Console.WriteLine($"{item.Name,-25} : {pct,6:F1}% (score: {item.Val:F4})");
            }
        }

        // --- ANALIZA KORELACJI ---
        public void PrintCorrelations(FastTensor<float> features)
        {
            var cols = features.GetDim(1);
            var rows = features.GetDim(0);
            var span = features.AsSpan();

            Console.WriteLine("\n=== SILNE KORELACJE MIĘDZY CECHAMI (>30%) ===");

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

        // Pomocnicze obliczanie straty bez nagrywania grafu
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
            using var l1 = TensorMath.ReLU(TensorMath.AddBias(TensorMath.MatMul(input, w1), b1));
            using var pred = TensorMath.AddBias(TensorMath.MatMul(l1, w2), b2);
            using var loss = TensorMath.MSELoss(pred, target);
            
            return loss.Data[0]; //
        }

        private float CalculatePearson(FastTensor<float> t, int colA, int colB)
        {
            var rows = t.GetDim(0);
            var cols = t.GetDim(1);
            var s = t.AsSpan();

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

            var num = (rows * sumAB) - (sumA * sumB);
            var den = MathF.Sqrt((rows * sumA2 - sumA * sumA) * (rows * sumB2 - sumB * sumB));
            
            return den == 0 ? 0 : num / den;
        }

        private FastTensor<float> CloneAndShuffleColumn(FastTensor<float> src, int colIdx)
        {
            var res = FastTensor<float>.SameShape(src, false);
            src.AsSpan().CopyTo(res.AsSpan());

            var rows = res.GetDim(0);
            var cols = res.GetDim(1);
            var span = res.AsSpan();

            // Fisher-Yates Shuffle tylko dla jednej kolumny
            for (var i = rows - 1; i > 0; i--)
            {
                var j = Random.Shared.Next(i + 1);
                
                (span[i * cols + colIdx], span[j * cols + colIdx]) = (span[j * cols + colIdx], span[i * cols + colIdx]);
            }
            return res;
        }
    }
}