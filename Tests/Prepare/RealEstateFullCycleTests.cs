using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.Data.Contracts;
using DevOnBike.Overfit.Data.Prepare;
using DevOnBike.Overfit.Optimizers;
using Xunit.Abstractions;
namespace DevOnBike.Overfit.Tests.Prepare
{
    public class RealEstateFullCycleTests
    {
        private readonly ITestOutputHelper _output;
        
        public RealEstateFullCycleTests(ITestOutputHelper output) => _output = output;

        [Fact]
        public void Training_Analysis_And_Prediction_Should_Work_EndToEnd()
        {
            // --- 1. PRZYGOTOWANIE DANYCH ---
            var rawData = GenerateDummyData(500);
            var schema = CreatePropertySchema();
            var converter = new TabularToTensorConverter<PropertyData>(schema);
            converter.Fit(rawData);
            var (rawX, rawY) = converter.Transform(rawData);

            var numericIndices = new List<int> { 0, 1, 4 };
            var pipeline = new DataPipeline()
                .AddLayer(new TechnicalSanityLayer())
                .AddLayer(new AnomalyFilterLayer())
                .AddLayer(new RobustScalingLayer(numericIndices));

            using var cleanContext = pipeline.Execute(rawX, rawY);

            // Skalowanie celu (Cena / 100,000)
            var targetSpan = cleanContext.Targets.AsSpan();
            float meanTarget = 0;
            for (var i = 0; i < targetSpan.Length; i++)
            {
                targetSpan[i] /= 100000f;
                meanTarget += targetSpan[i];
            }
            meanTarget /= targetSpan.Length;

            int inputSize = cleanContext.Features.GetDim(1);

            // --- 2. ARCHITEKTURA MODELU ---
            var w1 = new AutogradNode(new FastTensor<float>(inputSize, 32).Randomize(MathF.Sqrt(2.0f / inputSize)), true);
            var b1 = new AutogradNode(new FastTensor<float>(1, 32).Fill(0.1f), true); //
            var w2 = new AutogradNode(new FastTensor<float>(32, 1).Randomize(MathF.Sqrt(2.0f / 32)), true);
            var b2 = new AutogradNode(new FastTensor<float>(1, 1).Fill(meanTarget), true);

            var inputNode = new AutogradNode(cleanContext.Features, false);
            var targetNode = new AutogradNode(cleanContext.Targets, false);

            // --- 3. TRENING ---
            var parameters = new[] { w1, b1, w2, b2 };
            var optimizer = new Adam(parameters, learningRate: 0.005f); //
            var graph = ComputationGraph.Active;

            _output.WriteLine("=== START TRENINGU ===");
            for (var epoch = 0; epoch <= 400; epoch++)
            {
                graph.Reset(); //
                optimizer.ZeroGrad(); //

                using var l1 = TensorMath.ReLU(TensorMath.AddBias(TensorMath.MatMul(inputNode, w1), b1));
                using var prediction = TensorMath.AddBias(TensorMath.MatMul(l1, w2), b2);
                using var lossNode = TensorMath.MSELoss(prediction, targetNode); //

                graph.Backward(lossNode); //
                optimizer.Step(); //

                if (epoch % 100 == 0) _output.WriteLine($"Epoka {epoch:D3} | Loss: {lossNode.Forward():F6}");
            }

            // --- 4. ANALIZA ISTOTNOŚCI (PFI) ---
            var featureNames = new List<string> {
                "Powierzchnia", "Pietro", "CzyKamienica", "CzyMaKomorke", "PowKomorki",
                "M_Krakow", "M_Warszawa", "A_Premium", "A_Standard"
            };

            _output.WriteLine("\n=== ISTOTNOŚĆ CECH (Wpływ na wynik %) ===");
            CalculateImportance(w1, b1, w2, b2, cleanContext.Features, cleanContext.Targets, featureNames);

            // --- 5. ANALIZA KORELACJI (%) ---
            _output.WriteLine("\n=== KORELACJE MIĘDZY CECHAMI (%) ===");
            CalculateCorrelations(cleanContext.Features, featureNames);

            // --- 6. ANALIZA KIERUNKU WPŁYWU (Sensitivity) ---
            _output.WriteLine("\n=== KIERUNEK WPŁYWU NA CENĘ ===");
            AnalyzeDirection(w1, b1, w2, b2, cleanContext.Features, featureNames);

            // Czyszczenie
            foreach (var p in parameters) p.Dispose();
        }

        // --- LOGIKA ANALITYCZNA ---

        private void CalculateImportance(AutogradNode w1, AutogradNode b1, AutogradNode w2, AutogradNode b2,
                                        FastTensor<float> x, FastTensor<float> y, List<string> names)
        {
            ComputationGraph.Active.IsRecording = false; //
            float baselineLoss = GetEvalLoss(w1, b1, w2, b2, x, y);
            var impacts = new float[x.GetDim(1)];

            for (int c = 0; c < x.GetDim(1); c++)
            {
                using var shuffledX = FastTensor<float>.SameShape(x, false); //
                x.AsSpan().CopyTo(shuffledX.AsSpan());
                ShuffleColumn(shuffledX, c); // Permutacja kolumny

                float newLoss = GetEvalLoss(w1, b1, w2, b2, shuffledX, y);
                impacts[c] = MathF.Max(0, newLoss - baselineLoss);
            }

            float total = impacts.Sum();
            foreach (var r in impacts.Select((val, idx) => (Name: names[idx], Val: val)).OrderByDescending(r => r.Val))
            {
                float pct = (total > 0) ? (r.Val / total) * 100 : 0;
                _output.WriteLine($"{r.Name,-15} : {pct,6:F1}%");
            }
        }

        private void AnalyzeDirection(AutogradNode w1, AutogradNode b1, AutogradNode w2, AutogradNode b2,
                                     FastTensor<float> x, List<string> names)
        {
            // Sprawdzamy jak zmiana cechy o +0.1 (znormalizowane) wpływa na cenę
            for (int c = 0; c < x.GetDim(1); c++)
            {
                using var testX = FastTensor<float>.SameShape(x, false);
                x.AsSpan().CopyTo(testX.AsSpan());

                float priceBefore = GetAveragePrediction(w1, b1, w2, b2, testX);
                for (int r = 0; r < testX.GetDim(0); r++) testX.AsSpan()[r * x.GetDim(1) + c] += 0.1f;
                float priceAfter = GetAveragePrediction(w1, b1, w2, b2, testX);

                string direction = (priceAfter > priceBefore) ? "[+]" : "[-]";
                _output.WriteLine($"{names[c],-15} : {direction} (Zmiana o {((priceAfter - priceBefore) * 100000f):N0} PLN)");
            }
        }

        private void CalculateCorrelations(FastTensor<float> x, List<string> names)
        {
            int cols = x.GetDim(1);
            for (int i = 0; i < cols; i++)
            {
                for (int j = i + 1; j < cols; j++)
                {
                    float r = Pearson(x, i, j);
                    if (MathF.Abs(r) > 0.3f)
                        _output.WriteLine($"{names[i],-12} <-> {names[j],-12} : {r * 100,6:F1}%");
                }
            }
        }

        private float GetEvalLoss(AutogradNode w1, AutogradNode b1, AutogradNode w2, AutogradNode b2, FastTensor<float> x, FastTensor<float> y)
        {
            var inNode = new AutogradNode(x, false);
            var target = new AutogradNode(y, false);
            using var l1 = TensorMath.ReLU(TensorMath.AddBias(TensorMath.MatMul(inNode, w1), b1));
            using var pred = TensorMath.AddBias(TensorMath.MatMul(l1, w2), b2);
            using var loss = TensorMath.MSELoss(pred, target);
            return loss.Forward(); //
        }

        private float GetAveragePrediction(AutogradNode w1, AutogradNode b1, AutogradNode w2, AutogradNode b2, FastTensor<float> x)
        {
            var inNode = new AutogradNode(x, false);
            using var l1 = TensorMath.ReLU(TensorMath.AddBias(TensorMath.MatMul(inNode, w1), b1));
            using var pred = TensorMath.AddBias(TensorMath.MatMul(l1, w2), b2);
            float sum = 0;
            var span = pred.Data.AsSpan();
            for (int i = 0; i < span.Length; i++) sum += span[i];
            return sum / span.Length;
        }

        private float Pearson(FastTensor<float> t, int colA, int colB)
        {
            int rows = t.GetDim(0), cols = t.GetDim(1);
            var s = t.AsSpan(); //
            float sumA = 0, sumB = 0, sumAB = 0, sumA2 = 0, sumB2 = 0;
            for (int r = 0; r < rows; r++)
            {
                float a = s[r * cols + colA], b = s[r * cols + colB];
                sumA += a; sumB += b; sumAB += a * b; sumA2 += a * a; sumB2 += b * b;
            }
            float num = (rows * sumAB) - (sumA * sumB);
            float den = MathF.Sqrt((rows * sumA2 - sumA * sumA) * (rows * sumB2 - sumB * sumB));
            return den == 0 ? 0 : num / den;
        }

        private void ShuffleColumn(FastTensor<float> t, int colIdx)
        {
            int rows = t.GetDim(0), cols = t.GetDim(1);
            var span = t.AsSpan();
            for (int i = rows - 1; i > 0; i--)
            {
                int j = Random.Shared.Next(i + 1);
                float temp = span[i * cols + colIdx];
                span[i * cols + colIdx] = span[j * cols + colIdx];
                span[j * cols + colIdx] = temp;
            }
        }

        // --- GENEROWANIE DANYCH ---
        private List<PropertyData> GenerateDummyData(int count)
        {
            var data = new List<PropertyData>();
            var rnd = new Random(42);
            for (var i = 0; i < count; i++)
            {
                float pow = rnd.Next(25, 150);
                var cena = (pow * 11500f) + rnd.Next(-15000, 15000);
                data.Add(new PropertyData
                {
                    Powierzchnia = pow,
                    Pietro = rnd.Next(0, 10),
                    CzyKamienica = rnd.NextDouble() > 0.7,
                    CzyMaKomorke = true,
                    PowKomorki = rnd.Next(2, 10),
                    Miasto = rnd.NextDouble() > 0.5 ? "Warszawa" : "Krakow",
                    NazwaAgencji = "Premium",
                    Cena = cena
                });
            }
            return data;
        }

        private TableSchema CreatePropertySchema() => new TableSchema
        {
            Features = [
                new() { Name = "Powierzchnia", Type = ColumnType.Numeric },
                new() { Name = "Pietro", Type = ColumnType.Numeric },
                new() { Name = "CzyKamienica", Type = ColumnType.Binary },
                new() { Name = "CzyMaKomorke", Type = ColumnType.Binary },
                new() { Name = "PowKomorki", Type = ColumnType.Numeric },
                new() { Name = "Miasto", Type = ColumnType.Categorical },
                new() { Name = "NazwaAgencji", Type = ColumnType.Categorical }
            ],
            Target = new ColumnDefinition { Name = "Cena", Type = ColumnType.Numeric }
        };
    }
}