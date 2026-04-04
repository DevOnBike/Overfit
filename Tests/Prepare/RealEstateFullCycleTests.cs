using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.Data.Contracts;
using DevOnBike.Overfit.Data.Prepare;
using DevOnBike.Overfit.Optimizers;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.EndToEnd
{
    public class RealEstateFullCycleTests
    {
        private readonly ITestOutputHelper _output;

        public RealEstateFullCycleTests(ITestOutputHelper output) => _output = output;

        [Fact]
        public void Training_And_Prediction_Should_Work_EndToEnd()
        {
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

            var targetSpan = cleanContext.Targets.AsSpan();
            float meanTarget = 0;
            for (var i = 0; i < targetSpan.Length; i++)
            {
                targetSpan[i] /= 100000f;
                meanTarget += targetSpan[i];
            }
            meanTarget /= targetSpan.Length;

            var inputSize = cleanContext.Features.GetDim(1);

            var heScale1 = MathF.Sqrt(2.0f / inputSize);
            var heScale2 = MathF.Sqrt(2.0f / 32);

            var w1 = new AutogradNode(new FastTensor<float>(inputSize, 32).Randomize(heScale1), true);
            var b1 = new AutogradNode(new FastTensor<float>(1, 32).Fill(0.1f), true);
            var w2 = new AutogradNode(new FastTensor<float>(32, 1).Randomize(heScale2), true);
            var b2 = new AutogradNode(new FastTensor<float>(1, 1).Fill(meanTarget), true);

            var inputNode = new AutogradNode(cleanContext.Features, false);
            var targetNode = new AutogradNode(cleanContext.Targets, false);

            var parameters = new[] { w1, b1, w2, b2 };
            var optimizer = new Adam(parameters, learningRate: 0.005f);
            var scheduler = new LRScheduler(optimizer, parameters, msg => _output.WriteLine(msg), 0.5f, 50, 1e-6f, 0.001f);

            // JAWNY GRAF OBLICZENIOWY
            var graph = new ComputationGraph();

            _output.WriteLine("=== START TRENINGU ===");
            float initialLoss = 0;

            for (var epoch = 0; epoch <= 400; epoch++)
            {
                graph.Reset();
                optimizer.ZeroGrad();

                using var l1 = TensorMath.ReLU(graph, TensorMath.AddBias(graph, TensorMath.MatMul(graph, inputNode, w1), b1));
                using var prediction = TensorMath.AddBias(graph, TensorMath.MatMul(graph, l1, w2), b2);
                using var lossNode = TensorMath.MSELoss(graph, prediction, targetNode);

                var currentLoss = lossNode.Forward();
                if (epoch == 0) initialLoss = currentLoss;

                graph.Backward(lossNode);
                optimizer.Step();
                scheduler.Step(currentLoss);

                if (epoch % 100 == 0)
                    _output.WriteLine($"Epoka {epoch:D3} | Loss: {currentLoss:F6}");
            }

            _output.WriteLine("=== START PREDYKCJI ===");

            var testProperty = new PropertyData
            {
                Powierzchnia = 60,
                Pietro = 2,
                CzyKamienica = false,
                CzyMaKomorke = true,
                PowKomorki = 4,
                Miasto = "Warszawa",
                NazwaAgencji = "Premium"
            };

            var (valX, _) = converter.Transform(new List<PropertyData> { testProperty });
            using var valContext = pipeline.Execute(valX, new FastTensor<float>(1, 1));
            var valInput = new AutogradNode(valContext.Features, false);

            // INFERENCJA -> NULL GRAF
            using var pL1 = TensorMath.ReLU(null, TensorMath.AddBias(null, TensorMath.MatMul(null, valInput, w1), b1));
            using var pOut = TensorMath.AddBias(null, TensorMath.MatMul(null, pL1, w2), b2);

            var predictedPrice = pOut.Forward() * 100000f;

            _output.WriteLine($"Strata początkowa: {initialLoss:F6}");
            _output.WriteLine($"Predykcja dla 60m2 Warszawa: {predictedPrice:N2} PLN");

            foreach (var p in parameters) p.Dispose();
            inputNode.Dispose();
            targetNode.Dispose();
        }

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

    public class PropertyData
    {
        public float Powierzchnia { get; set; }
        public float Pietro { get; set; }
        public bool CzyKamienica { get; set; }
        public bool CzyMaKomorke { get; set; }
        public float PowKomorki { get; set; }
        public string Miasto { get; set; }
        public string NazwaAgencji { get; set; }
        public float Cena { get; set; }
    }
}