// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.Data;
using DevOnBike.Overfit.Data.Contracts;
using DevOnBike.Overfit.Data.Prepare;
using DevOnBike.Overfit.Optimizers;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.Prepare
{
    public class RealEstateFullCycleTests
    {
        private readonly ITestOutputHelper _output;

        public RealEstateFullCycleTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        public void Training_And_Prediction_Should_Work_EndToEnd()
        {
            var rawData = GenerateDummyData(500);
            var schema = CreatePropertySchema();

            var converter = new TabularToTensorConverter<PropertyData>(schema, valueExtractor: (item, propName) => propName switch
            {
                "Powierzchnia" => item.Powierzchnia,
                "Pietro" => item.Pietro,
                "CzyKamienica" => item.CzyKamienica,
                "CzyMaKomorke" => item.CzyMaKomorke,
                "PowKomorki" => item.PowKomorki,
                "Miasto" => item.Miasto,
                "NazwaAgencji" => item.NazwaAgencji,
                "Cena" => item.Cena,
                _ => null
            });

            converter.Fit(rawData);
            var (rawX, rawY) = converter.Transform(rawData);

            // Indeksy po konwersji:
            // [0] Powierzchnia  (Numeric)
            // [1] Pietro        (Numeric)
            // [2] CzyKamienica  (Binary)
            // [3] CzyMaKomorke  (Binary)
            // [4] PowKomorki    (Numeric)
            // [5] Krakow        (One-Hot)
            // [6] Warszawa      (One-Hot)
            // [7] Wroclaw       (One-Hot)
            // [8] Premium       (One-Hot)
            // [9] Standard      (One-Hot)
            var binaryAndOneHot = new HashSet<int>
            {
                2,
                3,
                5,
                6,
                7,
                8,
                9
            };
            var numericColumns = new HashSet<int>
            {
                0,
                1,
                4
            };

            var pipeline = new DataPipeline(log: msg => _output.WriteLine(msg))
                .AddLayer(new TechnicalSanityLayer(maxCorruptedRatio: 0.3f))
                .AddLayer(new DuplicateRowFilterLayer())
                .AddLayer(new ConstantColumnFilterLayer())
                .AddLayer(new OutlierClipLayer(
                0.01f,
                0.99f,
                excludedColumns: binaryAndOneHot))
                .AddLayer(new RobustScalingLayer(
                numericColumns,
                binaryAndOneHot));

            using var cleanContext = pipeline.Execute(rawX, rawY);

            _output.WriteLine($"Po pipeline: {cleanContext.Features.GetDim(0)} wierszy x {cleanContext.Features.GetDim(1)} kolumn");

            // Normalizacja Targetów
            var targetSpan = cleanContext.Targets.AsSpan();
            float meanTarget = 0;
            for (var i = 0; i < targetSpan.Length; i++)
            {
                targetSpan[i] /= 100000f;
                meanTarget += targetSpan[i];
            }
            meanTarget /= targetSpan.Length;

            var inputSize = cleanContext.Features.GetDim(1);

            // SIEĆ: 2-warstwowa MLP
            var heScale1 = MathF.Sqrt(2.0f / inputSize);
            var heScale2 = MathF.Sqrt(2.0f / 32);

            var w1 = new AutogradNode(new FastTensor<float>(inputSize, 32).Randomize(heScale1));
            var b1 = new AutogradNode(new FastTensor<float>(1, 32).Fill(0.1f));
            var w2 = new AutogradNode(new FastTensor<float>(32, 1).Randomize(heScale2));
            var b2 = new AutogradNode(new FastTensor<float>(1, 1).Fill(meanTarget));

            var inputNode = new AutogradNode(cleanContext.Features, false);
            var targetNode = new AutogradNode(cleanContext.Targets, false);

            var parameters = new[]
            {
                w1, b1, w2, b2
            };
            var optimizer = new Adam(parameters, 0.005f);
            var scheduler = new LRScheduler(optimizer, parameters, log: msg => _output.WriteLine(msg), 0.5f, 50, 1e-6f, 0.001f);

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
                if (epoch == 0)
                {
                    initialLoss = currentLoss;
                }

                graph.Backward(lossNode);
                optimizer.Step();
                scheduler.Step(currentLoss);

                if (epoch % 100 == 0)
                {
                    _output.WriteLine($"Epoka {epoch:D3} | Loss: {currentLoss:F6}");
                }
            }

            // PREDYKCJA
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

            var (valX, _) = converter.Transform(new List<PropertyData>
            {
                testProperty
            });
            using var valContext = pipeline.Execute(valX, new FastTensor<float>(1, 1));
            var valInput = new AutogradNode(valContext.Features, false);

            // INFERENCJA → null graf (bez autograd)
            using var pL1 = TensorMath.ReLU(null, TensorMath.AddBias(null, TensorMath.MatMul(null, valInput, w1), b1));
            using var pOut = TensorMath.AddBias(null, TensorMath.MatMul(null, pL1, w2), b2);

            var predictedPrice = pOut.Forward() * 100000f;

            _output.WriteLine($"Strata początkowa: {initialLoss:F6}");
            _output.WriteLine($"Predykcja dla 60m² Warszawa: {predictedPrice:N2} PLN");

            foreach (var p in parameters)
            {
                p.Dispose();
            }
            inputNode.Dispose();
            targetNode.Dispose();
        }

        private List<PropertyData> GenerateDummyData(int count)
        {
            var data = new List<PropertyData>(count);
            var rnd = new Random(42);
            var miasta = new[]
            {
                "Warszawa", "Krakow", "Wroclaw"
            };
            var agencje = new[]
            {
                "Premium", "Standard"
            };

            for (var i = 0; i < count; i++)
            {
                float pow = rnd.Next(25, 150);
                var miasto = miasta[rnd.Next(miasta.Length)];

                // Cena bazowa zależna od miasta
                var mnoznikMiasta = miasto switch
                {
                    "Warszawa" => 13000f,
                    "Krakow" => 11000f,
                    _ => 9500f
                };

                var czyKamienica = rnd.NextDouble() > 0.7;
                var czyMaKomorke = rnd.NextDouble() > 0.3;
                var powKomorki = czyMaKomorke ? rnd.Next(2, 10) : 0f;

                // Cena: baza + szum + korekta za kamienicę
                var cena = pow * mnoznikMiasta
                           + rnd.Next(-20000, 20000)
                           + (czyKamienica ? 15000f : 0f);

                data.Add(new PropertyData
                {
                    Powierzchnia = pow,
                    Pietro = rnd.Next(0, 10),
                    CzyKamienica = czyKamienica,
                    CzyMaKomorke = czyMaKomorke,
                    PowKomorki = powKomorki,
                    Miasto = miasto,
                    NazwaAgencji = agencje[rnd.Next(agencje.Length)],
                    Cena = cena
                });
            }

            return data;
        }

        private TableSchema CreatePropertySchema()
        {
            return new TableSchema
            {
                Features =
                [
                    new ColumnDefinition
                    {
                        Name = "Powierzchnia",
                        Type = ColumnType.Numeric
                    },
                    new ColumnDefinition
                    {
                        Name = "Pietro",
                        Type = ColumnType.Numeric
                    },
                    new ColumnDefinition
                    {
                        Name = "CzyKamienica",
                        Type = ColumnType.Binary
                    },
                    new ColumnDefinition
                    {
                        Name = "CzyMaKomorke",
                        Type = ColumnType.Binary
                    },
                    new ColumnDefinition
                    {
                        Name = "PowKomorki",
                        Type = ColumnType.Numeric
                    },
                    new ColumnDefinition
                    {
                        Name = "Miasto",
                        Type = ColumnType.Categorical
                    },
                    new ColumnDefinition
                    {
                        Name = "NazwaAgencji",
                        Type = ColumnType.Categorical
                    }
                ],
                Target = new ColumnDefinition
                {
                    Name = "Cena",
                    Type = ColumnType.Numeric
                }
            };
        }
    }
}