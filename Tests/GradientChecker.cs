// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.

using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.DeepLearning;

namespace DevOnBike.Overfit.Tests
{
    public static class GradientChecker
    {
        /// <summary>
        /// Weryfikuje gradienty analityczne modułu poprzez porównanie ich z numeryczną aproksymacją.
        /// </summary>
        /// <param name="module">Moduł (warstwa lub cała sieć) do przetestowania.</param>
        /// <param name="forwardAndLoss">Delegat wykonujący Forward Pass i zwracający węzeł (skalar) funkcji straty.</param>
        /// <param name="epsilon">Krok do numerycznej pochodnej (zalecane 1e-4 dla float32).</param>
        /// <param name="tolerance">Maksymalny dopuszczalny błąd względny.</param>
        public static void Verify(
            IModule module,
            Func<ComputationGraph, AutogradNode> forwardAndLoss,
            float epsilon = 1e-4f,
            float tolerance = 1e-3f)
        {
            var graph = new ComputationGraph();
            graph.IsRecording = true;

            // ====================================================================
            // 1. OBLICZENIE GRADIENTU ANALITYCZNEGO (Krok kontrolny)
            // ====================================================================
            var lossNode = forwardAndLoss(graph);
            graph.Backward(lossNode);

            // Kopiujemy wszystkie wyliczone gradienty analityczne do bezpiecznego magazynu,
            // ponieważ za chwilę będziemy resetować graf do testów numerycznych.
            var analyticalGradients = new List<float[]>();
            var parameters = module.Parameters().ToList();

            foreach (var param in parameters)
            {
                var gradCopy = new float[param.DataView.Size];
                param.GradView.AsReadOnlySpan().CopyTo(gradCopy);
                analyticalGradients.Add(gradCopy);
            }

            graph.Reset(); // Sprzątamy taśmę

            // ====================================================================
            // 2. OBLICZENIE GRADIENTU NUMERYCZNEGO I PORÓWNANIE
            // ====================================================================
            for (int pIdx = 0; pIdx < parameters.Count; pIdx++)
            {
                var param = parameters[pIdx];
                var dataSpan = param.DataView.AsSpan();
                var analyticalGradSpan = analyticalGradients[pIdx];

                for (int i = 0; i < param.DataView.Size; i++)
                {
                    float originalValue = dataSpan[i];

                    // Krok w przód ( + epsilon )
                    dataSpan[i] = originalValue + epsilon;
                    var lossPlusNode = forwardAndLoss(graph);
                    float lossPlus = lossPlusNode.DataView.AsReadOnlySpan()[0];
                    graph.Reset();

                    // Krok w tył ( - epsilon )
                    dataSpan[i] = originalValue - epsilon;
                    var lossMinusNode = forwardAndLoss(graph);
                    float lossMinus = lossMinusNode.DataView.AsReadOnlySpan()[0];
                    graph.Reset();

                    // Przywrócenie oryginalnej wartości
                    dataSpan[i] = originalValue;

                    // Obliczenie numerycznej pochodnej
                    float numericalGrad = (lossPlus - lossMinus) / (2f * epsilon);
                    float analyticalGrad = analyticalGradSpan[i];

                    // Obliczenie błędu względnego (Relative Error)
                    // Używamy max(1, ...) w mianowniku, aby uniknąć dzielenia przez bardzo małe liczby
                    float diff = MathF.Abs(analyticalGrad - numericalGrad);
                    float denominator = MathF.Max(1f, MathF.Max(MathF.Abs(analyticalGrad), MathF.Abs(numericalGrad)));
                    float relativeError = diff / denominator;

                    if (relativeError > tolerance)
                    {
                        throw new Exception(
                            $"GRADIENT CHECK FAILED!\n" +
                            $"Parametr: {pIdx}, Indeks: {i}\n" +
                            $"Analityczny: {analyticalGrad:E6}\n" +
                            $"Numeryczny:  {numericalGrad:E6}\n" +
                            $"Błąd względny: {relativeError:E6} (Tolerancja: {tolerance:E6})");
                    }
                }
            }

            Console.WriteLine("✅ Gradient Check Passed! Gradienty analityczne zgadzają się z numerycznymi.");
        }
    }
}