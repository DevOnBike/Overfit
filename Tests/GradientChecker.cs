// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;

namespace DevOnBike.Overfit.Tests
{
    public static class GradientChecker
    {
        /// <summary>
        /// Weryfikuje gradienty analityczne modułu poprzez porównanie ich z numeryczną aproksymacją.
        /// </summary>
        /// <param name="module">Moduł (warstwa lub cała sieć) do przetestowania.</param>
        /// <param name="forwardAndLoss">Delegat wykonujący Forward Pass i zwracający skalar funkcji straty.</param>
        /// <param name="epsilon">Krok do numerycznej pochodnej.</param>
        /// <param name="tolerance">Maksymalny dopuszczalny błąd względny.</param>
        /// <param name="maxChecksPerParameter">
        /// Maksymalna liczba sprawdzanych elementów na parametr.
        /// Gdy null albo <= 0, sprawdzane są wszystkie elementy.
        /// </param>
        /// <param name="seed">Seed do deterministycznego próbkowania indeksów.</param>
        public static void Verify(
            IModule module,
            Func<ComputationGraph, AutogradNode> forwardAndLoss,
            float epsilon = 1e-4f,
            float tolerance = 1e-3f,
            int? maxChecksPerParameter = 64,
            int seed = 12345)
        {
            ArgumentNullException.ThrowIfNull(module);
            ArgumentNullException.ThrowIfNull(forwardAndLoss);

            var parameters = module.Parameters().ToList();
            if (parameters.Count == 0)
            {
                throw new InvalidOperationException("GradientChecker: moduł nie zwraca żadnych parametrów.");
            }

            var graph = new ComputationGraph { IsRecording = true };

            // 1. Zerowanie gradientów parametrów przed analitycznym backwardem.
            foreach (var param in parameters)
            {
                if (param.RequiresGrad)
                {
                    param.ZeroGrad();
                }
            }

            // 2. Forward + Backward dla gradientów analitycznych.
            var lossNode = forwardAndLoss(graph);
            EnsureScalarLoss(lossNode);

            graph.Backward(lossNode);

            // 3. Kopiowanie gradientów analitycznych do bezpiecznego magazynu.
            var analyticalGradients = new List<float[]>(parameters.Count);
            foreach (var param in parameters)
            {
                if (!param.RequiresGrad)
                {
                    analyticalGradients.Add(Array.Empty<float>());
                    continue;
                }

                var gradCopy = new float[param.DataView.Size];
                param.GradView.AsReadOnlySpan().CopyTo(gradCopy);
                analyticalGradients.Add(gradCopy);
            }

            // Czyścimy tape i gradienty, żeby nic nie przeciekało do części numerycznej.
            graph.Reset();
            foreach (var param in parameters)
            {
                if (param.RequiresGrad)
                {
                    param.ZeroGrad();
                }
            }

            var rng = new Random(seed);

            // 4. Gradient numeryczny i porównanie.
            for (var pIdx = 0; pIdx < parameters.Count; pIdx++)
            {
                var param = parameters[pIdx];
                if (!param.RequiresGrad)
                {
                    continue;
                }

                var dataSpan = param.DataView.AsSpan();
                var analyticalGradSpan = analyticalGradients[pIdx];

                foreach (var i in GetIndicesToCheck(dataSpan.Length, maxChecksPerParameter, rng))
                {
                    var originalValue = dataSpan[i];

                    // f(x + eps)
                    dataSpan[i] = originalValue + epsilon;
                    var lossPlus = EvaluateScalarLoss(graph, forwardAndLoss);

                    // f(x - eps)
                    dataSpan[i] = originalValue - epsilon;
                    var lossMinus = EvaluateScalarLoss(graph, forwardAndLoss);

                    // Przywrócenie oryginalnej wartości
                    dataSpan[i] = originalValue;

                    var numericalGrad = (lossPlus - lossMinus) / (2f * epsilon);
                    var analyticalGrad = analyticalGradSpan[i];

                    var absError = MathF.Abs(analyticalGrad - numericalGrad);

                    // Bardziej czuły i standardowy mianownik niż max(1,...)
                    var relDenom = MathF.Max(1e-6f, MathF.Abs(analyticalGrad) + MathF.Abs(numericalGrad));
                    var relativeError = absError / relDenom;

                    if (relativeError > tolerance && absError > tolerance)
                    {
                        throw new Exception(
                            $"GRADIENT CHECK FAILED!\n" +
                            $"Parametr: {pIdx}, Indeks: {i}\n" +
                            $"Analityczny: {analyticalGrad:E6}\n" +
                            $"Numeryczny: {numericalGrad:E6}\n" +
                            $"Błąd bezwzględny: {absError:E6}\n" +
                            $"Błąd względny: {relativeError:E6} (Tolerancja: {tolerance:E6})");
                    }
                }
            }

            Console.WriteLine("✅ Gradient Check Passed! Gradienty analityczne zgadzają się z numerycznymi.");
        }

        private static float EvaluateScalarLoss(
            ComputationGraph graph,
            Func<ComputationGraph, AutogradNode> forwardAndLoss)
        {
            graph.Reset();

            var lossNode = forwardAndLoss(graph);
            EnsureScalarLoss(lossNode);

            var loss = lossNode.DataView.AsReadOnlySpan()[0];

            graph.Reset();
            return loss;
        }

        private static void EnsureScalarLoss(AutogradNode lossNode)
        {
            ArgumentNullException.ThrowIfNull(lossNode);

            var lossView = lossNode.DataView;
            if (lossView.Size != 1)
            {
                throw new InvalidOperationException(
                    $"GradientChecker wymaga skalarnej funkcji straty, ale otrzymał tensor o rozmiarze {lossView.Size}.");
            }
        }

        private static IEnumerable<int> GetIndicesToCheck(int size, int? maxChecksPerParameter, Random rng)
        {
            if (size <= 0)
            {
                yield break;
            }

            if (!maxChecksPerParameter.HasValue || maxChecksPerParameter.Value <= 0 || maxChecksPerParameter.Value >= size)
            {
                for (var i = 0; i < size; i++)
                {
                    yield return i;
                }

                yield break;
            }

            var count = maxChecksPerParameter.Value;
            var chosen = new HashSet<int>();

            while (chosen.Count < count)
            {
                chosen.Add(rng.Next(size));
            }

            foreach (var idx in chosen.OrderBy(x => x))
            {
                yield return idx;
            }
        }
    }
}