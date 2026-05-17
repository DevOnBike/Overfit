// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;

namespace DevOnBike.Overfit.Tests.TestSupport.GradientChecking
{
    public static class GradientChecker
    {
        /// <summary>
        /// Verifies the analytical gradients of a module by comparing them against a numerical approximation.
        /// </summary>
        /// <param name="module">The module (layer or full network) to test.</param>
        /// <param name="forwardAndLoss">Delegate that runs the forward pass and returns a scalar loss value.</param>
        /// <param name="epsilon">Step size for the numerical derivative.</param>
        /// <param name="tolerance">Maximum allowable relative error.</param>
        /// <param name="maxChecksPerParameter">
        /// Maximum number of elements to check per parameter.
        /// When null or &lt;= 0, all elements are checked.
        /// </param>
        /// <param name="seed">Seed for deterministic index sampling.</param>
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

            // 1. Zero out parameter gradients before the analytical backward pass.
            foreach (var param in parameters)
            {
                if (param.RequiresGrad)
                {
                    param.ZeroGrad();
                }
            }

            // 2. Forward + Backward for analytical gradients.
            var lossNode = forwardAndLoss(graph);
            EnsureScalarLoss(lossNode);

            graph.Backward(lossNode);

            // 3. Copy analytical gradients to a safe storage buffer.
            var analyticalGradients = new List<float[]>(parameters.Count);
            foreach (var param in parameters)
            {
                if (!param.RequiresGrad)
                {
                    analyticalGradients.Add([]);
                    continue;
                }

                var gradCopy = new float[param.DataView.Size];
                param.GradView.AsReadOnlySpan().CopyTo(gradCopy);
                analyticalGradients.Add(gradCopy);
            }

            // Clear the tape and gradients so nothing leaks into the numerical part.
            graph.Reset();
            foreach (var param in parameters)
            {
                if (param.RequiresGrad)
                {
                    param.ZeroGrad();
                }
            }

            var rng = new Random(seed);

            // 4. Numerical gradient and comparison.
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

                    // Restore the original value
                    dataSpan[i] = originalValue;

                    var numericalGrad = (lossPlus - lossMinus) / (2f * epsilon);
                    var analyticalGrad = analyticalGradSpan[i];

                    var absError = MathF.Abs(analyticalGrad - numericalGrad);

                    // More sensitive and standard denominator than max(1,...)
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