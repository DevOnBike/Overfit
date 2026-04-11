// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Optimizers;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests
{
    public class TicTacToeIntelligenceTests
    {
        private readonly ITestOutputHelper _output;
        private readonly Random _rng = new();

        public TicTacToeIntelligenceTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        public void Bestia_ShouldLearn_TicTacToe_LongTraining()
        {
            using var model = new Sequential(
            new LinearLayer(9, 128),
            new ReluActivation(),
            new LinearLayer(128, 64),
            new ReluActivation(),
            new LinearLayer(64, 9)
            );

            using var optimizer = new Adam(model.Parameters());

            var graph = new ComputationGraph();

            var totalGames = 50000;
            var epsilon = 1.0f;
            var epsilonDecay = 0.99991f;

            var wins = 0;
            var draws = 0;

            _output.WriteLine("=== Rozpoczynam morderczy trening Bestii (50k partii) ===");

            for (var i = 1; i <= totalGames; i++)
            {
                var board = new float[9];
                var gameOver = false;

                while (!gameOver)
                {
                    var stateBefore = (float[])board.Clone();

                    var action = ChooseAction(model, stateBefore, epsilon);

                    var legal = MakeMove(board, action, 1.0f);
                    var reward = EvaluateBoard(board, legal, out gameOver);

                    TrainStep(model, optimizer, graph, stateBefore, action, reward, board, gameOver);

                    if (gameOver)
                    {
                        if (reward > 0.5)
                        {
                            wins++;
                        }

                        if (reward == 0.5)
                        {
                            draws++;
                        }

                        break;
                    }

                    OpponentMove(board);
                    reward = EvaluateBoard(board, true, out gameOver);

                    if (gameOver)
                    {
                        TrainStep(model, optimizer, graph, stateBefore, action, reward, board, true);
                    }
                }

                epsilon = MathF.Max(0.01f, epsilon * epsilonDecay);

                if (i % 5000 == 0)
                {
                    var winRate = (float)wins / 5000;
                    _output.WriteLine($"Gry: {i:D5} | WinRate: {winRate:P2} | Remisy: {draws} | Eps: {epsilon:F3}");
                    wins = 0;
                    draws = 0;
                }
            }

            _output.WriteLine("=== Trening zakończony. Bestia jest gotowa do dominacji. ===");
            Assert.True(epsilon < 0.1, "Epsilon zbyt wysoki - Bestia nie skończyła nauki.");

            model.Save("BestiaTicTacToe.bin");
            _output.WriteLine("Mózg Bestii został zgrany na dysk.");
        }

        private void TrainStep(Sequential model, Adam opt, ComputationGraph graph, float[] s, int a, float r, float[] nextS, bool done)
        {
            graph.Reset();
            opt.ZeroGrad();

            using var inputMat = new FastTensor<float>(1, 9);
            s.CopyTo(inputMat.AsSpan());
            using var inputNode = new AutogradNode(inputMat, false);

            // Forward z grafem — trening, nowe tensory, using OK
            using var pred = model.Forward(graph, inputNode);

            using var targetMat = new FastTensor<float>(1, 9);
            pred.Data.AsSpan().CopyTo(targetMat.AsSpan());

            var targetValue = r;
            if (!done)
            {
                using var nextInputMat = new FastTensor<float>(1, 9);
                nextS.CopyTo(nextInputMat.AsSpan());
                using var nextInputNode = new AutogradNode(nextInputMat, false);

                // Forward(null) — inference, zwraca współdzielony bufor → BEZ using
                var nextQNode = model.Forward(null, nextInputNode);

                var maxNextQ = -float.MaxValue;
                var nqSpan = nextQNode.Data.AsSpan();
                for (var i = 0; i < nqSpan.Length; i++)
                {
                    if (nqSpan[i] > maxNextQ)
                    {
                        maxNextQ = nqSpan[i];
                    }
                }

                targetValue += 0.95f * maxNextQ;
            }

            targetMat[0, a] = targetValue;

            using var targetNode = new AutogradNode(targetMat, false);
            using var loss = TensorMath.MSELoss(graph, pred, targetNode);

            graph.Backward(loss);
            opt.Step();
        }

        private int ChooseAction(Sequential model, float[] board, float eps)
        {
            if (_rng.NextSingle() < eps)
            {
                var empty = board.Select((v, i) => v == 0 ? i : -1).Where(i => i != -1).ToArray();
                return empty.Length > 0 ? empty[_rng.Next(empty.Length)] : 0;
            }

            using var inputMat = new FastTensor<float>(1, 9);
            board.CopyTo(inputMat.AsSpan());
            using var inputNode = new AutogradNode(inputMat, false);

            // Forward(null) — inference, zwraca współdzielony bufor → BEZ using
            var output = model.Forward(null, inputNode);

            var bestIdx = 0;
            var maxVal = -float.MaxValue;
            var span = output.Data.AsSpan();
            for (var i = 0; i < span.Length; i++)
            {
                if (span[i] > maxVal)
                {
                    maxVal = span[i];
                    bestIdx = i;
                }
            }

            return bestIdx;
        }

        private bool MakeMove(float[] b, int i, float p)
        {
            if (i < 0 || i > 8 || b[i] != 0)
            {
                return false;
            }

            b[i] = p;
            return true;
        }

        private void OpponentMove(float[] b)
        {
            var empty = b.Select((v, i) => v == 0 ? i : -1).Where(i => i != -1).ToArray();
            if (empty.Length > 0)
            {
                b[empty[_rng.Next(empty.Length)]] = -1.0f;
            }
        }

        private float EvaluateBoard(float[] b, bool legal, out bool over)
        {
            over = false;

            if (!legal)
            {
                over = true;
                return -2.0f;
            }

            if (CheckWin(b, 1.0f))
            {
                over = true;
                return 1.0f;
            }

            if (CheckWin(b, -1.0f))
            {
                over = true;
                return -1.0f;
            }

            if (!b.Contains(0.0f))
            {
                over = true;
                return 0.5f;
            }

            return 0.0f;
        }

        private bool CheckWin(float[] b, float p)
        {
            int[,] lines =
            {
                {
                    0, 1, 2
                },
                {
                    3, 4, 5
                },
                {
                    6, 7, 8
                },
                {
                    0, 3, 6
                },
                {
                    1, 4, 7
                },
                {
                    2, 5, 8
                },
                {
                    0, 4, 8
                },
                {
                    2, 4, 6
                }
            };

            for (var i = 0; i < 8; i++)
            {
                if (b[lines[i, 0]] == p && b[lines[i, 1]] == p && b[lines[i, 2]] == p)
                {
                    return true;
                }
            }

            return false;
        }
    }
}