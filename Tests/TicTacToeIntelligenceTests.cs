// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.

using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Optimizers;
using Xunit;
using Xunit.Abstractions;
using System;
using System.Collections.Generic;
using System.Linq;

namespace DevOnBike.Overfit.Tests
{
    public class TicTacToeIntelligenceTests
    {
        private readonly ITestOutputHelper _output;
        private readonly Random _rng = new();

        public TicTacToeIntelligenceTests(ITestOutputHelper output) => _output = output;

        [Fact(Skip = "aaa")]
        public void Bestia_ShouldLearn_TicTacToe_LongTraining()
        {
            using var model = new Sequential(
                new LinearLayer(9, 128),
                new ReluActivation(),
                new LinearLayer(128, 64),
                new ReluActivation(),
                new LinearLayer(64, 9)
            );

            using var optimizer = new Adam(model.Parameters(), learningRate: 0.001f);
            var graph = new ComputationGraph();

            var totalGames = 50000;
            var epsilon = 1.0f;
            var epsilonDecay = 0.99991f;

            var wins = 0; var draws = 0; var losses = 0;

            for (var game = 0; game < totalGames; game++)
            {
                var board = new float[9];
                var isOver = false;

                while (!isOver)
                {
                    using var stateTensor = new FastTensor<float>(1, 9, clearMemory: false);
                    board.CopyTo(stateTensor.GetView().AsSpan());
                    using var stateNode = new AutogradNode(stateTensor, false);

                    // WAŻNE: Nie używamy 'using' dla prediction, bo to współdzielony bufor modelu!
                    var prediction = model.Forward(null, stateNode);
                    var qValues = prediction.DataView.AsReadOnlySpan();

                    var action = -1;
                    if (_rng.NextSingle() < epsilon)
                    {
                        do { action = _rng.Next(9); } while (board[action] != 0); // Tylko legalne w exploracji
                    }
                    else
                    {
                        var maxQ = float.MinValue;
                        for (var i = 0; i < 9; i++)
                        {
                            if (board[i] == 0 && qValues[i] > maxQ)
                            {
                                maxQ = qValues[i]; action = i;
                            }
                        }
                        if (action == -1)
                        {
                            action = _rng.Next(9);
                        }
                    }

                    var reward = ExecuteAction(board, action, out isOver);

                    // Jeśli gra trwa, rusza się przeciwnik
                    if (!isOver)
                    {
                        var enemyAction = RandomMove(board);
                        if (enemyAction != -1)
                        {
                            var enemyReward = ExecuteAction(board, enemyAction, out isOver, -1f);
                            if (enemyReward == 1f)
                            {
                                reward = -1f; // Przegraliśmy
                            }
                        }
                    }

                    // TRENING (po zakończeniu tury/gry)
                    using var targetTensor = new FastTensor<float>(1, 9, clearMemory: false);
                    qValues.CopyTo(targetTensor.GetView().AsSpan());
                    targetTensor.GetView().AsSpan()[action] = reward;
                    using var targetNode = new AutogradNode(targetTensor, false);

                    graph.Reset();
                    optimizer.ZeroGrad();
                    using var trainPred = model.Forward(graph, stateNode);
                    using var loss = TensorMath.MSELoss(graph, trainPred, targetNode);
                    graph.Backward(loss);
                    optimizer.Step();

                    if (isOver)
                    {
                        if (reward == 1f)
                        {
                            wins++;
                        }
                        else if (reward == -1f || reward == -2f)
                        {
                            losses++;
                        }
                        else
                        {
                            draws++;
                        }
                    }
                }
                epsilon *= epsilonDecay;
            }
            _output.WriteLine($"Final: Wins: {wins}, Draws: {draws}, Losses: {losses}");
            Assert.True(wins > losses, "Model powinien wygrywać częściej niż przegrywać z losowym botem.");
        }

        private int RandomMove(float[] b)
        {
            var empty = new List<int>();
            for (var i = 0; i < 9; i++)
            {
                if (b[i] == 0f)
                {
                    empty.Add(i);
                }
            }
            return empty.Count == 0 ? -1 : empty[_rng.Next(empty.Count)];
        }

        private float ExecuteAction(float[] b, int action, out bool over, float player = 1.0f)
        {
            if (b[action] != 0f) { over = true; return -2.0f; }
            b[action] = player;
            over = CheckWin(b, player) || !b.Contains(0.0f);
            if (CheckWin(b, player))
            {
                return 1.0f;
            }
            if (!b.Contains(0.0f))
            {
                return 0.5f;
            }
            return 0.0f;
        }

        private bool CheckWin(float[] b, float p)
        {
            int[,] lines = { { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 }, { 0, 3, 6 }, { 1, 4, 7 }, { 2, 5, 8 }, { 0, 4, 8 }, { 2, 4, 6 } };
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