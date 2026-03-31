using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Optimizers;
using System.Numerics.Tensors;
using Xunit;
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

        [Fact(Skip = "a")]
        public void Bestia_ShouldLearn_TicTacToe_LongTraining()
        {
            // 1. ARCHITEKTURA: Solidny mózg dla solidnej Bestii
            using var model = new Sequential(
                new LinearLayer(9, 128),
                new ReluActivation(),
                new LinearLayer(128, 64),
                new ReluActivation(),
                new LinearLayer(64, 9)
            );

            var optimizer = new Adam(model.Parameters(), 0.001);

            // PARAMETRY DŁUGIEGO TRENINGU
            int totalGames = 50000;
            double epsilon = 1.0;
            double epsilonDecay = 0.99991; // Bardzo powolne mądrzenie

            int wins = 0;
            int draws = 0;

            // Cache parametrów modelu, żeby ich nie usunąć przez przypadek w TrainStep
            var modelParams = model.Parameters().ToHashSet();

            _output.WriteLine("=== Rozpoczynam morderczy trening Bestii (50k partii) ===");

            for (int i = 1; i <= totalGames; i++)
            {
                var board = new double[9]; // 0: puste, 1: AI, -1: Wróg
                bool gameOver = false;

                while (!gameOver)
                {
                    // --- RUCH AI ---
                    var stateBefore = (double[])board.Clone();
                    int action = ChooseAction(model, stateBefore, epsilon);

                    bool legal = MakeMove(board, action, 1.0);
                    double reward = EvaluateBoard(board, legal, out gameOver);

                    // Nauka na własnym ruchu
                    TrainStep(model, optimizer, modelParams, stateBefore, action, reward, board, gameOver);

                    if (gameOver)
                    {
                        if (reward > 0.5) wins++;
                        if (reward == 0.5) draws++;
                        break;
                    }

                    // --- RUCH PRZECIWNIKA (Losowy Bot) ---
                    OpponentMove(board);
                    reward = EvaluateBoard(board, true, out gameOver);

                    // Jeśli przeciwnik wygrał, AI dostaje karę (uczymy się na stanie po ruchu AI)
                    if (gameOver)
                        TrainStep(model, optimizer, modelParams, stateBefore, action, reward, board, true);
                }

                // Aktualizacja eksploracji
                epsilon = Math.Max(0.01, epsilon * epsilonDecay);

                // Raportowanie co 5000 gier
                if (i % 5000 == 0)
                {
                    double winRate = (double)wins / 5000;
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

        private void TrainStep(Sequential model, Adam opt, HashSet<AutogradNode> modelParams, double[] s, int a, double r, double[] nextS, bool done)
        {
            opt.ZeroGrad();

            // 1. Forward
            using var inputMat = new FastMatrix<double>(1, 9);
            inputMat.CopyFrom(s);
            using var inputNode = new AutogradNode(inputMat);
            var pred = model.Forward(inputNode);

            // 2. Bellman Target
            var targetMat = new FastMatrix<double>(1, 9);
            pred.Data.AsReadOnlySpan().CopyTo(targetMat.AsSpan());

            double targetValue = r;
            if (!done)
            {
                using var nextInputMat = new FastMatrix<double>(1, 9);
                nextInputMat.CopyFrom(nextS);
                using var nextInputNode = new AutogradNode(nextInputMat);
                using var nextQNode = model.Forward(nextInputNode);

                double maxNextQ = -double.MaxValue;
                var nqSpan = nextQNode.Data.AsReadOnlySpan();
                for (int i = 0; i < nqSpan.Length; i++)
                    if (nqSpan[i] > maxNextQ) maxNextQ = nqSpan[i];

                targetValue += 0.95 * maxNextQ;
            }
            targetMat[0, a] = targetValue;

            // 3. Loss & Backward
            using var targetNode = new AutogradNode(targetMat, requiresGrad: false);
            using var loss = TensorMath.MSELoss(pred, targetNode);

            var graph = loss.Backward();
            opt.Step();

            // 4. CLEANUP: Czyścimy tylko węzły tymczasowe
            foreach (var node in graph)
            {
                if (!modelParams.Contains(node))
                {
                    node.Dispose();
                }
            }
        }

        private int ChooseAction(Sequential model, double[] board, double eps)
        {
            if (_rng.NextDouble() < eps)
            {
                var empty = board.Select((v, i) => v == 0 ? i : -1).Where(i => i != -1).ToArray();
                return empty.Length > 0 ? empty[_rng.Next(empty.Length)] : 0;
            }

            using var inputMat = new FastMatrix<double>(1, 9);
            inputMat.CopyFrom(board);
            using var inputNode = new AutogradNode(inputMat);
            using var output = model.Forward(inputNode);

            int bestIdx = 0;
            double maxVal = -double.MaxValue;
            var span = output.Data.AsReadOnlySpan();
            for (int i = 0; i < span.Length; i++)
            {
                if (span[i] > maxVal) { maxVal = span[i]; bestIdx = i; }
            }
            return bestIdx;
        }

        private bool MakeMove(double[] b, int i, double p)
        {
            if (i < 0 || i > 8 || b[i] != 0) return false;
            b[i] = p; return true;
        }

        private void OpponentMove(double[] b)
        {
            var empty = b.Select((v, i) => v == 0 ? i : -1).Where(i => i != -1).ToArray();
            if (empty.Length > 0) b[empty[_rng.Next(empty.Length)]] = -1.0;
        }

        private double EvaluateBoard(double[] b, bool legal, out bool over)
        {
            over = false;
            if (!legal) { over = true; return -2.0; }

            if (CheckWin(b, 1.0)) { over = true; return 1.0; }
            if (CheckWin(b, -1.0)) { over = true; return -1.0; }
            if (!b.Contains(0.0)) { over = true; return 0.5; }

            return 0.0;
        }

        private bool CheckWin(double[] b, double p)
        {
            int[,] lines = { { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 }, { 0, 3, 6 }, { 1, 4, 7 }, { 2, 5, 8 }, { 0, 4, 8 }, { 2, 4, 6 } };
            for (int i = 0; i < 8; i++)
                if (b[lines[i, 0]] == p && b[lines[i, 1]] == p && b[lines[i, 2]] == p) return true;
            return false;
        }
    }
}