using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Optimizers;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests
{
    public class TicTacToeIntelligenceTests : IDisposable
    {
        private readonly ITestOutputHelper _output;
        private readonly Random _rng = new();

        public TicTacToeIntelligenceTests(ITestOutputHelper output)
        {
            _output = output;
            // 1. ZAINICJOWANIE TAŚMY (Globalnego Grafu Obliczeniowego)
            ComputationGraph.Active = new ComputationGraph();
        }

        public void Dispose()
        {
            // Zwolnienie referencji do grafu po zakończeniu testów
            ComputationGraph.Active = null;
        }

        [Fact(Skip = "aaa")]
        public void Bestia_ShouldLearn_TicTacToe_LongTraining()
        {
            // ARCHITEKTURA: Solidny mózg dla solidnej Bestii
            using var model = new Sequential(
                new LinearLayer(9, 128),
                new ReluActivation(),
                new LinearLayer(128, 64),
                new ReluActivation(),
                new LinearLayer(64, 9)
            );

            // WAŻNE: 'using' przy Adamie! Nowy Adam wynajmuje FastBuffery (Zero-Alloc), 
            // które muszą wrócić do ArrayPoola po zakończeniu całego procesu.
            using var optimizer = new Adam(model.Parameters(), 0.001f);

            // PARAMETRY DŁUGIEGO TRENINGU
            var totalGames = 50000;
            var epsilon = 1.0f;
            var epsilonDecay = 0.99991f; // Bardzo powolne mądrzenie

            var wins = 0;
            var draws = 0;

            _output.WriteLine("=== Rozpoczynam morderczy trening Bestii (50k partii) ===");

            for (var i = 1; i <= totalGames; i++)
            {
                var board = new float[9]; // 0: puste, 1: AI, -1: Wróg
                var gameOver = false;

                while (!gameOver)
                {
                    // --- RUCH AI ---
                    var stateBefore = (float[])board.Clone();
                    var action = ChooseAction(model, stateBefore, epsilon);

                    var legal = MakeMove(board, action, 1.0f);
                    var reward = EvaluateBoard(board, legal, out gameOver);

                    // Nauka na własnym ruchu
                    TrainStep(model, optimizer, stateBefore, action, reward, board, gameOver);

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
                    {
                        TrainStep(model, optimizer, stateBefore, action, reward, board, true);
                    }
                }

                // Aktualizacja eksploracji
                epsilon = MathF.Max(0.01f, epsilon * epsilonDecay);

                // Raportowanie co 5000 gier
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

        private void TrainStep(Sequential model, Adam opt, float[] s, int a, float r, float[] nextS, bool done)
        {
            // RESET TAŚMY - Zwalnia operacje z poprzedniego kroku, zero alokacji!
            ComputationGraph.Active.Reset();
            opt.ZeroGrad();

            // 1. Forward 
            using var inputMat = new FloatFastMatrix(1, 9);
            inputMat.CopyFrom(s);
            using var inputNode = new AutogradNode(inputMat, requiresGrad: false);

            // Predykcja - te operacje zarejestrują się na taśmie (bo wagi mają RequiresGrad)
            using var pred = model.Forward(inputNode);

            // 2. Bellman Target
            using var targetMat = new FloatFastMatrix(1, 9);
            pred.Data.AsReadOnlySpan().CopyTo(targetMat.AsSpan());

            var targetValue = r;
            if (!done)
            {
                using var nextInputMat = new FloatFastMatrix(1, 9);
                nextInputMat.CopyFrom(nextS);
                using var nextInputNode = new AutogradNode(nextInputMat, requiresGrad: false);

                // --- WYŁĄCZAMY NAGRYWANIE ---
                ComputationGraph.Active.IsRecording = false;

                using var nextQNode = model.Forward(nextInputNode);

                // --- WŁĄCZAMY NAGRYWANIE Z POWROTEM ---
                ComputationGraph.Active.IsRecording = true;

                var maxNextQ = -float.MaxValue;
                var nqSpan = nextQNode.Data.AsReadOnlySpan();
                for (var i = 0; i < nqSpan.Length; i++)
                {
                    if (nqSpan[i] > maxNextQ) maxNextQ = nqSpan[i];
                }

                targetValue += 0.95f * maxNextQ;
            }
            targetMat[0, a] = targetValue;

            // 3. Loss & Backward
            // UWAGA: targetNode dostaje requiresGrad: false. Dzięki temu Taśma
            // podczas przechodzenia w tył natychmiast zatrzyma propagację błędu w stronę
            // obliczeń 'nextQNode' i wykona optymalizację wyłącznie dla 'pred'.
            using var targetNode = new AutogradNode(targetMat, requiresGrad: false);

            using var loss = TensorMath.MSELoss(pred, targetNode);

            // Zlecenie całej matematyki różniczkowej na zoptymalizowaną Taśmę
            ComputationGraph.Active.Backward(loss);

            opt.Step();

            // 4. CLEANUP (Węzłów-śmieci w grafie już nie ma, wszystko zwalnia "using var")
        }

        private int ChooseAction(Sequential model, float[] board, float eps)
        {
            if (_rng.NextSingle() < eps)
            {
                var empty = board.Select((v, i) => v == 0 ? i : -1).Where(i => i != -1).ToArray();
                return empty.Length > 0 ? empty[_rng.Next(empty.Length)] : 0;
            }

            // RESET TAŚMY - Inference też "brudzi" taśmę, jeśli wagi mają RequiresGrad.
            // Resetujemy graf, żeby nie wyciekała wirtualna pamięć w trakcie grania!
            ComputationGraph.Active.Reset();

            using var inputMat = new FloatFastMatrix(1, 9);
            inputMat.CopyFrom(board);

            // Wymuszamy wejściu brak gradientu
            using var inputNode = new AutogradNode(inputMat, requiresGrad: false);
            using var output = model.Forward(inputNode);

            var bestIdx = 0;
            var maxVal = -float.MaxValue;
            var span = output.Data.AsReadOnlySpan();
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
            if (i < 0 || i > 8 || b[i] != 0) return false;
            b[i] = p; return true;
        }

        private void OpponentMove(float[] b)
        {
            var empty = b.Select((v, i) => v == 0 ? i : -1).Where(i => i != -1).ToArray();
            if (empty.Length > 0) b[empty[_rng.Next(empty.Length)]] = -1.0f;
        }

        private float EvaluateBoard(float[] b, bool legal, out bool over)
        {
            over = false;
            if (!legal) { over = true; return -2.0f; }

            if (CheckWin(b, 1.0f)) { over = true; return 1.0f; }
            if (CheckWin(b, -1.0f)) { over = true; return -1.0f; }
            if (!b.Contains(0.0f)) { over = true; return 0.5f; }

            return 0.0f;
        }

        private bool CheckWin(float[] b, float p)
        {
            int[,] lines = { { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 }, { 0, 3, 6 }, { 1, 4, 7 }, { 2, 5, 8 }, { 0, 4, 8 }, { 2, 4, 6 } };
            for (var i = 0; i < 8; i++)
                if (b[lines[i, 0]] == p && b[lines[i, 1]] == p && b[lines[i, 2]] == p) return true;
            return false;
        }
    }
}