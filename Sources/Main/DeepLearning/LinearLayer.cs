using DevOnBike.Overfit.Core;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit.DeepLearning
{
    public sealed class LinearLayer : IModule
    {
        public AutogradNode Weights { get; private set; }
        public AutogradNode Biases { get; private set; }
        public bool IsTraining { get; private set; } = true;

        private readonly int _inputSize;
        private readonly int _outputSize;

        // Bufor inferencji (Zero-Allocation)
        private readonly AutogradNode _inferenceOutputNode;

        // Transponowane wagi dla SIMD inference — kształt (outputSize, inputSize)
        // Sekwencyjny dostęp w wewnętrznej pętli = pełne wykorzystanie SIMD
        private FastTensor<float> _weightsTransposed;

        public LinearLayer(int inputSize, int outputSize)
        {
            _inputSize = inputSize;
            _outputSize = outputSize;

            var wData = new FastTensor<float>(inputSize, outputSize);
            var stdDev = MathF.Sqrt(2f / inputSize);
            var wSpan = wData.AsSpan();
            for (var i = 0; i < wSpan.Length; i++)
            {
                wSpan[i] = MathUtils.NextGaussian() * stdDev;
            }

            Weights = new AutogradNode(wData, true);
            Biases = new AutogradNode(new FastTensor<float>(outputSize), true);

            var outData = new FastTensor<float>(1, outputSize);
            _inferenceOutputNode = new AutogradNode(outData, requiresGrad: false);
        }

        public void Train()
        {
            IsTraining = true;

            // Zwolnij transponowane wagi — w trybie treningowym nie są potrzebne
            _weightsTransposed?.Dispose();
            _weightsTransposed = null;
        }

        public void Eval()
        {
            IsTraining = false;

            // Pre-transpozycja wag: (inputSize, outputSize) → (outputSize, inputSize)
            // Robione raz przy przejściu do Eval, nie w każdym Forward
            RebuildTransposedWeights();
        }

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            if (graph == null || !IsTraining)
            {
                LinearInferenceSimd(
                    input.Data.AsReadOnlySpan(),
                    _weightsTransposed.AsReadOnlySpan(),
                    Biases.Data.AsReadOnlySpan(),
                    _inferenceOutputNode.Data.AsSpan());

                // Zwracamy bufor współdzielony — caller NIE powinien go disposować.
                // Wynik jest ważny tylko do następnego wywołania Forward.
                return _inferenceOutputNode;
            }

            return TensorMath.Linear(graph, input, Weights, Biases);
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            yield return Weights;
            yield return Biases;
        }

        public void Save(BinaryWriter bw)
        {
            var wSpan = Weights.Data.AsReadOnlySpan();
            for (var i = 0; i < wSpan.Length; i++)
            {
                bw.Write(wSpan[i]);
            }

            var bSpan = Biases.Data.AsReadOnlySpan();
            for (var i = 0; i < bSpan.Length; i++)
            {
                bw.Write(bSpan[i]);
            }
        }

        public void Load(BinaryReader br)
        {
            var wSpan = Weights.Data.AsSpan();
            for (var i = 0; i < wSpan.Length; i++)
            {
                wSpan[i] = br.ReadSingle();
            }

            var bSpan = Biases.Data.AsSpan();
            for (var i = 0; i < bSpan.Length; i++)
            {
                bSpan[i] = br.ReadSingle();
            }

            // Jeśli jesteśmy w trybie Eval, przebuduj transponowane wagi po Load
            if (!IsTraining)
            {
                RebuildTransposedWeights();
            }
        }

        public void Save(string path)
        {
            using var fs = new FileStream(path, FileMode.Create);
            using var bw = new BinaryWriter(fs);
            Save(bw);
        }

        public void Load(string path)
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException($"Brak pliku wag: {path}");
            }

            using var fs = new FileStream(path, FileMode.Open);
            using var br = new BinaryReader(fs);
            Load(br);
        }

        /// <summary>
        /// Transpozycja wag: W[inputSize, outputSize] → W_T[outputSize, inputSize].
        /// Po transpozycji wiersz W_T[j] = kolumna W[:,j] — dane dla outputu j
        /// leżą sekwencyjnie w pamięci = Vector.Dot na pełnej prędkości SIMD.
        /// </summary>
        private void RebuildTransposedWeights()
        {
            _weightsTransposed?.Dispose();
            _weightsTransposed = new FastTensor<float>(_outputSize, _inputSize);

            var src = Weights.Data.AsReadOnlySpan();
            var dst = _weightsTransposed.AsSpan();

            for (var i = 0; i < _inputSize; i++)
            {
                for (var j = 0; j < _outputSize; j++)
                {
                    dst[j * _inputSize + i] = src[i * _outputSize + j];
                }
            }
        }

        /// <summary>
        /// SIMD inference na pre-transponowanych wagach.
        /// Wewnętrzna pętla jest sekwencyjna po inputSize — 
        /// Vector.Dot przetwarza 8 floatów (AVX2) lub 16 (AVX-512) per instrukcję.
        /// 
        /// Dla 784→10 (MNIST): 10 × ceil(784/8) = 10 × 98 = 980 instrukcji SIMD.
        /// Bez transpozycji: stride access co 10 elementów = zero korzyści z SIMD.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void LinearInferenceSimd(
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weightsT,
            ReadOnlySpan<float> bias,
            Span<float> output)
        {
            var inputSize = input.Length;
            var outputSize = output.Length;
            var vCount = Vector<float>.Count;

            for (var j = 0; j < outputSize; j++)
            {
                var sum = Vector<float>.Zero;
                var wRow = weightsT.Slice(j * inputSize, inputSize);
                var i = 0;

                // SIMD główna pętla — Vector.Dot na blokach po vCount (8/16 floatów)
                for (; i <= inputSize - vCount; i += vCount)
                {
                    var vIn = new Vector<float>(input.Slice(i));
                    var vW = new Vector<float>(wRow.Slice(i));
                    sum += vIn * vW;
                }

                // Redukcja wektora SIMD do skalara
                var scalarSum = Vector.Dot(sum, Vector<float>.One) + bias[j];

                // Reszta (tail loop)
                for (; i < inputSize; i++)
                {
                    scalarSum += input[i] * wRow[i];
                }

                output[j] = scalarSum;
            }
        }

        public void Dispose()
        {
            Weights?.Dispose();
            Biases?.Dispose();
            _inferenceOutputNode?.Dispose();
            _weightsTransposed?.Dispose();
        }
    }
}