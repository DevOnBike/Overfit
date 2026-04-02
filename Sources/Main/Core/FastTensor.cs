using System.Buffers;

namespace DevOnBike.Overfit.Core
{
    public class FastTensor<T> : IDisposable
    {
        private T[] _data;
        private readonly bool _ownsData;

        public int[] Shape { get; private set; }
        public int[] Strides { get; private set; }
        public int Offset { get; private set; }

        public int Size { get; }
        public int Rank { get; }
        public bool IsContiguous { get; private set; }

        public FastTensor(params int[] shape)
        {
            Shape = (int[])shape.Clone();
            Strides = CalculateStrides(Shape);
            Offset = 0;
            Rank = Shape.Length;
            Size = CalculateSize(Shape);
            IsContiguous = true;

            _data = ArrayPool<T>.Shared.Rent(Size);

            _data.AsSpan(0, Size).Clear();

            _ownsData = true;
        }

        private FastTensor(T[] data, int[] shape, int[] strides, int offset, int size, bool isContiguous, bool ownsData = false)
        {
            _data = data;
            Shape = shape;
            Strides = strides;
            Offset = offset;
            Rank = shape.Length;
            Size = size;
            IsContiguous = isContiguous;
            _ownsData = ownsData;
        }

        private static int[] CalculateStrides(int[] shape)
        {
            var strides = new int[shape.Length];
            var currentStride = 1;
            for (var i = shape.Length - 1; i >= 0; i--)
            {
                strides[i] = currentStride;
                currentStride *= shape[i];
            }
            return strides;
        }

        private static int CalculateSize(int[] shape)
        {
            var size = 1;
            for (var i = 0; i < shape.Length; i++) size *= shape[i];
            return size;
        }

        // --- DOSTĘP DO DANYCH ---
        public T this[int i]
        {
            get => _data[Offset + i * Strides[0]];
            set => _data[Offset + i * Strides[0]] = value;
        }

        public T this[int i, int j]
        {
            get => _data[Offset + i * Strides[0] + j * Strides[1]];
            set => _data[Offset + i * Strides[0] + j * Strides[1]] = value;
        }

        public T this[int i, int j, int k]
        {
            get => _data[Offset + i * Strides[0] + j * Strides[1] + k * Strides[2]];
            set => _data[Offset + i * Strides[0] + j * Strides[1] + k * Strides[2]] = value;
        }

        public T this[int i, int j, int k, int l]
        {
            get => _data[Offset + i * Strides[0] + j * Strides[1] + k * Strides[2] + l * Strides[3]];
            set => _data[Offset + i * Strides[0] + j * Strides[1] + k * Strides[2] + l * Strides[3]] = value;
        }

        public Span<T> AsSpan()
        {
            if (!IsContiguous)
                throw new InvalidOperationException("Nie można utworzyć Span z nieciągłego (np. transponowanego) Tensora. Użyj ToContiguous() najpierw.");

            return new Span<T>(_data, Offset, Size);
        }

        // NAPRAWIONE: Implementacja fizycznego kopiowania danych do układu ciągłego
        public FastTensor<T> ToContiguous()
        {
            if (IsContiguous) return this;

            var newTensor = new FastTensor<T>(Shape);
            var targetSpan = newTensor.AsSpan();

            // Używamy iteratora N-wymiarowego, aby przejść po danych w logicznej kolejności
            // i zapisać je liniowo w nowym buforze.
            var indices = new int[Rank];
            for (var i = 0; i < Size; i++)
            {
                // Obliczamy fizyczny offset w nieciągłej pamięci źródłowej
                var sourceOffset = Offset;
                for (var d = 0; d < Rank; d++)
                {
                    sourceOffset += indices[d] * Strides[d];
                }

                targetSpan[i] = _data[sourceOffset];

                // Inkrementacja indeksów (licznik N-wymiarowy)
                for (var d = Rank - 1; d >= 0; d--)
                {
                    indices[d]++;
                    if (indices[d] < Shape[d]) break;
                    indices[d] = 0;
                }
            }

            return newTensor;
        }

        public FastTensor<T> Reshape(params int[] newShape)
        {
            var newSize = CalculateSize(newShape);
            if (newSize != Size) throw new ArgumentException("Shape mismatch.");
            if (!IsContiguous) throw new InvalidOperationException("Reshape nieciągłego tensora jest niedozwolone.");

            return new FastTensor<T>(_data, (int[])newShape.Clone(), CalculateStrides(newShape), Offset, Size, true, false);
        }

        public FastTensor<T> Transpose(int dim0, int dim1)
        {
            var newShape = (int[])Shape.Clone();
            var newStrides = (int[])Strides.Clone();

            (newShape[dim0], newShape[dim1]) = (newShape[dim1], newShape[dim0]);
            (newStrides[dim0], newStrides[dim1]) = (newStrides[dim1], newStrides[dim0]);

            return new FastTensor<T>(_data, newShape, newStrides, Offset, Size, isContiguous: false, ownsData: false);
        }

        public void Dispose()
        {
            if (_ownsData && _data != null)
            {
                ArrayPool<T>.Shared.Return(_data);
                _data = null!;
            }
        }
    }
}