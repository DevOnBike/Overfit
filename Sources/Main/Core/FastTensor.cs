using System.Buffers;
using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit.Core
{
    /// <summary>
    /// N-wymiarowy tensor z ArrayPool i inline Shape/Strides dla rank ≤ 4.
    /// </summary>
    public sealed class FastTensor<T> : IDisposable
    {
        private T[] _data;
        private int _disposed;   // 0 = alive, 1 = disposed — Interlocked wymaga int nie bool
        private readonly bool _ownsData;

        // ── Inline Shape/Strides dla rank ≤ 4 (eliminuje 2×new int[] per instancja) ──
        private int _s0, _s1, _s2, _s3;
        private int _st0, _st1, _st2, _st3;

        // Fallback dla rank > 4 (rzadkie — CNN najwyżej 4D)
        private readonly int[] _shapeOverflow;
        private readonly int[] _stridesOverflow;

        public int Offset { get; }
        public int Size { get; }
        public int Rank { get; }
        public bool IsContiguous { get; }

        // ── Dostęp O(1) bez alokacji (OSTATECZNA OPTYMALIZACJA) ────────────────────

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetDim(int index)
        {
            if (_shapeOverflow != null)
            {
                return _shapeOverflow[index];
            }

            return index switch
            {
                0 => _s0,
                1 => _s1,
                2 => _s2,
                3 => _s3,
                _ => throw new IndexOutOfRangeException()
            };
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetStride(int index)
        {
            if (_stridesOverflow != null)
            {
                return _stridesOverflow[index];
            }

            return index switch
            {
                0 => _st0,
                1 => _st1,
                2 => _st2,
                3 => _st3,
                _ => throw new IndexOutOfRangeException()
            };
        }

        // ── Właściwości Shape/Strides — alokują tablicę (Używać tylko do debugu!) ──
        public int[] Shape
        {
            get
            {
                if (_shapeOverflow != null)
                {
                    return _shapeOverflow;
                }

                var s = new int[Rank];

                if (Rank > 0) s[0] = _s0;
                if (Rank > 1) s[1] = _s1;
                if (Rank > 2) s[2] = _s2;
                if (Rank > 3) s[3] = _s3;

                return s;
            }
        }

        public int[] Strides
        {
            get
            {
                if (_stridesOverflow != null)
                {
                    return _stridesOverflow;
                }

                var st = new int[Rank];
                if (Rank > 0) st[0] = _st0;
                if (Rank > 1) st[1] = _st1;
                if (Rank > 2) st[2] = _st2;
                if (Rank > 3) st[3] = _st3;

                return st;
            }
        }

        // ── Konstruktory publiczne ────────────────────────────────────────────────────

        public FastTensor(params int[] shape) : this(true, shape) { }

        public FastTensor(bool clearMemory, params int[] shape)
        {
            Rank = shape.Length;
            Size = CalculateSize(shape);
            Offset = 0;
            IsContiguous = true;
            _ownsData = true;

            StoreShape(shape);
            StoreStrides(shape);

            _data = ArrayPool<T>.Shared.Rent(Size);

            if (clearMemory)
            {
                _data.AsSpan(0, Size).Clear();
            }
        }

        // Prywatny konstruktor dla widoków (Transpose, Reshape)
        private FastTensor(
            T[] data, int rank, int size, int offset, bool isContiguous, bool ownsData,
            int s0, int s1, int s2, int s3,
            int st0, int st1, int st2, int st3,
            int[] shapeOverflow, int[] stridesOverflow)
        {
            _data = data;
            Rank = rank;
            Size = size;
            Offset = offset;
            IsContiguous = isContiguous;
            _ownsData = ownsData;
            _s0 = s0; _s1 = s1; _s2 = s2; _s3 = s3;
            _st0 = st0; _st1 = st1; _st2 = st2; _st3 = st3;
            _shapeOverflow = shapeOverflow;
            _stridesOverflow = stridesOverflow;
        }

        // ── Indexery ─────────────────────────────────────────────────────────────────

        public T this[int i]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _data[Offset + i * _st0];
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => _data[Offset + i * _st0] = value;
        }

        public T this[int i, int j]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _data[Offset + i * _st0 + j * _st1];
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => _data[Offset + i * _st0 + j * _st1] = value;
        }

        public T this[int i, int j, int k]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _data[Offset + i * _st0 + j * _st1 + k * _st2];
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => _data[Offset + i * _st0 + j * _st1 + k * _st2] = value;
        }

        public T this[int i, int j, int k, int l]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _data[Offset + i * _st0 + j * _st1 + k * _st2 + l * _st3];
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => _data[Offset + i * _st0 + j * _st1 + k * _st2 + l * _st3] = value;
        }

        // ── Span access ──────────────────────────────────────────────────────────────

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Span<T> AsSpan()
        {
            ObjectDisposedException.ThrowIf(_disposed == 1, this);

            if (!IsContiguous)
            {
                throw new InvalidOperationException("Tensor nie jest ciągły. Wywołaj ToContiguous() najpierw.");
            }

            return new Span<T>(_data, Offset, Size);
        }

        // ── Operacje O(1) (widoki, zero kopii) ───────────────────────────────────────

        public FastTensor<T> Transpose(int dim0, int dim1)
        {
            // No-op gdy dim0 == dim1
            if (dim0 == dim1)
            {
                return this;
            }

            ReadShapeStrides(out var s, out var st);

            (s[dim0], s[dim1]) = (s[dim1], s[dim0]);
            (st[dim0], st[dim1]) = (st[dim1], st[dim0]);

            return MakeView(s, st, Offset, isContiguous: false);
        }

        public FastTensor<T> Reshape(params int[] newShape)
        {
            var newSize = CalculateSize(newShape);
            if (newSize != Size)
            {
                throw new ArgumentException($"Reshape: rozmiar {newSize} != {Size}.");
            }

            if (!IsContiguous)
            {
                throw new InvalidOperationException("Reshape nieciągłego tensora jest niedozwolone.");
            }

            var newStrides = BuildStrides(newShape);

            return MakeView(newShape, newStrides, Offset, isContiguous: true);
        }

        // ── Materializacja widoku ─────────────────────────────────────────────────────

        public FastTensor<T> ToContiguous()
        {
            if (IsContiguous)
            {
                return this;
            }

            var result = new FastTensor<T>(false, Shape); // Shape alokuje tu raz — rzadka operacja
            var dst = result.AsSpan();

            var indices = Rank <= 8 ? stackalloc int[Rank] : new int[Rank];

            ReadShapeStrides(out var shapeArr, out var stridesArr);

            for (var i = 0; i < Size; i++)
            {
                var srcOff = Offset;

                for (var d = 0; d < Rank; d++)
                {
                    srcOff += indices[d] * stridesArr[d];
                }

                dst[i] = _data[srcOff];

                for (var d = Rank - 1; d >= 0; d--)
                {
                    if (++indices[d] < shapeArr[d])
                    {
                        break;
                    }

                    indices[d] = 0;
                }
            }

            return result;
        }

        /// <summary>
        /// Zero-alloc factory — tworzy nowy tensor o tym samym kształcie co <paramref name="template"/>.
        /// Zastępuje wzorzec: new FastTensor&lt;T&gt;(template.Shape) który alokuje new int[Rank].
        ///
        /// Dla rank ≤ 4 (100% przypadków CNN): zero alokacji int[].
        /// Dla rank > 4: fallback na template.Shape (rzadkie, akceptowalne).
        /// </summary>
        public static FastTensor<T> SameShape(FastTensor<T> template, bool clearMemory = true)
        {
            return template.Rank switch
            {
                1 => new FastTensor<T>(clearMemory, template.GetDim(0)),
                2 => new FastTensor<T>(clearMemory, template.GetDim(0), template.GetDim(1)),
                3 => new FastTensor<T>(clearMemory, template.GetDim(0), template.GetDim(1), template.GetDim(2)),
                4 => new FastTensor<T>(clearMemory, template.GetDim(0), template.GetDim(1), template.GetDim(2), template.GetDim(3)),
                _ => new FastTensor<T>(clearMemory, template.Shape) // rank > 4 — Shape alokuje, ale to przypadek brzegowy
            };
        }

        public FastTensor<T> Zero()
        {
            AsSpan().Clear();

            return this;
        }

        // ── Dispose — Interlocked, double-dispose safe ───────────────────────────────

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 1)
            {
                return;
            }

            if (_ownsData)
            {
                var rented = Interlocked.Exchange(ref _data, null!);

                if (rented != null)
                {
                    ArrayPool<T>.Shared.Return(rented);
                }
            }
        }

        // ── Prywatne helpers ─────────────────────────────────────────────────────────

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CalculateSize(int[] shape)
        {
            var size = 1;

            for (var i = 0; i < shape.Length; i++)
            {
                size *= shape[i];
            }

            return size;
        }

        private static int[] BuildStrides(int[] shape)
        {
            var st = new int[shape.Length];
            var cur = 1;

            for (var i = shape.Length - 1; i >= 0; i--)
            {
                st[i] = cur;
                cur *= shape[i];
            }

            return st;
        }

        // Zapisuje shape do inline fields lub overflow array
        private void StoreShape(int[] shape)
        {
            if (shape.Length <= 4)
            {
                if (shape.Length > 0) _s0 = shape[0];
                if (shape.Length > 1) _s1 = shape[1];
                if (shape.Length > 2) _s2 = shape[2];
                if (shape.Length > 3) _s3 = shape[3];
                // _shapeOverflow pozostaje null
            }
            // else: _shapeOverflow ustawione przez MakeView lub konstruktor overflow (rank>4)
        }

        // Oblicza i zapisuje strides do inline fields
        private void StoreStrides(int[] shape)
        {
            var cur = 1;
            if (shape.Length > 3) { _st3 = cur; cur *= shape[3]; }
            if (shape.Length > 2) { _st2 = cur; cur *= shape[2]; }
            if (shape.Length > 1) { _st1 = cur; cur *= shape[1]; }
            if (shape.Length > 0) { _st0 = cur; }
        }

        // Odczytuje shape i strides do tymczasowych tablic (tylko dla operacji widoków)
        private void ReadShapeStrides(out int[] s, out int[] st)
        {
            if (_shapeOverflow != null)
            {
                s = (int[])_shapeOverflow.Clone();
                st = (int[])_stridesOverflow!.Clone();
                return;
            }

            s = new int[Rank];
            st = new int[Rank];

            if (Rank > 0) { s[0] = _s0; st[0] = _st0; }
            if (Rank > 1) { s[1] = _s1; st[1] = _st1; }
            if (Rank > 2) { s[2] = _s2; st[2] = _st2; }
            if (Rank > 3) { s[3] = _s3; st[3] = _st3; }
        }

        // Tworzy widok (Transpose/Reshape) z podanymi shape i strides
        private FastTensor<T> MakeView(int[] shape, int[] strides, int offset, bool isContiguous)
        {
            int s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            int st0 = 0, st1 = 0, st2 = 0, st3 = 0;
            int[] shapeOv = null, stridesOv = null;

            if (shape.Length <= 4)
            {
                if (shape.Length > 0) { s0 = shape[0]; st0 = strides[0]; }
                if (shape.Length > 1) { s1 = shape[1]; st1 = strides[1]; }
                if (shape.Length > 2) { s2 = shape[2]; st2 = strides[2]; }
                if (shape.Length > 3) { s3 = shape[3]; st3 = strides[3]; }
            }
            else
            {
                shapeOv = (int[])shape.Clone();
                stridesOv = (int[])strides.Clone();
            }

            return new FastTensor<T>(
                _data, shape.Length, Size, offset, isContiguous, ownsData: false,
                s0, s1, s2, s3, st0, st1, st2, st3,
                shapeOv, stridesOv);
        }
    }
}