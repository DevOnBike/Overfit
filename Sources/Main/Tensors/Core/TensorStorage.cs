using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit.Tensors.Core
{
    /// <summary>
    /// PURE DOD: Czysty magazyn pamięci. Zero pojęcia o geometrii, wymiarach czy matematyce.
    /// Zastępuje dawną klasę FastTensor w kwestii alokacji.
    /// </summary>
    public sealed unsafe class TensorStorage<T> : IDisposable where T : unmanaged
    {
        private T[] _data;
        private int _disposed;

        internal NativeBufferManaged<T> _buffer;
        private void* _nativePtr;
        internal readonly bool _isBorrowedMemory;

        public readonly int Length;

        // 1. Konstruktor dla standardowej sterty (OverfitPool)
        public TensorStorage(int length, bool clearMemory = true)
        {
            Length = length;
            
            _data = OverfitPool<T>.Shared.Rent(length);

            if (clearMemory)
            {
                _data.AsSpan(0, length).Clear();
            }
        }

        // 2. Konstruktor Zero-Alloc (NativeBufferManaged / Arena)
        public TensorStorage(NativeBufferManaged<T> buffer, int length)
        {
            Length = length;
            _buffer = buffer;
            _nativePtr = buffer.Allocate(length);
            _isBorrowedMemory = true;
        }

        // Wydawanie surowej pamięci - podstawa dla TensorView i TensorKernels
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Span<T> AsSpan()
        {
            ObjectDisposedException.ThrowIf(_disposed == 1, this);
            
            return _isBorrowedMemory ? new Span<T>(_nativePtr, Length) : _data.AsSpan(0, Length);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ReadOnlySpan<T> AsReadOnlySpan()
        {
            return AsSpan();
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 0)
            {
                if (!_isBorrowedMemory && _data != null)
                {
                    OverfitPool<T>.Shared.Return(_data);
                }

                _data = null;
                _nativePtr = null;
                _buffer = null;
            }
        }
    }
}