using System.Buffers;
using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit
{
    public sealed class FastBuffer<T> : IDisposable where T : struct
    {
        private T[] _rented;
        private bool _disposed;

        public int Length { get; }

        public FastBuffer(int length)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
            
            Length = length;

            _rented = ArrayPool<T>.Shared.Rent(length);
            
            _rented.AsSpan(0, length).Clear();
        }
        
        public ref T this[int index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                return ref _rented![index];
            }
        }

        /// <summary>Span dokładnie length elementów (nie nadmiarowych bajtów z puli).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Span<T> AsSpan()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            
            return _rented.AsSpan(0, Length);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ReadOnlySpan<T> AsReadOnlySpan()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            
            return _rented.AsSpan(0, Length);
        }

        public void Clear()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            
            AsSpan().Clear();
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, true))
            {
                return;
            }

            var rented = Interlocked.Exchange(ref _rented, null);

            if (rented != null)
            {
                ArrayPool<T>.Shared.Return(rented);
            }

            GC.SuppressFinalize(this);
        }
    }
}