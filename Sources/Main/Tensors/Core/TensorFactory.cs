using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit.Tensors.Core
{
    /// <summary>
    /// Pure DOD: creation, cloning and materialization logic separated from storage and views.
    /// </summary>
    public static class TensorFactory
    {
        /// <summary>
        /// Creates a new storage with the same length and memory mode as the template.
        /// For borrowed memory, the clone borrows from the same arena.
        /// For pooled memory, the clone rents a new managed buffer.
        /// </summary>
        public static TensorStorage<T> CloneStorage<T>(TensorStorage<T> template, bool clearMemory = true)
            where T : unmanaged
        {
            ArgumentNullException.ThrowIfNull(template);

            TensorStorage<T> result;

            if (template._isBorrowedMemory)
            {
                if (template._buffer is null)
                {
                    throw new InvalidOperationException("Template storage is marked as borrowed memory but has no arena.");
                }

                result = new TensorStorage<T>(template._buffer, template.Length);
            }
            else
            {
                result = new TensorStorage<T>(template.Length, clearMemory);
            }

            if (template._isBorrowedMemory && clearMemory)
            {
                result.AsSpan().Clear();
            }

            return result;
        }

        /// <summary>
        /// Materializes any tensor view into a new contiguous storage buffer.
        /// Supports rank 1..4.
        /// </summary>
        public static TensorStorage<T> Materialize<T>(TensorSpan<T> view)
            where T : unmanaged
        {
            var storage = new TensorStorage<T>(view.Size, clearMemory: false);
            var target = storage.AsSpan();

            if (view.IsContiguous)
            {
                view.AsReadOnlySpan().CopyTo(target);

                return storage;
            }

            switch (view.Rank)
            {
                case 1:
                    MaterializeRank1(view, target);
                    break;

                case 2:
                    MaterializeRank2(view, target);
                    break;

                case 3:
                    MaterializeRank3(view, target);
                    break;

                case 4:
                    MaterializeRank4(view, target);
                    break;

                default:
                    storage.Dispose();
                    throw new NotSupportedException($"Unsupported tensor rank: {view.Rank}");
            }

            return storage;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void MaterializeRank1<T>(TensorSpan<T> view, Span<T> target)
            where T : unmanaged
        {
            var idx = 0;

            for (var i = 0; i < view.Shape.D0; i++)
            {
                target[idx++] = view[i];
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void MaterializeRank2<T>(TensorSpan<T> view, Span<T> target)
            where T : unmanaged
        {
            var idx = 0;

            for (var i = 0; i < view.Shape.D0; i++)
            {
                for (var j = 0; j < view.Shape.D1; j++)
                {
                    target[idx++] = view[i, j];
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void MaterializeRank3<T>(TensorSpan<T> view, Span<T> target)
            where T : unmanaged
        {
            var idx = 0;

            for (var i = 0; i < view.Shape.D0; i++)
            {
                for (var j = 0; j < view.Shape.D1; j++)
                {
                    for (var k = 0; k < view.Shape.D2; k++)
                    {
                        target[idx++] = view[i, j, k];
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void MaterializeRank4<T>(TensorSpan<T> view, Span<T> target)
            where T : unmanaged
        {
            var idx = 0;

            for (var i = 0; i < view.Shape.D0; i++)
            {
                for (var j = 0; j < view.Shape.D1; j++)
                {
                    for (var k = 0; k < view.Shape.D2; k++)
                    {
                        for (var l = 0; l < view.Shape.D3; l++)
                        {
                            target[idx++] = view[i, j, k, l];
                        }
                    }
                }
            }
        }
    }
}