// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.InteropServices;

namespace DevOnBike.Overfit.Tensors
{
    /// <summary>
    /// A 64-byte-aligned unmanaged block with a <c>using</c>-shaped lifetime.
    ///
    /// <para><b>No production code uses this type.</b> The only reference in the tree is one test. It is
    /// kept because it is public API and removing it is a caller-visible break, not because anything
    /// depends on it — a reader looking for the arena the graph actually uses wants
    /// <c>NativeBufferManaged&lt;T&gt;</c>.</para>
    ///
    /// <para><b>Deliberately not <c>readonly</c>, so <see cref="Dispose"/> can clear its own fields.</b> A
    /// <c>readonly ref struct</c> cannot, so disposing the same variable twice — an explicit call inside a
    /// <c>using</c>, a retry path, a <c>finally</c> that duplicates the cleanup — passed the same pointer
    /// to <see cref="NativeMemory.AlignedFree"/> twice. Double-freeing unmanaged memory corrupts the
    /// allocator, and the damage surfaces in some unrelated allocation later, which is the one failure in
    /// this directory that hurts something other than the caller.</para>
    ///
    /// <para><b>What this does NOT close, stated because the obvious reading is that it does.</b> Copy the
    /// struct and dispose both copies and the pointer is still freed twice: clearing a field clears it in
    /// one copy. No value type can close that — <c>PooledBuffer&lt;T&gt;</c> has exactly the same limit for
    /// exactly the same reason. Closing it needs a single owner, which means a class, which means an
    /// allocation per buffer, which is the thing this type exists to avoid.</para>
    /// </summary>
    public unsafe ref struct NativeBuffer<T> where T : unmanaged
    {
        public Span<T> Span;
        private void* _ptr;

        public NativeBuffer(int size, bool clearMemory = true)
        {
            if (size <= 0)
            {
                _ptr = null;
                Span = [];

                return;
            }

            // checked, like NativeBufferManaged does for the identical expression. On a 32-bit runtime a
            // large element count times a wide T overflows nuint, and the allocation that follows is
            // smaller than every subsequent write assumes.
            var byteSize = checked((nuint)size * (nuint)sizeof(T));
            var paddedByteSize = (byteSize + 63) & ~(nuint)63;

            _ptr = NativeMemory.AlignedAlloc(paddedByteSize, 64);

            Span = new Span<T>(_ptr, size);

            if (clearMemory)
            {
                Span.Clear();
            }
        }

        /// <summary>
        /// Frees the block, once. A second call is a no-op rather than a second free, and the span is
        /// emptied so a use-after-dispose reads nothing instead of reading memory the allocator has
        /// handed to somebody else.
        /// </summary>
        public void Dispose()
        {
            if (_ptr != null)
            {
                NativeMemory.AlignedFree(_ptr);
                _ptr = null;
            }

            Span = [];
        }
    }
}