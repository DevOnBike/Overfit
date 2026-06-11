// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.InteropServices;

namespace DevOnBike.Overfit.Runtime
{
    /// <summary>
    /// A scoped <see cref="GCHandle"/> — wraps Alloc/Free in a <c>using</c> so call sites stay clean instead of a
    /// hand-rolled try/finally. A <c>ref struct</c>, so it can't escape its scope or be boxed. Two modes:
    /// <list type="bullet">
    ///   <item><b>Normal</b> (<c>new GCHandleScope(obj)</c>): keeps a managed object reachable and yields a stable
    ///   <see cref="Token"/> to hand to an unsafe <c>void*</c> context (e.g. an
    ///   <see cref="OverfitParallel"/> worker that must call a method on it), recovered with
    ///   <see cref="Recover{T}"/>.</item>
    ///   <item><b>Pinned</b> (<see cref="Pin"/>): pins the object at a fixed address (for native interop, e.g.
    ///   handing a buffer to a P/Invoke), read via <see cref="Address"/>.</item>
    /// </list>
    /// </summary>
    public ref struct GcHandleScope
    {
        // Not readonly: GCHandle.Free() mutates the handle, so a readonly field would force a defensive copy (the free would run on the copy, not this handle).
        private GCHandle _handle;

        /// <summary>Allocates a normal (non-pinning) handle that keeps <paramref name="target"/> reachable.</summary>
        public GcHandleScope(object target)
        {
            _handle = GCHandle.Alloc(target);
        }

        private GcHandleScope(GCHandle handle)
        {
            _handle = handle;
        }

        /// <summary>Allocates a pinned handle that fixes <paramref name="target"/> in memory for native interop.</summary>
        public static GcHandleScope Pin(object target)
        {
            return new GcHandleScope(GCHandle.Alloc(target, GCHandleType.Pinned));
        }

        /// <summary>The token to embed in an unsafe context; recover the object with <see cref="Recover{T}"/>.</summary>
        public IntPtr Token => GCHandle.ToIntPtr(_handle);

        /// <summary>The pinned object's fixed address (only valid for a handle created by <see cref="Pin"/>).</summary>
        public IntPtr Address => _handle.AddrOfPinnedObject();

        /// <summary>Recovers the original object from a <see cref="Token"/> inside an unsafe worker.</summary>
        public static T Recover<T>(IntPtr token) where T : class
        {
            return (T)GCHandle.FromIntPtr(token).Target!;
        }

        public void Dispose()
        {
            _handle.Free();
        }
    }
}
