// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers;
using System.IO.MemoryMappedFiles;
using DevOnBike.Overfit.Exceptions;

namespace DevOnBike.Overfit.LanguageModels.Loading
{
    /// <summary>
    /// A read-only memory map over a whole model file (GGUF). Hands out
    /// <see cref="ReadOnlyMemory{T}"/> slices that point straight into the mapped
    /// pages — <b>zero copy</b>, so a weight backed by one of these slices costs no
    /// managed-heap bytes; the OS pages it in on demand and counts it as
    /// shared/clean working set, not committed private RAM.
    ///
    /// Verbatim-layout quant weights (<see cref="Runtime.Q4KWeight"/>,
    /// <see cref="Runtime.Q6KWeight"/>) keep their on-disk block layout, so a tensor
    /// (or a contiguous per-head sub-range of one) maps to exactly one
    /// <see cref="Slice(long,int)"/>.
    ///
    /// The map owns the file handle for its whole lifetime — the engine that holds
    /// the mmap-backed weights must keep this alive and dispose it last
    /// (see <see cref="Runtime.CachedLlamaInferenceEngine"/>). Slices handed out are
    /// invalid once this is disposed; never read them afterwards.
    ///
    /// <b>A path through a Windows symbolic link works, deliberately.</b> There is exactly one
    /// source of truth for the length — the handle this type opens — because two sources
    /// disagree on a reparse point. Measured on <c>C:\qwen3b\qwen.q4km.gguf</c> (2026-08-25,
    /// Windows 11, .NET 10): <c>FileInfo.Length</c> is <c>2104932768</c> for the file and
    /// <c>0</c> through a file symbolic link to it, because it reads the reparse point's own
    /// metadata rather than the target's; <c>FileStream.Length</c> is <c>2104932768</c> through
    /// both. A <b>hard link</b>, a <b>directory junction</b> and a <b>directory symbolic link</b>
    /// all report <c>2104932768</c> from either API — only a symbolic link on the final path
    /// component carries the disagreement. The view's capacity is not usable as the length
    /// either: it is rounded up to the page, <c>2104934400</c> for the same file, which would
    /// let <see cref="Slice(long,int)"/> hand out bytes past the end of the file.
    /// </summary>
    public sealed unsafe class MemoryMappedModelFile : IDisposable
    {
        private readonly FileStream _file;
        private readonly MemoryMappedFile _mmf;
        private readonly MemoryMappedViewAccessor _view;
        private readonly byte* _base;
        private bool _disposed;

        public MemoryMappedModelFile(string path)
        {
            ArgumentException.ThrowIfNullOrEmpty(path);

            // Open the file ONCE and take both the length and the map from that one handle.
            // FileInfo.Length is not usable here: on a Windows reparse point it reads the link's
            // own metadata, so a file symbolic link reports 0 while the map below (capacity 0 =
            // "to end of file") spans the target's real bytes — see the type doc for the numbers.
            var file = new FileStream(
                path, FileMode.Open, FileAccess.Read, FileShare.Read, bufferSize: 1, FileOptions.None);
            MemoryMappedFile? mmf = null;
            MemoryMappedViewAccessor? view = null;

            try
            {
                Length = file.Length;

                if (Length == 0)
                {
                    throw new OverfitFormatException(
                        $"Model file '{path}' is 0 bytes, so there is nothing to map. "
                        + "If it is a symbolic link, its target may be missing, empty or on a "
                        + "drive that is not mounted; check the target before the link.");
                }

                // Read-only, whole-file map over the handle above. capacity 0 = "to end of file".
                mmf = MemoryMappedFile.CreateFromFile(
                    file, mapName: null, capacity: 0, MemoryMappedFileAccess.Read,
                    HandleInheritability.None, leaveOpen: true);
                view = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);

                byte* p = null;
                view.SafeMemoryMappedViewHandle.AcquirePointer(ref p);
                // PointerOffset is the gap the OS inserted for page alignment (0 here, but
                // honour it so file offset 0 maps to _base).
                _base = p + view.PointerOffset;
            }
            catch
            {
                view?.Dispose();
                mmf?.Dispose();
                file.Dispose();
                throw;
            }

            _file = file;
            _mmf = mmf;
            _view = view;
        }

        /// <summary>Mapped file length in bytes.</summary>
        public long Length
        {
            get;
        }

        /// <summary>
        /// A zero-copy <see cref="ReadOnlyMemory{T}"/> over <c>[offset, offset+length)</c>
        /// of the mapped file. The returned memory stays valid until this map is disposed.
        /// </summary>
        public ReadOnlyMemory<byte> Slice(long offset, int length)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            ArgumentOutOfRangeException.ThrowIfNegative(offset);
            ArgumentOutOfRangeException.ThrowIfNegative(length);
            if (offset + length > Length)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(length),
                    $"Slice [{offset}, {offset + length}) exceeds mapped length {Length}.");
            }

            // IDISP004: the SliceManager is deliberately not disposed here — its Memory keeps
            // it alive, it owns no resource (the parent map does), and disposing it would
            // invalidate the slice we hand back.
#pragma warning disable IDISP004
            return new SliceManager(_base + offset, length).Memory;
#pragma warning restore IDISP004
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;

            _view.SafeMemoryMappedViewHandle.ReleasePointer();
            _view.Dispose();
            _mmf.Dispose();
            // leaveOpen: true above, so the handle the length came from is ours to close, last.
            _file.Dispose();
        }

        /// <summary>
        /// A <see cref="MemoryManager{T}"/> over an unmanaged sub-range of the map. It does
        /// <b>not</b> own the mapping — the parent <see cref="MemoryMappedModelFile"/> does —
        /// so its own dispose is a no-op and the pages are already pinned (no GC interaction).
        /// </summary>
        private sealed unsafe class SliceManager : MemoryManager<byte>
        {
            private readonly byte* _ptr;
            private readonly int _length;

            public SliceManager(byte* ptr, int length)
            {
                _ptr = ptr;
                _length = length;
            }

            public override Span<byte> GetSpan() => new(_ptr, _length);

            public override MemoryHandle Pin(int elementIndex = 0)
            {
                if (elementIndex < 0 || elementIndex >= _length)
                {
                    throw new ArgumentOutOfRangeException(nameof(elementIndex));
                }
                return new MemoryHandle(_ptr + elementIndex);
            }

            public override void Unpin()
            {
            }

            // IDISP010: MemoryManager<T>.Dispose(bool) is abstract — there is no base body to
            // call. This slice owns nothing (the parent map owns the mapping), so it's a no-op.
#pragma warning disable IDISP010
            protected override void Dispose(bool disposing)
            {
            }
#pragma warning restore IDISP010
        }
    }
}
