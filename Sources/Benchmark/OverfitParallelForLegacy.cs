// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;

namespace Benchmarks
{
    /// <summary>
    /// Frozen pre-P1/P2 copy of <c>OverfitParallelFor</c> — the bulk-wake
    /// dispatcher BEFORE caller-participation, the workerCount&lt;=1 fast-path
    /// and the grain overload were added. Byte-for-byte the dispatch core as it
    /// stood at <c>HEAD</c> before this session's tuning.
    ///
    /// Exists ONLY so <see cref="OverfitParallelForBenchmark"/> can measure
    /// old-vs-new in a single run. NOT for production use.
    /// </summary>
    public static unsafe class OverfitParallelForLegacy
    {
        private const string WorkerCountEnvVar = "OVERFIT_PARALLEL_WORKERS";

        private static readonly int _workerCount = ResolveWorkerCount();

        private static int ResolveWorkerCount()
        {
            var procCount = Environment.ProcessorCount;
            var raw = Environment.GetEnvironmentVariable(WorkerCountEnvVar);
            if (string.IsNullOrEmpty(raw))
            {
                return procCount;
            }
            if (!int.TryParse(raw, out var requested) || requested <= 0)
            {
                return procCount;
            }
            return Math.Min(requested, procCount);
        }

        private static readonly SemaphoreSlim _startSemaphore;
        private static readonly CountdownEvent _completion;
        private static readonly object _gate = new();
        private static readonly ChunkState[] _chunks;
        private static int _chunkCount;
        private static PaddedCounter _nextChunk;

        static OverfitParallelForLegacy()
        {
            _startSemaphore = new SemaphoreSlim(0, _workerCount);
            _completion = new CountdownEvent(0);
            _chunks = new ChunkState[_workerCount];

            for (var i = 0; i < _workerCount; i++)
            {
                var thread = new Thread(WorkerLoop)
                {
                    IsBackground = true,
                    Name = $"OverfitParallelLegacy-{i}",
                };
                thread.Start();
            }
        }

        public static int WorkerCount => _workerCount;

        public static void For(
            int rangeStart,
            int rangeEnd,
            delegate*<int, int, void*, void> body,
            void* context)
        {
            if (body == null)
            {
                throw new ArgumentNullException(nameof(body));
            }

            var totalWork = rangeEnd - rangeStart;
            if (totalWork <= 0) { return; }

            if (totalWork == 1)
            {
                body(rangeStart, rangeEnd, context);
                return;
            }

            var chunkCount = Math.Min(_workerCount, totalWork);
            var perChunk = (totalWork + chunkCount - 1) / chunkCount;

            lock (_gate)
            {
                _nextChunk.Value = 0;
                _chunkCount = chunkCount;
                _completion.Reset(chunkCount);

                for (var i = 0; i < chunkCount; i++)
                {
                    var chunkStart = rangeStart + i * perChunk;
                    var chunkEnd = Math.Min(chunkStart + perChunk, rangeEnd);

                    _chunks[i].Start = chunkStart;
                    _chunks[i].End = chunkEnd;
                    _chunks[i].Body = body;
                    _chunks[i].Context = context;
                    _chunks[i].Error = null;
                }

                // Bulk wake — releases chunkCount tokens; the caller then only
                // waits (pre-P2: no caller participation).
                _startSemaphore.Release(chunkCount);

                _completion.Wait();

                for (var i = 0; i < chunkCount; i++)
                {
                    if (_chunks[i].Error != null)
                    {
                        _chunks[i].Error.Throw();
                    }
                }
            }
        }

        private static void WorkerLoop()
        {
            while (true)
            {
                _startSemaphore.Wait();

                var index = Interlocked.Increment(ref _nextChunk.Value) - 1;

                if (index < _chunkCount)
                {
                    try
                    {
                        _chunks[index].Body(_chunks[index].Start, _chunks[index].End, _chunks[index].Context);
                    }
                    catch (Exception ex)
                    {
                        _chunks[index].Error = ExceptionDispatchInfo.Capture(ex);
                    }
                    finally
                    {
                        _completion.Signal();
                    }
                }
                else
                {
                    Debug.Fail($"OverfitParallelForLegacy: claim index {index} >= chunkCount {_chunkCount}.");
                }
            }
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct ChunkState
        {
            public int Start;
            public int End;
            public void* Context;
            public delegate*<int, int, void*, void> Body;
            public ExceptionDispatchInfo? Error;

            private readonly long _pad1;
            private readonly long _pad2;
            private readonly long _pad3;
            private readonly long _pad4;
        }

        [StructLayout(LayoutKind.Explicit, Size = 128)]
        private struct PaddedCounter
        {
            [FieldOffset(64)]
            public int Value;
        }
    }
}
