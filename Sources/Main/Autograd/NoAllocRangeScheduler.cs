// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Threading;

namespace DevOnBike.Overfit.Runtime
{
    /// <summary>
    /// Persistent zero-allocation range scheduler.
    ///
    /// This intentionally does not use Parallel.For, Task, ThreadPool work items,
    /// lambdas, closures, CountdownEvent or per-call allocations.
    ///
    /// Threads are created once and reused.
    /// Each call partitions [fromInclusive, toExclusive) into workerCount + 1 chunks.
    /// The caller thread also executes one chunk.
    /// </summary>
    public sealed class NoAllocRangeScheduler
    {
        private static readonly Lazy<NoAllocRangeScheduler> SharedLazy =
            new(CreateDefault, LazyThreadSafetyMode.ExecutionAndPublication);

        private readonly Worker[] _workers;

        private IRangeJob? _job;
        private Exception? _workerException;

        private int _fromInclusive;
        private int _toExclusive;
        private int _partitionCount;
        private int _activeWorkerCount;
        private int _remainingWorkers;
        private int _running;
        private int _shutdown;

        private NoAllocRangeScheduler(int workerCount)
        {
            if (workerCount < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(workerCount));
            }

            _workers = new Worker[workerCount];

            for (var i = 0; i < workerCount; i++)
            {
                _workers[i] = new Worker(this, i);
            }
        }

        public static NoAllocRangeScheduler Shared => SharedLazy.Value;

        public int WorkerCount => _workers.Length;

        public void For(int fromInclusive, int toExclusive, IRangeJob job)
        {
            ArgumentNullException.ThrowIfNull(job);

            var length = toExclusive - fromInclusive;

            if (length <= 1 || _workers.Length == 0)
            {
                if (length > 0)
                {
                    job.Execute(fromInclusive, toExclusive);
                }

                return;
            }

            if (Interlocked.CompareExchange(ref _running, 1, 0) != 0)
            {
                job.Execute(fromInclusive, toExclusive);
                return;
            }

            try
            {
                var activeWorkers = Math.Min(_workers.Length, length - 1);

                if (activeWorkers <= 0)
                {
                    job.Execute(fromInclusive, toExclusive);
                    return;
                }

                _job = job;
                _workerException = null;
                _fromInclusive = fromInclusive;
                _toExclusive = toExclusive;
                _activeWorkerCount = activeWorkers;
                _partitionCount = activeWorkers + 1;

                Volatile.Write(ref _remainingWorkers, activeWorkers);

                for (var i = 0; i < activeWorkers; i++)
                {
                    _workers[i].Signal();
                }

                Exception? callerException = null;

                try
                {
                    ExecutePartition(0);
                }
                catch (Exception ex)
                {
                    callerException = ex;
                }

                WaitForWorkers();

                _job = null;

                if (callerException != null)
                {
                    throw callerException;
                }

                if (_workerException != null)
                {
                    throw _workerException;
                }
            }
            finally
            {
                _job = null;
                Volatile.Write(ref _running, 0);
            }
        }

        private void ExecuteWorkerPartition(int workerIndex)
        {
            if (Volatile.Read(ref _shutdown) != 0)
            {
                return;
            }

            try
            {
                ExecutePartition(workerIndex + 1);
            }
            catch (Exception ex)
            {
                Interlocked.CompareExchange(ref _workerException, ex, null);
            }
            finally
            {
                Interlocked.Decrement(ref _remainingWorkers);
            }
        }

        private void ExecutePartition(int partitionIndex)
        {
            var job = _job;

            if (job == null)
            {
                return;
            }

            var total = _toExclusive - _fromInclusive;
            var partitions = _partitionCount;

            var start = _fromInclusive + (int)((long)total * partitionIndex / partitions);
            var end = _fromInclusive + (int)((long)total * (partitionIndex + 1) / partitions);

            if (start < end)
            {
                job.Execute(start, end);
            }
        }

        private void WaitForWorkers()
        {
            var spinner = new SpinWait();

            while (Volatile.Read(ref _remainingWorkers) != 0)
            {
                spinner.SpinOnce();
            }
        }

        private static NoAllocRangeScheduler CreateDefault()
        {
            var workerCount = Math.Max(0, Environment.ProcessorCount - 1);
            return new NoAllocRangeScheduler(workerCount);
        }

        public void Shutdown()
        {
            if (Interlocked.Exchange(ref _shutdown, 1) != 0)
            {
                return;
            }

            for (var i = 0; i < _workers.Length; i++)
            {
                _workers[i].Signal();
            }
        }

        private sealed class Worker
        {
            private readonly NoAllocRangeScheduler _owner;
            private readonly int _index;
            private readonly AutoResetEvent _signal;
            private readonly Thread _thread;

            public Worker(NoAllocRangeScheduler owner, int index)
            {
                _owner = owner;
                _index = index;
                _signal = new AutoResetEvent(false);
                _thread = new Thread(Loop)
                {
                    IsBackground = true,
                    Name = $"Overfit compute worker {index}"
                };

                _thread.Start();
            }

            public void Signal()
            {
                _signal.Set();
            }

            private void Loop()
            {
                while (true)
                {
                    _signal.WaitOne();

                    if (Volatile.Read(ref _owner._shutdown) != 0)
                    {
                        return;
                    }

                    if (_index < Volatile.Read(ref _owner._activeWorkerCount))
                    {
                        _owner.ExecuteWorkerPartition(_index);
                    }
                }
            }
        }
    }

}