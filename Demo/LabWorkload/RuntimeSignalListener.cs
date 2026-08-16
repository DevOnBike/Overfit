// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics.Metrics;

namespace DevOnBike.Overfit.LabWorkload
{
    /// <summary>
    /// Reads instruments the .NET runtime and ASP.NET Core populate on their own, and makes them available
    /// to <see cref="WorkloadMetrics"/> for exposition.
    ///
    /// <para><b>Different in kind from everything else here.</b> The rest of this app records measurements
    /// its own code takes; this subscribes to meters somebody else owns. No application code can produce
    /// these numbers, which is exactly why they close gaps the app's own counters cannot.</para>
    ///
    /// <para><b>Instrument names are not trusted until seen.</b> This repository's own
    /// <c>MetricNameCatalog</c> records a mapping written by the author of the system that still left two
    /// channels unbound, reporting blind for hours. So the listener records which of the names it asked for
    /// actually arrived — see <see cref="SubscribedNames"/> and <see cref="IsPublished"/> — and the caller
    /// exposes that rather than assuming. A channel bound to an instrument that does not exist reads as a
    /// permanent zero, which is indistinguishable from health.</para>
    ///
    /// <para><b>Two instrument shapes, and treating them alike gives zero or nonsense.</b> A pushed
    /// instrument (<see cref="UpDownCounter{T}"/>, <see cref="Counter{T}"/>) sends deltas — <c>+1</c> on
    /// request start, <c>-1</c> on end — so its measurements must be SUMMED; storing the last value would
    /// report <c>1</c> or <c>-1</c> for ever. An <b>observable</b> instrument sends an absolute value and
    /// only when asked: it never fires on its own, so it must be POLLED through <see cref="Refresh"/>, and
    /// summing it would accumulate a running total of a running total.</para>
    ///
    /// <para><b>Measured 2026-08-09, which is why the distinction is written down rather than assumed.</b>
    /// <c>dotnet.monitor.lock_contentions</c> reported as published, the subscription looked correct, and
    /// the counter stayed at <b>0</b> through sixteen threads deliberately fighting over one lock — because
    /// nothing had asked it for a value. A clean-looking binding that silently reads zero is the failure
    /// shape this repository has now hit three times in two days.</para>
    /// </summary>
    internal sealed class RuntimeSignalListener : IDisposable
    {
        /// <summary>Requests in flight. Closes the hung-request gap: a request enters the duration histogram
        /// only when it FINISHES, so while one hangs every latency percentile stays quiet.</summary>
        public const string ActiveRequests = "http.server.active_requests";

        /// <summary>
        /// Monitor lock contentions, cumulative. Closes a different gap: contention produces a latency cliff
        /// with <b>normal CPU and normal GC</b>, so every channel the guard has today stays quiet through it
        /// — the threads are waiting, not working, and waiting costs no CPU.
        /// </summary>
        public const string LockContentions = "dotnet.monitor.lock_contentions";

        /// <summary>
        /// First-chance exceptions, cumulative. Distinct from <c>ErrorRate</c>, which counts 5xx responses:
        /// an exception that is caught and retried never becomes a 5xx, so the error rate stays flat while
        /// the process is already in trouble. It frequently precedes the failure the error rate eventually
        /// sees.
        /// </summary>
        public const string Exceptions = "dotnet.exceptions";

        /// <summary>
        /// Connections accepted but not yet being processed, because the server is at its concurrency
        /// limit. Saturation <b>before the application sees it</b>: a queued connection has sent no request
        /// yet, so it is invisible to every request-derived channel — latency, error rate, in-flight count.
        /// </summary>
        public const string QueuedConnections = "kestrel.queued_connections";

        /// <summary>
        /// Connections refused outright. A healthy server rejects none, so any non-zero value is itself the
        /// finding — the same shape as <c>OomEventsRate</c> and <c>ContainerRestarts</c>.
        /// </summary>
        public const string RejectedConnections = "kestrel.rejected_connections";

        private static readonly string[] Wanted =
            [ActiveRequests, LockContentions, Exceptions, QueuedConnections, RejectedConnections];

        private readonly MeterListener _listener;
        private readonly Dictionary<string, string> _subscribed = new(StringComparer.Ordinal);
        private readonly Dictionary<string, long> _totals = new(StringComparer.Ordinal);
        private readonly HashSet<string> _observable = new(StringComparer.Ordinal);
        private readonly object _gate = new();

        public RuntimeSignalListener()
        {
            foreach (var name in Wanted)
            {
                _totals[name] = 0L;
            }

            _listener = new MeterListener
            {
                InstrumentPublished = (instrument, listener) =>
                {
                    if (Array.IndexOf(Wanted, instrument.Name) < 0)
                    {
                        return;
                    }

                    lock (_gate)
                    {
                        // The meter name is recorded alongside, because "the instrument exists" and "it
                        // comes from the meter the documentation names" are different facts, and only the
                        // second survives a framework reorganisation.
                        _subscribed[instrument.Name] = instrument.Meter.Name;

                        if (instrument.IsObservable)
                        {
                            _observable.Add(instrument.Name);
                        }
                    }

                    listener.EnableMeasurementEvents(instrument);
                },
            };

            // long AND int: the published width is the framework's choice, not this project's, and a
            // callback registered for the wrong one is silently never invoked.
            _listener.SetMeasurementEventCallback<long>(
                (instrument, measurement, _, _) => Add(instrument.Name, measurement));

            _listener.SetMeasurementEventCallback<int>(
                (instrument, measurement, _, _) => Add(instrument.Name, measurement));

            _listener.Start();
        }

        /// <summary>Requests currently in flight, as the hosting layer counts them.</summary>
        public long ActiveRequestCount => Read(ActiveRequests);

        /// <summary>Lock contentions since this process started.</summary>
        public long LockContentionCount => Read(LockContentions);

        /// <summary>First-chance exceptions since this process started, thrown or not caught.</summary>
        public long ExceptionCount => Read(Exceptions);

        /// <summary>Connections waiting for a processing slot right now.</summary>
        public long QueuedConnectionCount => Read(QueuedConnections);

        /// <summary>Connections refused since this process started.</summary>
        public long RejectedConnectionCount => Read(RejectedConnections);

        /// <summary>
        /// Meter/instrument pairs that were actually published, so a name that does not exist on this
        /// runtime is visible as an absence rather than as a permanent zero.
        /// </summary>
        public IReadOnlyList<string> SubscribedNames
        {
            get
            {
                lock (_gate)
                {
                    var names = _subscribed.Select(p => $"{p.Value}/{p.Key}").ToList();

                    names.Sort(StringComparer.Ordinal);

                    return names;
                }
            }
        }

        /// <summary>Whether one named instrument arrived. Exposed per channel so a partial failure is
        /// visible: two of two is health, one of two is a channel that will never report.</summary>
        public bool IsPublished(string instrument)
        {
            lock (_gate)
            {
                return _subscribed.ContainsKey(instrument);
            }
        }

        public void Dispose()
        {
            _listener.Dispose();
        }

        /// <summary>
        /// Polls every observable instrument. Call immediately before reading, from whatever renders the
        /// exposition — an observable instrument has no value at all until something asks.
        /// </summary>
        public void Refresh()
        {
            _listener.RecordObservableInstruments();
        }

        private void Add(string instrument, long measurement)
        {
            lock (_gate)
            {
                // Observable instruments report where they are; pushed ones report how far they moved.
                _totals[instrument] = _observable.Contains(instrument)
                    ? measurement
                    : _totals.GetValueOrDefault(instrument) + measurement;
            }
        }

        private long Read(string instrument)
        {
            lock (_gate)
            {
                return _totals.GetValueOrDefault(instrument);
            }
        }
    }
}
