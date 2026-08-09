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
    /// its own code takes; this subscribes to meters somebody else owns. <c>http.server.active_requests</c>
    /// is maintained by the hosting layer as requests begin and end, and no application code can produce it
    /// — which is exactly why it closes a gap the request histogram cannot: <b>a histogram only records a
    /// request when it finishes</b>, so a hung request is invisible to every latency percentile while it is
    /// hanging. The in-flight count moves the moment it stalls.</para>
    ///
    /// <para><b>Instrument names are not trusted until seen.</b> This repository's own
    /// <c>MetricNameCatalog</c> records a mapping written by the author of the system that still left two
    /// channels unbound, reporting blind for hours. So the listener records which of the names it asked for
    /// actually arrived — see <see cref="SubscribedNames"/> — and the caller can expose that rather than
    /// assume.</para>
    ///
    /// <para><b>Why an <see cref="UpDownCounter{T}"/> needs a running total rather than a last value.</b>
    /// The hosting layer publishes deltas: +1 when a request starts, -1 when it ends. A listener that stored
    /// the most recent measurement would expose "1" or "-1" forever. The sum is the in-flight count.</para>
    /// </summary>
    internal sealed class RuntimeSignalListener : IDisposable
    {
        /// <summary>The instrument this closes the hung-request gap with, named as ASP.NET Core publishes it.</summary>
        public const string ActiveRequests = "http.server.active_requests";

        private readonly MeterListener _listener;
        private readonly List<string> _subscribed = [];
        private readonly object _gate = new();

        private long _activeRequests;

        public RuntimeSignalListener()
        {
            _listener = new MeterListener
            {
                InstrumentPublished = (instrument, listener) =>
                {
                    if (!string.Equals(instrument.Name, ActiveRequests, StringComparison.Ordinal))
                    {
                        return;
                    }

                    lock (_gate)
                    {
                        // The meter name is recorded alongside, because "the instrument exists" and "it comes
                        // from the meter the documentation names" are different facts and only the second
                        // survives a framework reorganisation.
                        _subscribed.Add($"{instrument.Meter.Name}/{instrument.Name}");
                    }

                    listener.EnableMeasurementEvents(instrument);
                },
            };

            _listener.SetMeasurementEventCallback<long>(
                (_, measurement, _, _) => Interlocked.Add(ref _activeRequests, measurement));

            // int as well as long: the published type is the framework's choice, not this project's, and a
            // callback registered for the wrong width is silently never invoked.
            _listener.SetMeasurementEventCallback<int>(
                (_, measurement, _, _) => Interlocked.Add(ref _activeRequests, measurement));

            _listener.Start();
        }

        /// <summary>Requests currently in flight, as the hosting layer counts them.</summary>
        public long ActiveRequestCount => Interlocked.Read(ref _activeRequests);

        /// <summary>
        /// Which meter/instrument pairs were actually published, so a name that does not exist on this
        /// runtime is visible as an empty list rather than as a permanent zero.
        /// </summary>
        public IReadOnlyList<string> SubscribedNames
        {
            get
            {
                lock (_gate)
                {
                    return [.. _subscribed];
                }
            }
        }

        public void Dispose()
        {
            _listener.Dispose();
        }
    }
}
