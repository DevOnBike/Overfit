// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics.Metrics;
using BenchmarkDotNet.Attributes;

namespace Benchmarks
{
    /// <summary>
    /// What a <see cref="MeterListener"/> costs on the workload it is measuring.
    ///
    /// <para><b>Why this is not a curiosity.</b> The lab workload is an INSTRUMENT: every anomaly-guard
    /// threshold in this repository was calibrated against what those twelve replicas do when nothing is
    /// wrong. Two runtime channels have now been added by attaching a listener to meters the framework owns,
    /// and if that attachment costs anything measurable per request then the baseline moved under every floor
    /// already calibrated — quietly, and in the direction that looks like the workload got slower.</para>
    ///
    /// <para><b>Two costs, and only one of them is on a hot path.</b> A PUSHED instrument
    /// (<c>http.server.active_requests</c>) fires a callback on every request start and end, so its cost is
    /// paid per request and is what could matter. An OBSERVABLE one
    /// (<c>dotnet.monitor.lock_contentions</c>) is polled once per scrape — every fifteen seconds — so it
    /// would have to be catastrophically slow to be worth anything. Both are measured rather than assumed,
    /// because the assumption in the other direction is what this whole subsystem keeps getting wrong.</para>
    ///
    /// <para><b>The baseline is the same instrument with NO listener attached</b>, not an empty method. An
    /// unobserved <see cref="Counter{T}"/> already checks whether anyone is listening, so measuring against
    /// nothing would charge this change for the instrument's own existence.</para>
    ///
    /// <para><see cref="SimpleJobAttribute"/> rather than the shared config, deliberately: this is
    /// nanosecond work and the shared <c>InvocationCount=1</c> exists for multi-millisecond model runs — it
    /// would leave this measuring timer noise. That trap has already produced a phantom 1.61x regression in
    /// this repository.</para>
    /// </summary>
    [MemoryDiagnoser]
    [SimpleJob]
    public class MeterListenerOverheadBenchmark
    {
        private Meter _quiet = null!;
        private Meter _observed = null!;
        private UpDownCounter<long> _quietCounter = null!;
        private UpDownCounter<long> _observedCounter = null!;
        private ObservableCounter<long> _observableCounter = null!;
        private MeterListener _listener = null!;

        private long _sink;

        /// <summary>Read by the observable callback. Written once so the value is not a constant the
        /// compiler could fold away, and so it does not warn as never-assigned.</summary>
        private long _observableValue = 1;

        [GlobalSetup]
        public void Setup()
        {
            // Distinct meters so the listener can subscribe to one and leave the other genuinely unobserved.
            _quiet = new Meter("overfit.bench.quiet");
            _observed = new Meter("overfit.bench.observed");

            _quietCounter = _quiet.CreateUpDownCounter<long>("bench.requests.active");
            _observedCounter = _observed.CreateUpDownCounter<long>("bench.requests.active");
            _observableCounter = _observed.CreateObservableCounter("bench.contentions",
                () => Interlocked.Read(ref _observableValue));

            _listener = new MeterListener
            {
                InstrumentPublished = (instrument, listener) =>
                {
                    if (ReferenceEquals(instrument.Meter, _observed))
                    {
                        listener.EnableMeasurementEvents(instrument);
                    }
                },
            };

            // The same shape the lab workload uses: accumulate rather than store, so the callback does real
            // work instead of being optimised into nothing.
            _listener.SetMeasurementEventCallback<long>(
                (_, measurement, _, _) => Interlocked.Add(ref _sink, measurement));

            _listener.Start();
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _listener.Dispose();
            _observed.Dispose();
            _quiet.Dispose();
        }

        /// <summary>One request's worth of instrument traffic with nobody listening.</summary>
        [Benchmark(Baseline = true)]
        public void RequestPair_NoListener()
        {
            _quietCounter.Add(1);
            _quietCounter.Add(-1);
        }

        /// <summary>The same, with the listener attached — this is the per-request cost of the change.</summary>
        [Benchmark]
        public void RequestPair_Listening()
        {
            _observedCounter.Add(1);
            _observedCounter.Add(-1);
        }

        /// <summary>
        /// The per-scrape cost: polling every observable instrument. Paid once per fifteen seconds on the
        /// lab, so it is here to bound the number rather than because it was expected to matter.
        /// </summary>
        [Benchmark]
        public void PollObservables()
        {
            _listener.RecordObservableInstruments();
        }
    }
}
