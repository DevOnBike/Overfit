// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Net.Sockets;

namespace DevOnBike.Overfit.Tests
{
    /// <summary>
    /// A <see cref="LongFact"/> for tests that need the local Kubernetes lab — and that must be
    /// <b>reported</b>, not quietly counted, when it is absent.
    ///
    /// <para><b>Three outcomes, three meanings, and this is the third.</b> A missing model fixture is an
    /// ordinary state on a machine that does not carry 27 GB of checkpoints, so <see cref="ModelFact"/>
    /// skips it without ceremony. A fixture that is present but inconsistent is a configuration defect and
    /// asserts. A missing <i>lab</i> is neither: it means a slice of the release gate did not run, and
    /// whether that is acceptable for a given release is a person's call, not the suite's.</para>
    ///
    /// <para>So these skip — they must never fail on a machine without a cluster, and on GitHub CI they
    /// must not even try — but <c>Scripts/longfact_gate.py</c> counts them <b>separately</b> from ordinary
    /// skips and says so at the end of the run. A lab skip hidden among two hundred fixture skips is a
    /// decision nobody made.</para>
    ///
    /// <para><b><c>OVERFIT_LAB</c> controls the probe.</b> Unset: probe with a short timeout. <c>0</c> or
    /// <c>off</c>: skip immediately without touching the network — this is what CI sets, so a runner with
    /// no cluster spends nothing discovering that. <c>1</c>: assume present and let the test fail on its
    /// own terms if it is not, which is what you want while fixing lab plumbing.</para>
    ///
    /// <para><b>The probe is a TCP connect with a 500 ms budget</b>, deliberately not an HTTP request.
    /// It runs at test-DISCOVERY time, once per attribute, and discovery must not hang: a dead forward
    /// answers a connect immediately with a refusal, while an HTTP call to a half-open tunnel can sit
    /// there. What it establishes is only "something is listening" — a test whose forward points at the
    /// wrong pod still fails, correctly, on its own assertions.</para>
    /// </summary>
    internal sealed class LabFact : LongFact
    {
        internal const string LabVariable = "OVERFIT_LAB";

        private const int ProbeMilliseconds = 500;

        public LabFact(LabEndpoint endpoint, string runtime = null)
            : base(runtime)
        {
            Endpoint = endpoint;

            // Already skipped as long-running: leave that reason, it is the more general one.
            if (Skip is not null)
            {
                return;
            }

            var configured = Environment.GetEnvironmentVariable(LabVariable);

            if (string.Equals(configured, "1", StringComparison.Ordinal))
            {
                return;
            }

            if (string.Equals(configured, "0", StringComparison.Ordinal)
                || string.Equals(configured, "off", StringComparison.OrdinalIgnoreCase))
            {
                Skip = Message(endpoint, $"{LabVariable}={configured} — the lab was not probed at all");

                return;
            }

            var (port, what) = Probe(endpoint);

            if (port > 0)
            {
                return;
            }

            Skip = Message(endpoint, what);
        }

        /// <summary>Which face of the lab this test needs.</summary>
        public LabEndpoint Endpoint { get; }

        /// <summary>
        /// The marker <c>Scripts/longfact_gate.py</c> greps for when it separates lab skips from fixture
        /// skips. Changing this string changes what the gate reports, so it lives in one place.
        /// </summary>
        internal const string Marker = "LAB NOT AVAILABLE";

        private static string Message(LabEndpoint endpoint, string detail)
        {
            var remedy = endpoint == LabEndpoint.Prometheus
                ? @"k8s\monitoring\forward.cmd  (Prometheus on 9090, 9098, 9099)"
                : @"k8s\overfit\forward-replicas.cmd  (one local port per replica)";

            return $"{Marker}: {detail}. This test did NOT run — it has checked nothing, and that is a "
                + $"deliberate skip rather than a pass. Bring the lab up with {remedy}, or decide that "
                + "this part of the gate is not needed for this release. Set OVERFIT_LAB=0 on runners "
                + "with no cluster (GitHub CI) so the probe is skipped entirely.";
        }

        /// <summary>Returns the first port that accepts a connection, or 0 with a description of what did not.</summary>
        private static (int Port, string Detail) Probe(LabEndpoint endpoint)
        {
            // Prometheus is forwarded on three ports because the diagnostics default to different ones;
            // any single answer means the tunnel is up.
            int[] ports = endpoint == LabEndpoint.Prometheus ? [9090, 9098, 9099] : [8081, 8082, 8083];

            foreach (var port in ports)
            {
                if (Accepts(port))
                {
                    return (port, null);
                }
            }

            return (0, $"nothing listening on 127.0.0.1:{string.Join('/', ports)}");
        }

        private static bool Accepts(int port)
        {
            try
            {
                using var client = new TcpClient();
                var connecting = client.ConnectAsync("127.0.0.1", port);

                return connecting.Wait(ProbeMilliseconds) && client.Connected;
            }
            catch (Exception)
            {
                // A refused connection, a DNS failure and a socket exhaustion all mean the same thing
                // here — the lab is not reachable — and none of them should break test discovery.
                return false;
            }
        }
    }
}
