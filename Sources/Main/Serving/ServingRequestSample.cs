// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Serving
{
    /// <summary>
    /// One request's measurements from a serving load test, fed to <see cref="ServingLoadReport"/>.
    /// A failed request (HTTP error, timeout, connection drop) is recorded with
    /// <see cref="Succeeded"/> = false and contributes only to the error rate.
    /// </summary>
    public readonly struct ServingRequestSample
    {
        public ServingRequestSample(bool succeeded, double timeToFirstTokenMs, double endToEndMs, int outputTokens)
        {
            Succeeded = succeeded;
            TimeToFirstTokenMs = timeToFirstTokenMs;
            EndToEndMs = endToEndMs;
            OutputTokens = outputTokens;
        }

        /// <summary>The request completed without an HTTP/transport error.</summary>
        public bool Succeeded { get; }

        /// <summary>Time to first token: submission → first streamed chunk, milliseconds.</summary>
        public double TimeToFirstTokenMs { get; }

        /// <summary>End-to-end latency: submission → last streamed chunk, milliseconds.</summary>
        public double EndToEndMs { get; }

        /// <summary>Number of output chunks/tokens streamed for this request.</summary>
        public int OutputTokens { get; }

        /// <summary>Mean inter-token latency for this request = (e2e − ttft) / (tokens − 1), the
        /// per-step decode gap. Zero when fewer than two tokens were produced (no gap to measure).</summary>
        public double InterTokenLatencyMs
            => OutputTokens > 1 ? (EndToEndMs - TimeToFirstTokenMs) / (OutputTokens - 1) : 0.0;

        /// <summary>A successful sample with measured latencies and token count.</summary>
        public static ServingRequestSample Success(double timeToFirstTokenMs, double endToEndMs, int outputTokens)
            => new(true, timeToFirstTokenMs, endToEndMs, outputTokens);

        /// <summary>A failed request — counted toward the error rate, excluded from latency percentiles.</summary>
        public static ServingRequestSample Failure()
            => new(false, 0.0, 0.0, 0);
    }
}
