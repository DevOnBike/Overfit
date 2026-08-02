// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;

namespace DevOnBike.Overfit.Cli
{
    /// <summary>
    /// Tells a running guard what an operator thought of an incident, and lists what is currently muted.
    ///
    /// <para><b>It talks to the process, not to the file, and that is the whole reason this is a network
    /// call.</b> The guard rewrites its learned state from memory every cycle, so a command that edited that
    /// file would have its work overwritten within one cadence — no error, no warning, the operator's
    /// judgement simply gone. The obvious implementation loses data and looks like it worked, which is worse
    /// than not having the command.</para>
    ///
    /// <para><b>There is no default for the judgement.</b> <c>--noise</c> silences a signal and
    /// <c>--real</c> pins it so no future threshold may hide it. Guessing between those two on the operator's
    /// behalf is not something to do quietly, so the command requires one of them.</para>
    /// </summary>
    public static class AnomalyAckCommand
    {
        /// <summary>Records a judgement about <paramref name="incidentId"/>.</summary>
        /// <param name="url">Base URL of the guard's metrics endpoint.</param>
        /// <param name="incidentId">The incident, as reported in the guard's log.</param>
        /// <param name="real">True for "this was correct", false for "this was noise".</param>
        /// <param name="duration">How long to mute it, e.g. <c>7d</c>. Ignored for a real judgement.</param>
        /// <param name="reason">What to record alongside it.</param>
        public static async Task<int> AckAsync(
            string url, long incidentId, bool real, string? duration, string? reason, CancellationToken ct)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(url);

            var query = $"?id={incidentId.ToString(CultureInfo.InvariantCulture)}"
                        + $"&kind={(real ? "real" : "noise")}";

            if (!real && !string.IsNullOrWhiteSpace(duration))
            {
                query += $"&for={Uri.EscapeDataString(duration)}";
            }

            if (!string.IsNullOrWhiteSpace(reason))
            {
                query += $"&reason={Uri.EscapeDataString(reason)}";
            }

            using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(15) };

            try
            {
                using var response = await http
                    .PostAsync($"{url.TrimEnd('/')}/ack{query}", content: null, ct)
                    .ConfigureAwait(false);

                var body = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);

                Console.Write(body);

                return response.IsSuccessStatusCode ? 0 : 1;
            }
            catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException)
            {
                // Named rather than swallowed: an operator who thinks they have muted something and has not
                // is worse off than one who was told the guard is unreachable.
                Console.Error.WriteLine(
                    $"Could not reach the guard at {url}: {ex.Message}. Nothing was recorded.");

                return 1;
            }
        }

        /// <summary>Lists every suppression currently muting something.</summary>
        public static async Task<int> ListAsync(string url, CancellationToken ct)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(url);

            using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(15) };

            try
            {
                var body = await http
                    .GetStringAsync($"{url.TrimEnd('/')}/suppressions", ct)
                    .ConfigureAwait(false);

                Console.Write(body);

                return 0;
            }
            catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException)
            {
                Console.Error.WriteLine($"Could not reach the guard at {url}: {ex.Message}");

                return 1;
            }
        }
    }
}
