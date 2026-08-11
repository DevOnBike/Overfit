// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using DevOnBike.Overfit.Demo.LocalAgent.Observability;
using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.Tests.Demo
{
    /// <summary>
    /// The local-agent demo's <c>/metrics</c> exposition — <c>XC-6</c>.
    ///
    /// <para><b>Why these tests exist at all.</b> The demo used to get this text from
    /// <c>OpenTelemetry.Exporter.Prometheus.AspNetCore</c>, which has been in prerelease since 2022-08-18 —
    /// 33 versions, not one stable, while the rest of its suite ships stable — and was the repository's only
    /// prerelease pin. Removing it replaced a package with real logic, and every rule below is one that
    /// makes Prometheus <b>reject the whole document</b> or silently mis-read it, while the endpoint keeps
    /// answering 200 and looks healthy from the outside.</para>
    ///
    /// <para>This is also the only Demo project the test assembly references; the reason is written at the
    /// reference in <c>Tests.csproj</c>.</para>
    /// </summary>
    public sealed class LocalAgentMetricsExpositionTests
    {
        /// <summary>
        /// <b>The rule that rejects a document.</b> Prometheus permits one <c># HELP</c> and one
        /// <c># TYPE</c> per metric name; a second pair makes it refuse everything, not just the duplicate.
        /// Emitting them per sample instead of per metric is the natural way to write this loop wrong.
        /// </summary>
        [Fact]
        public void EachMetricNameIsDeclaredExactlyOnce()
        {
            var text = Exercised().WriteExposition();

            foreach (var kind in new[] { "# HELP ", "# TYPE " })
            {
                var names = text.Split('\n')
                    .Where(line => line.StartsWith(kind, StringComparison.Ordinal))
                    .Select(line => line[kind.Length..].Split(' ')[0])
                    .ToList();

                Assert.NotEmpty(names);
                Assert.Equal(names.Count, names.Distinct(StringComparer.Ordinal).Count());
            }
        }

        /// <summary>
        /// Histogram buckets are CUMULATIVE — <c>le</c> means "less than or equal", so each includes every
        /// bucket below it. Emitting per-bucket counts instead produces a histogram that parses, renders a
        /// chart, and is wrong; nothing rejects it, which is why it is pinned here.
        /// </summary>
        [Fact]
        public void HistogramBucketsAreCumulativeAndEndAtTheCount()
        {
            var text = Exercised().WriteExposition();

            foreach (var metric in new[] { "overfit_rag_search_seconds", "overfit_decode_rate" })
            {
                var buckets = text.Split('\n')
                    .Where(line => line.StartsWith(metric + "_bucket{", StringComparison.Ordinal))
                    .Select(line => long.Parse(line.Split("} ")[1], CultureInfo.InvariantCulture))
                    .ToList();

                Assert.NotEmpty(buckets);

                for (var i = 1; i < buckets.Count; i++)
                {
                    Assert.True(buckets[i] >= buckets[i - 1],
                        $"{metric}: bucket {i} ({buckets[i]}) is below its predecessor ({buckets[i - 1]}) "
                        + "— these are cumulative, not per-bucket counts.");
                }

                var count = long.Parse(
                    text.Split('\n').First(l => l.StartsWith(metric + "_count ", StringComparison.Ordinal))
                        .Split(' ')[1],
                    CultureInfo.InvariantCulture);

                // The +Inf bucket is the last one and must equal _count, or the histogram is internally
                // inconsistent and quantile queries over it return nonsense.
                Assert.Equal(count, buckets[^1]);
            }
        }

        /// <summary>
        /// <b>Load-bearing on Windows specifically.</b> A model path is full of backslashes, and an
        /// unescaped one makes the exposition unparseable — so the demo would break on the developer
        /// machine it ships for and not on CI.
        /// </summary>
        [Fact]
        public void BackslashesInALabelValueAreEscaped()
        {
            var collector = new MetricsCollector
            {
                ModelFile = @"C:\qwen3b\qwen0.5b.q4km.gguf",
                ModelFingerprint = "abc123",
            };

            var line = collector.WriteExposition().Split('\n')
                .First(l => l.StartsWith("overfit_build_info{", StringComparison.Ordinal));

            Assert.Contains(@"C:\\qwen3b\\qwen0.5b.q4km.gguf", line, StringComparison.Ordinal);
            Assert.DoesNotContain(@"C:\q", line, StringComparison.Ordinal);
        }

        /// <summary>
        /// A counter that has never been incremented must still declare itself. A series that only appears
        /// once it is non-zero is indistinguishable from one that was never registered, so an alert written
        /// against it never fires and nobody finds out.
        /// </summary>
        [Fact]
        public void AMetricWithNoSamplesStillDeclaresItself()
        {
            var text = new MetricsCollector().WriteExposition();

            Assert.Contains("# TYPE overfit_tool_calls_total counter", text, StringComparison.Ordinal);
            Assert.Contains("# TYPE overfit_requests_total counter", text, StringComparison.Ordinal);
        }

        /// <summary>
        /// The numbers are the ones that were recorded — the shadow state the exposition reads is updated
        /// beside every <c>Meter</c> instrument, and a <c>Record</c> method that touches one and forgets the
        /// other reports a value that is quietly stale rather than missing.
        /// </summary>
        [Fact]
        public void RecordedTotalsReachTheExposition()
        {
            var text = Exercised().WriteExposition();

            Assert.Contains("overfit_generations_total 2", text, StringComparison.Ordinal);
            Assert.Contains("overfit_prompt_tokens_total 30", text, StringComparison.Ordinal);
            Assert.Contains("overfit_generated_tokens_total 12", text, StringComparison.Ordinal);
            Assert.Contains("overfit_requests_total{endpoint=\"chat\"} 1", text, StringComparison.Ordinal);
            Assert.Contains("overfit_requests_total{endpoint=\"rag\"} 1", text, StringComparison.Ordinal);
            Assert.Contains("overfit_tool_calls_total{tool=\"search\"} 2", text, StringComparison.Ordinal);
        }

        /// <summary>
        /// Every value line must be `name[{labels}] number` — a stray culture-formatted decimal comma is a
        /// parse error for the whole document, and this repository runs on a machine whose current culture
        /// formats <c>0.5</c> as <c>0,5</c>.
        /// </summary>
        [Fact]
        public void EverySampleLineParsesAsANumberUnderTheInvariantCulture()
        {
            var text = Exercised().WriteExposition();

            foreach (var line in text.Split('\n'))
            {
                if (line.Length == 0 || line.StartsWith('#'))
                {
                    continue;
                }

                var value = line[(line.LastIndexOf(' ') + 1)..];

                Assert.True(
                    double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out _),
                    $"'{line}' does not end in an invariant-culture number.");
            }
        }

        private static MetricsCollector Exercised()
        {
            var collector = new MetricsCollector
            {
                ModelFile = "model.gguf",
                ModelFingerprint = "deadbeef",
                MmapEnabled = true,
                ModelLoadSeconds = 0.5,
            };

            // TokensPerSecond is DERIVED from generated tokens and elapsed nanoseconds, not settable — so
            // the elapsed values below are chosen to produce a decode rate rather than asserted directly.
            // 5 tokens in 0.25 s and 7 in 2 s put the two samples in different histogram buckets, which is
            // what makes the cumulative-bucket test able to fail.
            collector.RecordGeneration("chat", new GenerationStats(
                promptTokens: 10, generatedTokens: 5,
                elapsedNanoseconds: 250_000_000, allocatedBytes: 64, usedKeyValueCache: true));

            collector.RecordGeneration("rag", new GenerationStats(
                promptTokens: 20, generatedTokens: 7,
                elapsedNanoseconds: 2_000_000_000, allocatedBytes: 128, usedKeyValueCache: true));

            collector.RecordToolCall("search");
            collector.RecordToolCall("search");
            collector.RecordRagSearch(0.012);
            collector.RecordRagSearch(0.4);

            return collector;
        }
    }
}
