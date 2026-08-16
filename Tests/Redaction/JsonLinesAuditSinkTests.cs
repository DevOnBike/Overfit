// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using DevOnBike.Overfit.Redaction;
using DevOnBike.Overfit.Server;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.Redaction
{
    /// <summary>
    /// The audit sink writes the line, and — since the entry carries no timestamp — it is the sink that
    /// decides when the redaction happened.
    ///
    /// <para><b>Why these tests exist at all.</b> The gateway used to stamp each audit entry with
    /// <c>DateTimeOffset.UtcNow</c> at three separate call sites, inside private static helpers where no test
    /// could reach them. The sweep moved the read into this sink behind an <c>IClock</c>; without a test that
    /// pins the stamp, that move would be unverified and the next refactor could quietly put the wall clock
    /// back.</para>
    /// </summary>
    public sealed class JsonLinesAuditSinkTests : IDisposable
    {
        private readonly string _path = Path.Combine(Path.GetTempPath(), "overfit-audit-" + Guid.NewGuid().ToString("N") + ".jsonl");

        public void Dispose()
        {
            if (File.Exists(_path))
            {
                File.Delete(_path);
            }
        }

        private static RedactionAuditEntry Entry(string requestId = "req-1", int total = 2)
        {
            return new RedactionAuditEntry(
                requestId, total, new Dictionary<string, int>(StringComparer.Ordinal) { ["EMAIL"] = 2 });
        }

        /// <summary>The claim the whole split rests on: the instant comes from the injected clock.</summary>
        [Fact]
        public void TheSinkStampsTheLineFromItsOwnClock()
        {
            var moment = new DateTimeOffset(2026, 8, 10, 6, 30, 0, TimeSpan.Zero);

            using (var sink = new JsonLinesAuditSink(_path, new ManualClock(moment)))
            {
                sink.Record(Entry());
            }

            var line = Assert.Single(File.ReadAllLines(_path));

            Assert.Contains(
                "\"timestamp\":\"" + moment.ToString("o", CultureInfo.InvariantCulture) + "\"",
                line,
                StringComparison.Ordinal);
        }

        /// <summary>
        /// Two entries recorded without the clock moving carry the same instant — which is the behaviour the
        /// old shape could not give, because each call site read the wall clock separately.
        /// </summary>
        [Fact]
        public void TwoEntriesRecordedAtTheSameInstantCarryTheSameTimestamp()
        {
            var clock = new ManualClock();

            using (var sink = new JsonLinesAuditSink(_path, clock))
            {
                sink.Record(Entry("req-1"));
                sink.Record(Entry("req-2"));
            }

            var lines = File.ReadAllLines(_path);

            Assert.Equal(2, lines.Length);
            Assert.Equal(Stamp(lines[0]), Stamp(lines[1]));
        }

        /// <summary>And it moves when the clock does, so the field is not simply frozen.</summary>
        [Fact]
        public void TheStampAdvancesWithTheClock()
        {
            var clock = new ManualClock();

            using (var sink = new JsonLinesAuditSink(_path, clock))
            {
                sink.Record(Entry("req-1"));
                clock.Advance(TimeSpan.FromMinutes(7));
                sink.Record(Entry("req-2"));
            }

            var lines = File.ReadAllLines(_path);

            Assert.NotEqual(Stamp(lines[0]), Stamp(lines[1]));
        }

        /// <summary>The counts are the caller's and must survive the sink untouched.</summary>
        [Fact]
        public void TheEntrysCountsAreWrittenVerbatim()
        {
            using (var sink = new JsonLinesAuditSink(_path, new ManualClock()))
            {
                sink.Record(Entry("req-42", total: 5));
            }

            var line = Assert.Single(File.ReadAllLines(_path));

            Assert.Contains("\"requestId\":\"req-42\"", line, StringComparison.Ordinal);
            Assert.Contains("\"totalRedactions\":5", line, StringComparison.Ordinal);
            Assert.Contains("\"EMAIL\":2", line, StringComparison.Ordinal);
        }

        /// <summary>
        /// An audit log must be safe to retain, so the sink may never see a value to write in the first
        /// place: <see cref="RedactionAuditEntry"/> carries counts and nothing else.
        /// </summary>
        [Fact]
        public void TheLineCarriesNoRedactedValues()
        {
            var result = Redactor.CreateDefault().Redact("mail me at alice@example.com");

            using (var sink = new JsonLinesAuditSink(_path, new ManualClock()))
            {
                sink.Record(RedactionAuditEntry.FromResult("req-1", result));
            }

            var line = Assert.Single(File.ReadAllLines(_path));

            Assert.DoesNotContain("alice@example.com", line, StringComparison.OrdinalIgnoreCase);
        }

        private static string Stamp(string line)
        {
            var start = line.IndexOf("\"timestamp\":\"", StringComparison.Ordinal) + 13;

            return line[start..line.IndexOf('"', start)];
        }
    }
}
