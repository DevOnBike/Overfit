// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using DevOnBike.Overfit.Redaction;
using DevOnBike.Overfit.Text;

namespace Benchmarks
{
    /// <summary>
    ///     `XC-45`: does swapping <c>Redactor</c>'s <c>StringBuilder</c> for <c>ValueStringBuilder</c> buy
    ///     anything <b>visible in the enclosing request</b>?
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         <b>The question deliberately is not "which builder is faster".</b> `XC-44` settled that and the
    ///         answer is `ValueStringBuilder`, at every length measured. The trap this class exists to avoid is
    ///         the one that inventory documented and then nearly fell into: <c>Redactor.Redact</c> also
    ///         allocates a <c>List&lt;RedactionMatch&gt;</c>, a <c>Dictionary&lt;string,int&gt;</c>, one
    ///         interpolated placeholder string per matched span, and runs a regex over the whole input. A 20 ns
    ///         saving inside a request that costs microseconds of regex is not a finding, and the only way to
    ///         know which it is, is to measure both and divide.
    ///     </para>
    ///     <para>
    ///         <b>Two categories, and the arithmetic that connects them is the deliverable.</b>
    ///         <c>operation</c> is the whole public call as it ships today — that is the denominator.
    ///         <c>assembly</c> is the segment-and-placeholder loop lifted out of
    ///         <c>Redactor.Redact</c> (<c>Redactor.cs:70-89</c>) and run over the span list that call
    ///         produced, in both builder shapes — that difference is the numerator. The migration is worth
    ///         proposing only if numerator/denominator is a number anyone would act on.
    ///     </para>
    ///     <para>
    ///         <b>This is an ablation and it cannot separate everything.</b> The <c>assembly</c> arms re-run the
    ///         loop over a span list captured beforehand, so they exclude detection (regex + validators) by
    ///         construction — which is the point — but they also exclude the per-match placeholder
    ///         interpolation at <c>Redactor.cs:82</c>, because the captured matches already carry their
    ///         placeholder string. So <c>assembly</c> is a <b>lower bound</b> on what the real loop costs, and
    ///         the builder's true share of the operation is at most what this reports. Stated here rather than
    ///         discovered later.
    ///     </para>
    ///     <para>
    ///         <c>Matches</c> sweeps the number of PII spans in one request, since the loop runs once per span
    ///         and a request with two is a different shape from one with sixteen.
    ///     </para>
    ///     <para><b>NOT MEASURED YET</b> — written 2026-08-13. Fill in with numbers, box and build.</para>
    /// </remarks>
    // Same job reasoning as ValueStringBuilderBenchmark: the shared BenchmarkConfig pins InvocationCount=1,
    // which is wrong for work at this scale. BDN's default job pilots the invocation count instead.
    [SimpleJob(warmupCount: 6, iterationCount: 15)]
    [MemoryDiagnoser]
    [CategoriesColumn]
    [GroupBenchmarksBy(BenchmarkLogicalGroupRule.ByCategory)]
    public class RedactionRequestBenchmark
    {
        [Params(2, 16)]
        public int Matches
        {
            get; set;
        }

        private Redactor _redactor = null!;
        private string _input = string.Empty;

        /// <summary>
        /// Materialised to an array on purpose: <c>RedactionResult.Matches</c> is an
        /// <c>IReadOnlyList&lt;T&gt;</c>, and iterating through the interface costs measurably more than
        /// iterating the array (2.4x, measured — see docs/measured-baselines.md). Both assembly arms would pay
        /// it equally so the ratio would survive, but it would inflate the denominator's share and make the
        /// builder look smaller than it is.
        /// </summary>
        private RedactionMatch[] _spans = [];

        [GlobalSetup]
        public void Setup()
        {
            _redactor = new Redactor(DefaultRedactionRules.All());
            _input = BuildRequest(Matches);

            var result = _redactor.Redact(_input);
            var captured = new RedactionMatch[result.Matches.Count];

            for (var i = 0; i < result.Matches.Count; i++)
            {
                captured[i] = result.Matches[i];
            }

            _spans = captured;

            if (_spans.Length == 0)
            {
                throw new InvalidOperationException(
                    "The synthetic request produced no redaction spans, so the assembly arms would measure "
                    + "an empty loop and the whole comparison would be vacuous. Fix BuildRequest.");
            }
        }

        /// <summary>
        /// A support-ticket-shaped prompt with <paramref name="count"/> e-mail addresses embedded in prose —
        /// the shape the gateway actually sees. Prose between the matches matters: it is what the
        /// segment-copying half of the loop copies, and a request that is nothing but PII would measure only
        /// the placeholder half.
        /// </summary>
        private static string BuildRequest(int count)
        {
            var text = new StringBuilder(2048);

            text.Append("Hi, I am forwarding the thread about the failed migration last night. ");

            for (var i = 0; i < count; i++)
            {
                text.Append("The report was sent to operator")
                    .Append(i)
                    .Append(".oncall@example-corp.com and bounced with a transient error, ")
                    .Append("so please re-send it once the relay is healthy again. ");
            }

            text.Append("Thanks, and sorry for the noise on a Friday.");

            return text.ToString();
        }

        // ── The denominator: the whole operation as it ships ──────────────────────────────────────────

        [BenchmarkCategory("operation")]
        [Benchmark(Baseline = true, Description = "operation: Redactor.Redact (as shipped)")]
        public int FullRedact()
        {
            return _redactor.Redact(_input).Text.Length;
        }

        // ── The numerator: the assembly loop only, both builder shapes ────────────────────────────────

        [BenchmarkCategory("assembly")]
        [Benchmark(Baseline = true, Description = "assembly: StringBuilder (as shipped)")]
        public int AssembleWithStringBuilder()
        {
            var builder = new StringBuilder(_input.Length);
            var cursor = 0;

            foreach (var span in _spans)
            {
                if (span.Start < cursor)
                {
                    continue;
                }

                builder.Append(_input, cursor, span.Start - cursor);
                builder.Append(span.Placeholder);
                cursor = span.Start + span.Length;
            }

            builder.Append(_input, cursor, _input.Length - cursor);

            return builder.ToString().Length;
        }

        [BenchmarkCategory("assembly")]
        [Benchmark(Description = "assembly: ValueStringBuilder, pooled exact")]
        public int AssembleWithValueStringBuilder()
        {
            var text = new ValueStringBuilder(_input.Length);
            var cursor = 0;

            foreach (var span in _spans)
            {
                if (span.Start < cursor)
                {
                    continue;
                }

                text.Append(_input.AsSpan(cursor, span.Start - cursor));
                text.Append(span.Placeholder);
                cursor = span.Start + span.Length;
            }

            text.Append(_input.AsSpan(cursor, _input.Length - cursor));

            return text.ToString().Length;
        }
    }
}
