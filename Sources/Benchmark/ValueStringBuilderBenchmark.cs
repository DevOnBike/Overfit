// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using DevOnBike.Overfit.Text;

namespace Benchmarks
{
    /// <summary>
    ///     Prices <c>ValueStringBuilder</c> against <c>StringBuilder</c> so its own doc comment stops being a
    ///     hypothesis. That comment makes two claims and neither has a number behind it: that the type is
    ///     worth reaching for only when a <see cref="string"/> is produced <b>and</b> the build takes several
    ///     growth steps, and that <b>small builds can be slower</b> than <c>StringBuilder</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         <b>The question is where the crossover is, not which is faster.</b> Both answers are known to
    ///         be "it depends on the length", so a single size would produce a ratio that is true and useless.
    ///         The sweep is therefore over total output length, with the number of growth steps a
    ///         deterministic function of it.
    ///     </para>
    ///     <para>
    ///         <b>The stack buffer is 256 chars = 512 B</b>, which is not an arbitrary pick: it is the
    ///         <c>OVERFIT025</c> budget, and the BCL's own <c>StackallocCharBufferSizeLimit</c>. Against it,
    ///         <c>ValueStringBuilder.Grow</c> doubles, so the parameters below sit at known growth counts:
    ///     </para>
    ///     <list type="table">
    ///         <item><term>64</term><description>0 growths — a quarter of the stack buffer used.</description></item>
    ///         <item><term>256</term><description>0 growths — fits the stack buffer exactly.</description></item>
    ///         <item><term>512</term><description>1 growth.</description></item>
    ///         <item><term>2048</term><description>3 growths.</description></item>
    ///         <item><term>16384</term><description>6 growths.</description></item>
    ///     </list>
    ///     <para>
    ///         <b>Stated before measuring, so it can be refuted.</b> The two types grow by opposite mechanisms
    ///         and that — not the stack buffer — is expected to dominate at length. <c>StringBuilder</c> chains
    ///         a new chunk and copies <i>nothing</i>, paying one copy of everything at <c>ToString</c>;
    ///         <c>ValueStringBuilder</c> rents a bigger buffer and copies everything written so far on
    ///         <i>every</i> growth, so its copy work is ~2n against ~n, in exchange for renting from the pool
    ///         instead of allocating ~log(n) chunks on the heap. The prediction is therefore: ValueStringBuilder
    ///         wins on <b>bytes allocated</b> at every size, wins on <b>time</b> only where it does not grow,
    ///         and the time ratio moves against it as the growth count rises. A result that contradicts this is
    ///         a reason to suspect this harness first — <c>ElseRefactorBenchmark</c> in this directory documents
    ///         what a backwards result looked like the last time.
    ///     </para>
    ///     <para>
    ///         <b>Two categories, because one ratio column cannot answer two questions.</b> The
    ///         <c>string</c> category ends in <c>ToString</c> on every arm, so the final string allocation is
    ///         common to all four and the ratio isolates the cost of <i>building</i>. The <c>span</c> category
    ///         ends in a copy into a caller-owned <c>char[]</c> and allocates no string at all — that is the
    ///         house pattern the doc calls "strictly better", and it is measured separately precisely so its
    ///         number is never read as a speedup over the string arms. It is not: it produces a different
    ///         thing. Its own baseline is <c>StringBuilder</c> copying into the same destination, which is the
    ///         only fair comparison available.
    ///     </para>
    ///     <para>
    ///         Appends are fixed 16-char pieces, so every arm at a given <c>Length</c> performs the identical
    ///         number of appends and the only variable is what the builder does with them. Every method returns
    ///         a value so nothing is dead-code eliminated.
    ///     </para>
    ///     <para>
    ///         <b>NOT MEASURED YET.</b> Written 2026-08-13; the box was running another agent's build and test
    ///         suite, and a benchmark on a loaded box is not a measurement. Fill this paragraph in with the
    ///         numbers, the box and the build when it has actually been run — and if the result contradicts the
    ///         prediction above, leave the prediction in place and say what was wrong with it.
    ///     </para>
    /// </remarks>
    // Deliberately NOT [Config(typeof(BenchmarkConfig))]. That shared config pins InvocationCount=1 /
    // UnrollFactor=1, which suits multi-millisecond model workloads and leaves work at this scale — the 64-char
    // arm is tens of nanoseconds — measuring timer resolution rather than the code. The same trap produced a
    // RatioSD of 0.44 and a phantom 1.61x regression in ElseRefactorBenchmark. BDN's default job runs a pilot
    // and auto-tunes the invocation count, which is what work this small needs.
    [SimpleJob(warmupCount: 8, iterationCount: 20)]
    [MemoryDiagnoser]
    [CategoriesColumn]
    [GroupBenchmarksBy(BenchmarkLogicalGroupRule.ByCategory)]
    public class ValueStringBuilderBenchmark
    {
        /// <summary>512 B — the OVERFIT025 budget and the BCL's own char-stackalloc limit.</summary>
        private const int StackChars = 256;

        private const int PieceChars = 16;

        [Params(64, 256, 512, 2048, 16384)]
        public int Length
        {
            get; set;
        }

        private string _piece = string.Empty;
        private int _pieces;

        /// <summary>
        /// The destination the <c>span</c> category writes into — allocated once, owned by the caller, and
        /// deliberately oversized so a too-small destination never turns into a measurement of the failure path.
        /// </summary>
        private char[] _destination = [];

        [GlobalSetup]
        public void Setup()
        {
            _piece = new string('x', PieceChars);
            _pieces = Length / PieceChars;
            _destination = new char[Length + PieceChars];
        }

        // ── Category "string": a string is produced, which is the only case the doc endorses ──────────

        [BenchmarkCategory("string")]
        [Benchmark(Baseline = true, Description = "string: StringBuilder, default capacity")]
        public int StringBuilderDefault()
        {
            var builder = new StringBuilder();

            for (var i = 0; i < _pieces; i++)
            {
                builder.Append(_piece);
            }

            return builder.ToString().Length;
        }

        /// <summary>
        /// The shape most existing sites in this repository actually use — a capacity is passed. Separated from
        /// the default-capacity arm because otherwise "ValueStringBuilder is faster" would be indistinguishable
        /// from "pre-sizing is faster", and the second is free to apply without changing type.
        /// </summary>
        [BenchmarkCategory("string")]
        [Benchmark(Description = "string: StringBuilder, pre-sized")]
        public int StringBuilderPresized()
        {
            var builder = new StringBuilder(Length);

            for (var i = 0; i < _pieces; i++)
            {
                builder.Append(_piece);
            }

            return builder.ToString().Length;
        }

        [BenchmarkCategory("string")]
        [Benchmark(Description = "string: ValueStringBuilder, 256-char stack")]
        public int ValueStringBuilderStack()
        {
            Span<char> scratch = stackalloc char[StackChars];
            var text = new ValueStringBuilder(scratch);

            for (var i = 0; i < _pieces; i++)
            {
                text.Append(_piece);
            }

            // ToString disposes. That is the documented contract of the type, not an omission here.
            return text.ToString().Length;
        }

        /// <summary>
        /// The pooled constructor sized to the answer — no stack involvement and no growth at any parameter.
        /// It isolates the stack buffer as a variable: the difference between this and the stack arm at 512+ is
        /// the cost of growing, and at 64 it is the cost of the stack reservation itself.
        /// </summary>
        [BenchmarkCategory("string")]
        [Benchmark(Description = "string: ValueStringBuilder, pooled exact")]
        public int ValueStringBuilderPooled()
        {
            var text = new ValueStringBuilder(Length);

            for (var i = 0; i < _pieces; i++)
            {
                text.Append(_piece);
            }

            return text.ToString().Length;
        }

        // ── Category "span": no string produced. NOT comparable to the arms above — see the class remarks ──

        [BenchmarkCategory("span")]
        [Benchmark(Baseline = true, Description = "span: StringBuilder -> CopyTo")]
        public int StringBuilderToSpan()
        {
            var builder = new StringBuilder();

            for (var i = 0; i < _pieces; i++)
            {
                builder.Append(_piece);
            }

            builder.CopyTo(0, _destination, 0, builder.Length);

            return builder.Length;
        }

        [BenchmarkCategory("span")]
        [Benchmark(Description = "span: ValueStringBuilder -> TryCopyTo")]
        public int ValueStringBuilderToSpan()
        {
            Span<char> scratch = stackalloc char[StackChars];
            var text = new ValueStringBuilder(scratch);

            for (var i = 0; i < _pieces; i++)
            {
                text.Append(_piece);
            }

            // TryCopyTo deliberately does not dispose — the retry it enables is the reason. Dispose explicitly.
            text.TryCopyTo(_destination, out var written);
            text.Dispose();

            return written;
        }
    }
}
