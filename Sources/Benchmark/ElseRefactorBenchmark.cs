// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;

namespace Benchmarks
{
    /// <summary>
    ///     A/B for the OVERFIT021 sweep: does rewriting <c>else</c> / <c>else if</c> into guard clauses,
    ///     <c>continue</c> and ternaries cost anything at runtime?
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         The sweep uses four shapes, and they carry very different risk, which is why they are measured
    ///         separately rather than as one blended number:
    ///     </para>
    ///     <list type="bullet">
    ///         <item><b>Ternary</b> — two-way assignment of one variable (the <c>BoostedTreeModel</c> NaN check).
    ///               Expected to be identical; both forms usually become the same branch or a cmov.</item>
    ///         <item><b>Continue</b> — an if/else-if/else chain inside a loop rewritten as guard clauses with
    ///               <c>continue</c> (the IBAN validator shape). Expected identical: same branches, different
    ///               polarity.</item>
    ///         <item><b>Inverted guard</b> — the rare-branch-first inversion used in
    ///               <c>OverfitParallel.WorkerLoop</c>. Polarity change only, but it is the one place where a
    ///               static branch-prediction hint could in principle differ.</item>
    ///         <item><b>Method extraction</b> — the genuinely risky one. Where inversion is not enough
    ///               (<c>DataParallelTrainer</c>, <c>OverfitLicense</c>) the branch body moves into its own
    ///               method. Measured in three variants — plain, <c>AggressiveInlining</c>, and
    ///               <c>NoInlining</c> as the pessimistic bound — because a method the JIT declines to inline
    ///               is the only way this refactor can actually cost something.</item>
    ///     </list>
    ///     <para>
    ///         <b>Measured 2026-07-19 (net10.0, Release).</b> Assign 1.01, Chain 1.00, Guard 1.01 — the
    ///         inversion / ternary / continue rewrites are free. Extraction: free (1.01) when the JIT inlines
    ///         the method, <b>2.25x</b> in a tight loop when it does not.
    ///     </para>
    ///     <para>
    ///         <b>Resolved anomaly, kept as a warning.</b> The first version of the Extract category reported
    ///         the <c>NoInlining</c> variant at 0.63 — reliably FASTER than the fully inline body, which is
    ///         backwards. The cause was the saturating <c>float</c>-&gt;<c>long</c> cast in the synthetic
    ///         branch body, not anything about calls: that cast is a multi-instruction sequence
    ///         (<c>vcvttss2si</c> + <c>vucomiss</c> + saturation select) and when inlined it extends the
    ///         dependency chain through the accumulator, costing ~4.7 us; inside a non-inlined callee it hides
    ///         entirely behind call overhead. Proof is the NoCast category: the same shapes without the cast
    ///         give 1.00 / 1.01 / 2.25, and the NoInlining timing barely moves (4.04 vs 4.11 us) whether the
    ///         cast is present or not. Lesson: a microbenchmark whose branch body contains an expensive,
    ///         placement-sensitive operation measures that operation, not the shape under test.
    ///     </para>
    ///     <para>
    ///         Each pair is <c>[Benchmark(Baseline = true)]</c> on the <c>else</c> form so the ratio column is
    ///         the answer. Results are accumulated and returned so nothing is dead-code eliminated, and the
    ///         input data is deliberately branch-unfriendly (pseudo-random NaNs / character classes) — a
    ///         perfectly predictable input would hide any branch-layout difference, which is precisely what
    ///         this benchmark exists to detect.
    ///     </para>
    /// </remarks>
    // Deliberately NOT [Config(typeof(BenchmarkConfig))]: that shared config pins InvocationCount=1 /
    // UnrollFactor=1, which suits multi-millisecond model workloads but leaves a ~15 us microbenchmark
    // measuring mostly timer noise (the first run of this file produced RatioSD up to 0.44 and a bogus
    // 1.61x "regression"). BDN's default job auto-tunes the invocation count instead.
    [SimpleJob(warmupCount: 8, iterationCount: 20)]
    [MemoryDiagnoser]
    [CategoriesColumn]
    [GroupBenchmarksBy(BenchmarkLogicalGroupRule.ByCategory)]
    public class ElseRefactorBenchmark
    {
        private const int N = 4096;

        private float[] _values = [];
        private byte[] _defaultLeft = [];
        private float[] _thresholds = [];
        private char[] _chars = [];
        private int[] _indices = [];

        [GlobalSetup]
        public void Setup()
        {
            var rng = new Random(20260719);

            _values = new float[N];
            _defaultLeft = new byte[N];
            _thresholds = new float[N];
            _chars = new char[N];
            _indices = new int[N];

            for (var i = 0; i < N; i++)
            {
                // ~15% NaN: frequent enough that the branch is not free, rare enough to stay realistic.
                _values[i] = rng.NextDouble() < 0.15 ? float.NaN : (float)((rng.NextDouble() * 2.0) - 1.0);
                _defaultLeft[i] = (byte)(rng.Next(2));
                _thresholds[i] = (float)((rng.NextDouble() * 2.0) - 1.0);

                var roll = rng.Next(100);
                _chars[i] = roll < 55 ? (char)('0' + rng.Next(10))
                    : roll < 97 ? (char)('A' + rng.Next(26))
                    : '-';

                // Mostly in-range, occasionally not — the WorkerLoop claim-index shape.
                _indices[i] = rng.Next(100) < 96 ? rng.Next(N) : N + rng.Next(8);
            }
        }

        // ── Shape 1: two-way assignment — if/else vs ternary ────────────────────────────────

        [Benchmark(Baseline = true, Description = "assign: if/else")]
        [BenchmarkCategory("Assign")]
        public int AssignIfElse()
        {
            var left = 0;
            for (var i = 0; i < N; i++)
            {
                var value = _values[i];
                bool goLeft;

                if (float.IsNaN(value))
                {
                    goLeft = _defaultLeft[i] != 0;
                }
                else
                {
                    goLeft = value < _thresholds[i];
                }

                if (goLeft)
                {
                    left++;
                }
            }
            return left;
        }

        [Benchmark(Description = "assign: ternary")]
        [BenchmarkCategory("Assign")]
        public int AssignTernary()
        {
            var left = 0;
            for (var i = 0; i < N; i++)
            {
                var value = _values[i];
                var goLeft = float.IsNaN(value) ? _defaultLeft[i] != 0 : value < _thresholds[i];

                if (goLeft)
                {
                    left++;
                }
            }
            return left;
        }

        // ── Shape 2: chain inside a loop — else-if vs continue guards ───────────────────────

        [Benchmark(Baseline = true, Description = "chain: if/else if/else")]
        [BenchmarkCategory("Chain")]
        public long ChainElseIf()
        {
            long remainder = 0;
            for (var i = 0; i < N; i++)
            {
                var c = _chars[i];

                if (char.IsDigit(c))
                {
                    remainder = ((remainder * 10) + (c - '0')) % 97;
                }
                else if (c is >= 'A' and <= 'Z')
                {
                    remainder = ((remainder * 100) + (c - 'A' + 10)) % 97;
                }
                else
                {
                    remainder = (remainder + 1) % 97;
                }
            }
            return remainder;
        }

        [Benchmark(Description = "chain: continue guards")]
        [BenchmarkCategory("Chain")]
        public long ChainContinue()
        {
            long remainder = 0;
            for (var i = 0; i < N; i++)
            {
                var c = _chars[i];

                if (char.IsDigit(c))
                {
                    remainder = ((remainder * 10) + (c - '0')) % 97;
                    continue;
                }

                if (c is >= 'A' and <= 'Z')
                {
                    remainder = ((remainder * 100) + (c - 'A' + 10)) % 97;
                    continue;
                }

                remainder = (remainder + 1) % 97;
            }
            return remainder;
        }

        // ── Shape 3: rare-branch-first inversion (WorkerLoop) ───────────────────────────────

        [Benchmark(Baseline = true, Description = "guard: common-first + else")]
        [BenchmarkCategory("Guard")]
        public long GuardCommonFirst()
        {
            long acc = 0;
            for (var i = 0; i < N; i++)
            {
                var index = _indices[i];

                if (index < N)
                {
                    acc += index;
                }
                else
                {
                    acc -= 1;
                }
            }
            return acc;
        }

        [Benchmark(Description = "guard: rare-first + continue")]
        [BenchmarkCategory("Guard")]
        public long GuardRareFirst()
        {
            long acc = 0;
            for (var i = 0; i < N; i++)
            {
                var index = _indices[i];

                if (index >= N)
                {
                    acc -= 1;
                    continue;
                }

                acc += index;
            }
            return acc;
        }

        // ── Shape 4: method extraction — the only shape that can genuinely cost ─────────────

        [Benchmark(Baseline = true, Description = "extract: inline else body")]
        [BenchmarkCategory("Extract")]
        public long ExtractInline()
        {
            long acc = 0;
            for (var i = 0; i < N; i++)
            {
                var value = _values[i];

                if (float.IsNaN(value))
                {
                    acc += _defaultLeft[i];
                }
                else
                {
                    // The "long branch body" that forced extraction at the real call sites. The threshold is
                    // hoisted into a local so this does exactly ONE array load, matching what the extracted
                    // variants get for free by taking it as a parameter — without this the categories compare
                    // different amounts of work, not different call shapes.
                    var threshold = _thresholds[i];
                    var scaled = value * threshold;
                    var shifted = scaled + threshold;
                    acc += (long)(shifted * 8f) & 0xFF;
                }
            }
            return acc;
        }

        [Benchmark(Description = "extract: method (default JIT)")]
        [BenchmarkCategory("Extract")]
        public long ExtractMethod()
        {
            long acc = 0;
            for (var i = 0; i < N; i++)
            {
                var value = _values[i];

                if (float.IsNaN(value))
                {
                    acc += _defaultLeft[i];
                    continue;
                }

                acc += Blend(value, _thresholds[i]);
            }
            return acc;
        }

        [Benchmark(Description = "extract: method (AggressiveInlining)")]
        [BenchmarkCategory("Extract")]
        public long ExtractMethodInlined()
        {
            long acc = 0;
            for (var i = 0; i < N; i++)
            {
                var value = _values[i];

                if (float.IsNaN(value))
                {
                    acc += _defaultLeft[i];
                    continue;
                }

                acc += BlendInlined(value, _thresholds[i]);
            }
            return acc;
        }

        [Benchmark(Description = "extract: method (NoInlining — worst case)")]
        [BenchmarkCategory("Extract")]
        public long ExtractMethodNotInlined()
        {
            long acc = 0;
            for (var i = 0; i < N; i++)
            {
                var value = _values[i];

                if (float.IsNaN(value))
                {
                    acc += _defaultLeft[i];
                    continue;
                }

                acc += BlendNotInlined(value, _thresholds[i]);
            }
            return acc;
        }

        // ── Shape 4b: identical structure, but WITHOUT the saturating float->long cast ─────
        // Isolates the one instruction sequence that differs in placement between the inline and
        // NoInlining disassembly. If the anomaly vanishes here, the cast is the cause, not the call.

        [Benchmark(Baseline = true, Description = "nocast: inline else body")]
        [BenchmarkCategory("NoCast")]
        public float NoCastInline()
        {
            var acc = 0f;
            for (var i = 0; i < N; i++)
            {
                var value = _values[i];

                if (float.IsNaN(value))
                {
                    acc += _defaultLeft[i];
                }
                else
                {
                    var threshold = _thresholds[i];
                    var scaled = value * threshold;
                    acc += (scaled + threshold) * 8f;
                }
            }
            return acc;
        }

        [Benchmark(Description = "nocast: method (default JIT)")]
        [BenchmarkCategory("NoCast")]
        public float NoCastMethod()
        {
            var acc = 0f;
            for (var i = 0; i < N; i++)
            {
                var value = _values[i];

                if (float.IsNaN(value))
                {
                    acc += _defaultLeft[i];
                    continue;
                }

                acc += BlendFloat(value, _thresholds[i]);
            }
            return acc;
        }

        [Benchmark(Description = "nocast: method (NoInlining)")]
        [BenchmarkCategory("NoCast")]
        public float NoCastMethodNotInlined()
        {
            var acc = 0f;
            for (var i = 0; i < N; i++)
            {
                var value = _values[i];

                if (float.IsNaN(value))
                {
                    acc += _defaultLeft[i];
                    continue;
                }

                acc += BlendFloatNotInlined(value, _thresholds[i]);
            }
            return acc;
        }

        private static float BlendFloat(float value, float threshold)
        {
            var scaled = value * threshold;
            return (scaled + threshold) * 8f;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static float BlendFloatNotInlined(float value, float threshold)
        {
            var scaled = value * threshold;
            return (scaled + threshold) * 8f;
        }

        private static long Blend(float value, float threshold)
        {
            var scaled = value * threshold;
            var shifted = scaled + threshold;
            return (long)(shifted * 8f) & 0xFF;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long BlendInlined(float value, float threshold)
        {
            var scaled = value * threshold;
            var shifted = scaled + threshold;
            return (long)(shifted * 8f) & 0xFF;
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static long BlendNotInlined(float value, float threshold)
        {
            var scaled = value * threshold;
            var shifted = scaled + threshold;
            return (long)(shifted * 8f) & 0xFF;
        }
    }
}
