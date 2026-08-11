// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit.Tests
{
    /// <summary>
    /// A <see cref="FactAttribute"/> for tests too slow for the default suite: real models loaded from
    /// <c>C:\qwen3b</c> and friends, training loops, profilers, PyTorch-parity diagnostics — anything over
    /// roughly ten seconds on the dev box. Skipped by default, so <c>dotnet test -c Release</c> stays fast and
    /// holds only correctness checks.
    ///
    /// <para><b>Set <c>OVERFIT_RUN_LONG=1</c> to run them.</b> Until 2026-08-06 there was no switch: the skip
    /// was unconditional and its own message told the reader to <i>remove the Skip property</i> — that is, to
    /// edit source in order to run a test. Counted from the test runner's own results on 2026-08-07:
    /// <b>256 of these against 1715 ordinary facts</b>, 195 under <c>LanguageModels</c>, which is the model
    /// loaders and the runtime. Nothing ran them on any schedule. <b>A test that never runs is worse than no
    /// test, because it looks like coverage</b> — and editing an attribute to run one has already gone wrong
    /// here, leaving a <c>[LongFact]</c> flipped to <c>[Fact]</c> until somebody noticed it by hand.
    ///
    /// <para>An earlier version of this paragraph said 358, from counting occurrences of the text
    /// <c>[LongFact</c> — which also counts every mention in a comment or a doc block, 103 of them. The
    /// number that reconciles is the runner's: 256 skipped here plus 5 deliberate <c>[Fact(Skip=...)]</c>
    /// equals the 261 it reports. <b>Count what executes, not what the source mentions.</b></para>
    ///
    /// <para>The environment variable is the pattern this repository already uses twice for a deliberate
    /// override of exactly this kind: <see cref="MeasurementExclusion"/> reads
    /// <c>OVERFIT_ALLOW_CONCURRENT_MEASUREMENT</c>, and <c>SmallModelFact</c> sets <c>Skip</c> conditionally
    /// on a fixture being present rather than unconditionally.</para>
    ///
    /// <para><b>The default is unchanged, deliberately.</b> Without the variable these skip exactly as before,
    /// so no existing run becomes slower by accident. Note also that a long run still takes the machine-wide
    /// measurement mutex through <see cref="MeasurementExclusion"/>: it refuses to start while a benchmark or
    /// an anomaly-guard measurement holds the box, which is correct — 256 model loads inside somebody's
    /// sampling window produces two wrong answers instead of one result.</para>
    ///
    /// <para><b>The optional runtime.</b> <c>[LongFact("32s")]</c> records how long the test was measured to
    /// take. "Run these before a release" means something very different at 115 ms than at 15 minutes, and
    /// until 2026-08-07 nobody had a single number for any of them, because none had ever executed. The one
    /// that turned out to take <b>15min51s</b> sits in a file that gave no hint of it.</para>
    ///
    /// <para><b>Read every value as one sample, not an average.</b> They come from a single run on one dev
    /// box (2026-08-07), each test in its own process, and they include a cold process start plus reading
    /// multi-gigabyte weights off disk. A second run on a warm file cache is faster and that is not captured
    /// here. A value is absent when the test has never completed — a failed test has a duration and no
    /// meaning, and the heavy group in <c>Scripts/longfact_heavy.txt</c> is deliberately not run per merge.</para>
    /// </summary>
    internal class LongFact : FactAttribute
    {
        /// <summary>Set to <c>1</c> to run long tests instead of skipping them.</summary>
        internal const string RunVariable = "OVERFIT_RUN_LONG";

        /// <param name="runtime">
        /// Measured wall-clock in human notation — <c>"115ms"</c>, <c>"32s"</c>, <c>"4min"</c>,
        /// <c>"15min51s"</c>, <c>"2h15min"</c>. Human first on purpose: this is read far more often than it
        /// is parsed, and a reader deciding whether to run a subset wants "15min51s", not 951. Machine
        /// readers use <see cref="TryParseRuntime"/>. Leave it off when the test has never been measured;
        /// an absent value says "unknown", and a wrong one says something worse.
        /// </param>
        public LongFact(
            string runtime = null,
            [CallerFilePath] string sourceFilePath = null,
            [CallerLineNumber] int sourceLineNumber = -1)
            : base(sourceFilePath, sourceLineNumber)
        {
            Runtime = runtime;

            // Polarity matters more here than anything else in the file: skipping is the DEFAULT and only an
            // explicit "1" lifts it. Inverting this would silently pull 256 model-loading tests into every
            // `dotnet test` and turn a fast suite into an hours-long one — a failure that looks like the
            // suite simply got slow.
            if (Environment.GetEnvironmentVariable(RunVariable) == "1")
            {
                return;
            }

            Skip = "Long-running: skipped by default. Set OVERFIT_RUN_LONG=1 to run these — they load real "
                + "models, take minutes, and hold the machine measurement mutex while they do.";
        }

        /// <summary>Measured wall-clock in human notation, or <see langword="null"/> if never measured.</summary>
        public string Runtime
        {
            get;
        }

        /// <summary>
        /// Parses the notation written by <c>Scripts/longfact_annotate.py</c>: an optional hours part, an
        /// optional minutes part and an optional seconds part (<c>"2h15min"</c>, <c>"15min51s"</c>,
        /// <c>"32s"</c>), or a bare milliseconds value (<c>"115ms"</c>).
        ///
        /// <para>It exists so the value is <b>data rather than decoration</b> — a gate can sum it to say what
        /// it costs, sort by it to report the worst offenders, or filter on it. A field nothing can read back
        /// is a comment with extra syntax.</para>
        /// </summary>
        /// <returns><see langword="false"/> on null, empty or unrecognised input; <paramref name="value"/> is
        /// then <see cref="TimeSpan.Zero"/>. Unparseable input is not an exception: these strings are written
        /// by tooling into source, and a malformed one should make a report say "unknown", not fail a run.</returns>
        public static bool TryParseRuntime(string runtime, out TimeSpan value)
        {
            value = TimeSpan.Zero;

            if (string.IsNullOrWhiteSpace(runtime))
            {
                return false;
            }

            var text = runtime.Trim();
            var total = TimeSpan.Zero;
            var position = 0;
            var matched = false;

            // Milliseconds are exclusive — "115ms" is a whole value, never a suffix on something larger,
            // so it is handled before the h/min/s walk rather than inside it. Checked first because "ms"
            // ends in "s" and would otherwise be read as seconds with a stray "m".
            if (text.EndsWith("ms", StringComparison.OrdinalIgnoreCase))
            {
                if (!double.TryParse(text[..^2], System.Globalization.NumberStyles.Float,
                        System.Globalization.CultureInfo.InvariantCulture, out var milliseconds))
                {
                    return false;
                }

                value = TimeSpan.FromMilliseconds(milliseconds);

                return true;
            }

            // BOUND: at most three units (h, min, s), so at most three passes; `position` strictly advances
            // because every accepted unit consumes at least one digit and one letter.
            foreach (var (suffix, factor) in new[] { ("h", 3600.0), ("min", 60.0), ("s", 1.0) })
            {
                var digits = position;

                while (digits < text.Length && (char.IsDigit(text[digits]) || text[digits] == '.'))
                {
                    digits++;
                }

                if (digits == position || !text.AsSpan(digits).StartsWith(suffix,
                        StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                if (!double.TryParse(text[position..digits], System.Globalization.NumberStyles.Float,
                        System.Globalization.CultureInfo.InvariantCulture, out var quantity))
                {
                    return false;
                }

                total += TimeSpan.FromSeconds(quantity * factor);
                position = digits + suffix.Length;
                matched = true;
            }

            if (!matched || position != text.Length)
            {
                return false;
            }

            value = total;

            return true;
        }
    }
}
