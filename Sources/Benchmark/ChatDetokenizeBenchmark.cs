// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using DevOnBike.Overfit.LanguageModels.Chat;
using DevOnBike.Overfit.LanguageModels.Tokenizers;

namespace Benchmarks
{
    /// <summary>
    ///     `XC-45`: what does the streaming detokenize step cost per reply on a tokenizer that has <b>no</b>
    ///     zero-allocation decode — i.e. how much is riding on one <c>StringBuilder</c> at
    ///     <c>HuggingFaceBpeTokenizer.cs:182</c>?
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         <b>The site is per TOKEN, not per decode call, and that is what `XC-44` got wrong.</b> The chain
    ///         is three lines and it is worth naming exactly: <c>IncrementalDetokenizer.Decode</c>
    ///         (<c>IncrementalDetokenizer.cs:110-112</c>) branches on
    ///         <c>ITokenizer.SupportsZeroAllocationDecode</c> and, when it is <see langword="false"/>, calls
    ///         <c>DecodeToString</c>. <c>HuggingFaceBpeTokenizer</c> returns <see langword="false"/>
    ///         (<c>HuggingFaceBpeTokenizer.cs:71</c>). And the detokenizer decodes the <b>whole run so far</b>
    ///         on every token, by design — a byte-level BPE can re-render earlier text once a following byte
    ///         arrives, so decoding only the newest id is wrong. So one generated token builds a string of the
    ///         entire reply to date, with an unsized <c>StringBuilder</c> plus a <c>List&lt;byte&gt;</c>. The
    ///         cost is quadratic in the reply length by construction, which the detokenizer's own doc comment
    ///         states.
    ///     </para>
    ///     <para>
    ///         <b>What this measures and what it deliberately leaves out.</b> One benchmark call is a whole
    ///         reply: <c>Tokens</c> successive <c>TryAdvance</c> calls over a growing prefix, which is exactly
    ///         the loop <c>ChatSession.Generate</c> runs. It excludes the model forward, and that exclusion is
    ///         the point — the forward is milliseconds per token on any real model, so anything here is
    ///         invisible in wall time and the honest question is <b>bytes</b>. Read the Allocated column as
    ///         garbage per reply, and compare it against the project's zero-allocation-decode claim rather
    ///         than against the clock.
    ///     </para>
    ///     <para>
    ///         <b>No A/B arm, on purpose.</b> The counterfactual — the same path with a
    ///         <c>ValueStringBuilder</c> — cannot be run without changing production code, and `XC-45` is a
    ///         proposal task, not a migration. Pairing this absolute number with the
    ///         <c>ValueStringBuilderBenchmark</c> sweep at the matching character length gives the expected
    ///         saving; that is an inference from two measurements and must be reported as one, not as a
    ///         measured A/B. Substituting a different tokenizer as the "fast arm" would be worse than nothing:
    ///         a different vocabulary decodes different text, so the lever would not be the builder.
    ///     </para>
    ///     <para>
    ///         <b>Needs a real fixture</b> — a HuggingFace tokenizer directory, default <c>C:\qwen3b</c>,
    ///         overridable with <c>OVERFIT_QWEN3B_DIR</c>. It is 152k vocabulary, so <c>Load</c> is slow and
    ///         belongs in <c>GlobalSetup</c> where it is not timed.
    ///     </para>
    ///     <para><b>NOT MEASURED YET</b> — written 2026-08-13. Fill in with numbers, box and build.</para>
    /// </remarks>
    [SimpleJob(warmupCount: 3, iterationCount: 10)]
    [MemoryDiagnoser]
    public class ChatDetokenizeBenchmark
    {
        /// <summary>Generated tokens in the reply. A short answer and a paragraph are different shapes.</summary>
        [Params(64, 256)]
        public int Tokens
        {
            get; set;
        }

        private HuggingFaceBpeTokenizer _tokenizer = null!;
        private int[] _run = [];

        [GlobalSetup]
        public void Setup()
        {
            var directory = Environment.GetEnvironmentVariable("OVERFIT_QWEN3B_DIR");

            if (string.IsNullOrWhiteSpace(directory))
            {
                directory = @"C:\qwen3b";
            }

            if (!Directory.Exists(directory))
            {
                throw new DirectoryNotFoundException(
                    $"Tokenizer fixture directory '{directory}' not found. Set OVERFIT_QWEN3B_DIR. This "
                    + "benchmark needs a real HuggingFace tokenizer because the cost being measured is the "
                    + "tokenizer's own decode, and a toy vocabulary would not reproduce it.");
            }

            _tokenizer = HuggingFaceBpeTokenizer.Load(directory);

            if (_tokenizer.SupportsZeroAllocationDecode)
            {
                throw new InvalidOperationException(
                    "This tokenizer reports SupportsZeroAllocationDecode = true, so IncrementalDetokenizer "
                    + "will take the span path and this benchmark would measure the OPPOSITE of its subject. "
                    + "Refusing rather than reporting a number for the wrong branch.");
            }

            _run = BuildRun(Tokens);
        }

        /// <summary>
        /// A realistic reply, encoded with the same tokenizer that will decode it, truncated or padded by
        /// repetition to exactly <paramref name="count"/> ids. Encoding real prose rather than picking ids at
        /// random matters: random ids across a 152k vocabulary hit byte-fallback pieces far more often than
        /// real text does, which would change what the decode loop actually executes.
        /// </summary>
        private int[] BuildRun(int count)
        {
            const string Reply =
                "The migration failed because the connection pool was exhausted before the schema lock was "
                + "released. I have raised the pool ceiling and re-run the job; it completed in four minutes "
                + "and the row counts now match the source system exactly. No data was lost at any point, and "
                + "the retry was idempotent, so a second run would have been safe as well.";

            var scratch = new int[Reply.Length + 16];
            var written = _tokenizer.Encode(Reply.AsSpan(), scratch);

            if (written <= 0)
            {
                throw new InvalidOperationException("Encoding the sample reply produced no tokens.");
            }

            var run = new int[count];

            for (var i = 0; i < count; i++)
            {
                run[i] = scratch[i % written];
            }

            return run;
        }

        /// <summary>
        /// One call = one whole reply of <see cref="Tokens"/> tokens, streamed. The returned character count
        /// keeps the deltas from being eliminated.
        /// </summary>
        [Benchmark(Description = "detokenize a whole reply, token by token (HF BPE, no zero-alloc decode)")]
        public int StreamWholeReply()
        {
            using var detokenizer = new IncrementalDetokenizer();
            var emitted = 0;

            for (var i = 1; i <= _run.Length; i++)
            {
                if (detokenizer.TryAdvance(_tokenizer, _run.AsSpan(0, i), out var delta))
                {
                    emitted += delta.Length;
                }
            }

            return emitted;
        }
    }
}
