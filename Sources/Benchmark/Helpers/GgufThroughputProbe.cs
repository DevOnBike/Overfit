// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Runtime;

namespace Benchmarks.Helpers
{
    /// <summary>
    /// End-to-end prefill and decode of a real GGUF, in the two shapes <c>llama-bench</c> measures, so the
    /// two engines' numbers describe the same thing.
    ///
    /// <para><b>Why the quantities are llama.cpp's and not this project's.</b> <c>pp512</c> is the rate of
    /// processing a 512-token prompt in one go; <c>tg128</c> is the rate of generating 128 tokens one at a
    /// time from an empty context. A metric only this repository computes cannot be compared with anything,
    /// which is the position <c>XC-76</c> found the project in: sixty-plus benchmark classes and not one
    /// that loads a GGUF and measures decode end to end.</para>
    ///
    /// <para><b>Synthetic token ids, deliberately, and that is what llama-bench does too.</b> Its
    /// <c>test_prompt</c> fills the batch with pseudo-random ids drawn against the vocabulary size. Tokenizing
    /// real text would put the tokenizer inside a measurement of the transformer, and would make the token
    /// count depend on the text. The ids here come from a fixed seed so two runs feed identical input.</para>
    ///
    /// <para><b>What is inside the timer and what is not.</b> llama-bench clears the KV cache before it
    /// starts its clock and times only the decode calls, so this does the same:
    /// <see cref="ResetForPrefill"/> and <see cref="ResetForDecode"/> are the untimed setup and
    /// <see cref="RunPrefill"/> / <see cref="RunDecode"/> are the timed bodies. The decode setup prefills a
    /// single token because a session with an empty cache has no logits to sample from; llama.cpp's decode
    /// loop starts at position 0 and this one at position 1, which is one position of KV attention out of
    /// 128 and is stated rather than hidden.</para>
    ///
    /// <para><b>Greedy sampling is included in the decode timing and llama.cpp's equivalent is not.</b> Its
    /// <c>test_gen</c> calls <c>llama_decode</c> and never chooses a token, while this must, because
    /// <see cref="CachedLlamaSession.GenerateNextToken(in SamplingOptions)"/> is the API a caller uses. An
    /// argmax over the vocabulary is the smaller side of that difference by three orders of magnitude
    /// against a ~32 ms token, but it is a difference and it favours llama.cpp.</para>
    /// </summary>
    internal sealed class GgufThroughputProbe : IDisposable
    {
        /// <summary>Fixed so two runs feed byte-identical input; the value itself has no meaning.</summary>
        private const int TokenSeed = 20260825;

        private readonly CachedLlamaInferenceEngine _engine;
        private readonly CachedLlamaSession _session;
        private readonly int[] _promptTokens;
        private readonly int[] _decodeSeedToken;
        private readonly int _generateTokens;

        public GgufThroughputProbe(string modelPath, int promptTokens, int generateTokens)
        {
            ModelPath = modelPath;
            PromptTokens = promptTokens;
            _generateTokens = generateTokens;

            _engine = CachedLlamaInferenceEngine.LoadGguf(modelPath);

            // Matches llama-bench, which sizes the context to the test it is about to run rather than to
            // the model's maximum. A larger KV cache is a different memory footprint and therefore a
            // different measurement.
            var context = Math.Max(promptTokens, generateTokens + 1) + 1;
            _session = _engine.CreateSession(context);

            var vocabulary = _engine.Config.VocabSize;
            var random = new Random(TokenSeed);

            _promptTokens = new int[Math.Max(promptTokens, 1)];

            for (var i = 0; i < _promptTokens.Length; i++)
            {
                _promptTokens[i] = random.Next(vocabulary);
            }

            _decodeSeedToken = [_promptTokens[0]];
        }

        public string ModelPath
        {
            get;
        }

        public int PromptTokens
        {
            get;
        }

        /// <summary>Workers the general parallel pool resolved to — what a caller's thread request became.</summary>
        public static int WorkerCount => OverfitParallel.WorkerCount;

        /// <summary>
        /// Workers the <b>decode</b> dispatch resolved to, which is a different number from
        /// <see cref="WorkerCount"/> and is why both are reported.
        ///
        /// <para><c>ResolveDecodeMaxWorkers</c> caps decode at <c>min(workers - 1, 10)</c> unless
        /// <c>OVERFIT_DECODE_WORKERS</c> overrides it, so a request of 16 threads produces 16 general workers
        /// and 10 decode workers. Reporting one number while setting two is a defect this project has already
        /// paid for: on 2026-08-20 a comparison harness did exactly that and two configurations six-fold
        /// apart were recorded as the same one.</para>
        /// </summary>
        public static int DecodeWorkerCount => OverfitParallel.DecodeMaxWorkers;

        /// <summary>Untimed: llama-bench clears the cache before it starts its clock.</summary>
        public void ResetForPrefill()
        {
            _session.Reset();
        }

        /// <summary>The timed body of a <c>pp</c> repetition.</summary>
        public void RunPrefill()
        {
            _session.Prefill(_promptTokens);
        }

        /// <summary>
        /// Untimed: clears the cache and seeds one token, because a session at position 0 has no logits to
        /// sample from and <see cref="CachedLlamaSession.GenerateNextToken(in SamplingOptions)"/> refuses.
        /// </summary>
        public void ResetForDecode()
        {
            _session.Reset();
            _session.Prefill(_decodeSeedToken);
        }

        /// <summary>The timed body of a <c>tg</c> repetition: one forward pass per generated token.</summary>
        public int RunDecode()
        {
            var sampling = SamplingOptions.Greedy;
            var last = 0;

            for (var i = 0; i < _generateTokens; i++)
            {
                last = _session.GenerateNextToken(in sampling);
            }

            return last;
        }

        public void Dispose()
        {
            _session.Dispose();
            _engine.Dispose();
        }
    }
}
