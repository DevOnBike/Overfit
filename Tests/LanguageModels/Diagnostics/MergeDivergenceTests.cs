// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Audio.Tts.Orpheus;
using DevOnBike.Overfit.LanguageModels.LoRA;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Localizes where the LoRA-merged inference engine (the `--fast` clone path) diverges from the trainable model
    /// that produced it. Decodes the SAME prompt through both and compares the final hidden state (post-stack,
    /// pre-final-norm). If the hiddens match, the merge forward is faithful and the garbage is downstream
    /// (sampling/LM-head); if they diverge, the fast engine's numerical path (Q8 requant + reassociated attention)
    /// does not preserve the fine-tune. [LongFact] — needs C:\orpheus + C:\myvoice\myvoice_v2.adapter.
    /// </summary>
    public sealed class MergeDivergenceTests
    {
        private const string Orpheus = @"C:\orpheus\orpheus-3b-0.1-ft-q4_k_m.gguf";
        private const string Adapter = @"C:\myvoice\myvoice_v2.adapter";
        private readonly ITestOutputHelper _out;
        public MergeDivergenceTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Merged_Vs_Trainable_FinalHidden_Diff()
        {
            if (!File.Exists(Orpheus) || !File.Exists(Adapter)) { _out.WriteLine("missing orpheus/adapter"); return; }

            using var trainer = new VoiceCloneTrainer(Orpheus, maxSeqLen: 256, new QLoRAOptions());
            trainer.LoadAdapter(Adapter);

            var dModel = trainer.Config.DModel;
            var audioBase = AudioBase(trainer);
            var prompt = OrpheusPrompt.BuildPromptTokens(trainer.Tokenizer, "Hello, this is a test.", "myvoice");
            _out.WriteLine($"prompt {prompt.Length} tok, dModel {dModel}, audioBase {audioBase}");

            // Final-hidden sanity (one engine only to avoid the shared-weight double-dispose).
            var tHidden = new float[dModel];
            trainer.DecodePromptHidden(prompt, tHidden);

            const int K = 24;
            // Trainable greedy tokens (eos set unreachable so it doesn't stop early).
            var tTokens = trainer.Generate(prompt, K, eosTokenId: -999, temperature: 0f, repeatPenalty: 1f, seed: 1);

            using var merged = trainer.BuildMergedEngine(mergeLora: true);
            using (var s = merged.CreateSession())
            {
                s.Prefill(prompt);
                Report("FULL merge vs trainable (final hidden)", tHidden, s.LastHiddenState);
            }

            // Merged greedy tokens, WITHOUT any mask (full vocab) and WITH the audio-vocab mask.
            var mFree = MergedGreedy(merged, prompt, K, audioBase, mask: false);
            var mMask = MergedGreedy(merged, prompt, K, audioBase, mask: true);

            _out.WriteLine($"trainable : [{string.Join(",", tTokens)}]");
            _out.WriteLine($"merged-free: [{string.Join(",", mFree)}]");
            _out.WriteLine($"merged-mask: [{string.Join(",", mMask)}]");
            _out.WriteLine($"first divergence (free): {FirstDiff(tTokens, mFree)}  (mask): {FirstDiff(tTokens, mMask)}");
        }

        private static int FirstDiff(int[] a, int[] b)
        {
            var n = Math.Min(a.Length, b.Length);
            for (var i = 0; i < n; i++) { if (a[i] != b[i]) { return i; } }
            return n;
        }

        private static int AudioBase(VoiceCloneTrainer t)
        {
            Span<int> ids = stackalloc int[8];
            var n = t.Tokenizer.Encode("<custom_token_0>", ids);
            return ids[0];
        }

        private static int[] MergedGreedy(DevOnBike.Overfit.LanguageModels.Runtime.CachedLlamaInferenceEngine eng,
            int[] prompt, int k, int audioBase, bool mask)
        {
            using var s = eng.CreateSession();
            s.Prefill(prompt);
            var sampling = DevOnBike.Overfit.LanguageModels.Contracts.SamplingOptions.GreedyWithPenalty(1f);
            var outp = new int[k];
            for (var i = 0; i < k; i++)
            {
                outp[i] = mask
                    ? s.GenerateNextToken(in sampling, new MaskBelow(audioBase))
                    : s.GenerateNextToken(in sampling);
            }
            return outp;
        }

        private sealed class MaskBelow : DevOnBike.Overfit.LanguageModels.Contracts.ITokenConstraint
        {
            private readonly int _base;
            public MaskBelow(int b) => _base = b;
            public bool IsComplete => false;
            public void Accept(int token) { }
            public void ApplyMask(Span<float> logits)
            {
                var n = Math.Min(_base, logits.Length);
                for (var i = 0; i < n; i++) { logits[i] = float.NegativeInfinity; }
            }
        }

        private void Report(string label, ReadOnlySpan<float> a, ReadOnlySpan<float> b)
        {
            var n = Math.Min(a.Length, b.Length);
            a = a[..n]; b = b[..n];
            var dot = TensorPrimitives.Dot(a, b);
            var na = MathF.Sqrt(TensorPrimitives.Dot(a, a));
            var nb = MathF.Sqrt(TensorPrimitives.Dot(b, b));
            var cos = dot / (na * nb + 1e-9f);
            var maxAbs = 0f;
            for (var i = 0; i < n; i++) { maxAbs = MathF.Max(maxAbs, MathF.Abs(a[i] - b[i])); }
            _out.WriteLine($"  [{label}] cos={cos:F5} |a|={na:F2} |b|={nb:F2} maxAbsDiff={maxAbs:F4}");
        }
    }
}
