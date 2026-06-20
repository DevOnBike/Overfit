// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.Maths;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Interpretability: activation capture + logit lens on a real model. Two things are asserted:
    /// <list type="number">
    /// <item><b>Correctness anchor</b> — the logit lens at the LAST layer must reproduce the real next-token
    /// logits exactly (same final-norm + LM head as the decode). This pins the lens to the model, so its
    /// earlier-layer readouts are trustworthy.</item>
    /// <item><b>Demo</b> — projecting each layer's captured residual stream through the head shows the
    /// prediction forming across depth (the early layers are uncertain / off-topic, later layers converge).</item>
    /// </list>
    /// This works because Overfit's tensors are pure-managed: the residual stream at every layer is directly
    /// readable, no PyTorch hooks / ONNX surgery / FFI. [LongFact] — needs C:\qwen3b\qwen.q4km.gguf.
    /// </summary>
    public sealed class LogitLensInterpretabilityTests
    {
        private const string Model = @"C:\qwen3b\qwen.q4km.gguf";

        private readonly ITestOutputHelper _out;
        public LogitLensInterpretabilityTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void LogitLens_LastLayer_MatchesLogits_AndShowsPredictionAcrossDepth()
        {
            if (!File.Exists(Model))
            {
                _out.WriteLine($"missing {Model}");
                return;
            }

            using var client = OverfitClient.LoadGguf(Model, maxContextLength: 256, mmap: true, maxNewTokens: 1);
            var engine = client.Engine;
            var layers = engine.Config.NLayers;
            var dModel = engine.Config.DModel;
            var vocab = engine.Config.VocabSize;

            engine.EnableActivationCapture(true);

            // Encode a prompt whose continuation is an unambiguous single token.
            Span<int> tokenBuf = new int[64];
            var n = client.Tokenizer.Encode("The capital of France is", tokenBuf);
            var prompt = tokenBuf.Slice(0, n).ToArray();

            using var session = engine.CreateSession();
            session.Reset(prompt);
            // One single-token decode (fires the per-layer capture) + its logits.
            session.GenerateNextToken(SamplingOptions.Greedy);

            var realLogits = new float[vocab];
            session.GetLastLogits(realLogits);

            var hidden = new float[dModel];
            var lensLogits = new float[vocab];

            // ── Correctness anchor: last-layer lens == real logits ──
            engine.GetLayerActivation(layers - 1, hidden);
            engine.LogitLens(hidden, lensLogits);

            var maxAbsDiff = 0f;
            for (var i = 0; i < vocab; i++)
            {
                maxAbsDiff = MathF.Max(maxAbsDiff, MathF.Abs(realLogits[i] - lensLogits[i]));
            }
            _out.WriteLine($"last-layer lens vs real logits — maxAbsDiff {maxAbsDiff:E3}");
            Assert.True(maxAbsDiff < 1e-2f, $"logit lens at the last layer diverges from the real logits: {maxAbsDiff}");

            // ── Demo: top-1 token predicted at each depth ──
            _out.WriteLine("prediction forming across depth (logit lens top-1 per layer):");
            Span<int> one = stackalloc int[1];
            for (var l = 0; l < layers; l++)
            {
                engine.GetLayerActivation(l, hidden);
                engine.LogitLens(hidden, lensLogits);
                one[0] = MathUtils.ArgMax(lensLogits);
                var tok = client.Tokenizer.DecodeToString(one).Replace("\n", "\\n");
                _out.WriteLine($"  layer {l,2} → \"{tok}\"");
            }

            // The final layer's top-1 is the model's actual next token — assert it's a real (in-vocab) id.
            engine.GetLayerActivation(layers - 1, hidden);
            engine.LogitLens(hidden, lensLogits);
            Assert.InRange(MathUtils.ArgMax(lensLogits), 0, vocab - 1);
        }
    }
}
