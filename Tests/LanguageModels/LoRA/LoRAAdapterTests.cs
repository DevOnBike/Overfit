// Copyright (c) 2026 DevOnBike. AGPLv3.

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.LoRA;
using DevOnBike.Overfit.LanguageModels.Runtime;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.LoRA
{
    [Trait("Category", "LoRA")]
    public sealed class LoRAAdapterTests
    {
        private readonly ITestOutputHelper _out;
        public LoRAAdapterTests(ITestOutputHelper output) => _out = output;

        private const string ModelPath = "d:/qwen/qwen.bin";

        // ── LoRAWeight unit tests (no model needed) ───────────────────────────

        [Fact]
        public void LoRAWeight_Init_BIsZero()
        {
            var w = new LoRAWeight(inDim: 8, outDim: 4, rank: 2);
            Assert.Equal(8 * 4, w.InDim * w.OutDim);
            Assert.All(w.B.ToArray(), v => Assert.Equal(0f, v));  // B must be zero
            // A must contain at least one non-zero element
            var hasNonZero = false;
            foreach (var v in w.A) if (v != 0f) { hasNonZero = true; break; }
            Assert.True(hasNonZero, "A must be initialized to non-zero values");
        }

        [Fact]
        public void LoRAWeight_DeltaIsATimesB()
        {
            // A = identity-like [4×2], B = identity-like [2×4]
            // delta = A @ B should be roughly identity
            var w = new LoRAWeight(inDim: 4, outDim: 4, rank: 2);
            // Set A = [[1,0],[0,1],[0,0],[0,0]], B = [[1,0,0,0],[0,1,0,0]]
            var a = w.AMutable; a.Clear();
            a[0] = 1f; a[3] = 1f;  // row0=[1,0], row1=[0,1], row2,3 = 0
            var b = w.BMutable; b.Clear();
            b[0] = 1f; b[5] = 1f;  // row0=[1,0,0,0], row1=[0,1,0,0]

            var delta = new float[4 * 4];
            w.ComputeDelta(delta);

            _out.WriteLine($"delta = [{string.Join(", ", delta)}]");
            // delta[0,0] = sum_k(A[0,k]*B[k,0]) = A[0,0]*B[0,0] = 1*1 = 1
            Assert.Equal(1f, delta[0], precision: 4);
            // delta[1,1] = sum_k(A[1,k]*B[k,1]) = A[1,1]*B[1,1] = 1*1 = 1
            Assert.Equal(1f, delta[5], precision: 4);
            // delta[2,2] = 0 (A[2,:] = 0)
            Assert.Equal(0f, delta[10], precision: 4);
        }

        [Fact]
        public void LoRAWeight_ForwardAdd_MatchesManual()
        {
            var w = new LoRAWeight(inDim: 4, outDim: 2, rank: 2);
            var a = w.AMutable; a.Clear();
            var b = w.BMutable; b.Clear();
            // A[0,0]=1, A[1,1]=1 → A selects x[0] and x[1] into r[0], r[1]
            a[0] = 1f; a[3] = 1f;
            // B[0,0]=2, B[1,1]=3 → r[0]*2 → out[0], r[1]*3 → out[1]
            b[0] = 2f; b[3] = 3f;

            float[] x      = { 5f, 7f, 0f, 0f };
            float[] result = new float[2];
            w.ForwardAdd(x, result, scale: 1f);

            _out.WriteLine($"result = [{result[0]}, {result[1]}]");
            // out[0] = scale * B[0,0] * A[0,0] * x[0] = 1*2*1*5 = 10
            // out[1] = scale * B[1,1] * A[1,1] * x[1] = 1*3*1*7 = 21
            Assert.Equal(10f, result[0], precision: 3);
            Assert.Equal(21f, result[1], precision: 3);
        }

        [Fact]
        public void LoRAWeight_SaveLoad_Roundtrip()
        {
            var w = new LoRAWeight(8, 4, 2, new Random(1));
            var path = Path.GetTempFileName();
            try
            {
                using (var fs = File.OpenWrite(path))
                using (var bw = new BinaryWriter(fs))
                    w.Save(bw);

                LoRAWeight loaded;
                using (var fs = File.OpenRead(path))
                using (var br = new BinaryReader(fs))
                    loaded = LoRAWeight.Load(br);

                Assert.Equal(w.InDim, loaded.InDim);
                Assert.Equal(w.OutDim, loaded.OutDim);
                Assert.Equal(w.Rank, loaded.Rank);

                var maxDiff = w.A.ToArray().Zip(loaded.A.ToArray())
                               .Select(p => MathF.Abs(p.First - p.Second)).Max();
                Assert.True(maxDiff < 1e-6f, $"A max_diff={maxDiff}");
            }
            finally { File.Delete(path); }
        }

        // ── Integration tests (require model) ─────────────────────────────────

        [Fact]
        public void LoRA_CreateAdapter_HasCorrectParamCount()
        {
            if (!File.Exists(ModelPath)) return;
            var engine = CachedLlamaInferenceEngine.Load(ModelPath);
            using (engine)
            {
                var opts = new LoRAOptions(
                    rank: 8,
                    alpha: 16f,
                    dropout: 0f,
                    LoRATargetModules.Query | LoRATargetModules.Value);

                using var adapter = engine.CreateLoRAAdapter("test", opts);

                _out.WriteLine($"LoRA params: {adapter.TrainableParameterCount:N0}");
                _out.WriteLine($"Adapter enabled: {adapter.IsEnabled}");
                _out.WriteLine($"Adapter name: {adapter.Name}");

                // For Qwen2.5-0.5B: 24 layers, 14 Q heads, 2 KV heads, dModel=896, headDim=64, rank=8
                // Q: 24 * 14 * (896+64) * 8 = 24 * 14 * 7680 = 2,580,480
                // V: 24 *  2 * (896+64) * 8 = 24 *  2 * 7680 =   368,640
                _out.WriteLine($"Expected ~{(24 * 14 + 24 * 2) * (896 + 64) * 8:N0} params");

                Assert.True(adapter.TrainableParameterCount > 0);
                Assert.False(adapter.IsEnabled);
            }
        }

        [Fact]
        public void LoRA_EnableDisable_LogitsUnchanged()
        {
            if (!File.Exists(ModelPath)) return;
            var engine = CachedLlamaInferenceEngine.Load(ModelPath);
            using (engine)
            {
                var opts = new LoRAOptions(4, 8f, 0f, LoRATargetModules.Query);
                using var adapter = engine.CreateLoRAAdapter("test", opts);
                using var session = engine.CreateSession(32);
                var sampling = SamplingOptions.Greedy;

                session.Reset(new[] { 151643 });
                var before = session.LastLogits.ToArray();

                // Enable LoRA (zero-init B → delta = 0 → no change)
                adapter.Enable();
                session.Reset(new[] { 151643 });
                var during = session.LastLogits.ToArray();
                adapter.Disable();

                session.Reset(new[] { 151643 });
                var after = session.LastLogits.ToArray();

                var maxDiffDuring = before.Zip(during).Select(p => MathF.Abs(p.First - p.Second)).Max();
                var maxDiffAfter  = before.Zip(after) .Select(p => MathF.Abs(p.First - p.Second)).Max();

                _out.WriteLine($"Max diff during Enable (B=0, delta=0): {maxDiffDuring:E3}");
                _out.WriteLine($"Max diff after Disable (restored):      {maxDiffAfter:E3}");

                // B is zero-initialized → delta = A@B = 0 → no change
                Assert.True(maxDiffDuring < 1e-4f,
                    $"B=0 should mean zero delta but got {maxDiffDuring:E3}");

                // After disable, weights must be exactly restored
                Assert.True(maxDiffAfter < 1e-6f,
                    $"Disable must restore exact weights, got {maxDiffAfter:E3}");

                _out.WriteLine("✓ LoRA Enable/Disable roundtrip correct");
            }
        }

        [Fact]
        public void LoRA_DeltaIsNonZeroWhenBNonZero()
        {
            // Unit test — no model needed
            var w = new LoRAWeight(inDim: 8, outDim: 4, rank: 2);

            // Set B to non-zero
            var b = w.BMutable;
            var rng = new Random(1);
            for (var i = 0; i < b.Length; i++)
                b[i] = (float)(rng.NextDouble() * 0.1 - 0.05);

            var delta = new float[8 * 4];
            w.ComputeDelta(delta);

            var norm = 0f;
            foreach (var v in delta) norm += v * v;
            norm = MathF.Sqrt(norm);

            _out.WriteLine($"Delta L2 norm = {norm:F6}");
            Assert.True(norm > 1e-6f, $"Delta must be non-zero when B≠0, got norm={norm}");
        }

        [Fact]
        public void LoRA_ForwardAdd_ChangesOutput()
        {
            // Verifies ForwardAdd produces delta without merging
            var w = new LoRAWeight(inDim: 4, outDim: 2, rank: 2);
            var b = w.BMutable;
            b[0] = 0.1f; b[1] = 0.2f;  // B[0,:] = [0.1, 0.2]
            b[2] = 0.3f; b[3] = 0.4f;  // B[1,:] = [0.3, 0.4]

            float[] x      = { 1f, 1f, 0f, 0f };
            float[] result = new float[2];
            w.ForwardAdd(x, result, scale: 1f);

            _out.WriteLine($"ForwardAdd result = [{result[0]:F4}, {result[1]:F4}]");
            var norm = MathF.Sqrt(result[0]*result[0] + result[1]*result[1]);
            Assert.True(norm > 1e-6f, $"ForwardAdd result must be non-zero when B≠0");
        }

        [Fact]
        public void LoRA_AfterTraining_LogitsChange()
        {
            if (!File.Exists(ModelPath)) return;
            var engine = CachedLlamaInferenceEngine.Load(ModelPath);
            using (engine)
            {
                var opts = new LoRAOptions(4, 32f, 0f, LoRATargetModules.Query);
                using var adapter = engine.CreateLoRAAdapter("test", opts);
                using var session = engine.CreateSession(32);

                // Baseline with B=0 (delta=0, logits unchanged)
                session.Reset(new[] { 151643 });
                var baseline = session.LastLogits.ToArray();

                // Directly modify B in all Q LoRAWeights via GetAllWeights
                var count = 0;
                foreach (var w in GetAllWeights(adapter))
                {
                    var b = w.BMutable;
                    // Large values to ensure visible impact
                    for (var i = 0; i < b.Length; i++)
                        b[i] = (i % 2 == 0) ? 0.1f : -0.1f;
                    count++;
                }
                _out.WriteLine($"Modified B in {count} LoRAWeight objects");
                Assert.True(count > 0, "Adapter must have weights");

                // Verify delta is non-zero for first weight
                LoRAWeight? firstW = null;
                foreach (var ww in GetAllWeights(adapter)) { firstW = ww; break; }
                Assert.NotNull(firstW);
                var delta  = new float[firstW.InDim * firstW.OutDim];
                firstW.ComputeDelta(delta);
                var deltaNorm = 0f;
                foreach (var v in delta) deltaNorm += v * v;
                deltaNorm = MathF.Sqrt(deltaNorm);
                _out.WriteLine($"First weight delta L2 norm: {deltaNorm:F4}");
                Assert.True(deltaNorm > 0.1f, $"Delta must be large, got {deltaNorm:F4}");

                // CRITICAL DIAGNOSTIC: read TensorStorage value before/after Enable
                var beforeNorm = adapter.ReadBaseWeightNorm(0, LoRATargetModules.Query, 0);
                var beforeVal0 = adapter.ReadBaseWeight(0, LoRATargetModules.Query, 0, 0);

                adapter.Enable();

                var afterEnableNorm = adapter.ReadBaseWeightNorm(0, LoRATargetModules.Query, 0);
                var afterEnableVal0 = adapter.ReadBaseWeight(0, LoRATargetModules.Query, 0, 0);

                _out.WriteLine($"=== TensorStorage state via _baseRefs ===");
                _out.WriteLine($"Q[L0,H0] L2 norm  before/after Enable: {beforeNorm:F4} / {afterEnableNorm:F4}  diff={afterEnableNorm-beforeNorm:F4}");
                _out.WriteLine($"Q[L0,H0][0]       before/after Enable: {beforeVal0:F6} / {afterEnableVal0:F6}  diff={afterEnableVal0-beforeVal0:F6}");

                if (Math.Abs(afterEnableVal0 - beforeVal0) < 1e-6f)
                    _out.WriteLine("✗ TensorStorage NOT modified despite delta=14.18 — bug in MultiplyAdd or AsSpan");
                else
                    _out.WriteLine("✓ TensorStorage IS modified — bug must be in inference reading separate copy");

                session.Reset(new[] { 151643 });
                var modified = session.LastLogits.ToArray();
                adapter.Disable();

                // Restore check
                session.Reset(new[] { 151643 });
                var restored = session.LastLogits.ToArray();

                var maxDiffEnabled  = 0f;
                var maxDiffRestored = 0f;
                for (var i = 0; i < baseline.Length; i++)
                {
                    maxDiffEnabled  = Math.Max(maxDiffEnabled,  MathF.Abs(baseline[i] - modified[i]));
                    maxDiffRestored = Math.Max(maxDiffRestored, MathF.Abs(baseline[i] - restored[i]));
                }

                _out.WriteLine($"Max logit diff when enabled:  {maxDiffEnabled:F6}");
                _out.WriteLine($"Max logit diff after restore: {maxDiffRestored:F6}");
                _out.WriteLine($"scale = {opts.Scale:F2}  (alpha={opts.Alpha}/rank={opts.Rank})");
                _out.WriteLine($"BaseRefCount: {adapter.BaseRefCount}");
                _out.WriteLine($"LastApplyMatchCount: {adapter.LastApplyMatchCount}");
                _out.WriteLine($"_weights count: {count} (=nLayers*nHeads = 24*14 = 336)");

                // Jeżeli LastApplyMatchCount=0 → klucze _baseRefs nie pasują do _weights
                Assert.True(adapter.LastApplyMatchCount > 0,
                    $"ApplyDelta matched 0 entries — key mismatch between _weights and _baseRefs. BaseRefCount={adapter.BaseRefCount}");

                Assert.True(maxDiffEnabled  > 0.001f, $"Enable must change logits, got {maxDiffEnabled:F6}");
                Assert.True(maxDiffRestored < 1e-4f,  $"Disable must restore weights, got {maxDiffRestored:F6}");
            }
        }

        [Fact]
        public void LoRA_SaveLoad_Roundtrip()
        {
            if (!File.Exists(ModelPath)) return;
            var engine = CachedLlamaInferenceEngine.Load(ModelPath);
            using (engine)
            {
                var opts = new LoRAOptions(4, 8f, 0f, LoRATargetModules.Query);
                using var adapter = engine.CreateLoRAAdapter("test", opts);

                // Fill B with non-zero values
                foreach (var w in GetAllWeights(adapter))
                {
                    var rng = new Random(99);
                    var b = w.BMutable;
                    for (var i = 0; i < b.Length; i++)
                        b[i] = (float)(rng.NextDouble() * 0.01);
                }

                var path = Path.GetTempFileName() + ".lora";
                try
                {
                    adapter.Save(path);
                    _out.WriteLine($"Saved: {new FileInfo(path).Length:N0} bytes");

                    // Create new adapter and load
                    using var adapter2 = engine.CreateLoRAAdapter("test2", opts);
                    adapter2.Load(path);

                    // Both adapters should produce same logits
                    using var session = engine.CreateSession(32);

                    adapter.Enable();
                    session.Reset(new[] { 151643 });
                    var logits1 = session.LastLogits.ToArray();
                    adapter.Disable();

                    adapter2.Enable();
                    session.Reset(new[] { 151643 });
                    var logits2 = session.LastLogits.ToArray();
                    adapter2.Disable();

                    var maxDiff = logits1.Zip(logits2)
                        .Select(p => MathF.Abs(p.First - p.Second)).Max();
                    _out.WriteLine($"Max diff after save/load: {maxDiff:E3}");
                    Assert.True(maxDiff < 1e-4f, $"Save/Load should preserve weights, got {maxDiff:E3}");
                    _out.WriteLine("✓ LoRA Save/Load roundtrip correct");
                }
                finally { File.Delete(path); }
            }
        }

        // ── Helper to access LoRAWeight instances ─────────────────────────────

        private static IEnumerable<LoRAWeight> GetAllWeights(LlamaLoRAAdapter adapter)
        {
            // Access via reflection for tests — in production use adapter.ForwardAdd
            var field = typeof(LlamaLoRAAdapter)
                .GetField("_weights",
                    System.Reflection.BindingFlags.NonPublic |
                    System.Reflection.BindingFlags.Instance)!;

            var dict = (Dictionary<(int, LoRATargetModules, int), LoRAWeight>)field.GetValue(adapter)!;
            return dict.Values;
        }

    }
}
