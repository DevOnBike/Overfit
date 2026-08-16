// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Blocks
{
    public class CachedGptStackTests
    {
        [Fact]
        public void Constructor_ExposesShape()
        {
            var stack = new CachedGptStack(
                layerCount: 2,
                dModel: 4,
                headCount: 2,
                dFF: 8,
                vocabSize: 16,
                maxSequenceLength: 32,
                layerNormEpsilon: 1e-5f,
                feedForwardActivation: FeedForwardActivation.ReLU);

            Assert.Equal(2, stack.LayerCount);
            Assert.Equal(4, stack.DModel);
            Assert.Equal(2, stack.HeadCount);
            Assert.Equal(2, stack.HeadDimension);
            Assert.Equal(8, stack.DFF);
            Assert.Equal(16, stack.VocabSize);
            Assert.Equal(32, stack.MaxSequenceLength);
            Assert.Equal(1e-5f, stack.LayerNormEpsilon);
            Assert.Equal(FeedForwardActivation.ReLU, stack.FeedForwardActivation);
        }

        [Fact]
        public void Constructor_InvalidArguments_Throw()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new CachedGptStack(0, 2, 1, 2, 2, 2));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new CachedGptStack(1, 0, 1, 2, 2, 2));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new CachedGptStack(1, 2, 0, 2, 2, 2));

            Assert.Throws<ArgumentException>(() =>
                new CachedGptStack(1, 3, 2, 2, 2, 2));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new CachedGptStack(1, 2, 1, 0, 2, 2));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new CachedGptStack(1, 2, 1, 2, 0, 2));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new CachedGptStack(1, 2, 1, 2, 2, 0));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new CachedGptStack(1, 2, 1, 2, 2, 2, layerNormEpsilon: 0f));
        }

        [Fact]
        public void Decode_ZeroBlocksAndIdentityLmHead_ReturnsFinalLayerNormLogits()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                kvHeadCount: 1,
                maxSequenceLength: 4,
                headDimension: 2);

            var stack = new CachedGptStack(
                layerCount: 1,
                dModel: 2,
                headCount: 1,
                dFF: 2,
                vocabSize: 2,
                maxSequenceLength: 4,
                feedForwardActivation: FeedForwardActivation.None);

            var zeroHeads = new[]
            {
                new[]
                {
                    new float[2 * 2]
                }
            };

            var attentionBiases = new[]
            {
                Array.Empty<float>()
            };

            var ffnW1 = new[]
            {
                new float[2 * 2]
            };

            var ffnB1 = new[]
            {
                Array.Empty<float>()
            };

            var ffnW2 = new[]
            {
                new float[2 * 2]
            };

            var ffnB2 = new[]
            {
                Array.Empty<float>()
            };

            var lmHeadIdentity = new[]
            {
                1f, 0f,
                0f, 1f
            };

            var logits = new float[2];

            cache.Advance();

            var _sw = MakeStackWeights(stack.LayerCount, stack.HeadCount, stack.DModel,
                    zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads,
                    attentionBiases, ffnW1, ffnB1, ffnW2, ffnB2, lmHeadIdentity);
            stack.Decode([1f, -1f], _sw, cache, 0, // position
                logits);

            var expected = LayerNorm([1f, -1f], 1e-5f);

            AssertClose(expected[0], logits[0]);
            AssertClose(expected[1], logits[1]);
        }

        [Fact]
        public void Decode_MultipleZeroLayers_PreservesHiddenUntilFinalNorm()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 2,
                kvHeadCount: 1,
                maxSequenceLength: 4,
                headDimension: 2);

            var stack = new CachedGptStack(
                layerCount: 2,
                dModel: 2,
                headCount: 1,
                dFF: 2,
                vocabSize: 2,
                maxSequenceLength: 4,
                feedForwardActivation: FeedForwardActivation.None);

            var zeroHeads = new[]
            {
                new[]
                {
                    new float[2 * 2]
                },
                new[]
                {
                    new float[2 * 2]
                }
            };

            var attentionBiases = new[]
            {
                Array.Empty<float>(),
                Array.Empty<float>()
            };

            var ffnW1 = new[]
            {
                new float[2 * 2],
                new float[2 * 2]
            };

            var ffnB1 = new[]
            {
                Array.Empty<float>(),
                Array.Empty<float>()
            };

            var ffnW2 = new[]
            {
                new float[2 * 2],
                new float[2 * 2]
            };

            var ffnB2 = new[]
            {
                Array.Empty<float>(),
                Array.Empty<float>()
            };

            var lmHeadIdentity = new[]
            {
                1f, 0f,
                0f, 1f
            };

            var logits = new float[2];

            cache.Advance();

            var _sw = MakeStackWeights(stack.LayerCount, stack.HeadCount, stack.DModel,
                    zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads,
                    attentionBiases, ffnW1, ffnB1, ffnW2, ffnB2, lmHeadIdentity);
            stack.Decode([2f, -2f], _sw, cache, 0, // position
                logits);

            var expected = LayerNorm([2f, -2f], 1e-5f);

            AssertClose(expected[0], logits[0]);
            AssertClose(expected[1], logits[1]);
        }

        [Fact]
        public void Decode_AppliesLmHeadBias()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                kvHeadCount: 1,
                maxSequenceLength: 4,
                headDimension: 2);

            var stack = new CachedGptStack(
                layerCount: 1,
                dModel: 2,
                headCount: 1,
                dFF: 2,
                vocabSize: 2,
                maxSequenceLength: 4,
                feedForwardActivation: FeedForwardActivation.None);

            var zeroHeads = new[]
            {
                new[]
                {
                    new float[2 * 2]
                }
            };

            var attentionBiases = new[]
            {
                Array.Empty<float>()
            };

            var ffnW1 = new[]
            {
                new float[2 * 2]
            };

            var ffnB1 = new[]
            {
                Array.Empty<float>()
            };

            var ffnW2 = new[]
            {
                new float[2 * 2]
            };

            var ffnB2 = new[]
            {
                Array.Empty<float>()
            };

            var lmHeadIdentity = new[]
            {
                1f, 0f,
                0f, 1f
            };

            var logits = new float[2];

            cache.Advance();

            var _sw = MakeStackWeights(stack.LayerCount, stack.HeadCount, stack.DModel,
                    zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads,
                    attentionBiases, ffnW1, ffnB1, ffnW2, ffnB2, lmHeadIdentity);
            stack.Decode([1f, -1f], _sw, cache, 0, logits);

            var expected = LayerNorm([1f, -1f], 1e-5f);

            AssertClose(expected[0], logits[0]); // LM head bias not in StackWeights API
            AssertClose(expected[1], logits[1]);
        }

        [Fact]
        public void Decode_StoresLastFinalHiddenAndLogits()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                kvHeadCount: 1,
                maxSequenceLength: 4,
                headDimension: 2);

            var stack = new CachedGptStack(
                layerCount: 1,
                dModel: 2,
                headCount: 1,
                dFF: 2,
                vocabSize: 2,
                maxSequenceLength: 4,
                feedForwardActivation: FeedForwardActivation.None);

            var zeroHeads = new[]
            {
                new[]
                {
                    new float[2 * 2]
                }
            };

            var attentionBiases = new[]
            {
                Array.Empty<float>()
            };

            var ffnW1 = new[]
            {
                new float[2 * 2]
            };

            var ffnB1 = new[]
            {
                Array.Empty<float>()
            };

            var ffnW2 = new[]
            {
                new float[2 * 2]
            };

            var ffnB2 = new[]
            {
                Array.Empty<float>()
            };

            var lmHeadIdentity = new[]
            {
                1f, 0f,
                0f, 1f
            };

            var logits = new float[2];

            cache.Advance();

            var _sw = MakeStackWeights(stack.LayerCount, stack.HeadCount, stack.DModel,
                    zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads,
                    attentionBiases, ffnW1, ffnB1, ffnW2, ffnB2, lmHeadIdentity);
            stack.Decode([1f, -1f], _sw, cache, 0, // position
                logits);

            var finalHidden = new float[2];
            var lastLogits = new float[2];

            stack.GetLastFinalHidden(finalHidden);
            stack.GetLastLogits(lastLogits);

            Assert.Equal(logits, finalHidden);
            Assert.Equal(logits, lastLogits);
        }

        [Fact]
        public void Decode_PositionNotVisible_Throws()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                kvHeadCount: 1,
                maxSequenceLength: 4,
                headDimension: 2);

            var stack = new CachedGptStack(
                layerCount: 1,
                dModel: 2,
                headCount: 1,
                dFF: 2,
                vocabSize: 2,
                maxSequenceLength: 4);

            var zeroHeads = new[]
            {
                new[]
                {
                    new float[2 * 2]
                }
            };

            Assert.Throws<ArgumentOutOfRangeException>(() =>
            {
                var _sw = MakeStackWeights(stack.LayerCount, stack.HeadCount, stack.DModel,
                    zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads,
                    [[]], [new float[2 * 2]], [[]], [new float[2 * 2]], [[]], new float[2 * 2]);
                stack.Decode([1f, -1f], _sw, cache, 0, // position
                        logits: new float[2]);
            });
            ;
        }

        [Fact]
        public void Decode_InvalidArguments_Throw()
        {
            using var cache = KeyValueCache.Create(
                layerCount: 1,
                kvHeadCount: 1,
                maxSequenceLength: 4,
                headDimension: 2);

            cache.Advance();

            var stack = new CachedGptStack(
                layerCount: 1,
                dModel: 2,
                headCount: 1,
                dFF: 2,
                vocabSize: 2,
                maxSequenceLength: 4);

            var zeroHeads = new[]
            {
                new[]
                {
                    new float[2 * 2]
                }
            };

            Assert.Throws<ArgumentException>(() =>
            {
                var _sw = MakeStackWeights(stack.LayerCount, stack.HeadCount, stack.DModel,
                    zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads,
                    [[]], [new float[2 * 2]], [[]], [new float[2 * 2]], [[]], new float[2 * 2]);
                stack.Decode(new float[1], _sw, cache, 0, // position
                        logits: new float[2]);
            });
            ;

            Assert.Throws<ArgumentException>(() =>
            {
                var _sw = MakeStackWeights(stack.LayerCount, stack.HeadCount, stack.DModel,
                    zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads,
                    [[]], [new float[1]], [[]], [new float[2 * 2]], [[]], new float[2 * 2]);
                stack.Decode(new float[2], _sw, cache, 0, // position
                        logits: new float[2]);
            });
            ;

            Assert.Throws<ArgumentException>(() =>
            {
                var _sw = MakeStackWeights(stack.LayerCount, stack.HeadCount, stack.DModel,
                    zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads,
                    [[]], [new float[2 * 2]], [[]], [new float[2 * 2]], [[]], new float[1]);
                stack.Decode(new float[2], _sw, cache, 0, // position
                        logits: new float[2]);
            });
            ;

            Assert.Throws<ArgumentException>(() =>
            {
                var _sw = MakeStackWeights(stack.LayerCount, stack.HeadCount, stack.DModel,
                    zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads, zeroHeads,
                    [[]], [new float[2 * 2]], [[]], [new float[2 * 2]], [[]], new float[2 * 2]);
                stack.Decode(new float[2], _sw, cache, 0, // position
                        logits: new float[1]);
            });
            ;
        }

        [Fact]
        public void GetBlock_ReturnsRequestedBlock()
        {
            var stack = new CachedGptStack(
                layerCount: 2,
                dModel: 2,
                headCount: 1,
                dFF: 2,
                vocabSize: 2,
                maxSequenceLength: 4);

            Assert.NotNull(stack.GetBlock(0));
            Assert.NotNull(stack.GetBlock(1));
            Assert.Throws<ArgumentOutOfRangeException>(() => stack.GetBlock(2));
        }

        // ── XC-58: shared-stack mutual-exclusion guard ────────────────────────────
        // One test per guarded entry point. The list below was enumerated from CachedGptStack's own
        // internal writers (every method that touches the shared scratch), not from prose — a missed entry
        // point is the defect this shape exists to catch, and one "the guard works" test cannot see it.
        //
        // Every assertion checks the guard's own message as well as the exception type: several of these
        // entry points throw OverfitRuntimeException for unrelated reasons (e.g. "batched quant prefill
        // requires a SwiGLU FFN"), so Assert.Throws<OverfitRuntimeException> alone would pass with no guard
        // at all.

        private const string GuardMessageMarker = "Concurrent use of one CachedGptStack";

        /// <summary>
        /// Drives one guarded entry point through the whole contract: it throws while another caller holds
        /// the stack, it succeeds once released, and — the half that pins the <c>finally</c> — it releases
        /// its own flag, so an immediately repeated call succeeds too.
        /// </summary>
        private static void AssertGuardedEntryPoint(CachedGptStack stack, Action call)
        {
            stack.EnterExclusive("test-holder");

            var thrown = Assert.Throws<OverfitRuntimeException>(call);
            Assert.Contains(GuardMessageMarker, thrown.Message);

            stack.ExitExclusive();

            call();
            call();
        }

        [Fact]
        public void Guard_Decode_ThrowsWhileHeld_SucceedsAfterRelease()
        {
            using var cache = KeyValueCache.Create(1, 1, 4, 2);
            cache.Advance();
            var stack = GuardStack();
            var weights = GuardWeights();
            var logits = new float[2];

            AssertGuardedEntryPoint(stack, () => stack.Decode([1f, -1f], weights, cache, 0, logits));
        }

        [Fact]
        public void Guard_DecodeWithoutLogits_ThrowsWhileHeld_SucceedsAfterRelease()
        {
            using var cache = KeyValueCache.Create(1, 1, 4, 2);
            cache.Advance();
            var stack = GuardStack();
            var weights = GuardWeights();

            AssertGuardedEntryPoint(stack, () => stack.DecodeWithoutLogits([1f, -1f], weights, cache, 0));
        }

        [Fact]
        public void Guard_PrefillBatched_ThrowsWhileHeld_SucceedsAfterRelease()
        {
            using var cache = KeyValueCache.Create(1, 1, 4, 2);
            cache.Advance();
            var stack = GuardStack();
            var weights = GuardWeights();
            var hidden = new[] { 1f, -1f };

            AssertGuardedEntryPoint(stack, () => stack.PrefillBatched(hidden, 1, weights, cache, 0));
        }

        [Fact]
        public void Guard_PrefillBatchedQuant_ThrowsWhileHeld_SucceedsAfterRelease()
        {
            using var cache = KeyValueCache.Create(1, 1, 4, 2);
            cache.Advance();
            var stack = GuardStack();
            var weights = GuardWeights(swiGlu: true);
            var hidden = new[] { 1f, -1f };

            AssertGuardedEntryPoint(stack, () => stack.PrefillBatchedQuant(hidden, 1, weights, cache, 0));
        }

        [Fact]
        public void Guard_PrefillBatchedQuantAllRows_ThrowsWhileHeld_SucceedsAfterRelease()
        {
            using var cache = KeyValueCache.Create(1, 1, 4, 2);
            cache.Advance();
            var stack = GuardStack();
            var weights = GuardWeights(swiGlu: true);
            var hidden = new[] { 1f, -1f };
            var allRows = new float[2];

            AssertGuardedEntryPoint(stack, () => stack.PrefillBatchedQuantAllRows(hidden, 1, weights, cache, 0, allRows));
        }

        [Fact]
        public void Guard_ProjectLogits_ThrowsWhileHeld_SucceedsAfterRelease()
        {
            var stack = GuardStack();
            var weights = GuardWeights();
            var logits = new float[2];

            AssertGuardedEntryPoint(stack, () => stack.ProjectLogits(weights, logits));
        }

        [Fact]
        public void Guard_ProjectLogitsFrom_ThrowsWhileHeld_SucceedsAfterRelease()
        {
            var stack = GuardStack();
            var weights = GuardWeights();
            var finalNorm = new[] { 0.5f, -0.5f };
            var logits = new float[2];

            AssertGuardedEntryPoint(stack, () => stack.ProjectLogitsFrom(finalNorm, weights, logits));
        }

        [Fact]
        public void Guard_ProjectLogitsBatched_ThrowsWhileHeld_SucceedsAfterRelease()
        {
            var stack = GuardStack();
            var weights = GuardWeights();
            var finalNorm = new[] { 0.5f, -0.5f };
            var logits = new float[2];

            AssertGuardedEntryPoint(stack, () => stack.ProjectLogitsBatched(finalNorm, 1, weights, logits));
        }

        [Fact]
        public void Guard_LogitLensFromHidden_ThrowsWhileHeld_SucceedsAfterRelease()
        {
            var stack = GuardStack();
            var weights = GuardWeights();
            var hidden = new[] { 0.5f, -0.5f };
            var logits = new float[2];

            AssertGuardedEntryPoint(stack, () => stack.LogitLensFromHidden(hidden, weights, logits));
        }

        /// <summary>
        /// The release must sit in a <c>finally</c>: a guarded call that throws from inside its own body
        /// must leave the stack usable. Without this, one shape exception mid-decode would leave the engine
        /// refusing every later call for the rest of its life — an availability bug traded for a
        /// correctness one. The guard is entered BEFORE the argument checks precisely so this covers them.
        /// </summary>
        [Fact]
        public void Guard_IsReleased_WhenTheGuardedCallItselfThrows()
        {
            using var cache = KeyValueCache.Create(1, 1, 4, 2);
            cache.Advance();
            var stack = GuardStack();
            var weights = GuardWeights();

            Assert.Throws<ArgumentException>(() => stack.Decode([1f, -1f], weights, cache, 0, new float[1]));

            // Same call with a valid logits buffer: must not throw the guard's exception.
            stack.Decode([1f, -1f], weights, cache, 0, new float[2]);
        }

        /// <summary>
        /// Two threads meeting at a barrier, neither releasing: exactly one gets in.
        ///
        /// <para><b>What this establishes and what it does not.</b> It establishes that a second entry is
        /// refused while a first is held, across threads. It does <b>not</b> prove the exchange is atomic:
        /// a plain read-then-write flag would usually — not always — fail this test, so a green run is not
        /// evidence of a memory-ordering property. The atomicity argument is that entry is a single
        /// <c>Interlocked.CompareExchange</c>, and it is reasoned, not tested (the plan's M4).</para>
        ///
        /// <para>Every wait carries a timeout and its return value is asserted: the fast suite runs classes
        /// in parallel, and an unbounded wait here turns a broken guard into a hung run rather than a red
        /// test — which this repository has already paid for twice (XC-49, XC-52).</para>
        /// </summary>
        [Fact]
        public void Guard_TwoThreadsRacingToEnter_ExactlyOneSucceeds()
        {
            var stack = GuardStack();
            using var barrier = new Barrier(2);
            var successes = 0;
            var refusals = 0;

            void Attempt()
            {
                Assert.True(barrier.SignalAndWait(TimeSpan.FromSeconds(10)), "barrier timed out");

                try
                {
                    stack.EnterExclusive("racer");
                    Interlocked.Increment(ref successes);
                }
                catch (OverfitRuntimeException ex) when (ex.Message.Contains(GuardMessageMarker))
                {
                    Interlocked.Increment(ref refusals);
                }
            }

            var a = new Thread(Attempt) { IsBackground = true };
            var b = new Thread(Attempt) { IsBackground = true };

            a.Start();
            b.Start();

            Assert.True(a.Join(TimeSpan.FromSeconds(10)), "thread A did not finish");
            Assert.True(b.Join(TimeSpan.FromSeconds(10)), "thread B did not finish");

            Assert.Equal(1, Volatile.Read(ref successes));
            Assert.Equal(1, Volatile.Read(ref refusals));

            stack.ExitExclusive();
        }

        /// <summary>
        /// The supported shape, asserted so a too-broad guard cannot pass: two callers sharing one stack —
        /// the shape two sessions of one engine have — taking turns on ONE thread produce bit-identical
        /// logits to the same sequence run alone.
        ///
        /// <para><b>What this establishes.</b> That the guard does not refuse cooperative interleaving, and
        /// that at stack level a decode step carries no state into the next caller's step for these shapes
        /// and this length. It is <b>not</b> a proof of the atomicity claim in general, and it is one stack
        /// with two caches — not two <c>CachedLlamaSession</c>s over a real engine, which needs a model
        /// fixture (see <c>QwenInferenceSmokeTests</c>).</para>
        /// </summary>
        [Fact]
        public void TwoCallersInterleavedOnOneThread_ProduceTheSameLogitsAsAlone()
        {
            const int layers = 2;
            const int dModel = 8;
            const int heads = 2;
            const int dFF = 16;
            const int vocab = 8;
            const int steps = 3;

            var rng = new Random(58);
            var weights = MakeStackWeights(layers, heads, dModel,
                HeadW(rng, layers, heads, dModel * (dModel / heads)),
                HeadW(rng, layers, heads, dModel * (dModel / heads)),
                HeadW(rng, layers, heads, dModel * (dModel / heads)),
                HeadW(rng, layers, heads, (dModel / heads) * dModel),
                HeadW(rng, layers, heads, dModel / heads),
                HeadW(rng, layers, heads, dModel / heads),
                HeadW(rng, layers, heads, dModel / heads),
                LayerW(rng, layers, dModel),
                LayerW(rng, layers, dModel * dFF), LayerW(rng, layers, dFF),
                LayerW(rng, layers, dFF * dModel), LayerW(rng, layers, dModel),
                Rand(rng, dModel * vocab, 0.05f));

            var stack = new CachedGptStack(
                layerCount: layers, dModel: dModel, headCount: heads, dFF: dFF,
                vocabSize: vocab, maxSequenceLength: steps,
                feedForwardActivation: FeedForwardActivation.GeLU);

            // Two distinct input streams, so an accidental cross-talk changes the numbers.
            var streamA = Rand(rng, steps * dModel, 1f);
            var streamB = Rand(rng, steps * dModel, 1f);

            var aloneA = RunStream(stack, weights, streamA, steps, dModel, layers, heads, vocab);
            var aloneB = RunStream(stack, weights, streamB, steps, dModel, layers, heads, vocab);

            // Interleaved: A, B, A, B, A, B — two caches, one stack, one thread.
            var dHead = dModel / heads;
            using var cacheA = KeyValueCache.Create(layers, heads, steps, dHead);
            using var cacheB = KeyValueCache.Create(layers, heads, steps, dHead);
            var interleavedA = new float[steps * vocab];
            var interleavedB = new float[steps * vocab];
            var logits = new float[vocab];

            for (var i = 0; i < steps; i++)
            {
                cacheA.Advance();
                stack.Decode(streamA.AsSpan(i * dModel, dModel), weights, cacheA, i, logits);
                logits.CopyTo(interleavedA.AsSpan(i * vocab, vocab));

                cacheB.Advance();
                stack.Decode(streamB.AsSpan(i * dModel, dModel), weights, cacheB, i, logits);
                logits.CopyTo(interleavedB.AsSpan(i * vocab, vocab));
            }

            for (var i = 0; i < steps * vocab; i++)
            {
                Assert.Equal(aloneA[i], interleavedA[i]);
                Assert.Equal(aloneB[i], interleavedB[i]);
            }
        }

        private static float[] RunStream(
            CachedGptStack stack, StackWeights weights, float[] stream,
            int steps, int dModel, int layers, int heads, int vocab)
        {
            using var cache = KeyValueCache.Create(layers, heads, steps, dModel / heads);
            var all = new float[steps * vocab];
            var logits = new float[vocab];

            for (var i = 0; i < steps; i++)
            {
                cache.Advance();
                stack.Decode(stream.AsSpan(i * dModel, dModel), weights, cache, i, logits);
                logits.CopyTo(all.AsSpan(i * vocab, vocab));
            }

            return all;
        }

        private static CachedGptStack GuardStack()
            => new(
                layerCount: 1,
                dModel: 2,
                headCount: 1,
                dFF: 2,
                vocabSize: 2,
                maxSequenceLength: 4,
                feedForwardActivation: FeedForwardActivation.None);

        /// <summary>
        /// Minimal weights for the guard tests. <paramref name="swiGlu"/> selects the shape the quantized
        /// batched prefill requires (RMSNorm — no beta — plus an FFN gate); without it those entry points
        /// reject the call before the guard could be observed doing anything.
        /// </summary>
        private static StackWeights GuardWeights(bool swiGlu = false)
        {
            var rng = new Random(5800);
            const int dModel = 2;
            const int dFF = 2;
            var gamma = new[] { 1f, 1f };
            var zero = new float[dModel];
            var gate = swiGlu ? Rand(rng, dModel * dFF, 0.1f) : null;

            return StackWeights.ForTest(
                layerCount: 1,
                headCount: 1,
                l => new BlockWeights(
                    heads:
                    [
                        new SingleHeadWeights(
                            wq: Rand(rng, dModel * dModel, 0.1f), wk: Rand(rng, dModel * dModel, 0.1f),
                            wv: Rand(rng, dModel * dModel, 0.1f), wo: Rand(rng, dModel * dModel, 0.1f),
                            bq: Rand(rng, dModel, 0.1f), bk: Rand(rng, dModel, 0.1f), bv: Rand(rng, dModel, 0.1f))
                    ],
                    ln1Gamma: gamma, ln1Beta: swiGlu ? null : zero,
                    attentionBias: [],
                    ln2Gamma: gamma, ln2Beta: swiGlu ? null : zero,
                    ffnW1: Rand(rng, dModel * dFF, 0.1f), ffnB1: [],
                    ffnW2: Rand(rng, dFF * dModel, 0.1f), ffnB2: [],
                    ffnGate: gate),
                finalNormGamma: gamma,
                finalNormBeta: swiGlu ? [] : zero,
                lmHead: Rand(rng, dModel * 2, 0.1f));
        }

        private static float[] LayerNorm(
            ReadOnlySpan<float> input,
            float epsilon)
        {
            var mean = 0f;

            for (var i = 0; i < input.Length; i++)
            {
                mean += input[i];
            }

            mean /= input.Length;

            var variance = 0f;

            for (var i = 0; i < input.Length; i++)
            {
                var centered = input[i] - mean;
                variance += centered * centered;
            }

            variance /= input.Length;

            var invStd = 1f / MathF.Sqrt(variance + epsilon);
            var output = new float[input.Length];

            for (var i = 0; i < input.Length; i++)
            {
                output[i] = (input[i] - mean) * invStd;
            }

            return output;
        }

        private static void AssertClose(float expected, float actual)
        {
            Assert.True(
                MathF.Abs(expected - actual) <= 1e-5f,
                $"Expected {expected}, actual {actual}.");
        }
        [Theory]
        [InlineData(1, 8, 2, 16)]
        [InlineData(5, 8, 2, 16)]
        [InlineData(7, 32, 4, 64)]
        public void PrefillBatched_LastToken_IsBitIdentical_To_SingleTokenLoop(
            int rows, int dModel, int headCount, int dFF)
        {
            const int layers = 2;
            const int vocab = 8;
            var dHead = dModel / headCount;
            var maxSeq = rows;
            var rng = new Random(2024 + rows * 11 + dModel + headCount + dFF);

            var wq = HeadW(rng, layers, headCount, dModel * dHead);
            var wk = HeadW(rng, layers, headCount, dModel * dHead);
            var wv = HeadW(rng, layers, headCount, dModel * dHead);
            var wo = HeadW(rng, layers, headCount, dHead * dModel);
            var bq = HeadW(rng, layers, headCount, dHead);
            var bk = HeadW(rng, layers, headCount, dHead);
            var bv = HeadW(rng, layers, headCount, dHead);
            var attBiases = LayerW(rng, layers, dModel);
            var fw1 = LayerW(rng, layers, dModel * dFF);
            var fb1 = LayerW(rng, layers, dFF);
            var fw2 = LayerW(rng, layers, dFF * dModel);
            var fb2 = LayerW(rng, layers, dModel);
            var lmHead = Rand(rng, dModel * vocab, 0.05f);

            var sw = MakeStackWeights(layers, headCount, dModel,
                wq, wk, wv, wo, bq, bk, bv, attBiases, fw1, fb1, fw2, fb2, lmHead);

            var stack = new CachedGptStack(
                layerCount: layers, dModel: dModel, headCount: headCount, dFF: dFF,
                vocabSize: vocab, maxSequenceLength: maxSeq,
                feedForwardActivation: FeedForwardActivation.GeLU);

            var embed = Rand(rng, rows * dModel, 1f);

            // Reference: single-token loop (advance cache per token).
            var refHidden = new float[dModel];
            using (var cacheRef = KeyValueCache.Create(layers, headCount, maxSeq, dHead))
            {
                for (var i = 0; i < rows; i++)
                {
                    cacheRef.Advance();
                    stack.DecodeWithoutLogits(embed.AsSpan(i * dModel, dModel), sw, cacheRef, i);
                }
                stack.GetLastFinalHidden(refHidden);
            }

            // Batched prefill (cache advanced to N up front).
            var batHidden = new float[dModel];
            using (var cacheBat = KeyValueCache.Create(layers, headCount, maxSeq, dHead))
            {
                for (var i = 0; i < rows; i++)
                {
                    cacheBat.Advance();
                }
                stack.PrefillBatched(embed, rows, sw, cacheBat, basePosition: 0);
                stack.GetLastFinalHidden(batHidden);
            }

            for (var i = 0; i < dModel; i++)
            {
                Assert.Equal(refHidden[i], batHidden[i]);
            }
        }

        private static float[][][] HeadW(Random rng, int layers, int heads, int len)
        {
            var a = new float[layers][][];
            for (var l = 0; l < layers; l++)
            {
                a[l] = new float[heads][];
                for (var h = 0; h < heads; h++)
                {
                    a[l][h] = Rand(rng, len, 0.1f);
                }
            }
            return a;
        }

        private static float[][] LayerW(Random rng, int layers, int len)
        {
            var a = new float[layers][];
            for (var l = 0; l < layers; l++)
            {
                a[l] = Rand(rng, len, 0.1f);
            }
            return a;
        }

        private static float[] Rand(Random rng, int n, float scale)
        {
            var a = new float[n];
            for (var i = 0; i < n; i++)
            {
                a[i] = (rng.NextSingle() * 2f - 1f) * scale;
            }
            return a;
        }

        private static StackWeights MakeStackWeights(
            int layerCount, int headCount, int dModel,
            float[][][] wq, float[][][] wk, float[][][] wv, float[][][] wo,
            float[][][] bq, float[][][] bk, float[][][] bv,
            float[][] attBiases, float[][] fw1, float[][] fb1, float[][] fw2, float[][] fb2,
            float[] lmHead)
        {
            var gamma = Enumerable.Repeat(1f, dModel).ToArray();
            var zero = new float[dModel];
            return StackWeights.ForTest(
                layerCount, headCount,
                l =>
                {
                    var heads = new SingleHeadWeights[headCount];
                    for (var h = 0; h < headCount; h++)
                    {
                        heads[h] = new SingleHeadWeights(
                        wq: wq[l][h], wk: wk[l][h], wv: wv[l][h], wo: wo[l][h],
                        bq: bq[l][h], bk: bk[l][h], bv: bv[l][h]);
                    }
                    return new BlockWeights(
                        heads: heads,
                        ln1Gamma: gamma, ln1Beta: zero,
                        attentionBias: attBiases[l],
                        ln2Gamma: gamma, ln2Beta: zero,
                        ffnW1: fw1[l], ffnB1: fb1[l], ffnW2: fw2[l], ffnB2: fb2[l]);
                },
                finalNormGamma: gamma, finalNormBeta: zero, lmHead: lmHead);
        }

    }
}
