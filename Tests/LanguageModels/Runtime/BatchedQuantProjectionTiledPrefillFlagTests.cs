// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.Intrinsics.X86;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    /// <summary>
    /// Pins the two halves of <see cref="BatchedQuantProjection.UseTiledPrefillQ4K"/>'s contract, which pull in
    /// opposite directions: a write must be invisible to every other thread (two test classes write it in
    /// opposite directions while xunit runs their collections in parallel), and a thread that has NOT written
    /// must still read the <c>OVERFIT_TILED_PREFILL</c> default (the reason the flag could not simply take the
    /// <c>[ThreadStatic]</c> attribute its sibling <c>DisableRepackedKernelsForParity</c> carries — a
    /// <c>[ThreadStatic]</c> initialiser runs on the first thread only).
    /// </summary>
    public sealed class BatchedQuantProjectionTiledPrefillFlagTests
    {
        private static readonly TimeSpan JoinTimeout = TimeSpan.FromSeconds(30);

        /// <summary>
        /// Half 1: two threads holding opposite values never observe each other's. The barrier makes it
        /// deterministic rather than a race — thread B's write is ordered strictly between thread A's write and
        /// thread A's read, so a process-wide field fails this every time rather than one run in six.
        /// </summary>
        [Fact]
        public void Flag_WrittenOnOneThread_IsNotObservedByAnother()
        {
            using var gate = new Barrier(2);
            var readByA = false;
            var readByB = true;

            var a = new Thread(() =>
            {
                BatchedQuantProjection.UseTiledPrefillQ4K = true;
                gate.SignalAndWait();  // A has written
                gate.SignalAndWait();  // B has written the opposite
                readByA = BatchedQuantProjection.UseTiledPrefillQ4K;
            });

            var b = new Thread(() =>
            {
                gate.SignalAndWait();
                BatchedQuantProjection.UseTiledPrefillQ4K = false;
                gate.SignalAndWait();
                readByB = BatchedQuantProjection.UseTiledPrefillQ4K;
            });

            a.Start();
            b.Start();
            Assert.True(a.Join(JoinTimeout), "thread A did not finish");
            Assert.True(b.Join(JoinTimeout), "thread B did not finish");

            Assert.True(readByA);   // B's false must not have reached A
            Assert.False(readByB);  // A's true must not have reached B
        }

        /// <summary>
        /// Half 2: a thread that has never written reads the <c>OVERFIT_TILED_PREFILL</c> default, not
        /// <c>default(bool)</c>. The current thread touches the class first on purpose: under the rejected
        /// <c>[ThreadStatic] ... = TiledPrefillEnabled</c> shape the static constructor runs on the first thread
        /// to touch the type, and the spawned threads would then read <c>false</c>.
        ///
        /// <para><b>Reach, stated because it is narrower than it looks:</b> this discriminates only when
        /// <c>OVERFIT_TILED_PREFILL=1</c> (and AVX2 is present), because with the env flag unset the correct
        /// default and the broken one are both <c>false</c> and no in-process observation can tell them apart.
        /// Run the suite once with that variable set to exercise the discriminating arm.</para>
        /// </summary>
        [Fact]
        public void Flag_DefaultsToTheEnvironmentValue_OnEveryThread()
        {
            var expected = Q4KGemvKernel.TiledPrefillEnabled;
            Assert.Equal(expected, BatchedQuantProjection.UseTiledPrefillQ4K);

            var reads = new bool[4];
            var threads = new Thread[reads.Length];
            for (var i = 0; i < threads.Length; i++)
            {
                var index = i;
                threads[i] = new Thread(() => reads[index] = BatchedQuantProjection.UseTiledPrefillQ4K);
                threads[i].Start();
            }

            for (var i = 0; i < threads.Length; i++)
            {
                Assert.True(threads[i].Join(JoinTimeout), $"reader thread {i} did not finish");
            }

            for (var i = 0; i < reads.Length; i++)
            {
                Assert.Equal(expected, reads[i]);
            }
        }

        /// <summary>
        /// The second defect in the same code, independent of the read: two save/restore pairs interleaving on
        /// one shared location lose an update. B saves the value A is holding, A restores, then B restores A's
        /// value over it — and the flag stays wrong for every test that runs afterwards. The barrier forces
        /// exactly that order. A third thread, which never wrote, is the observer, because it is the position
        /// every later test is in.
        ///
        /// <para>This one discriminates in both environment arms: the leaked value is always the negation of the
        /// default, whatever the default is.</para>
        /// </summary>
        [Fact]
        public void InterleavedSaveRestore_OnTwoThreads_CannotLeakToAThirdThread()
        {
            var expected = Q4KGemvKernel.TiledPrefillEnabled;
            using var gate = new Barrier(2);

            var a = new Thread(() =>
            {
                var saved = BatchedQuantProjection.UseTiledPrefillQ4K;
                BatchedQuantProjection.UseTiledPrefillQ4K = !expected;
                gate.SignalAndWait();  // B saves now, and on a shared static it saves A's value
                gate.SignalAndWait();
                BatchedQuantProjection.UseTiledPrefillQ4K = saved;
                gate.SignalAndWait();  // A has restored; B restores after it
            });

            var b = new Thread(() =>
            {
                gate.SignalAndWait();
                var saved = BatchedQuantProjection.UseTiledPrefillQ4K;
                gate.SignalAndWait();
                gate.SignalAndWait();
                BatchedQuantProjection.UseTiledPrefillQ4K = saved;
            });

            a.Start();
            b.Start();
            Assert.True(a.Join(JoinTimeout), "thread A did not finish");
            Assert.True(b.Join(JoinTimeout), "thread B did not finish");

            var observed = !expected; // so a reader thread that never ran fails rather than passes
            var reader = new Thread(() => observed = BatchedQuantProjection.UseTiledPrefillQ4K);
            reader.Start();
            Assert.True(reader.Join(JoinTimeout), "reader thread did not finish");

            Assert.Equal(expected, observed);
            Assert.Equal(expected, BatchedQuantProjection.UseTiledPrefillQ4K);
        }

        /// <summary>
        /// The flag is still live at its one production read (<c>DispatchQ4K</c>): setting it <c>true</c> on the
        /// calling thread selects the register-tiled GEMM for a weight that is NOT prepacked, proven by
        /// bit-equality against <see cref="Q4KGemvKernel.GemmTiled"/> — the weight-stationary kernel reassociates
        /// its reduction and would not match. Its sibling
        /// <c>BatchedQuantProjectionTiledDispatchTests.Dispatch_PrepackedWeight_UsesTiled_EvenWithFlagOff</c>
        /// cannot cover this: it reaches the tiled path through <c>IsPrepacked</c>, so it stays green even if the
        /// flag is dead.
        /// </summary>
        [Fact]
        public void FlagTrue_OnTheWritingThread_SelectsTheTiledKernel()
        {
            if (!Avx2.IsSupported || !Fma.IsSupported)
            {
                return;
            }

            const int inputSize = 512;
            const int outputSize = 64;
            const int rows = 8;
            var spr = inputSize / 256;
            var bsumsPerRow = spr * Q4KDotKernel.GroupsPerSuperBlock;

            var rng = new Random(11);
            var wF32 = new float[outputSize * inputSize];
            for (var i = 0; i < wF32.Length; i++)
            {
                wF32[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }
            var q4k = GgmlQuant.QuantizeQ4_K(wF32, inputSize, outputSize);
            var repacked = Q4KRepack.RepackMatrix(q4k, outputSize, inputSize);

            var weight = new Q4KWeight(q4k, inputSize, outputSize);
            Assert.False(weight.IsPrepacked); // the flag, and only the flag, can select tiled here

            var input = new float[rows * inputSize];
            for (var i = 0; i < input.Length; i++)
            {
                input[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }

            using var flagOn = new TiledPrefillQ4KScope(true);

            DecodeWeight dw = weight;
            var got = new float[rows * outputSize];
            BatchedQuantProjection.Dispatch(input, rows, in dw, ReadOnlySpan<float>.Empty, got, inputSize, outputSize);

            var aq = new sbyte[rows * inputSize];
            var asc = new float[rows * spr];
            var ab = new short[rows * bsumsPerRow];
            for (var n = 0; n < rows; n++)
            {
                Q4KDotKernel.QuantizeActivationQ8K(
                    input.AsSpan(n * inputSize, inputSize),
                    aq.AsSpan(n * inputSize, inputSize),
                    asc.AsSpan(n * spr, spr),
                    ab.AsSpan(n * bsumsPerRow, bsumsPerRow));
            }
            var expected = new float[rows * outputSize];
            Q4KGemvKernel.GemmTiled(repacked, outputSize, inputSize, rows, aq, asc, ab, expected);

            for (var i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i], got[i]);
            }
        }
    }
}
