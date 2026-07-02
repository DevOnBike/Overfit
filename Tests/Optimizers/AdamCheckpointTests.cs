// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System;
using System.IO;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.Tests.Optimizers
{
    /// <summary>
    /// Pins <see cref="Adam.SaveState"/>/<see cref="Adam.LoadState"/> — the core of QLoRA resume-checkpointing.
    /// A restored optimizer must be byte-identical to the source: it holds the same step counter and per-param
    /// moments (m, v), so continuing from a checkpoint produces the exact same next update as an uninterrupted
    /// run — momentum intact, not just the weights.
    /// </summary>
    public sealed class AdamCheckpointTests
    {
        private const int N = 48;

        [Fact]
        public void SaveThenLoadState_ResumesExactly()
        {
            var init = new float[N];
            for (var i = 0; i < N; i++)
            {
                init[i] = (i * 0.013f) - 0.3f;
            }

            using var pA = CreateParameter(init, new float[N]);
            using var optA = new Adam(new[] { pA }, learningRate: 0.01f) { Epsilon = 1e-6f };

            // Warm up the optimizer state (non-trivial m/v/t) over a fixed gradient sequence.
            const int warmup = 6;
            for (var k = 0; k < warmup; k++)
            {
                Gradient(k).CopyTo(pA.GradView.AsSpan());
                optA.Step();
            }

            // Serialise A's state, then load it into a fresh optimizer over a param initialised to A's CURRENT
            // weights (this is exactly what resume does: LoadAdapter restores the weights, LoadState the moments).
            byte[] blob;
            using (var ms = new MemoryStream())
            {
                using (var w = new BinaryWriter(ms))
                {
                    optA.SaveState(w);
                }
                blob = ms.ToArray();
            }

            using var pC = CreateParameter(pA.DataView.AsSpan(), new float[N]);
            using var optC = new Adam(new[] { pC }, learningRate: 0.01f) { Epsilon = 1e-6f };
            using (var r = new BinaryReader(new MemoryStream(blob)))
            {
                optC.LoadState(r);
            }

            // One more step with the SAME gradient on both — restored optimizer must match the native one.
            var g = Gradient(warmup);
            g.CopyTo(pA.GradView.AsSpan());
            optA.Step();
            g.CopyTo(pC.GradView.AsSpan());
            optC.Step();

            var a = pA.DataView.AsSpan();
            var c = pC.DataView.AsSpan();
            for (var i = 0; i < N; i++)
            {
                Assert.Equal(a[i], c[i]); // exact — the checkpoint restored the full state, not an approximation
            }
        }

        [Fact]
        public void LoadState_MismatchedParamCount_Throws()
        {
            using var p2 = CreateParameter(new float[2], new float[2]);
            using var src = new Adam(new[] { p2 }, 0.01f);
            src.Step();

            byte[] blob;
            using (var ms = new MemoryStream())
            {
                using (var w = new BinaryWriter(ms))
                {
                    src.SaveState(w);
                }
                blob = ms.ToArray();
            }

            using var pA = CreateParameter(new float[2], new float[2]);
            using var pB = CreateParameter(new float[2], new float[2]);
            using var wrong = new Adam(new[] { pA, pB }, 0.01f); // 2 params vs the saved 1
            using var r = new BinaryReader(new MemoryStream(blob));
            Assert.ThrowsAny<Exception>(() => wrong.LoadState(r));
        }

        private static float[] Gradient(int step)
        {
            var g = new float[N];
            for (var i = 0; i < N; i++)
            {
                g[i] = MathF.Sin((step * 0.7f) + (i * 0.31f)) * 0.05f;
            }
            return g;
        }

        private static AutogradNode CreateParameter(ReadOnlySpan<float> data, ReadOnlySpan<float> grad)
        {
            var storage = new TensorStorage<float>(data.Length, clearMemory: false);
            data.CopyTo(storage.AsSpan());
            var node = new AutogradNode(storage, TensorShape.Vector(data.Length), requiresGrad: true);
            grad.CopyTo(node.GradView.AsSpan());
            return node;
        }
    }
}
