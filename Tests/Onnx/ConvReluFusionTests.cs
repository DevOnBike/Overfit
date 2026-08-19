// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Onnx;

namespace DevOnBike.Overfit.Tests.Onnx
{
    /// <summary>
    /// Folding a <c>Relu</c> into the preceding convolution's epilogue.
    ///
    /// <para><b>What has to be proved, and in which order.</b> Parity is the cheap half — a fused model that
    /// produced wrong numbers would fail the existing ONNX comparison tests too. The half only these tests
    /// cover is the <b>guard</b>: a convolution whose pre-activation output is read by something as well as
    /// by the <c>Relu</c> must not be clamped in place, and no fixture in this repository has that shape, so
    /// the graph is built by hand to produce it.</para>
    ///
    /// <para><b>A graph where nothing fuses is the silent failure.</b> Fusion that never fires is
    /// indistinguishable from fusion that fires and costs nothing — both read as "green, no change" — so the
    /// positive tests assert the node count actually dropped rather than only that the output is right.</para>
    /// </summary>
    public sealed class ConvReluFusionTests
    {
        private static string FixturePath =>
            Path.Combine(AppContext.BaseDirectory, "test_fixtures", "mnist_cnn.onnx");

        /// <summary>
        /// The MNIST CNN fixture loses exactly one <c>Relu</c> per fused convolution.
        ///
        /// <para>Both halves are needed. The node count alone would pass if some unrelated node vanished;
        /// the flag alone would pass if the <c>Relu</c> were left in place beside a convolution that now
        /// clamps — which applies the activation twice and still produces the right answer, because
        /// <c>max(0, max(0, x))</c> is <c>max(0, x)</c>. That is the one wrong implementation parity cannot
        /// see.</para>
        /// </summary>
        [Fact]
        public void Importing_TheMnistCnn_FoldsReluIntoConvolution()
        {
            Assert.True(File.Exists(FixturePath), $"missing fixture {FixturePath}");

            using var fused = OnnxGraphImporter.LoadFused(FixturePath, 28 * 28, 10);
            using var plain = OnnxGraphImporter.LoadUnfused(FixturePath, 28 * 28, 10);

            var fusedRelus = CountRelus(fused);
            var plainRelus = CountRelus(plain);
            var fusedConvs = CountFusedConvolutions(fused);

            Assert.True(plainRelus > 0, "the fixture has no Relu node, so it cannot demonstrate fusion");
            Assert.True(fusedConvs > 0, "no convolution reports a fused Relu");
            Assert.Equal(plainRelus - fusedConvs, fusedRelus);
            Assert.Equal(0, CountFusedConvolutions(plain));
        }

        /// <summary>
        /// The two arms agree bit for bit, because fusion changes when the clamp runs, not what it is.
        ///
        /// <para>Exact equality rather than a tolerance: the bias is added in the same order and
        /// <c>max(0, x)</c> introduces no rounding, so any difference at all is a defect rather than
        /// accumulated error.</para>
        /// </summary>
        [Fact]
        public void Fusing_DoesNotChangeTheOutput()
        {
            Assert.True(File.Exists(FixturePath), $"missing fixture {FixturePath}");

            var input = new float[28 * 28];
            var rng = new Random(20260819);

            for (var i = 0; i < input.Length; i++)
            {
                input[i] = (float)((rng.NextDouble() * 2.0) - 1.0);
            }

            var fusedOut = new float[10];
            var plainOut = new float[10];

            using (var fused = OnnxGraphImporter.LoadFused(FixturePath, 28 * 28, 10))
            {
                fused.Eval();
                fused.RunInference(input, fusedOut);
            }

            using (var plain = OnnxGraphImporter.LoadUnfused(FixturePath, 28 * 28, 10))
            {
                plain.Eval();
                plain.RunInference(input, plainOut);
            }

            for (var i = 0; i < fusedOut.Length; i++)
            {
                Assert.Equal(plainOut[i], fusedOut[i]);
            }
        }

        /// <summary>
        /// A convolution whose output has a second reader is left alone.
        ///
        /// <para>This is the correctness condition of the whole change, and no fixture here exercises it, so
        /// the graph is built directly. Fusing this shape would hand the second consumer activated values
        /// where it asked for pre-activation ones — a wrong answer no parity test against a
        /// skip-connection-free model could ever show.</para>
        /// </summary>
        [Fact]
        public void AConvolutionWhoseOutputHasASecondReader_IsNotFused()
        {
            using var conv = new ConvLayer(1, 1, 4, 4, 3);
            using var relu = new ReluActivation();
            using var tail = new ReluActivation();

            // Both read slot 1: the Relu that could fuse, and a second consumer that must keep seeing the
            // values the convolution actually produced.
            var nodes = new List<OnnxGraphNode>
            {
                new(conv, [0], 1, 4),
                new(relu, [1], 2, 4),
                new(tail, [1], 3, 4),
            };

            Assert.False(OnnxGraphImporter.CanFuseConvRelu(nodes, 0));

            OnnxGraphImporter.FuseConvRelu(nodes);

            Assert.False(conv.FusedRelu);
            Assert.Equal(3, nodes.Count);
        }

        /// <summary>
        /// The single-reader shape does fuse, so the test above is refusing for the right reason.
        ///
        /// <para>Without this, a <c>CanFuseConvRelu</c> that returned <see langword="false"/> unconditionally
        /// would pass every other test in this class that asserts a refusal.</para>
        /// </summary>
        [Fact]
        public void AConvolutionWhoseOutputHasOneReader_IsFused()
        {
            using var conv = new ConvLayer(1, 1, 4, 4, 3);
            using var relu = new ReluActivation();
            using var tail = new ReluActivation();

            var nodes = new List<OnnxGraphNode>
            {
                new(conv, [0], 1, 4),
                new(relu, [1], 2, 4),
                new(tail, [2], 3, 4),
            };

            Assert.True(OnnxGraphImporter.CanFuseConvRelu(nodes, 0));

            OnnxGraphImporter.FuseConvRelu(nodes);

            Assert.True(conv.FusedRelu);
            Assert.Equal(2, nodes.Count);

            // The tail read the Relu's slot; with the Relu gone it must read the convolution's.
            Assert.Equal(1, nodes[1].InputSlots[0]);
        }

        /// <summary>
        /// A fused convolution applies the clamp itself, and an unfused one does not.
        ///
        /// <para>Asserted on the layer rather than on a graph, because this is the half of the change that
        /// does the arithmetic. A negative input is used deliberately: with an all-positive input,
        /// <c>max(0, x)</c> is the identity and a fusion flag that did nothing would still pass.</para>
        /// </summary>
        [Fact]
        public void AFusedConvolution_ClampsItsOwnOutput()
        {
            var input = new float[9];
            var fusedOut = new float[1];
            var plainOut = new float[1];

            for (var i = 0; i < input.Length; i++)
            {
                input[i] = 1f;
            }

            using var plain = new ConvLayer(1, 1, 3, 3, 3);
            plain.LoadParameters(NegativeKernel(), [0f]);
            plain.Eval();
            plain.ForwardInference(input, plainOut);

            using var fused = new ConvLayer(1, 1, 3, 3, 3)
            {
                FusedRelu = true,
            };

            fused.LoadParameters(NegativeKernel(), [0f]);
            fused.Eval();
            fused.ForwardInference(input, fusedOut);

            Assert.True(plainOut[0] < 0f, $"the unfused output was {plainOut[0]}, so the clamp cannot be seen");
            Assert.Equal(0f, fusedOut[0]);
        }

        /// <summary>Nine weights of -1, so a positive input produces a negative output the clamp must remove.</summary>
        private static float[] NegativeKernel()
        {
            var kernel = new float[9];

            for (var i = 0; i < kernel.Length; i++)
            {
                kernel[i] = -1f;
            }

            return kernel;
        }

        private static int CountRelus(OnnxGraphModel model)
        {
            var count = 0;

            foreach (var node in model.Nodes)
            {
                if (node.Module is ReluActivation)
                {
                    count++;
                }
            }

            return count;
        }

        private static int CountFusedConvolutions(OnnxGraphModel model)
        {
            var count = 0;

            foreach (var node in model.Nodes)
            {
                if (node.Module is ConvLayer { FusedRelu: true })
                {
                    count++;
                }
            }

            return count;
        }
    }
}
