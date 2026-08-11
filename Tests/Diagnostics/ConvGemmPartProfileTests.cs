// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Diagnostics;
using DevOnBike.Overfit.Inference;
using DevOnBike.Overfit.Kernels;
using DevOnBike.Overfit.Onnx;
using DevOnBike.Overfit.Runtime;

namespace DevOnBike.Overfit.Tests.Diagnostics
{
    /// <summary>
    /// Splits convolution time into the <b>im2col patch gather</b> versus the <b>GEMM</b>, on a real
    /// ImageNet-sized CNN.
    ///
    /// <para><b>The question.</b> A per-operator profile put ConvLayer at 90.7% of VGG-16's 79 ms — but that
    /// does not say which half of the conv. The gather moves <c>O(K·N)</c> floats; the GEMM does
    /// <c>O(M·N·K)</c> FLOPs, and their ratio swings by an order of magnitude across VGG's layers (early
    /// layers: huge N, small K; late layers: small N, large K). Whether the next optimisation target is the
    /// micro-kernel or the gather depends entirely on this split, and it is cheap to measure and expensive to
    /// guess.</para>
    ///
    /// <para>Reference points for reading the output on the dev box: the conv GEMMs total ≈30.7 GFLOP per
    /// VGG-16 inference, the machine sustains ≈2190 GFLOP/s of float FMA across all cores
    /// (<c>MachineRooflineBenchmark</c>), and ONNX Runtime runs the whole model in ≈10–12 ms.</para>
    ///
    /// <para>Needs an exported VGG-16 (<c>python Scripts/export_cnn_onnx.py --arch vgg16</c>); logs and
    /// returns without it, so CI stays green.</para>
    /// </summary>
    public sealed class ConvGemmPartProfileTests
    {
        private const int InputSize = 3 * 224 * 224;
        private const int OutputSize = 1000;
        private const int Runs = 10;

        private readonly ITestOutputHelper _out;

        public ConvGemmPartProfileTests(ITestOutputHelper output) => _out = output;

        [LongFact("3s")]
        public void Conv_Im2ColVersusGemm_Split()
        {
            var path = Environment.GetEnvironmentVariable(OverfitEnvironment.CnnOnnx)
                ?? @"C:\onnxmodels\cnn.onnx";

            if (!File.Exists(path))
            {
                _out.WriteLine($"missing {path} — export with Scripts/export_cnn_onnx.py --arch vgg16");
                return;
            }

            using var model = OnnxGraphImporter.Load(path, InputSize, OutputSize);
            model.Eval();
            using var engine = InferenceEngine.FromBackend(new OnnxGraphInferenceBackend(model));

            var input = new float[InputSize];
            var output = new float[OutputSize];
            var rng = new Random(1234);
            for (var i = 0; i < input.Length; i++)
            {
                input[i] = (float)rng.NextDouble();
            }

            // Warm up outside the measured region: JIT, the pooled im2col buffer, page-in.
            for (var i = 0; i < 3; i++)
            {
                engine.Run(input, output);
            }

            Conv2DGemmKernels.ResetPartProfile();
            Conv2DGemmKernels.ProfileParts = true;
            var started = ValueStopwatch.StartNew();
            try
            {
                for (var i = 0; i < Runs; i++)
                {
                    engine.Run(input, output);
                }
            }
            finally
            {
                Conv2DGemmKernels.ProfileParts = false;
            }

            var wallMs = started.GetElapsedTime().TotalMilliseconds / Runs;

            _out.WriteLine($"wall            : {wallMs,8:F2} ms/run");
            _out.WriteLine($"conv parts (×{Runs}): {Conv2DGemmKernels.PartProfileReport()}");
            _out.WriteLine("conv GEMM work  : 30.7 GFLOP/run | machine 2190 GFLOP/s | ORT whole model ~10-12 ms");

            // Per layer, with each conv's own GFLOP/s — the view that locates a shape-dependent deficit.
            model.ResetNodeProfile();
            OnnxGraphModel.ProfileNodes = true;
            try
            {
                for (var i = 0; i < Runs; i++)
                {
                    engine.Run(input, output);
                }
            }
            finally
            {
                OnnxGraphModel.ProfileNodes = false;
            }

            _out.WriteLine(model.PerNodeProfileReport());
            _out.WriteLine(VggConvShapeTable());

            Assert.True(wallMs > 0, "no inference time recorded");
        }

        /// <summary>
        /// Splits the GEMM itself into <b>B-panel packing</b> versus the <b>micro-kernel</b>, by ablation.
        ///
        /// <para>The micro-kernel was measured in isolation at 148 GFLOP/s per core — essentially the AVX2
        /// hardware peak — yet the production GEMM reaches only 566 GFLOP/s across 16 cores, about 24% of what
        /// that kernel can do. So the loss is not the arithmetic. Packing is the suspect: it reads
        /// <c>B[kk·n + n0]</c> with stride <c>n</c> (200 KB apart on VGG's early layers) one scalar at a time,
        /// and over a whole GEMM it moves the entire im2col matrix a second time.</para>
        ///
        /// <para>Ablation rather than timers, because timestamps inside the parallel region would need
        /// per-thread accumulation and would perturb the thing being measured. Each arm produces wrong output
        /// by construction — this measures cost, never correctness. Read the two as upper bounds: removing one
        /// side also frees the other's cache pressure.</para>
        /// </summary>
        [LongFact("3s")]
        public void ConvGemm_PackVersusMicroKernel_Split()
        {
            var path = Environment.GetEnvironmentVariable(OverfitEnvironment.CnnOnnx)
                ?? @"C:\onnxmodels\cnn.onnx";

            if (!File.Exists(path))
            {
                _out.WriteLine($"missing {path}");
                return;
            }

            using var model = OnnxGraphImporter.Load(path, InputSize, OutputSize);
            model.Eval();
            using var engine = InferenceEngine.FromBackend(new OnnxGraphInferenceBackend(model));

            var input = new float[InputSize];
            var output = new float[OutputSize];
            var rng = new Random(1234);
            for (var i = 0; i < input.Length; i++)
            {
                input[i] = (float)rng.NextDouble();
            }

            for (var i = 0; i < 3; i++)
            {
                engine.Run(input, output);
            }

            var baseline = TimeArm(engine, input, output, pack: true, micro: true);
            var packOnly = TimeArm(engine, input, output, pack: true, micro: false);
            var microOnly = TimeArm(engine, input, output, pack: false, micro: true);

            _out.WriteLine($"baseline (pack + micro) : {baseline,8:F2} ms/run");
            _out.WriteLine($"pack only               : {packOnly,8:F2} ms/run   ({100 * packOnly / baseline,5:F1}% of baseline)");
            _out.WriteLine($"micro only              : {microOnly,8:F2} ms/run   ({100 * microOnly / baseline,5:F1}% of baseline)");
            _out.WriteLine($"unattributed            : {baseline - packOnly - microOnly,8:F2} ms/run");

            Assert.True(baseline > 0, "no inference time recorded");
        }

        /// <summary>
        /// VGG-16's thirteen conv layers as GEMM shapes, in graph order, so the per-node timings above can be
        /// read as GFLOP/s per layer. M = output channels, K = inChannels·3·3, N = outH·outW.
        /// </summary>
        private static string VggConvShapeTable()
        {
            (int M, int K, int N)[] layers =
            [
                (64, 3 * 9, 224 * 224), (64, 64 * 9, 224 * 224),
                (128, 64 * 9, 112 * 112), (128, 128 * 9, 112 * 112),
                (256, 128 * 9, 56 * 56), (256, 256 * 9, 56 * 56), (256, 256 * 9, 56 * 56),
                (512, 256 * 9, 28 * 28), (512, 512 * 9, 28 * 28), (512, 512 * 9, 28 * 28),
                (512, 512 * 9, 14 * 14), (512, 512 * 9, 14 * 14), (512, 512 * 9, 14 * 14),
            ];

            var sb = new System.Text.StringBuilder();
            sb.AppendLine("=== VGG-16 conv shapes, in graph order (match against the conv nodes above) ===");

            for (var i = 0; i < layers.Length; i++)
            {
                var (m, k, n) = layers[i];
                var gflop = 2.0 * m * k * n / 1e9;
                var panels = (n + 7) / 8;
                sb.AppendLine(
                    $"  conv{i + 1,-2} M={m,4} K={k,5} N={n,6}  {gflop,6:F2} GFLOP   "
                    + $"panels={panels,5}  A={m * k * 4 / 1024,6} KB re-read per panel");
            }

            return sb.ToString();
        }

        private static double TimeArm(
            InferenceEngine engine, float[] input, float[] output, bool pack, bool micro)
        {
            Conv2DGemmKernels.AblatePackB = !pack;
            Conv2DGemmKernels.AblateMicroKernel = !micro;
            try
            {
                // One warm pass under this arm's configuration, so a branch-predictor or cache state change
                // between arms is not charged to the measured loop.
                engine.Run(input, output);

                var started = ValueStopwatch.StartNew();
                for (var i = 0; i < Runs; i++)
                {
                    engine.Run(input, output);
                }

                return started.GetElapsedTime().TotalMilliseconds / Runs;
            }
            finally
            {
                Conv2DGemmKernels.AblatePackB = false;
                Conv2DGemmKernels.AblateMicroKernel = false;
            }
        }
    }
}
