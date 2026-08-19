// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using DevOnBike.Overfit.Onnx;
using DevOnBike.Overfit.Runtime;

namespace DevOnBike.Overfit.Tests.Diagnostics
{
    /// <summary>
    /// Resident bytes after importing a model, so `XC-82`'s claim can be settled with a measurement rather
    /// than with arithmetic off the source.
    ///
    /// <para><b>The claim.</b> <c>LinearLayer</c> allocates <c>_weightsTransposed</c> — a full second copy of
    /// the weight matrix — in its constructor, unconditionally. Both readers of that copy
    /// (<c>ForwardOutputMajorTiled</c> and <c>ForwardOutputMajorDot</c>) are reached only when
    /// <c>outputSize &lt; 32</c>, and VGG-16's dense layers have 4096, 4096 and 1000 outputs. If the reading
    /// is right, <b>494 MiB is allocated and never read</b>.</para>
    ///
    /// <para>Working set is reported alongside managed heap because the storage may be native: a managed-only
    /// figure would miss it entirely and read as "no problem".</para>
    /// </summary>
    public sealed class DenseWeightResidencyDiagnostics
    {
        private const int InputSize = 3 * 224 * 224;
        private const int OutputSize = 1000;

        private readonly ITestOutputHelper _out;

        public DenseWeightResidencyDiagnostics(ITestOutputHelper output)
        {
            _out = output;
        }

        /// <summary>
        /// The same measurement on the MNIST CNN fixture, which matters for the opposite reason.
        ///
        /// <para>Its classifier has <b>ten</b> outputs, so <c>outputSize &lt; 32</c> and the output-major
        /// copy is still allocated - this exercises the branch VGG-16 cannot reach. A model that keeps the
        /// copy must keep working, and its cost must stay proportionate: ten outputs is kilobytes, not the
        /// 471.6 MiB VGG-16 was paying for a copy nothing could read.</para>
        /// </summary>
        [LongFact("2s")]
        public void ResidentBytes_AfterImportingTheMnistCnn()
        {
            var path = Path.Combine(AppContext.BaseDirectory, "test_fixtures", "mnist_cnn.onnx");

            if (!File.Exists(path))
            {
                _out.WriteLine($"missing {path}");
                return;
            }

            Report(path, inputSize: 28 * 28, outputSize: 10);
        }

        /// <summary>Loads the model and reports what it costs, managed and resident.</summary>
        private void Report(string path, int inputSize, int outputSize)
        {
            var process = Process.GetCurrentProcess();

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            process.Refresh();

            var managedBefore = GC.GetTotalMemory(forceFullCollection: true);
            var workingBefore = process.WorkingSet64;

            using (var model = OnnxGraphImporter.Load(path, inputSize, outputSize))
            {
                model.Eval();

                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();

                process.Refresh();

                var managedAfter = GC.GetTotalMemory(forceFullCollection: true);
                var workingAfter = process.WorkingSet64;

                const double Mib = 1024.0 * 1024.0;

                _out.WriteLine($"{Path.GetFileName(path)}");
                _out.WriteLine($"  managed heap : {(managedAfter - managedBefore) / Mib,10:F2} MiB");
                _out.WriteLine($"  working set  : {(workingAfter - workingBefore) / Mib,10:F2} MiB");
                _out.WriteLine($"  file on disk : {new FileInfo(path).Length / Mib,10:F2} MiB");

                Assert.True(managedAfter >= managedBefore, "importing the model freed managed memory");
            }
        }

        [LongFact("6s")]
        public void ResidentBytes_AfterImportingVgg16()
        {
            var path = Environment.GetEnvironmentVariable(OverfitEnvironment.CnnOnnx)
                ?? @"C:\onnxmodels\vgg16.onnx";

            if (!File.Exists(path))
            {
                _out.WriteLine($"missing {path} — set OVERFIT_CNN_ONNX");
                return;
            }

            var process = Process.GetCurrentProcess();

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            process.Refresh();

            var managedBefore = GC.GetTotalMemory(forceFullCollection: true);
            var workingBefore = process.WorkingSet64;

            using var model = OnnxGraphImporter.Load(path, InputSize, OutputSize);
            model.Eval();

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            process.Refresh();

            var managedAfter = GC.GetTotalMemory(forceFullCollection: true);
            var workingAfter = process.WorkingSet64;

            const double Mib = 1024.0 * 1024.0;

            _out.WriteLine($"managed heap : {(managedAfter - managedBefore) / Mib,10:F1} MiB");
            _out.WriteLine($"working set  : {(workingAfter - workingBefore) / Mib,10:F1} MiB");
            _out.WriteLine($"file on disk : {new FileInfo(path).Length / Mib,10:F1} MiB");

            Assert.True(workingAfter > workingBefore, "importing the model did not increase the working set");
        }
    }
}
