// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Exceptions;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.Onnx;
using DevOnBike.Overfit.Onnx.Schema;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// Sizes and paths that come out of a model file, and must not be believed.
    ///
    /// <para><b>One defect, found three times in three modules on the same day.</b> A product of dimensions
    /// read from a file, multiplied unchecked: <c>GgufTensorInfo.ElementCount</c>,
    /// <c>OnnxTensor.ElementCount</c> and the ONNX graph importer's size computation. The crash is not the
    /// interesting outcome — a wrapped product is small, positive and plausible, so it can <b>pass</b> a
    /// later shape comparison while the real on-disk layout disagrees. That loads a model with silently
    /// wrong weights and no exception at all, and the only symptom is output that is a bit worse than it
    /// should be.</para>
    ///
    /// <para>The path half is the same shape of trust: a location string from the file, used to open a file
    /// on the host.</para>
    /// </summary>
    public sealed class MalformedSizeAndPathTests
    {
        /// <summary>
        /// Two dimensions whose product overflows <see cref="long"/>. Before the fix this returned a
        /// wrapped value with no complaint.
        /// </summary>
        [Fact]
        public void GgufElementCountRefusesAnOverflowingProduct()
        {
            var info = new GgufTensorInfo(
                "blk.0.attn_q.weight",
                [0x8000_0000_0000UL, 0x8000_0000UL],
                GgmlType.F32,
                offset: 0);

            var error = Assert.Throws<OverfitFormatException>(() => info.ElementCount);

            Assert.Contains("overflow", error.Message, StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>An ordinary shape still computes, so the guard did not cost the normal case.</summary>
        [Fact]
        public void GgufElementCountStillMultipliesOrdinaryShapes()
        {
            var info = new GgufTensorInfo("w", [4096UL, 11008UL], GgmlType.F32, offset: 0);

            Assert.Equal(4096L * 11008L, info.ElementCount);
        }

        /// <summary>
        /// The ONNX side of the identical defect. Its dimensions are <see cref="int"/>, so the product is
        /// tested against <see cref="int.MaxValue"/> rather than against overflow of the accumulator.
        /// </summary>
        [Fact]
        public void OnnxElementCountRefusesAProductThatCannotBeAddressed()
        {
            var tensor = new OnnxTensor
            {
                Name = "weight",
                DataType = OnnxDataType.Float,
                Dims = [70000, 70000],
            };

            Assert.Throws<OverfitFormatException>(() => tensor.ElementCount);
        }

        [Fact]
        public void OnnxElementCountRefusesANegativeDimension()
        {
            var tensor = new OnnxTensor
            {
                Name = "weight",
                DataType = OnnxDataType.Float,
                Dims = [4, -1],
            };

            Assert.Throws<OverfitFormatException>(() => tensor.ElementCount);
        }

        /// <summary>
        /// An absolute external-data location must be refused rather than followed.
        ///
        /// <para><b>This is the one with a security shape.</b> <c>Path.Combine</c> returns its second
        /// argument whole when that argument is rooted — documented behaviour — so a model naming an
        /// absolute path is not combined with the model directory at all: the process reads that file and
        /// hands its bytes back as weights. The graph importer, which is the one required for
        /// skip-connection models, had no check at all.</para>
        /// </summary>
        [Fact]
        public void ExternalDataRefusesAnAbsolutePath()
        {
            var absolute = OperatingSystem.IsWindows()
                ? @"C:\Windows\System32\drivers\etc\hosts"
                : "/etc/passwd";

            var error = Assert.Throws<OverfitFormatException>(
                () => OnnxExternalData.ResolvePath(Path.GetTempPath(), absolute, "w"));

            Assert.Contains("absolute", error.Message, StringComparison.OrdinalIgnoreCase);
        }

        [Fact]
        public void ExternalDataRefusesAPathThatEscapesTheModelDirectory()
        {
            var error = Assert.Throws<OverfitFormatException>(
                () => OnnxExternalData.ResolvePath(
                    Path.GetTempPath(), Path.Combine("..", "..", "..", "secrets.bin"), "w"));

            Assert.Contains("escapes", error.Message, StringComparison.OrdinalIgnoreCase);
        }

        [Fact]
        public void ExternalDataRefusesAnEmptyLocation()
        {
            Assert.Throws<OverfitFormatException>(
                () => OnnxExternalData.ResolvePath(Path.GetTempPath(), "   ", "w"));
        }

        /// <summary>A location beside the model still resolves, so the guard did not close the normal case.</summary>
        [Fact]
        public void ExternalDataAcceptsASiblingFile()
        {
            var dir = Path.GetTempPath();
            var resolved = OnnxExternalData.ResolvePath(dir, "model.onnx.data", "w");

            Assert.Equal(Path.GetFullPath(Path.Combine(dir, "model.onnx.data")), resolved);
        }

        /// <summary>
        /// A sidecar whose header declares a name longer than the file. The caller's promise is what makes
        /// this matter: <c>TryOpenSidecar</c> says a corrupt sidecar must never block loading, and catches
        /// <see cref="OverfitFormatException"/> and <see cref="IOException"/> — so an unbounded read that
        /// raised <see cref="OutOfMemoryException"/> was the one case that comment existed for and did not
        /// cover.
        /// </summary>
        [Fact]
        public void RepackedSidecarRefusesANameLongerThanTheFile()
        {
            var path = Path.Combine(Path.GetTempPath(), $"overfit_bad_repack_{Guid.NewGuid():N}.repack");

            try
            {
                using (var stream = File.Create(path))
                using (var writer = new BinaryWriter(stream))
                {
                    writer.Write("OVFRPK"u8.ToArray());
                    writer.Write((byte)1);
                    writer.Write((byte)0);
                    writer.Write(1);                 // one entry
                    writer.Write(int.MaxValue);      // ... whose name claims two gigabytes
                }

                Assert.Throws<OverfitFormatException>(() => RepackedWeightsFile.Open(path));
            }
            finally
            {
                File.Delete(path);
            }
        }

        /// <summary>An entry count larger than the remaining bytes can hold is a malformed header.</summary>
        [Fact]
        public void RepackedSidecarRefusesAnImpossibleEntryCount()
        {
            var path = Path.Combine(Path.GetTempPath(), $"overfit_bad_repack_{Guid.NewGuid():N}.repack");

            try
            {
                using (var stream = File.Create(path))
                using (var writer = new BinaryWriter(stream))
                {
                    writer.Write("OVFRPK"u8.ToArray());
                    writer.Write((byte)1);
                    writer.Write((byte)0);
                    writer.Write(100_000_000);
                }

                Assert.Throws<OverfitFormatException>(() => RepackedWeightsFile.Open(path));
            }
            finally
            {
                File.Delete(path);
            }
        }
    }
}
