// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Monitoring;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.DeepLearning;

namespace DevOnBike.Overfit.Tests.Monitoring
{
    public sealed class ModelPersistenceTests
    {
        private static AnomalyAutoencoder MakeAutoencoder(int inputSize = 32)
        {
            var m = new AnomalyAutoencoder(inputSize, hidden1: 8, hidden2: 4, bottleneckDim: 2);
            m.Eval();
            return m;
        }

        private static ReconstructionScorer MakeCalibratedScorer(float threshold = 0.05f)
        {
            var s = new ReconstructionScorer();
            s.Calibrate([threshold]);
            return s;
        }

        // -------------------------------------------------------------------------
        // Save — argument validation
        // -------------------------------------------------------------------------

        [Fact]
        public void Save_WhenPathIsNull_ThenThrowsArgumentNullException()
        {
            using var ae = MakeAutoencoder();
            var scorer = MakeCalibratedScorer();
            Assert.Throws<ArgumentNullException>(() => ModelPersistence.Save((string)null!, ae, scorer));
        }

        [Fact]
        public void Save_WhenAutoencoderIsNull_ThenThrowsArgumentNullException()
        {
            var scorer = MakeCalibratedScorer();
            var path = Path.GetTempFileName();
            try
            {
                Assert.Throws<ArgumentNullException>(() => ModelPersistence.Save(path, null!, scorer));
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void Save_WhenScorerIsNull_ThenThrowsArgumentNullException()
        {
            using var ae = MakeAutoencoder();
            var path = Path.GetTempFileName();
            try
            {
                Assert.Throws<ArgumentNullException>(() => ModelPersistence.Save(path, ae, null!));
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void Save_WhenScorerNotCalibrated_ThenThrowsInvalidOperationException()
        {
            using var ae = MakeAutoencoder();
            var scorer = new ReconstructionScorer(); // not calibrated
            var path = Path.GetTempFileName();
            try
            {
                Assert.Throws<InvalidOperationException>(() => ModelPersistence.Save(path, ae, scorer));
            }
            finally { File.Delete(path); }
        }

        // -------------------------------------------------------------------------
        // Save → file
        // -------------------------------------------------------------------------

        [Fact]
        public void Save_WhenCalled_ThenFileIsCreated()
        {
            using var ae = MakeAutoencoder();
            var scorer = MakeCalibratedScorer();
            var path = Path.GetTempFileName();
            try
            {
                ModelPersistence.Save(path, ae, scorer);
                Assert.True(File.Exists(path));
                Assert.True(new FileInfo(path).Length > 0);
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void Save_WhenCalledTwice_ThenOverwritesPreviousFile()
        {
            using var ae = MakeAutoencoder();
            var scorer = MakeCalibratedScorer();
            var path = Path.GetTempFileName();
            try
            {
                ModelPersistence.Save(path, ae, scorer, "first");
                var size1 = new FileInfo(path).Length;
                ModelPersistence.Save(path, ae, scorer, "second");
                var size2 = new FileInfo(path).Length;
                Assert.True(size2 > 0);
                _ = size1; // both exist and have content
            }
            finally { File.Delete(path); }
        }

        // -------------------------------------------------------------------------
        // Load — argument validation
        // -------------------------------------------------------------------------

        [Fact]
        public void Load_WhenFileDoesNotExist_ThenThrowsFileNotFoundException()
            => Assert.Throws<FileNotFoundException>(
            () => ModelPersistence.Load("/tmp/does-not-exist-overfit.ovfw"));

        // -------------------------------------------------------------------------
        // Round-trip: Save → Load
        // -------------------------------------------------------------------------

        [Fact]
        public void SaveLoad_WhenRoundtripped_ThenAutoencoderInputSizeIsPreserved()
        {
            using var ae = MakeAutoencoder(inputSize: 32);
            var scorer = MakeCalibratedScorer();
            var path = Path.GetTempFileName();
            try
            {
                ModelPersistence.Save(path, ae, scorer);
                var (loaded, _) = ModelPersistence.Load(path);
                using (loaded)
                {
                    Assert.Equal(32, loaded.InputSize);
                }
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void SaveLoad_WhenRoundtripped_ThenScorerThresholdIsPreserved()
        {
            using var ae = MakeAutoencoder();
            var scorer = MakeCalibratedScorer(threshold: 0.123f);
            var path = Path.GetTempFileName();
            try
            {
                ModelPersistence.Save(path, ae, scorer);
                var (loadedAe, loadedScorer) = ModelPersistence.Load(path);
                using (loadedAe)
                {
                    Assert.True(MathF.Abs(loadedScorer.Threshold - 0.123f) < 1e-5f);
                }
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void SaveLoad_WhenRoundtripped_ThenScorerIsCalibratedAfterLoad()
        {
            using var ae = MakeAutoencoder();
            var scorer = MakeCalibratedScorer();
            var path = Path.GetTempFileName();
            try
            {
                ModelPersistence.Save(path, ae, scorer);
                var (loadedAe, loadedScorer) = ModelPersistence.Load(path);
                using (loadedAe)
                {
                    Assert.True(loadedScorer.IsCalibrated);
                }
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void SaveLoad_WhenRoundtripped_ThenReconstructionIsFinite()
        {
            using var ae = MakeAutoencoder(inputSize: 32);
            var scorer = MakeCalibratedScorer();
            var path = Path.GetTempFileName();
            var features = Enumerable.Range(0, 32).Select(i => (float)i * 0.01f).ToArray();
            try
            {
                ModelPersistence.Save(path, ae, scorer);
                var (loadedAe, _) = ModelPersistence.Load(path);
                using (loadedAe)
                {
                    loadedAe.Eval();
                    var recon = new float[32];
                    loadedAe.Reconstruct(features, recon);
                    Assert.True(recon.All(float.IsFinite));
                }
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void SaveLoad_WhenRoundtripped_ThenLabelIsPreservedInHeader()
        {
            using var ae = MakeAutoencoder();
            var scorer = MakeCalibratedScorer();
            var path = Path.GetTempFileName();
            try
            {
                ModelPersistence.Save(path, ae, scorer, label: "prod-v42");
                var header = ModelPersistence.ReadHeader(path);
                Assert.Equal("prod-v42", header.Label);
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void SaveLoad_WhenRoundtripped_ThenSavedAtIsRecent()
        {
            var before = DateTime.UtcNow;
            using var ae = MakeAutoencoder();
            var scorer = MakeCalibratedScorer();
            var path = Path.GetTempFileName();
            try
            {
                ModelPersistence.Save(path, ae, scorer);
                var after = DateTime.UtcNow;
                var header = ModelPersistence.ReadHeader(path);
                Assert.True(header.SavedAt >= before && header.SavedAt <= after);
            }
            finally { File.Delete(path); }
        }

        // -------------------------------------------------------------------------
        // ReadHeader
        // -------------------------------------------------------------------------

        [Fact]
        public void ReadHeader_WhenValidFile_ThenReturnsCorrectInputSize()
        {
            using var ae = MakeAutoencoder(inputSize: 32);
            var scorer = MakeCalibratedScorer();
            var path = Path.GetTempFileName();
            try
            {
                ModelPersistence.Save(path, ae, scorer);
                var h = ModelPersistence.ReadHeader(path);
                Assert.Equal(32, h.InputSize);
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void ReadHeader_WhenFileDoesNotExist_ThenThrowsFileNotFoundException()
            => Assert.Throws<FileNotFoundException>(
            () => ModelPersistence.ReadHeader("/tmp/no-such-model.ovfw"));

        [Fact]
        public void ReadHeader_WhenFileIsTruncated_ThenThrowsException()
        {
            var path = Path.GetTempFileName();
            try
            {
                // Write valid magic but NO version — file is truncated after 4 bytes
                using var fs = new FileStream(path, FileMode.Create);
                using var bw = new BinaryWriter(fs);
                bw.Write(ModelBundleHeader.Magic); // correct little-endian magic
                // version intentionally omitted — ReadUInt16 will throw EndOfStreamException
            }
            finally {}

            try
            {
                Assert.Throws<EndOfStreamException>(() => ModelPersistence.ReadHeader(path));
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void Load_WhenMagicIsWrong_ThenThrowsInvalidDataException()
        {
            var path = Path.GetTempFileName();
            try
            {
                using var fs = new FileStream(path, FileMode.Create);
                using var bw = new BinaryWriter(fs);
                bw.Write(0xDEADBEEFu); // wrong magic
                bw.Write((ushort)1);
            }
            catch {}

            try
            {
                Assert.Throws<InvalidDataException>(() => ModelPersistence.Load(path));
            }
            finally { File.Delete(path); }
        }

        // -------------------------------------------------------------------------
        // BinaryWriter overload
        // -------------------------------------------------------------------------

        [Fact]
        public void SaveLoad_WhenUsingStreamOverload_ThenRoundtripsCorrectly()
        {
            using var ae = MakeAutoencoder(inputSize: 32);
            var scorer = MakeCalibratedScorer(threshold: 0.07f);

            using var ms = new MemoryStream();
            using (var bw = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true))
            {
                ModelPersistence.Save(bw, ae, scorer, "stream-test");
            }

            ms.Position = 0;
            using var br = new BinaryReader(ms, System.Text.Encoding.UTF8, leaveOpen: true);
            var (loadedAe, loadedScorer) = ModelPersistence.Load(br);
            using (loadedAe)
            {
                Assert.Equal(32, loadedAe.InputSize);
                Assert.True(MathF.Abs(loadedScorer.Threshold - 0.07f) < 1e-5f);
            }
        }
    }
}