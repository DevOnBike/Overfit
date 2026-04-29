// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Parameters;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;
using Xunit;

namespace DevOnBike.Overfit.Tests
{
    public class ParameterTests
    {
        // ─────────────────────────────────────────────────────────────────────
        // Construction
        // ─────────────────────────────────────────────────────────────────────

        [Fact]
        public void Parameter_RequiresGrad_AllocatesGradBuffer()
        {
            using var p = new Parameter(new TensorShape(10), requiresGrad: true);

            Assert.True(p.RequiresGrad);
            Assert.NotNull(p.Grad);
            Assert.Equal(10, p.Grad!.Length);
            Assert.All(p.GradSpan.ToArray(), v => Assert.Equal(0f, v));
        }

        [Fact]
        public void Parameter_NoGrad_GradIsNull()
        {
            using var p = new Parameter(new TensorShape(10), requiresGrad: false);

            Assert.False(p.RequiresGrad);
            Assert.Null(p.Grad);
            Assert.Throws<InvalidOperationException>(() => _ = p.GradSpan);
        }

        [Fact]
        public void Parameter_ClearData_StartsAsZero()
        {
            using var p = new Parameter(new TensorShape(16), clearData: true);

            Assert.All(p.DataSpan.ToArray(), v => Assert.Equal(0f, v));
        }

        // ─────────────────────────────────────────────────────────────────────
        // Data access
        // ─────────────────────────────────────────────────────────────────────

        [Fact]
        public void Parameter_LoadData_CopiesCorrectly()
        {
            using var p = new Parameter(new TensorShape(4), clearData: false);
            var data = new float[] { 1f, 2f, 3f, 4f };

            p.LoadData(data);

            Assert.Equal(data, p.DataSpan.ToArray());
        }

        [Fact]
        public void Parameter_LoadData_WrongLength_Throws()
        {
            using var p = new Parameter(new TensorShape(4));

            Assert.Throws<ArgumentException>(() => p.LoadData(new float[3]));
        }

        [Fact]
        public void Parameter_DataReadOnlySpan_MatchesDataSpan()
        {
            using var p = new Parameter(new TensorShape(8), clearData: false);
            p.LoadData(Enumerable.Range(0, 8).Select(i => (float)i).ToArray());

            Assert.Equal(p.DataSpan.ToArray(), p.DataReadOnlySpan.ToArray());
        }

        // ─────────────────────────────────────────────────────────────────────
        // ZeroGrad
        // ─────────────────────────────────────────────────────────────────────

        [Fact]
        public void Parameter_ZeroGrad_ClearsGradBuffer()
        {
            using var p = new Parameter(new TensorShape(4), requiresGrad: true);

            // Write garbage into grad
            var grad = p.GradSpan;
            for (var i = 0; i < grad.Length; i++) grad[i] = i + 1f;

            p.ZeroGrad();

            Assert.All(p.GradSpan.ToArray(), v => Assert.Equal(0f, v));
        }

        [Fact]
        public void Parameter_ZeroGrad_NoGrad_IsNoOp()
        {
            using var p = new Parameter(new TensorShape(4), requiresGrad: false);

            // Should not throw
            p.ZeroGrad();
        }

        // ─────────────────────────────────────────────────────────────────────
        // AsNode
        // ─────────────────────────────────────────────────────────────────────

        [Fact]
        public void Parameter_AsNode_ReturnsNodeWithParameterOwnership()
        {
            using var p = new Parameter(new TensorShape(4, 8));
            p.LoadData(Enumerable.Range(0, 32).Select(i => (float)i).ToArray());

            using var node = p.AsNode();

            Assert.Equal(AutogradNodeOwnership.Parameter, node.Ownership);
            Assert.Equal(p.Shape, node.Shape);
        }

        [Fact]
        public void Parameter_AsNode_SharesDataStorage()
        {
            using var p = new Parameter(new TensorShape(4), clearData: true);
            p.DataSpan[0] = 42f;

            using var node = p.AsNode();

            // Node reads from same underlying storage
            Assert.Equal(42f, node.DataView.AsReadOnlySpan()[0]);

            // Write via node, read back via parameter
            node.DataView.AsSpan()[1] = 99f;
            Assert.Equal(99f, p.DataSpan[1]);
        }

        [Fact]
        public void Parameter_AsNode_RequiresGrad_NodeAlsoRequiresGrad()
        {
            using var p = new Parameter(new TensorShape(4), requiresGrad: true);

            using var node = p.AsNode();

            Assert.True(node.RequiresGrad);
        }

        // ─────────────────────────────────────────────────────────────────────
        // Serialisation
        // ─────────────────────────────────────────────────────────────────────

        [Fact]
        public void Parameter_SaveLoad_RoundTrip()
        {
            using var original = new Parameter(new TensorShape(3, 3), clearData: false);
            var data = Enumerable.Range(1, 9).Select(i => (float)i).ToArray();
            original.LoadData(data);

            using var ms = new MemoryStream();
            using (var writer = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true))
            {
                original.Save(writer);
            }

            ms.Position = 0;

            using var loaded = new Parameter(new TensorShape(3, 3), clearData: true);
            using (var reader = new BinaryReader(ms))
            {
                loaded.Load(reader);
            }

            Assert.Equal(data, loaded.DataSpan.ToArray());
        }

        [Fact]
        public void Parameter_Load_WrongSize_Throws()
        {
            // Save a 4-element parameter
            using var original = new Parameter(new TensorShape(4));
            using var ms = new MemoryStream();
            using (var writer = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true))
            {
                original.Save(writer);
            }

            ms.Position = 0;

            // Try to load into a 9-element parameter
            using var target = new Parameter(new TensorShape(9));
            using var reader = new BinaryReader(ms);
            Assert.Throws<InvalidDataException>(() => target.Load(reader));
        }

        [Fact]
        public void Parameter_AsNode_RequiresGrad_SharesGradStorage()
        {
            using var p = new Parameter(new TensorShape(4), requiresGrad: true);

            using var node = p.AsNode();

            // Backward accumulates into node.GradView
            var gradSpan = node.GradView.AsSpan();
            gradSpan[0] = 7f;
            gradSpan[1] = 8f;

            // Optimizer reads it directly from Parameter.Grad — no copy needed
            Assert.Equal(7f, p.GradSpan[0]);
            Assert.Equal(8f, p.GradSpan[1]);
        }

        [Fact]
        public void Parameter_Dispose_DoesNotDisposeNodeDataStorage()
        {
            // Node borrows Parameter storage — disposing Parameter must not break
            // the node that was already handed out (lifecycle: node disposed first).
            var p = new Parameter(new TensorShape(4));
            p.LoadData([1f, 2f, 3f, 4f]);

            var node = p.AsNode();

            // Dispose Parameter first (unusual order, but must not corrupt node)
            // In practice the layer disposes Parameter, InferenceEngine may still
            // hold the node briefly — this test documents the expected behavior.
            // NOTE: accessing node after p.Dispose() is undefined in production.
            // Here we just assert node holds a valid reference to its own view.
            p.Dispose();
            node.Dispose(); // must not throw
        }

        [Fact]
        public void AutogradNode_CreateBorrowed_DoesNotOwnStorage()
        {
            var storage = new TensorStorage<float>(4, clearMemory: false);
            storage.AsSpan().Fill(5f);

            var node = AutogradNode.CreateBorrowed(storage, new TensorShape(4));

            Assert.Equal(AutogradNodeOwnership.ExternalBorrowed, node.Ownership);

            // Disposing node must not dispose storage
            node.Dispose();

            // Storage still usable
            Assert.Equal(5f, storage.AsSpan()[0]);
            storage.Dispose();
        }

        // ─────────────────────────────────────────────────────────────────────
        // Dispose / lifecycle
        // ─────────────────────────────────────────────────────────────────────

        [Fact]
        public void Parameter_DisposeTwice_IsIdempotent()
        {
            var p = new Parameter(new TensorShape(4));
            p.Dispose();
            p.Dispose(); // must not throw
        }

        [Fact]
        public void Parameter_AccessAfterDispose_Throws()
        {
            var p = new Parameter(new TensorShape(4));
            p.Dispose();

            Assert.Throws<ObjectDisposedException>(() => _ = p.DataSpan);
            Assert.Throws<ObjectDisposedException>(() => p.LoadData(new float[4]));
            Assert.Throws<ObjectDisposedException>(() => p.AsNode());
        }
    }
}
