// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.Parameters
{
    /// <summary>
    /// Long-lived trainable model state. Owns its data and gradient storage independently
    /// of any <see cref="ComputationGraph"/>.
    ///
    /// Lifecycle contrast with <see cref="AutogradNode"/>:
    /// <list type="bullet">
    ///   <item><see cref="AutogradNode"/> with <c>Ownership = GraphTemporary</c> lives for one
    ///   forward pass and is disposed by <see cref="ComputationGraph.Reset"/>.</item>
    ///   <item><see cref="Parameter"/> lives for the lifetime of the layer/model and is
    ///   disposed only when the layer/model itself is disposed.</item>
    /// </list>
    ///
    /// Optimizers should accept <c>IEnumerable&lt;Parameter&gt;</c>, not
    /// <c>IEnumerable&lt;AutogradNode&gt;</c>. This is enforced in Etap 6 of the
    /// architecture refactor plan.
    /// </summary>
    public sealed class Parameter : IDisposable
    {
        private int _disposed;

        /// <param name="shape">Tensor dimensions.</param>
        /// <param name="requiresGrad">
        /// When <c>true</c>, a zero-initialised gradient buffer is allocated alongside the data
        /// buffer. Set to <c>false</c> for frozen / non-trainable parameters.
        /// </param>
        /// <param name="clearData">
        /// When <c>true</c>, the data buffer is zero-initialised. Set to <c>false</c> when you
        /// will overwrite it immediately (e.g., via <see cref="LoadData"/>).
        /// </param>
        public Parameter(TensorShape shape, bool requiresGrad = true, bool clearData = true)
        {
            if (!shape.IsValid)
            {
                throw new ArgumentException("Shape is invalid.", nameof(shape));
            }

            Shape = shape;
            RequiresGrad = requiresGrad;

            Data = new TensorStorage<float>(shape.Size, clearMemory: clearData);

            if (requiresGrad)
            {
                Grad = new TensorStorage<float>(shape.Size, clearMemory: true);
            }
        }

        /// <summary>Shape of the parameter tensor.</summary>
        public TensorShape Shape { get; }

        /// <summary>Whether this parameter participates in gradient computation.</summary>
        public bool RequiresGrad { get; }

        /// <summary>Raw data buffer. Caller may read and write via <see cref="Span"/>.</summary>
        public TensorStorage<float> Data { get; }

        /// <summary>
        /// Gradient buffer. Non-null only when <see cref="RequiresGrad"/> is <c>true</c>.
        /// </summary>
        public TensorStorage<float>? Grad { get; }

        /// <summary>Mutable view over the data buffer.</summary>
        public Span<float> DataSpan
        {
            get
            {
                ObjectDisposedException.ThrowIf(_disposed == 1, this);
                return Data.AsSpan();
            }
        }

        /// <summary>Read-only view over the data buffer.</summary>
        public ReadOnlySpan<float> DataReadOnlySpan
        {
            get
            {
                ObjectDisposedException.ThrowIf(_disposed == 1, this);
                return Data.AsReadOnlySpan();
            }
        }

        /// <summary>Mutable view over the gradient buffer.</summary>
        /// <exception cref="InvalidOperationException">
        /// Thrown when <see cref="RequiresGrad"/> is <c>false</c>.
        /// </exception>
        public Span<float> GradSpan
        {
            get
            {
                ObjectDisposedException.ThrowIf(_disposed == 1, this);

                if (!RequiresGrad || Grad is null)
                {
                    throw new InvalidOperationException(
                        "This parameter does not track gradients (RequiresGrad = false).");
                }

                return Grad.AsSpan();
            }
        }

        /// <summary>
        /// Zeros out the gradient buffer. Called by optimizers at the start of each step.
        /// No-op when <see cref="RequiresGrad"/> is <c>false</c>.
        /// </summary>
        public void ZeroGrad()
        {
            Grad?.AsSpan().Clear();
        }

        /// <summary>
        /// Copies <paramref name="data"/> into this parameter's data buffer.
        /// Typically called by layer constructors after weight initialisation or by
        /// the ONNX importer after loading weights from a checkpoint.
        /// </summary>
        public void LoadData(ReadOnlySpan<float> data)
        {
            ObjectDisposedException.ThrowIf(_disposed == 1, this);

            if (data.Length != Shape.Size)
            {
                throw new ArgumentException(
                    $"Data length {data.Length} does not match parameter size {Shape.Size}.",
                    nameof(data));
            }

            data.CopyTo(Data.AsSpan());
        }

        /// <summary>
        /// Returns an <see cref="AutogradNode"/> that wraps this parameter's storage.
        /// The node does not own the storage — the <see cref="Parameter"/> remains the owner.
        ///
        /// Use this to pass a parameter into graph operations while the graph is active.
        /// The returned node is tagged <see cref="AutogradNodeOwnership.Parameter"/> so
        /// that <see cref="ComputationGraph.Reset"/> will not dispose it.
        /// </summary>
        public AutogradNode AsNode()
        {
            ObjectDisposedException.ThrowIf(_disposed == 1, this);

            if (RequiresGrad)
            {
                if (Grad is null)
                {
                    throw new InvalidOperationException(
                        "RequiresGrad is true but Grad storage is null. This is a bug in Parameter construction.");
                }

                // CreateParameterView shares both Data and Grad with this Parameter.
                // ownsDataStorage = false, ownsGradStorage = false.
                // Backward accumulates directly into Parameter.Grad — no post-backward sync needed.
                // Optimizer reads Parameter.Grad without any copy.
                return AutogradNode.CreateParameterView(Data, Grad, Shape);
            }

            // Non-trainable parameter: borrow data without grad.
            return AutogradNode.CreateBorrowed(Data, Shape, requiresGrad: false);
        }

        /// <summary>
        /// Saves the parameter data to a <see cref="BinaryWriter"/> in a simple
        /// length-prefixed float32 format compatible with the existing
        /// <c>LinearLayer.Save</c> / <c>Load</c> convention.
        /// </summary>
        public void Save(BinaryWriter writer)
        {
            ObjectDisposedException.ThrowIf(_disposed == 1, this);

            writer.Write(Shape.Size);

            foreach (var value in Data.AsReadOnlySpan())
            {
                writer.Write(value);
            }
        }

        /// <summary>
        /// Loads parameter data from a <see cref="BinaryReader"/>.
        /// </summary>
        public void Load(BinaryReader reader)
        {
            ObjectDisposedException.ThrowIf(_disposed == 1, this);

            var length = reader.ReadInt32();

            if (length != Shape.Size)
            {
                throw new InvalidDataException(
                    $"Checkpoint size {length} does not match parameter size {Shape.Size}.");
            }

            var span = Data.AsSpan();

            for (var i = 0; i < length; i++)
            {
                span[i] = reader.ReadSingle();
            }
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
            {
                return;
            }

            Data.Dispose();
            Grad?.Dispose();
        }
    }
}
