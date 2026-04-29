// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.Autograd
{
    /// <summary>
    /// WÄ™zeÅ‚ grafu obliczeniowego.
    /// Przechowuje fizycznÄ… pamiÄ™Ä‡ (TensorStorage) dla danych i gradientÃ³w.
    /// UdostÄ™pnia bezalokacyjne widoki (TensorSpan) dla logiki matematycznej.
    /// </summary>
    public sealed class AutogradNode : IDisposable
    {
        private TensorStorage<float>? _dataStorage;
        private TensorStorage<float>? _gradStorage;
        private readonly bool _ownsDataStorage;
        private readonly bool _ownsGradStorage;
        private int _disposed;

        public bool RequiresGrad { get; }

        public TensorShape Shape { get; }

        /// <summary>
        /// Lifecycle and ownership classification for this node.
        /// Used to determine which nodes graph.Reset() may dispose.
        /// Defaults to <see cref="AutogradNodeOwnership.Unknown"/> for backward compatibility.
        /// </summary>
        public AutogradNodeOwnership Ownership { get; internal set; } = AutogradNodeOwnership.Unknown;

        /// <summary>
        /// Zwraca widok na dane z Forward Pass.
        /// </summary>
        public TensorSpan<float> DataView
        {
            get
            {
                ObjectDisposedException.ThrowIf(_disposed == 1, this);
                return new TensorSpan<float>(_dataStorage!.AsSpan(), Shape);
            }
        }

        /// <summary>
        /// Zwraca widok na gradienty z Backward Pass.
        /// </summary>
        public TensorSpan<float> GradView
        {
            get
            {
                ObjectDisposedException.ThrowIf(_disposed == 1, this);

                if (!RequiresGrad || _gradStorage == null)
                {
                    throw new InvalidOperationException("Ten wÄ™zeÅ‚ nie Å›ledzi gradientÃ³w (RequiresGrad = false).");
                }

                return new TensorSpan<float>(_gradStorage.AsSpan(), Shape);
            }
        }

        /// <summary>
        /// Standard constructor — node owns data storage, allocates its own grad storage when requiresGrad=true.
        /// Used by the graph factory (CreateTemporary) and standalone inference nodes.
        /// </summary>
        public AutogradNode(TensorStorage<float> data, TensorShape shape, bool requiresGrad = false)
            : this(data, shape, requiresGrad,
                   ownsDataStorage: true,
                   externalGrad: null,
                   ownsGradStorage: false)
        {
        }

        /// <summary>
        /// Full-control private constructor. All ownership flags are explicit.
        /// Use via static factory methods to ensure Ownership is set correctly.
        /// </summary>
        private AutogradNode(
            TensorStorage<float> data,
            TensorShape shape,
            bool requiresGrad,
            bool ownsDataStorage,
            TensorStorage<float>? externalGrad,
            bool ownsGradStorage)
        {
            ArgumentNullException.ThrowIfNull(data);

            if (!shape.IsValid)
            {
                throw new ArgumentException("Shape is invalid.", nameof(shape));
            }

            if (data.Length < shape.Size)
            {
                throw new ArgumentException(
                    $"Storage length {data.Length} is smaller than required tensor size {shape.Size}.",
                    nameof(data));
            }

            _dataStorage = data;
            _ownsDataStorage = ownsDataStorage;
            Shape = shape;
            RequiresGrad = requiresGrad;

            if (requiresGrad)
            {
                if (externalGrad != null)
                {
                    // Shared grad storage (e.g., Parameter.AsNode): optimizer reads Parameter.Grad directly.
                    _gradStorage = externalGrad;
                    _ownsGradStorage = ownsGradStorage;
                }
                else
                {
                    // Node-owned grad storage: the default for graph temporaries.
                    _gradStorage = TensorFactory.CloneStorage(data, clearMemory: true);
                    _ownsGradStorage = true;
                }
            }
        }

        // ─────────────────────────────────────────────────────────────────
        // Factory methods — preferred over public constructor for
        // non-standard ownership scenarios
        // ─────────────────────────────────────────────────────────────────

        /// <summary>
        /// Creates a node that references a Parameter's data and gradient storage.
        /// The node does NOT own either storage — the Parameter remains the owner.
        /// Ownership is set to <see cref="AutogradNodeOwnership.Parameter"/>.
        /// Backward writes into <paramref name="grad"/> directly, so the optimizer
        /// can read Parameter.Grad without any post-backward sync copy.
        /// </summary>
        internal static AutogradNode CreateParameterView(
            TensorStorage<float> data,
            TensorStorage<float> grad,
            TensorShape shape)
        {
            ArgumentNullException.ThrowIfNull(data);
            ArgumentNullException.ThrowIfNull(grad);

            return new AutogradNode(
                data, shape,
                requiresGrad: true,
                ownsDataStorage: false,
                externalGrad: grad,
                ownsGradStorage: false)
            {
                Ownership = AutogradNodeOwnership.Parameter,
            };
        }

        /// <summary>
        /// Creates a node that wraps externally-owned data storage without taking ownership.
        /// The graph will NOT dispose this storage on Reset().
        /// Ownership is set to <see cref="AutogradNodeOwnership.ExternalBorrowed"/>.
        /// </summary>
        internal static AutogradNode CreateBorrowed(
            TensorStorage<float> data,
            TensorShape shape,
            bool requiresGrad = false)
        {
            ArgumentNullException.ThrowIfNull(data);

            return new AutogradNode(
                data, shape,
                requiresGrad,
                ownsDataStorage: false,
                externalGrad: null,
                ownsGradStorage: false)
            {
                Ownership = AutogradNodeOwnership.ExternalBorrowed,
            };
        }

        /// <summary>
        /// Tworzy view-node na tym samym data storage.
        /// Data storage pozostaje wÅ‚asnoÅ›ciÄ… source.
        /// Grad storage jest osobny, bo backward zapisuje gradient w ksztaÅ‚cie view.
        /// </summary>
        internal static AutogradNode ViewOf(AutogradNode source, TensorShape shape, bool requiresGrad)
        {
            ArgumentNullException.ThrowIfNull(source);
            ObjectDisposedException.ThrowIf(source._disposed == 1, source);

            if (!shape.IsValid)
            {
                throw new ArgumentException("Shape is invalid.", nameof(shape));
            }

            if (shape.Size != source.Shape.Size)
            {
                throw new ArgumentException(
                    $"View shape size {shape.Size} does not match source size {source.Shape.Size}.",
                    nameof(shape));
            }

            return new AutogradNode(
                source._dataStorage!,
                shape,
                requiresGrad,
                ownsDataStorage: false,
                externalGrad: null,
                ownsGradStorage: false)
            {
                Ownership = AutogradNodeOwnership.View,
            };
        }

        /// <summary>
        /// Zeruje gradienty przed nowÄ… epokÄ… lub batchem.
        /// </summary>
        public void ZeroGrad()
        {
            if (RequiresGrad && _gradStorage != null)
            {
                _gradStorage.AsSpan().Clear();
            }
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
            {
                return;
            }

            if (_ownsDataStorage)
            {
                _dataStorage?.Dispose();
            }

            if (_ownsGradStorage)
            {
                _gradStorage?.Dispose();
            }

            _dataStorage = null;
            _gradStorage = null;
        }
    }
}