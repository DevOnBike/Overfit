// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core; // Wpinamy nowy Core!

namespace DevOnBike.Overfit.Autograd
{
    /// <summary>
    /// Węzeł grafu obliczeniowego. 
    /// Przechowuje fizyczną pamięć (TensorStorage) dla danych i gradientów.
    /// Udostępnia bezalokacyjne widoki (TensorSpan) dla logiki matematycznej.
    /// </summary>
    public sealed class AutogradNode : IDisposable
    {
        // 1. PAMIĘĆ FIZYCZNA NA STERCIE / ARENIE (Magazyny)
        private TensorStorage<float>? _dataStorage;
        private TensorStorage<float>? _gradStorage;
        private int _disposed;

        public bool RequiresGrad { get; }

        // NOWOŚĆ: Węzeł sam pamięta swój kształt!
        public TensorShape Shape { get; }

        // ========================================================================
        // 2. BEZALOKACYJNE WIDOKI (Bramy do matematyki)
        // ========================================================================

        /// <summary> Zwraca widok na dane z Forward Pass. Używane przez TensorMath. </summary>
        public TensorSpan<float> DataView
        {
            get
            {
                ObjectDisposedException.ThrowIf(_disposed == 1, this);

                // ZERO-ALLOC: Składamy widok na stosie za każdym razem w ułamek nanosekundy!
                return new TensorSpan<float>(_dataStorage!.AsSpan(), Shape);
            }
        }

        /// <summary> Zwraca widok na gradienty z Backward Pass. Używane przez TensorMath. </summary>
        public TensorSpan<float> GradView
        {
            get
            {
                ObjectDisposedException.ThrowIf(_disposed == 1, this);

                if (!RequiresGrad || _gradStorage == null)
                {
                    throw new InvalidOperationException("Ten węzeł nie śledzi gradientów (RequiresGrad = false).");
                }

                return new TensorSpan<float>(_gradStorage.AsSpan(), Shape);
            }
        }

        // ========================================================================
        // KONSTRUKTOR
        // ========================================================================

        public AutogradNode(TensorStorage<float> data, TensorShape shape, bool requiresGrad = false)
        {
            _dataStorage = data ?? throw new ArgumentNullException(nameof(data));
            Shape = shape;
            RequiresGrad = requiresGrad;

            if (requiresGrad)
            {
                // Używamy nowej Fabryki DOD: sama zadba o to, czy sklonować to na Arenie, czy w Puli!
                _gradStorage = TensorFactory.CloneStorage(data, clearMemory: true);
            }
        }

        // ========================================================================
        // OPERACJE NARZĘDZIOWE
        // ========================================================================

        /// <summary>
        /// Zeruje gradienty przed nową epoką lub batchem.
        /// Wykorzystuje sprzętowe czyszczenie pamięci Span.
        /// </summary>
        public void ZeroGrad()
        {
            if (RequiresGrad && _gradStorage != null)
            {
                _gradStorage.AsSpan().Clear();
            }
        }

        // ========================================================================
        // SPRZĄTANIE (KRYTYCZNE)
        // ========================================================================

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 0)
            {
                _dataStorage?.Dispose();
                _dataStorage = null;

                _gradStorage?.Dispose();
                _gradStorage = null;
            }
        }
    }
}