// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.

namespace DevOnBike.Overfit.Core
{
    /// <summary>
    /// Węzeł grafu obliczeniowego. 
    /// Przechowuje fizyczną pamięć (FastTensor) dla danych i gradientów.
    /// Udostępnia bezalokacyjne widoki (TensorView) dla logiki matematycznej.
    /// </summary>
    public sealed class AutogradNode : IDisposable
    {
        // 1. PAMIĘĆ FIZYCZNA NA STERCIE (Magazyny)
        private FastTensor<float>? _dataTensor;
        private FastTensor<float>? _gradTensor;
        private int _disposed;

        public bool RequiresGrad { get; }

        // ========================================================================
        // 2. BEZALOKACYJNE WIDOKI (Bramy do matematyki)
        // ========================================================================

        /// <summary> Zwraca widok na dane z Forward Pass. Używane przez TensorMath. </summary>
        public TensorView<float> DataView
        {
            get
            {
                ObjectDisposedException.ThrowIf(_disposed == 1, this);

                return _dataTensor!.GetView();
            }
        }

        /// <summary> Zwraca widok na gradienty z Backward Pass. Używane przez TensorMath. </summary>
        public TensorView<float> GradView
        {
            get
            {
                ObjectDisposedException.ThrowIf(_disposed == 1, this);
                if (!RequiresGrad || _gradTensor == null)
                {
                    throw new InvalidOperationException("Ten węzeł nie śledzi gradientów (RequiresGrad = false).");
                }

                return _gradTensor.GetView();
            }
        }

        // ========================================================================
        // KONSTRUKTOR
        // ========================================================================

        public AutogradNode(FastTensor<float> data, bool requiresGrad = false)
        {
            _dataTensor = data ?? throw new ArgumentNullException(nameof(data));
            RequiresGrad = requiresGrad;

            if (requiresGrad)
            {
                // Alokujemy magazyn na gradienty od razu przyłączając go do węzła.
                // Używamy clearMemory: true, bo gradienty muszą startować od 0.
                _gradTensor = FastTensor<float>.SameShape(data, clearMemory: true);
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
            if (RequiresGrad && _gradTensor != null)
            {
                _gradTensor.GetView().AsSpan().Clear();
            }
        }

        // ========================================================================
        // SPRZĄTANIE (KRYTYCZNE)
        // ========================================================================

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 0)
            {
                // Oddajemy oba magazyny do ArrayPool
                _dataTensor?.Dispose();
                _dataTensor = null;

                _gradTensor?.Dispose();
                _gradTensor = null;
            }
        }
    }
}