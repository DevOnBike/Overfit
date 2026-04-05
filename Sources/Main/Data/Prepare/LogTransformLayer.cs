// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Data.Contracts;

namespace DevOnBike.Overfit.Data.Prepare
{
    /// <summary>
    /// Transformacja logarytmiczna dla kolumn z mocno skośnym rozkładem.
    /// Typowe zastosowania w danych nieruchomościowych:
    /// - Cena transakcyjna (długi prawy ogon: kawalerki vs penthouse'y)
    /// - Powierzchnia użytkowa (30m² vs 300m²)  
    /// - Odległość od centrum (wykładniczy spadek cen)
    /// - Czynsz administracyjny
    ///
    /// Stabilizuje wariancję, redukuje wpływ outlierów i przybliża rozkład do normalnego,
    /// co poprawia konwergencję gradientową w treningu.
    ///
    /// Obsługuje trzy warianty transformacji:
    /// - Log1p: log(1 + x) — bezpieczna dla wartości bliskich zeru i zerowych
    /// - SignedLog1p: sign(x) * log(1 + |x|) — zachowuje znak dla danych z ujemnymi wartościami
    /// - LogEps: log(x + epsilon) — klasyczna, dla danych ściśle dodatnich
    /// </summary>
    public sealed class LogTransformLayer : IDataLayer
    {
        private readonly List<int> _columnIndices;
        private readonly LogMode _mode;
        private readonly float _epsilon;

        public LogTransformLayer(
            List<int> columnIndices,
            LogMode mode = LogMode.Log1p,
            float epsilon = 1e-7f)
        {
            if (columnIndices == null || columnIndices.Count == 0)
            {
                throw new ArgumentException(
                "Lista kolumn do transformacji nie może być pusta.", nameof(columnIndices));
            }

            if (epsilon <= 0f)
            {
                throw new ArgumentOutOfRangeException(
                nameof(epsilon), "Epsilon musi być dodatni.");
            }

            _columnIndices = columnIndices;
            _mode = mode;
            _epsilon = epsilon;
        }

        public PipelineContext Process(PipelineContext context)
        {
            var rows = context.Features.GetDim(0);
            var cols = context.Features.GetDim(1);

            if (rows == 0 || cols == 0)
            {
                return context;
            }

            // Walidacja indeksów kolumn przed transformacją
            foreach (var c in _columnIndices)
            {
                if (c < 0 || c >= cols)
                {
                    throw new InvalidOperationException(
                    $"Indeks kolumny {c} wykracza poza zakres tensora (0–{cols - 1}).");
                }
            }

            var span = context.Features.AsSpan();

            // Dispatch per tryb — unikamy brancha wewnątrz hot-loop
            switch (_mode)
            {
                case LogMode.Log1p:
                    ApplyLog1p(span, rows, cols);
                    break;

                case LogMode.SignedLog1p:
                    ApplySignedLog1p(span, rows, cols);
                    break;

                case LogMode.LogEps:
                    ApplyLogEps(span, rows, cols);
                    break;

                default:
                    throw new InvalidOperationException($"Nieobsługiwany tryb: {_mode}");
            }

            return context;
        }

        /// <summary>
        /// log(1 + x) — standardowy wybór dla danych nieujemnych.
        /// Bezpieczna dla x = 0 (log(1) = 0).
        /// Wymaga x >= 0; ujemne wartości sygnalizują błąd w danych
        /// (cena/powierzchnia nie powinny być ujemne).
        /// </summary>
        private void ApplyLog1p(Span<float> span, int rows, int cols)
        {
            foreach (var c in _columnIndices)
            {
                for (var r = 0; r < rows; r++)
                {
                    ref var val = ref span[r * cols + c];

                    if (val < 0f)
                    {
                        // Ujemna cena/powierzchnia = prawdopodobnie błąd w danych.
                        // Zamiast cicho zamiatać pod dywan, zerujemy — TechnicalSanityLayer
                        // powinien to wyłapać wcześniej, ale tu mamy dodatkową ochronę.
                        val = 0f;
                        continue;
                    }

                    // MathF.Log(1 + x) zamiast MathF.Log1P (niedostępny w .NET)
                    // Dla małych x (< 1e-4) tracimy precyzję, ale przy float32 to akceptowalne
                    val = MathF.Log(1f + val);
                }
            }
        }

        /// <summary>
        /// sign(x) * log(1 + |x|) — zachowuje znak.
        /// Dla danych, które mogą być ujemne, np.:
        /// - Zmiana ceny r/r (może być -15%)
        /// - Bilans cieplny budynku (zysk/strata)
        /// - Różnica ceny vs mediana dzielnicy
        /// </summary>
        private void ApplySignedLog1p(Span<float> span, int rows, int cols)
        {
            foreach (var c in _columnIndices)
            {
                for (var r = 0; r < rows; r++)
                {
                    ref var val = ref span[r * cols + c];

                    if (val == 0f)
                    {
                        continue;
                    }

                    // sign(x) * log(1 + |x|)
                    // Rozkłada symetryczny ogon na obie strony
                    val = MathF.Sign(val) * MathF.Log(1f + MathF.Abs(val));
                }
            }
        }

        /// <summary>
        /// log(x + epsilon) — klasyczna transformacja dla danych ściśle dodatnich.
        /// Epsilon chroni przed log(0) = -∞.
        /// Wybieraj ten tryb gdy dane nie zawierają zer i chcesz zachować
        /// pełną rozdzielczość logarytmu (log1p "spłaszcza" małe wartości).
        /// </summary>
        private void ApplyLogEps(Span<float> span, int rows, int cols)
        {
            foreach (var c in _columnIndices)
            {
                for (var r = 0; r < rows; r++)
                {
                    ref var val = ref span[r * cols + c];

                    if (val < 0f)
                    {
                        val = MathF.Log(_epsilon);
                        continue;
                    }

                    val = MathF.Log(val + _epsilon);
                }
            }
        }
    }
}