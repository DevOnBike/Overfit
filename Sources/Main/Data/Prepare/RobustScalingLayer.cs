using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.Data.Contracts;

namespace DevOnBike.Overfit.Data.Prepare
{
    /// <summary>
    /// Skalowanie odporne na outliery — normalizacja przez medianę i rozstęp międzykwartylowy (IQR).
    /// Formuła: x' = (x - median) / IQR
    ///
    /// Przewaga nad StandardScaler (mean/std):
    /// - Mediana jest odporna na pojedynczy penthouse za 15M w datasecie z medianą 500k
    /// - IQR ignoruje ekstrema — std jest wrażliwe na kwadrat odchylenia
    ///
    /// Dla danych nieruchomościowych to krytyczne: rozkład cen jest mocno skośny,
    /// a StandardScaler "ściąga" wszystkie normalne mieszkania do wąskiego zakresu
    /// i daje outlierom nieproporcjonalny wpływ na gradienty.
    ///
    /// Warstwa powinna być stosowana PO winsoryzacji (OutlierClipLayer)
    /// i PO transformacji logarytmicznej (LogTransformLayer), ale PRZED selekcją cech.
    /// </summary>
    public sealed class RobustScalingLayer : IDataLayer
    {
        private readonly HashSet<int> _columnIndices;
        private readonly HashSet<int> _excludedColumns;
        private readonly float _fallbackIqr;
        private readonly bool _centerByMedian;

        // Wyliczone statystyki z Fit — reużywane w Transform (inference)
        private float[] _medians;
        private float[] _iqrs;
        private bool _fitted;

        /// <param name="columnIndices">
        /// Indeksy kolumn do skalowania. Null = skaluj wszystkie (poza wykluczonymi).
        /// </param>
        /// <param name="excludedColumns">
        /// Kolumny wykluczone ze skalowania (np. binarne, one-hot encoded).
        /// Skalowanie kolumny 0/1 przez IQR da nieokreślone wyniki (IQR = 0).
        /// </param>
        /// <param name="fallbackIqr">
        /// Wartość IQR zastępcza gdy prawdziwy IQR = 0 (kolumna near-constant).
        /// Domyślnie 1.0 — centruje przez medianę bez skalowania.
        /// </param>
        /// <param name="centerByMedian">
        /// Czy odejmować medianę (centrowanie). False = tylko dzielenie przez IQR.
        /// Przydatne gdy dane zostały już wycentrowane wcześniej w potoku.
        /// </param>
        public RobustScalingLayer(
            HashSet<int> columnIndices = null,
            HashSet<int> excludedColumns = null,
            float fallbackIqr = 1f,
            bool centerByMedian = true)
        {
            if (fallbackIqr <= 0f)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(fallbackIqr), "Zastępczy IQR musi być dodatni.");
            }

            _columnIndices = columnIndices;
            _excludedColumns = excludedColumns ?? new HashSet<int>();
            _fallbackIqr = fallbackIqr;
            _centerByMedian = centerByMedian;
        }

        public PipelineContext Process(PipelineContext context)
        {
            var rows = context.Features.GetDim(0);
            var cols = context.Features.GetDim(1);

            if (rows == 0 || cols == 0)
            {
                return context;
            }

            var span = context.Features.AsSpan();

            // Fit: wyliczamy statystyki — wymaga >= 2 wierszy
            if (!_fitted)
            {
                if (rows < 2)
                {
                    // Za mało danych do wyliczenia mediany/IQR — nie możemy skalować
                    return context;
                }

                Fit(span, rows, cols);
            }

            // Transform: zawsze stosujemy zapamiętane statystyki, nawet na 1 wierszu
            Transform(span, rows, cols);

            return context;
        }

        /// <summary>
        /// Wylicza medianę i IQR per kolumna.
        /// Statystyki są zapamiętywane — przy ponownym wywołaniu (inference)
        /// stosujemy te same progi co na danych treningowych.
        /// </summary>
        private void Fit(ReadOnlySpan<float> span, int rows, int cols)
        {
            _medians = new float[cols];
            _iqrs = new float[cols];

            // Domyślne wartości: median=0, iqr=1 (tożsamość)
            Array.Fill(_iqrs, _fallbackIqr);

            using var sortBuffer = new FastBuffer<float>(rows);
            var bufferSpan = sortBuffer.AsSpan();

            for (var c = 0; c < cols; c++)
            {
                if (!ShouldScaleColumn(c, cols))
                {
                    continue;
                }

                // Kopiujemy kolumnę do bufora
                for (var r = 0; r < rows; r++)
                {
                    bufferSpan[r] = span[r * cols + c];
                }

                bufferSpan.Sort();

                _medians[c] = InterpolatePercentile(bufferSpan, rows, 0.5f);

                var q1 = InterpolatePercentile(bufferSpan, rows, 0.25f);
                var q3 = InterpolatePercentile(bufferSpan, rows, 0.75f);
                var iqr = q3 - q1;

                _iqrs[c] = iqr > 0f ? iqr : _fallbackIqr;
            }

            _fitted = true;
        }

        /// <summary>
        /// Stosuje zapamiętane statystyki do danych in-place.
        /// </summary>
        private void Transform(Span<float> span, int rows, int cols)
        {
            for (var c = 0; c < cols; c++)
            {
                if (!ShouldScaleColumn(c, cols))
                {
                    continue;
                }

                var median = _medians[c];
                var iqr = _iqrs[c];

                // Prekomputujemy 1/IQR — mnożenie jest szybsze niż dzielenie w pętli
                var invIqr = 1f / iqr;

                if (_centerByMedian)
                {
                    for (var r = 0; r < rows; r++)
                    {
                        ref var val = ref span[r * cols + c];
                        val = (val - median) * invIqr;
                    }
                }
                else
                {
                    for (var r = 0; r < rows; r++)
                    {
                        ref var val = ref span[r * cols + c];
                        val *= invIqr;
                    }
                }
            }
        }

        private bool ShouldScaleColumn(int colIndex, int totalCols)
        {
            if (_excludedColumns.Contains(colIndex))
            {
                return false;
            }

            // Null = skaluj wszystkie (poza wykluczonymi)
            if (_columnIndices == null)
            {
                return true;
            }

            return _columnIndices.Contains(colIndex);
        }

        /// <summary>
        /// Interpolacja liniowa percentyla — zgodna z numpy.percentile(interpolation='linear').
        /// rank = p * (N - 1), wynik = lerp między dwoma sąsiednimi wartościami.
        /// </summary>
        private static float InterpolatePercentile(ReadOnlySpan<float> sorted, int count, float percentile)
        {
            var rank = percentile * (count - 1);
            var lowerIdx = (int)rank;
            var upperIdx = lowerIdx + 1;

            if (upperIdx >= count)
            {
                return sorted[lowerIdx];
            }

            var fraction = rank - lowerIdx;

            return sorted[lowerIdx] * (1f - fraction) + sorted[upperIdx] * fraction;
        }

        /// <summary>
        /// Resetuje zapamiętane statystyki.
        /// Wymusza ponowne Fit przy następnym wywołaniu Process.
        /// Przydatne przy retreningu na nowych danych.
        /// </summary>
        public void Reset()
        {
            _medians = null;
            _iqrs = null;
            _fitted = false;
        }
    }
}