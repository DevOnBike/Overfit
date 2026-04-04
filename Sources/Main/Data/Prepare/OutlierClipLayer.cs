using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.Data.Contracts;
using DevOnBike.Overfit.Data.Prepare;

public sealed class OutlierClipLayer : IDataLayer
{
    private readonly float _lowerPercentile;
    private readonly float _upperPercentile;
    private readonly Dictionary<int, (float Lower, float Upper)> _columnOverrides;
    private readonly HashSet<int> _excludedColumns;

    // Zapamiętane progi z Fit
    private float[] _lowThresholds;
    private float[] _highThresholds;
    private bool _fitted;

    public OutlierClipLayer(
        float lowerPercentile = 0.01f,
        float upperPercentile = 0.99f,
        Dictionary<int, (float Lower, float Upper)> columnOverrides = null,
        HashSet<int> excludedColumns = null)
    {
        if (lowerPercentile < 0f || lowerPercentile >= upperPercentile)
        {
            throw new ArgumentOutOfRangeException(
                nameof(lowerPercentile),
                "Dolny percentyl musi być >= 0 i mniejszy od górnego.");
        }

        if (upperPercentile > 1f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(upperPercentile),
                "Górny percentyl musi być <= 1.");
        }

        _lowerPercentile = lowerPercentile;
        _upperPercentile = upperPercentile;
        _columnOverrides = columnOverrides ?? new Dictionary<int, (float, float)>();
        _excludedColumns = excludedColumns ?? new HashSet<int>();

        foreach (var (col, (lower, upper)) in _columnOverrides)
        {
            if (lower < 0f || lower >= upper || upper > 1f)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(columnOverrides),
                    $"Nieprawidłowy zakres percentyli ({lower}, {upper}) dla kolumny {col}.");
            }
        }
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

        // Fit: wyliczamy progi — wymaga >= 2 wierszy
        if (!_fitted)
        {
            if (rows < 2)
            {
                return context;
            }

            Fit(span, rows, cols);
        }

        // Transform: przycinamy na zapamiętanych progach
        ClipAll(span, rows, cols);

        return context;
    }

    private void Fit(ReadOnlySpan<float> span, int rows, int cols)
    {
        _lowThresholds = new float[cols];
        _highThresholds = new float[cols];

        using var sortBuffer = new FastBuffer<float>(rows);
        var bufferSpan = sortBuffer.AsSpan();

        for (var c = 0; c < cols; c++)
        {
            if (_excludedColumns.Contains(c))
            {
                _lowThresholds[c] = float.MinValue;
                _highThresholds[c] = float.MaxValue;
                continue;
            }

            var (lowerPct, upperPct) = ResolvePercentiles(c);

            for (var r = 0; r < rows; r++)
            {
                bufferSpan[r] = span[r * cols + c];
            }

            bufferSpan.Sort();

            var lowVal = InterpolatePercentile(bufferSpan, rows, lowerPct);
            var highVal = InterpolatePercentile(bufferSpan, rows, upperPct);

            if (lowVal >= highVal)
            {
                _lowThresholds[c] = float.MinValue;
                _highThresholds[c] = float.MaxValue;
            }
            else
            {
                _lowThresholds[c] = lowVal;
                _highThresholds[c] = highVal;
            }
        }

        _fitted = true;
    }

    private void ClipAll(Span<float> span, int rows, int cols)
    {
        for (var c = 0; c < cols; c++)
        {
            var lowVal = _lowThresholds[c];
            var highVal = _highThresholds[c];

            // MinValue/MaxValue = kolumna wykluczona lub stała → skip
            if (lowVal == float.MinValue && highVal == float.MaxValue)
            {
                continue;
            }

            for (var r = 0; r < rows; r++)
            {
                ref var val = ref span[r * cols + c];

                if (val < lowVal)
                {
                    val = lowVal;
                }
                else if (val > highVal)
                {
                    val = highVal;
                }
            }
        }
    }

    private (float Lower, float Upper) ResolvePercentiles(int columnIndex)
    {
        if (_columnOverrides.TryGetValue(columnIndex, out var overrides))
        {
            return overrides;
        }

        return (_lowerPercentile, _upperPercentile);
    }

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

    public void Reset()
    {
        _lowThresholds = null;
        _highThresholds = null;
        _fitted = false;
    }
}