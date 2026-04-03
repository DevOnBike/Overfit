using System.Buffers;
namespace DevOnBike.Overfit.Data.Prepare;

public class RobustScalingLayer : IDataLayer
{
    private readonly List<int> _numericColumnIndices;

    // Przekazujemy indeksy kolumn, które faktycznie wymagają skalowania (np. 0, 1, 4)
    public RobustScalingLayer(List<int> numericColumnIndices)
    {
        _numericColumnIndices = numericColumnIndices;
    }

    public PipelineContext Process(PipelineContext context)
    {
        var rows = context.Features.GetDim(0);
        var cols = context.Features.GetDim(1);
        var span = context.Features.AsSpan();

        foreach (var c in _numericColumnIndices)
        {
            var buffer = ArrayPool<float>.Shared.Rent(rows);
            try
            {
                for (var r = 0; r < rows; r++) buffer[r] = span[r * cols + c];

                Array.Sort(buffer, 0, rows);
                var median = buffer[rows / 2];
                var iqr = buffer[rows * 3 / 4] - buffer[rows / 4];
                if (iqr == 0) iqr = 1f;

                for (var r = 0; r < rows; r++)
                {
                    span[r * cols + c] = (span[r * cols + c] - median) / iqr;
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(buffer);
            }
        }
        return context;
    }
}