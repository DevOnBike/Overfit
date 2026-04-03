namespace DevOnBike.Overfit.Data.Prepare
{
    public class NaNImputerLayer : IDataLayer
    {
        public PipelineContext Process(PipelineContext context)
        {
            var span = context.Features.AsSpan();
            for (var i = 0; i < span.Length; i++)
            {
                if (float.IsNaN(span[i]) || float.IsInfinity(span[i]))
                    span[i] = 0f; // In-place: zero alokacji
            }
            return context;
        }
    }
}