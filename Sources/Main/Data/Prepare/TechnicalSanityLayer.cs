namespace DevOnBike.Overfit.Data.Prepare
{
    public class TechnicalSanityLayer : IDataLayer
    {
        public PipelineContext Process(PipelineContext context)
        {
            // Operujemy in-place na Spanach, co jest ekstremalnie szybkie i nie generuje śmieci (GC)
            CleanSpan(context.Features.AsSpan());
            CleanSpan(context.Targets.AsSpan());

            return context;
        }

        private void CleanSpan(Span<float> span)
        {
            for (var i = 0; i < span.Length; i++)
            {
                // Zastępujemy NaN i nieskończoności zerem (imputacja stałą)
                if (float.IsNaN(span[i]) || float.IsInfinity(span[i]))
                {
                    span[i] = 0f;
                }
            }
        }
    }
}
