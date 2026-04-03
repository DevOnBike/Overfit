using DevOnBike.Overfit.Core;
namespace DevOnBike.Overfit.Data.Prepare
{
    public class CorrelationFilterLayer : IDataLayer
    {
        private readonly float _threshold;

        public CorrelationFilterLayer(float threshold = 0.98f) => _threshold = threshold;

        public PipelineContext Process(PipelineContext context)
        {
            var activeIndices = IdentifyUniqueFeatures(context.Features);

            if (activeIndices.Count == context.Features.GetDim(1))
                return context; // Nic nie usuwamy

            // Tworzymy NOWY tensor o mniejszej liczbie kolumn
            var newFeatures = ExtractColumns(context.Features, activeIndices);

            // KLUCZOWE: Zwalniamy stary tensor, bo stworzyliśmy nową kopię!
            context.Features.Dispose();

            return new PipelineContext(newFeatures, context.Targets);
        }

        private List<int> IdentifyUniqueFeatures(FastTensor<float> features)
        {
            var cols = features.GetDim(1);
            var keep = new List<int>();
            var remove = new HashSet<int>();

            for (var i = 0; i < cols; i++)
            {
                if (remove.Contains(i)) continue;
                keep.Add(i);

                for (var j = i + 1; j < cols; j++)
                {
                    // Wykorzystujemy Twoją matematykę wektorową do liczenia korelacji
                    var r = CalculateCorrelation(features, i, j);
                    if (MathF.Abs(r) > _threshold) remove.Add(j);
                }
            }
            return keep;
        }

        private float CalculateCorrelation(FastTensor<float> t, int colA, int colB)
        {
            // Tutaj implementacja Pearsona wykorzystująca TensorPrimitives.Dot
            // Twoja "Bestia" policzy to w mikrosekundy dzięki AVX-512.
            return 0.99f; // Placeholder dla logiki z poprzednich kroków
        }

        private FastTensor<float> ExtractColumns(FastTensor<float> src, List<int> indices)
        {
            var result = new FastTensor<float>(src.GetDim(0), indices.Count);
            // ... logika przepisywania kolumn ...
            return result;
        }
    }
}
