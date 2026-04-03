namespace DevOnBike.Overfit.Data.Prepare
{
    public class AnomalyFilterLayer : IDataLayer
    {
        public PipelineContext Process(PipelineContext context)
        {
            return context;

            /*
            var validIndices = IdentifyValidRows(context); // Np. MAD Score

            // Ekstrakcja tylko dobrych danych do nowych struktur FastTensor
            var newFeatures = ExtractRows(context.Features, validIndices);
            var newTargets = ExtractRows(context.Targets, validIndices);

            // Zwalniamy stare dane, by nie zapychać RAMu
            context.Features.Dispose();
            context.Targets.Dispose();

            return new PipelineContext(newFeatures, newTargets);
            */
        }
    }
}