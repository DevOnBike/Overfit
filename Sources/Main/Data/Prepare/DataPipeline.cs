using DevOnBike.Overfit.Core;
namespace DevOnBike.Overfit.Data.Prepare
{
    public class DataPipeline
    {
        private readonly List<IDataLayer> _layers = [];

        public DataPipeline AddLayer(IDataLayer layer)
        {
            _layers.Add(layer);
            return this;
        }

        public PipelineContext Execute(FastTensor<float> features, FastTensor<float> targets)
        {
            var current = new PipelineContext(features, targets);

            foreach (var layer in _layers)
            {
                current = layer.Process(current);
            }

            return current;
        }
    }
}