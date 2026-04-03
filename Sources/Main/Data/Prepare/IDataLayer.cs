namespace DevOnBike.Overfit.Data.Prepare
{
    public interface IDataLayer
    {
        PipelineContext Process(PipelineContext context);
    }
}