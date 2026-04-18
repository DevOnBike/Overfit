namespace DevOnBike.Overfit.Evolutionary.Abstractions
{
    public interface IEvolutionCheckpoint
    {
        void Save(BinaryWriter writer);
        void Load(BinaryReader reader);
    }
}