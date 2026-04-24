namespace DevOnBike.Overfit.Evolutionary.Storage
{
    public enum EliteInsertStatus
    {
        Rejected = 0,
        InsertedNewCell = 1,
        ReplacedExistingCell = 2,
        OutOfBounds = 3
    }
}