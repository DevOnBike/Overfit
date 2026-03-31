using DevOnBike.Overfit.Core;
namespace DevOnBike.Overfit
{
    public interface ILossFunction
    {
        AutogradNode Compute(AutogradNode predictions, AutogradNode targets);
    }
}