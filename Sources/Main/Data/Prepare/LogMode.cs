using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Data.Prepare
{

    public enum LogMode
    {
        /// <summary>log(1 + x) — bezpieczna, dla danych >= 0</summary>
        Log1p,

        /// <summary>sign(x) * log(1 + |x|) — zachowuje znak, dla danych z ujemnymi wartościami</summary>
        SignedLog1p,

        /// <summary>log(x + epsilon) — klasyczna, dla danych ściśle dodatnich</summary>
        LogEps
    }
}