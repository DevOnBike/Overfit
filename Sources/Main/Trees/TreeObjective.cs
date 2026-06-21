// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Trees
{
    /// <summary>
    /// Output transform applied to the summed raw tree margins of a gradient-boosted ensemble,
    /// mirroring the XGBoost objective that produced the model.
    /// </summary>
    public enum TreeObjective
    {
        /// <summary>Raw margin, no transform — regression (<c>reg:squarederror</c>, <c>reg:linear</c>).</summary>
        Identity,

        /// <summary>Sigmoid over a single margin — binary classification (<c>binary:logistic</c>).</summary>
        Logistic,

        /// <summary>Softmax over per-class margins — multiclass (<c>multi:softprob</c>, <c>multi:softmax</c>).</summary>
        Softmax
    }
}
