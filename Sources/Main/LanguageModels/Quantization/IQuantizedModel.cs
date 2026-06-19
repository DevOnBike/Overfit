// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.LanguageModels.Quantization
{
    public interface IQuantizedModel : ISlmModel
    {
        QuantizationOptions Quantization
        {
            get;
        }

        long QuantizedParameterBytes
        {
            get;
        }

        long OriginalParameterBytes
        {
            get;
        }

        double CompressionRatio
        {
            get;
        }
    }
}
