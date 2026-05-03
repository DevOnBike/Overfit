namespace DevOnBike.Overfit.LanguageModels.Quantization
{

    public readonly struct QuantizationOptions
    {
        public QuantizationOptions(
            QuantizationKind kind,
            QuantizationScaleKind scaleKind,
            int groupSize = 0,
            bool symmetric = true)
        {
            Kind = kind;
            ScaleKind = scaleKind;
            GroupSize = groupSize;
            Symmetric = symmetric;
        }

        public QuantizationKind Kind { get; }

        public QuantizationScaleKind ScaleKind { get; }

        public int GroupSize { get; }

        public bool Symmetric { get; }
    }
}
