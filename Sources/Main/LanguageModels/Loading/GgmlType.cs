namespace DevOnBike.Overfit.LanguageModels.Loading
{
    /// <summary>
    /// GGML tensor element types. Etap A supports F32 and F16 only.
    /// Quantized types (Q4_K_M, Q8_0 etc.) will be added in Etap B.
    /// </summary>
    public enum GgmlType : uint
    {
        F32  = 0,
        F16  = 1,
        Q4_0 = 2,
        Q4_1 = 3,
        Q5_0 = 6,
        Q5_1 = 7,
        Q8_0 = 8,
        Q8_1 = 9,
        Q2_K = 10,
        Q3_K = 11,
        Q4_K = 12,
        Q5_K = 13,
        Q6_K = 14,
        Q8_K = 15,
        BF16 = 30,
    }
}