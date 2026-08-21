// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// The outcome of comparing one arm's output against the CPU F32 reference. A fast wrong kernel is
    /// the failure mode this probe is most exposed to, so a failed check withholds the timing rather
    /// than sitting beside it (plan, section 3.6).
    /// </summary>
    internal sealed class ParityResult
    {
        private ParityResult(bool passed, double cosine, double maxRelative, double refMax, string detail)
        {
            Passed = passed;
            Cosine = cosine;
            MaxRelative = maxRelative;
            ReferenceMax = refMax;
            Detail = detail;
        }

        /// <summary>Cosine must reach this. Below it the kernel is computing something else.</summary>
        public const double CosineFloor = 0.9999;

        /// <summary>
        /// Maximum relative error, on elements whose REFERENCE magnitude exceeds 1. Bit-parity is not
        /// demanded: a serial dot product and a parallel tree reduction accumulate in different orders,
        /// so bit-parity would fail a correct kernel.
        /// </summary>
        public const double MaxRelativeCeiling = 1e-3;

        public bool Passed { get; }

        public double Cosine { get; }

        public double MaxRelative { get; }

        public double ReferenceMax { get; }

        public string Detail { get; }

        public static ParityResult NotRun(string reason) => new(false, 0, 0, 0, reason);

        public static ParityResult Compare(ReadOnlySpan<float> reference, ReadOnlySpan<float> candidate)
        {
            if (reference.Length != candidate.Length)
            {
                return NotRun($"length mismatch: reference {reference.Length}, candidate {candidate.Length}");
            }

            double dot = 0, na = 0, nb = 0, maxRel = 0, refMax = 0;
            var nonFinite = 0;

            for (var i = 0; i < reference.Length; i++)
            {
                double a = reference[i], b = candidate[i];
                if (!double.IsFinite(b))
                {
                    nonFinite++;
                    continue;
                }

                dot += a * b;
                na += a * a;
                nb += b * b;
                refMax = Math.Max(refMax, Math.Abs(a));

                if (Math.Abs(a) > 1.0)
                {
                    maxRel = Math.Max(maxRel, Math.Abs(a - b) / Math.Abs(a));
                }
            }

            var cosine = na > 0 && nb > 0 ? dot / (Math.Sqrt(na) * Math.Sqrt(nb)) : 0;
            var passed = nonFinite == 0 && cosine >= CosineFloor && maxRel < MaxRelativeCeiling;

            var detail = nonFinite > 0
                ? $"{nonFinite} non-finite output element(s)"
                : passed ? "ok" : "outside tolerance";

            return new ParityResult(passed, cosine, maxRel, refMax, detail);
        }
    }
}
