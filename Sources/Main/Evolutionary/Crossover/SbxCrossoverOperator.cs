using DevOnBike.Overfit.Evolutionary.Abstractions;

namespace DevOnBike.Overfit.Evolutionary.Crossover
{
    /// <summary>
    ///     Simulated Binary Crossover (SBX) over real-valued genomes, as introduced by
    ///     Deb &amp; Agrawal (1995) and used as the default recombination operator in NSGA-II.
    ///     Given two parent genomes, produces two child genomes whose arithmetic mean
    ///     equals the mean of the parents; the spread around that mean is controlled by
    ///     the distribution index <see cref="DistributionIndex"/> (η).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         For each gene position, with probability <see cref="PerGeneProbability"/>
    ///         SBX produces two offspring values symmetric about the parent midpoint:
    ///         <list type="bullet">
    ///             <item>child₁ = ½·((1 + β)·p₁ + (1 − β)·p₂)</item>
    ///             <item>child₂ = ½·((1 − β)·p₁ + (1 + β)·p₂)</item>
    ///         </list>
    ///         where β is drawn from a polynomial distribution parameterised by η. Small η
    ///         (e.g. 2) makes β spread widely, producing children far from the parents
    ///         (exploration). Large η (e.g. 20+) keeps β close to 1, producing children
    ///         close to the parents (exploitation). The NSGA-II default is η = 15, which is
    ///         also the default here.
    ///     </para>
    ///     <para>
    ///         Positions not selected for crossover (probability 1 − p) are copied through
    ///         from parent₁ to child₁ and parent₂ to child₂ unchanged.
    ///     </para>
    ///     <para>
    ///         Zero-allocation. No trigonometry or transcendentals except one
    ///         <see cref="MathF.Pow"/> per mutated gene, which is unavoidable in SBX's
    ///         inverse-CDF sampling.
    ///     </para>
    /// </remarks>
    public sealed class SbxCrossoverOperator : ICrossoverOperator
    {
        private readonly float _distributionIndex;
        private readonly float _perGeneProbability;
        private readonly float _inverseExponent;

        /// <summary>
        ///     Creates an SBX operator.
        /// </summary>
        /// <param name="distributionIndex">Non-negative η controlling spread. Smaller values
        ///     explore farther from parents; larger values exploit near parents. NSGA-II
        ///     uses 15. Must be ≥ 0.</param>
        /// <param name="perGeneProbability">Per-gene crossover probability. Genes not selected
        ///     are copied through unchanged. Must lie in [0, 1]. The NSGA-II default is 1.0
        ///     (every gene crosses over on every call).</param>
        public SbxCrossoverOperator(float distributionIndex = 15f, float perGeneProbability = 1f)
        {
            if (distributionIndex < 0f)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(distributionIndex),
                    "Distribution index must be non-negative.");
            }

            if (perGeneProbability is < 0f or > 1f)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(perGeneProbability),
                    "Per-gene probability must lie in [0, 1].");
            }

            _distributionIndex = distributionIndex;
            _perGeneProbability = perGeneProbability;

            // Pre-compute the inverse exponent 1 / (η + 1). This appears in both branches of
            // the β formula and is fixed for the lifetime of the operator.
            _inverseExponent = 1f / (distributionIndex + 1f);
        }

        public float DistributionIndex => _distributionIndex;

        public float PerGeneProbability => _perGeneProbability;

        public void Crossover(
            ReadOnlySpan<float> parent1,
            ReadOnlySpan<float> parent2,
            Span<float> child1,
            Span<float> child2,
            Random rng)
        {
            if (parent1.Length != parent2.Length ||
                parent1.Length != child1.Length ||
                parent1.Length != child2.Length)
            {
                throw new ArgumentException("All four genome spans must have the same length.");
            }

            var probability = _perGeneProbability;
            var inverseExponent = _inverseExponent;

            for (var i = 0; i < parent1.Length; i++)
            {
                var p1 = parent1[i];
                var p2 = parent2[i];

                // Per-gene branching: either SBX or straight copy.
                if (rng.NextSingle() >= probability)
                {
                    child1[i] = p1;
                    child2[i] = p2;
                    continue;
                }

                // Polynomial-distributed β. The two branches are structured so that β is
                // symmetric around 1 on a log scale: u=0.5 gives β=1 (child = midpoint),
                // u→0 gives β→0 (child = other parent), u→1 gives β→∞ (child = extrapolation).
                var u = rng.NextSingle();

                float beta;

                if (u <= 0.5f)
                {
                    beta = MathF.Pow(2f * u, inverseExponent);
                }
                else
                {
                    beta = MathF.Pow(1f / (2f * (1f - u)), inverseExponent);
                }

                // Symmetric recombination. child1 + child2 = p1 + p2 by construction, so the
                // arithmetic mean is preserved — a defining property of SBX.
                var half = 0.5f;
                child1[i] = half * (((1f + beta) * p1) + ((1f - beta) * p2));
                child2[i] = half * (((1f - beta) * p1) + ((1f + beta) * p2));
            }
        }
    }
}