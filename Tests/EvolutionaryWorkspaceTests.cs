using DevOnBike.Overfit.Evolutionary.Storage;

namespace DevOnBike.Overfit.Tests
{
    public sealed class EvolutionaryWorkspaceTests
    {
        [Fact]
        public void Constructor_InitializesExpectedShapes()
        {
            using var workspace = new EvolutionWorkspace(populationSize: 4, genomeSize: 3);

            Assert.Equal(4, workspace.PopulationSize);
            Assert.Equal(3, workspace.GenomeSize);
            Assert.Equal(12, workspace.Population.GetView().AsReadOnlySpan().Length);
            Assert.Equal(12, workspace.NextPopulation.GetView().AsReadOnlySpan().Length);
            Assert.Equal(4, workspace.Fitness.GetView().AsReadOnlySpan().Length);
            Assert.Equal(4, workspace.ShapedFitness.GetView().AsReadOnlySpan().Length);
            Assert.Equal(4, workspace.Ranking.Length);
            Assert.Equal(4, workspace.EliteIndices.Length);
        }

        [Fact]
        public void SwapPopulations_CopiesNextPopulationIntoPopulation()
        {
            using var workspace = new EvolutionWorkspace(populationSize: 2, genomeSize: 3);

            var current = workspace.Population.GetView().AsSpan();
            var next = workspace.NextPopulation.GetView().AsSpan();

            current.Clear();
            next[0] = 1f;
            next[1] = 2f;
            next[2] = 3f;
            next[3] = 4f;
            next[4] = 5f;
            next[5] = 6f;

            workspace.SwapPopulations();

            Assert.Equal([1f, 2f, 3f, 4f, 5f, 6f], workspace.Population.GetView().AsReadOnlySpan().ToArray());
        }
    }
}
