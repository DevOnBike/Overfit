// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.DeepLearning.Abstractions;

namespace DevOnBike.Overfit.Tests.DeepLearning
{
    /// <summary>
    /// The walk <see cref="CheckpointedModule"/> runs over its segment looking for a layer that draws
    /// randomness — <c>NR-4</c>, and the tests exist because nothing covered it at all.
    ///
    /// <para><b>What it used to be.</b> Plain recursion under a <c>#pragma warning disable OVERFIT022</c>
    /// whose stated bound was "Sequential nesting … a handful of levels at most" — a statement about how the
    /// author expected the API to be used, not a proof. <c>Sequential.Add</c> null-checks its argument and
    /// nothing else, so <c>s.Add(s)</c> builds fine and takes the host process down with a
    /// <c>StackOverflowException</c> .NET cannot catch.</para>
    ///
    /// <para><b><see cref="AModuleReusedAtTwoPositionsIsNotACycle"/> is the important one</b>, and it is here
    /// because the first version of the fix would have failed it: one global visited set throwing on any
    /// repeat rejects a model that reuses one stateless layer instance twice, which is legal and common. That
    /// version passed every other test in this file.</para>
    ///
    /// <para><b>A regression in <see cref="ADeeplyNestedSegmentDoesNotExhaustTheStack"/> does not fail — it
    /// kills the test host.</b> That is not a flaw in the test; it is the failure mode the fix exists to
    /// remove, and there is no way to observe it politely. 512 levels is chosen to clear the old recursion's
    /// frame comfortably while staying instant.</para>
    /// </summary>
    public sealed class CheckpointedModuleSegmentWalkTests
    {
        private static CheckpointedModule Wrap(IModule inner)
        {
            return new CheckpointedModule(inner, subArenaElements: 16);
        }

        [Fact]
        public void ASegmentWithoutRandomnessIsAccepted()
        {
            using var inner = new Sequential(new LinearLayer(2, 2), new LinearLayer(2, 2));
            using var checkpointed = Wrap(inner);

            Assert.NotNull(checkpointed);
        }

        [Fact]
        public void ADropoutNestedInsideTheSegmentIsRefused()
        {
            using var inner = new Sequential(
                new LinearLayer(2, 2),
                new Sequential(new LinearLayer(2, 2), new DropoutLayer(0.5f)));

            var error = Assert.Throws<ArgumentException>(() => Wrap(inner));

            Assert.Contains(nameof(DropoutLayer), error.Message, StringComparison.Ordinal);
        }

        /// <summary>
        /// Depth-first in model order, so the message names the layer a caller would move first. Pinned
        /// because the iterative rewrite has to push children in an order that preserves it, and getting that
        /// backwards is invisible unless two candidates exist.
        /// </summary>
        [Fact]
        public void TheFirstOffenderInModelOrderIsTheOneNamed()
        {
            using var inner = new Sequential(
                new Sequential(new LinearLayer(2, 2), new Dropout2DLayer(0.5f)),
                new DropoutLayer(0.5f));

            var error = Assert.Throws<ArgumentException>(() => Wrap(inner));

            Assert.Contains(nameof(Dropout2DLayer), error.Message, StringComparison.Ordinal);
        }

        /// <summary>
        /// The defect NR-4 names: legal to build, and it used to end the process rather than the call.
        /// </summary>
        [Fact]
        public void ASegmentContainingItselfIsReportedRatherThanKillingTheProcess()
        {
            var cyclic = new Sequential(new LinearLayer(2, 2));
            cyclic.Add(cyclic);

            var error = Assert.Throws<ArgumentException>(() => Wrap(cyclic));

            Assert.Contains("cycle", error.Message, StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>Two containers holding each other — a cycle no single-instance check would see.</summary>
        [Fact]
        public void TwoSegmentsContainingEachOtherAreReported()
        {
            var outer = new Sequential(new LinearLayer(2, 2));
            var inner = new Sequential(new LinearLayer(2, 2));
            outer.Add(inner);
            inner.Add(outer);

            var error = Assert.Throws<ArgumentException>(() => Wrap(outer));

            Assert.Contains("cycle", error.Message, StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>
        /// <b>The false positive the fix nearly shipped.</b> One stateless layer instance at two positions is
        /// a legal model, not a cycle — it is only a cycle if a container is its own ancestor.
        /// </summary>
        [Fact]
        public void AModuleReusedAtTwoPositionsIsNotACycle()
        {
            var shared = new LinearLayer(2, 2);
            using var inner = new Sequential(shared, new LinearLayer(2, 2), shared);
            using var checkpointed = Wrap(inner);

            Assert.NotNull(checkpointed);
        }

        /// <summary>The same, one level up: a shared sub-model, which the walk must not mistake for a loop.</summary>
        [Fact]
        public void ASharedSubSegmentAtTwoPositionsIsNotACycle()
        {
            var shared = new Sequential(new LinearLayer(2, 2));
            using var inner = new Sequential(shared, new Sequential(shared));
            using var checkpointed = Wrap(inner);

            Assert.NotNull(checkpointed);
        }

        [Fact]
        public void ADeeplyNestedSegmentDoesNotExhaustTheStack()
        {
            var deepest = new Sequential(new LinearLayer(2, 2));
            var current = deepest;

            for (var i = 0; i < 512; i++)
            {
                current = new Sequential(current);
            }

            using var checkpointed = Wrap(current);

            Assert.NotNull(checkpointed);
        }

        /// <summary>Depth must not blind the walk: the dropout is 512 levels down and still has to be found.</summary>
        [Fact]
        public void ADropoutAtTheBottomOfADeepSegmentIsStillFound()
        {
            IModule current = new Sequential(new LinearLayer(2, 2), new DropoutLayer(0.5f));

            for (var i = 0; i < 512; i++)
            {
                current = new Sequential(current);
            }

            var error = Assert.Throws<ArgumentException>(() => Wrap(current));

            Assert.Contains(nameof(DropoutLayer), error.Message, StringComparison.Ordinal);
        }

        /// <summary>
        /// The escape hatch still works, and it is checked before the walk's own findings are believed —
        /// a test that only ever exercises the refusing path cannot tell a strict guard from a stuck one.
        /// </summary>
        [Fact]
        public void RandomnessIsAcceptedWhenTheCallerOptsIn()
        {
            using var inner = new Sequential(new LinearLayer(2, 2), new DropoutLayer(0.5f));
            using var checkpointed = new CheckpointedModule(
                inner, subArenaElements: 16, allowNonDeterministic: true);

            Assert.NotNull(checkpointed);
        }
    }
}
