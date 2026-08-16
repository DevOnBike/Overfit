// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    /// <summary>
    /// Pins the two properties that make <see cref="NonRepackedKernelScope"/> a scope rather than a switch.
    /// Both are model-free, deterministic and fast, and each exists because the shape it forbids shipped:
    /// the two private near-twins this type replaced reset the flag to <c>false</c> on dispose, and the flag
    /// itself was a plain <c>static</c> read by tests running in parallel — measured as a
    /// <c>6.63813305</c> vs <c>6.63813257</c> reduction-order failure in a test that passed when run alone.
    /// </summary>
    public sealed class NonRepackedKernelScopeTests
    {
        [Fact]
        public void NonRepackedKernelScope_Nested_RestoresOuterValue()
        {
            var initial = BatchedQuantProjection.DisableRepackedKernelsForParity;
            using (new NonRepackedKernelScope())
            {
                Assert.True(BatchedQuantProjection.DisableRepackedKernelsForParity);

                using (new NonRepackedKernelScope())
                {
                    Assert.True(BatchedQuantProjection.DisableRepackedKernelsForParity);
                }

                // The inner scope must restore what it FOUND (true), not the constant false. A reset to
                // false here would silently put the rest of the outer scope back on the repacked kernels.
                Assert.True(BatchedQuantProjection.DisableRepackedKernelsForParity);
            }

            Assert.Equal(initial, BatchedQuantProjection.DisableRepackedKernelsForParity);
        }

        [Fact]
        public void NonRepackedKernelScope_IsNotVisibleFromAnotherThread()
        {
            var observedElsewhere = true;

            using (new NonRepackedKernelScope())
            {
                Assert.True(BatchedQuantProjection.DisableRepackedKernelsForParity);

                // The scope binds the thread that opened it: a concurrently running test must keep the
                // kernel layout it chose. Without [ThreadStatic] on the flag this reads true.
                var other = new Thread(() => observedElsewhere = BatchedQuantProjection.DisableRepackedKernelsForParity);
                other.Start();
                other.Join();
            }

            Assert.False(observedElsewhere);
        }
    }
}
