// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Analyzers;

namespace DevOnBike.Overfit.Tests.Analyzers
{
    /// <summary>
    /// OVERFIT038 — the semantic half of NASA Power of 10 rule 2: a count read from a file must not size an
    /// allocation or bound a loop before something has bounded it.
    ///
    /// <para><b>The negative tests are the ones that decide whether the rule is usable</b>, and they are not
    /// invented. A crude text sweep on 2026-08-02 found 19 read-then-use sites across the tree and flagged 11;
    /// reading each one left <b>7 real defects and 4 false positives</b>, and all four false positives were the
    /// same shape — <c>if (length != expectedSize) throw</c> — which the scan missed because it looked for
    /// <c>&lt;</c> and <c>&gt;</c>. That shape is pinned here twice, because a rule with a 36% false-positive
    /// rate gets suppressed wholesale and then protects nothing.</para>
    ///
    /// <para><b><see cref="TheGuardBelowTheAllocationDoesNotCount"/> is why this is a Roslyn rule</b> rather
    /// than a review question. In <c>ModelSerializer</c> the bound existed and ran after the array had already
    /// been allocated and filled — no review found it, because a human sees the check a few lines below and
    /// marks it done.</para>
    /// </summary>
    public sealed class UnvalidatedExternalSizeAnalyzerTests
    {
        private static IReadOnlyList<string> Run(string body)
        {
            var source = $$"""
                using System;
                using System.IO;

                namespace N
                {
                    public static class C
                    {
                        public static void M(BinaryReader br)
                        {
                {{body}}
                        }

                        private static void RequireInRange(int value, int max)
                        {
                            if (value < 0 || value > max)
                            {
                                throw new InvalidDataException();
                            }
                        }
                    }
                }
                """;

            return AnalyzerHarness.Run(new UnvalidatedExternalSizeAnalyzer(), source);
        }

        /// <summary>
        /// The shape of four of the seven real defects — fifteen consecutive lines of a ggml reader with no
        /// bound on anything. A header declaring two billion tokens costs four bytes in the file.
        /// </summary>
        [Fact]
        public void ACountReadFromTheFileSizingAnArrayIsReported()
        {
            var ids = Run("""
                            var nTokens = br.ReadInt32();
                            var tokens = new string[nTokens];
                """);

            Assert.Equal(["OVERFIT038"], ids);
        }

        [Fact]
        public void AReadUsedDirectlyAsTheSizeIsReported()
        {
            var ids = Run("""
                            var buffer = new byte[br.ReadInt32()];
                """);

            Assert.Equal(["OVERFIT038"], ids);
        }

        /// <summary><c>RepackedWeightsFile</c>'s shape: a name length straight into a counted read.</summary>
        [Fact]
        public void ACountedReadOnAnUnvalidatedLengthIsReported()
        {
            var ids = Run("""
                            var nameLen = br.ReadInt32();
                            var name = br.ReadBytes(nameLen);
                """);

            Assert.Equal(["OVERFIT038"], ids);
        }

        /// <summary><c>LlamaLoRAAdapter</c>'s shape: the count bounds a loop rather than an allocation.</summary>
        [Fact]
        public void ACountBoundingALoopIsReported()
        {
            var ids = Run("""
                            var count = br.ReadInt32();

                            for (var i = 0; i < count; i++)
                            {
                                _ = br.ReadSingle();
                            }
                """);

            Assert.Equal(["OVERFIT038"], ids);
        }

        /// <summary>
        /// <b>The defect no review found, and the reason a text scan cannot replace this.</b> The guard is
        /// present and correct; it runs after the array has been allocated and filled.
        /// </summary>
        [Fact]
        public void TheGuardBelowTheAllocationDoesNotCount()
        {
            var ids = Run("""
                            var rank = br.ReadInt32();
                            var fileShape = new int[rank];

                            if (rank != 4)
                            {
                                throw new InvalidDataException();
                            }
                """);

            Assert.Equal(["OVERFIT038"], ids);
        }

        /// <summary>
        /// All four false positives of the 2026-08-02 sweep were this line. An equality check against a known
        /// size is a bound, and a rule that calls it a defect is a rule nobody leaves enabled.
        /// </summary>
        [Fact]
        public void AnEqualityCheckAgainstAKnownSizeIsABound()
        {
            var ids = Run("""
                            var length = br.ReadInt32();

                            if (length != 512)
                            {
                                throw new InvalidDataException();
                            }

                            var buffer = new byte[length];
                """);

            Assert.Empty(ids);
        }

        [Fact]
        public void ARangeCheckIsABound()
        {
            var ids = Run("""
                            var count = br.ReadInt32();

                            if (count < 0 || count > 4096)
                            {
                                throw new InvalidDataException();
                            }

                            for (var i = 0; i < count; i++)
                            {
                                _ = br.ReadSingle();
                            }
                """);

            Assert.Empty(ids);
        }

        /// <summary>The house style in the loaders that get this right — a named validator.</summary>
        [Fact]
        public void ACallToAGuardHelperIsABound()
        {
            var ids = Run("""
                            var count = br.ReadInt32();
                            RequireInRange(count, 4096);
                            var buffer = new byte[count];
                """);

            Assert.Empty(ids);
        }

        [Fact]
        public void AFrameworkThrowHelperIsABound()
        {
            var ids = Run("""
                            var count = br.ReadInt32();
                            ArgumentOutOfRangeException.ThrowIfGreaterThan(count, 4096);
                            var buffer = new byte[count];
                """);

            Assert.Empty(ids);
        }

        /// <summary>Capping is as good as rejecting — the file can no longer choose the allocation.</summary>
        [Fact]
        public void CappingTheValueIsABound()
        {
            var ids = Run("""
                            var count = Math.Min(br.ReadInt32(), 4096);
                            var buffer = new byte[count];
                """);

            Assert.Empty(ids);
        }

        [Fact]
        public void CappingAtTheAllocationIsABound()
        {
            var ids = Run("""
                            var count = br.ReadInt32();
                            var buffer = new byte[Math.Clamp(count, 0, 4096)];
                """);

            Assert.Empty(ids);
        }

        /// <summary>
        /// A branch that only logs is not a bound. Pinned because "there is an if about it" is the cheapest
        /// possible way for this rule to be fooled, and the fix would be invisible in review.
        /// </summary>
        [Fact]
        public void ABranchThatDoesNotLeaveIsNotABound()
        {
            var ids = Run("""
                            var suspicious = false;
                            var count = br.ReadInt32();

                            if (count > 4096)
                            {
                                suspicious = true;
                            }

                            var buffer = new byte[count];
                            _ = suspicious;
                """);

            Assert.Equal(["OVERFIT038"], ids);
        }

        /// <summary>
        /// <b>A read that does not return a number cannot be a count.</b> Found by running the rule over the
        /// tree rather than by unit testing it: the first inventory reported 30 sites and 13 of them were
        /// <c>while (reader.Read())</c>, the standard JSON pull loop, whose <c>Read</c> returns bool and means
        /// "is there more". A third of the output was noise, which is the ratio at which a rule gets
        /// suppressed wholesale instead of fixed. <c>ReadBoolean</c> stands in for it here because the test
        /// harness's reference set is deliberately minimal and has no System.Text.Json; the mechanism is the
        /// same — a <c>Read…</c> method on an untrusted reader whose return type is not an integer.
        /// </summary>
        [Fact]
        public void AReadThatDoesNotReturnANumberIsNotACount()
        {
            var ids = Run("""
                            while (br.ReadBoolean())
                            {
                                _ = br.ReadSingle();
                            }
                """);

            Assert.Empty(ids);
        }

        /// <summary>A size the program chose itself is not this rule's business.</summary>
        [Fact]
        public void ASizeThatDoesNotComeFromTheFileIsNotReported()
        {
            var ids = Run("""
                            var count = 4096;
                            var buffer = new byte[count];
                """);

            Assert.Empty(ids);
        }

        /// <summary>
        /// One assignment hop, because <c>var n = header.Count;</c> is common enough to matter and stopping at
        /// the declaration would make the rule trivially avoidable.
        /// </summary>
        [Fact]
        public void TaintSurvivesOneAssignment()
        {
            var ids = Run("""
                            var declared = br.ReadInt32();
                            var count = declared;
                            var buffer = new byte[count];
                """);

            Assert.Equal(["OVERFIT038"], ids);
        }
    }
}
