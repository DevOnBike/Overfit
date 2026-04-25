// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Evolutionary.Storage;

namespace DevOnBike.Overfit.Tests
{
    public sealed class MapElitesArchiveTests
    {
        [Fact]
        public void Insert_IntoEmptyCell_OccupiesCellAndUpdatesMetrics()
        {
            using var archive = CreateArchive(parameterCount: 4);

            float[] parameters = [1f, 2f, 3f, 4f];
            float[] descriptor = [0.10f, 0.20f];

            var status = archive.Insert(parameters, fitness: 5f, descriptor);

            Assert.Equal(EliteInsertStatus.InsertedNewCell, status);
            Assert.Equal(1, archive.OccupiedCount);
            Assert.Equal(1f / archive.CellCount, archive.Coverage, 6);
            Assert.Equal(5f, archive.QdScore, 6);

            Assert.True(archive.TryGetCellIndex(descriptor, out int cellIndex));
            Assert.True(archive.IsOccupied(cellIndex));
            Assert.Equal(5f, archive.GetFitness(cellIndex), 6);
            Assert.Equal(parameters, archive.GetParameters(cellIndex).ToArray());
            Assert.Equal(descriptor, archive.GetDescriptor(cellIndex).ToArray());
        }

        [Fact]
        public void Insert_WorseCandidateIntoOccupiedCell_IsRejected()
        {
            using var archive = CreateArchive(parameterCount: 3);

            float[] descriptor = [0.35f, 0.65f];

            var first = archive.Insert([1f, 1f, 1f], fitness: 10f, descriptor);
            var second = archive.Insert([9f, 9f, 9f], fitness: 7f, descriptor);

            Assert.Equal(EliteInsertStatus.InsertedNewCell, first);
            Assert.Equal(EliteInsertStatus.Rejected, second);
            Assert.Equal(1, archive.OccupiedCount);
            Assert.Equal(10f, archive.QdScore, 6);

            Assert.True(archive.TryGetCellIndex(descriptor, out int cellIndex));
            Assert.Equal(10f, archive.GetFitness(cellIndex), 6);
            Assert.Equal(new[] { 1f, 1f, 1f }, archive.GetParameters(cellIndex).ToArray());
        }

        [Fact]
        public void Insert_BetterCandidateIntoOccupiedCell_ReplacesElite()
        {
            using var archive = CreateArchive(parameterCount: 2);

            float[] descriptor = [0.42f, 0.71f];

            archive.Insert([1f, 2f], fitness: 3f, descriptor);
            var status = archive.Insert([7f, 8f], fitness: 11f, descriptor);

            Assert.Equal(EliteInsertStatus.ReplacedExistingCell, status);
            Assert.Equal(1, archive.OccupiedCount);
            Assert.Equal(11f, archive.QdScore, 6);

            Assert.True(archive.TryGetCellIndex(descriptor, out int cellIndex));
            Assert.Equal(11f, archive.GetFitness(cellIndex), 6);
            Assert.Equal(new[] { 7f, 8f }, archive.GetParameters(cellIndex).ToArray());
        }

        [Fact]
        public void Insert_OutOfBoundsDescriptor_ReturnsOutOfBounds()
        {
            using var archive = CreateArchive(parameterCount: 2);

            var statusLow = archive.Insert([1f, 2f], fitness: 1f, descriptor: [-0.01f, 0.50f]);
            var statusHigh = archive.Insert([1f, 2f], fitness: 1f, descriptor: [0.50f, 1.01f]);

            Assert.Equal(EliteInsertStatus.OutOfBounds, statusLow);
            Assert.Equal(EliteInsertStatus.OutOfBounds, statusHigh);
            Assert.Equal(0, archive.OccupiedCount);
            Assert.Equal(0f, archive.Coverage, 6);
            Assert.Equal(0f, archive.QdScore, 6);
        }

        [Fact]
        public void TryGetCellIndex_PlacesMinAndMaxIntoValidCells()
        {
            using var archive = new GridEliteArchive(
                parameterCount: 2,
                binsPerDimension: stackalloc[] { 4, 5 },
                descriptorMin: stackalloc[] { 0f, -1f },
                descriptorMax: stackalloc[] { 1f, 1f });

            Assert.True(archive.TryGetCellIndex([0f, -1f], out int minCell));
            Assert.True(archive.TryGetCellIndex([1f, 1f], out int maxCell));

            Assert.InRange(minCell, 0, archive.CellCount - 1);
            Assert.InRange(maxCell, 0, archive.CellCount - 1);
            Assert.NotEqual(minCell, maxCell);
        }

        [Fact]
        public void TrySampleOccupiedCell_ReturnsOnlyOccupiedCells()
        {
            using var archive = CreateArchive(parameterCount: 2);

            archive.Insert([1f, 1f], 1f, [0.10f, 0.10f]);
            archive.Insert([2f, 2f], 2f, [0.90f, 0.90f]);

            uint state = 123u;

            for (var i = 0; i < 64; i++)
            {
                Assert.True(archive.TrySampleOccupiedCell(ref state, out int cellIndex));
                Assert.True(archive.IsOccupied(cellIndex));
            }
        }

        private static GridEliteArchive CreateArchive(int parameterCount)
        {
            return new GridEliteArchive(
                parameterCount: parameterCount,
                binsPerDimension: stackalloc[] { 8, 8 },
                descriptorMin: stackalloc[] { 0f, 0f },
                descriptorMax: stackalloc[] { 1f, 1f });
        }
    }
}