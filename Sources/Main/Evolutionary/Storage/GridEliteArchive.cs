// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Evolutionary.Abstractions;

namespace DevOnBike.Overfit.Evolutionary.Storage
{
    /// <summary>
    /// Fixed-grid MAP-Elites archive.
    /// Stores one elite per descriptor cell.
    /// </summary>
    public sealed class GridEliteArchive : IEliteArchive
    {
        private const uint DefaultNonZeroSeed = 0x6D2B79F5u;

        private readonly int _parameterCount;
        private readonly int[] _binsPerDimension;
        private readonly int[] _cellStrides;
        private readonly float[] _descriptorMin;
        private readonly float[] _descriptorMax;
        private readonly float[] _descriptorInvRange;

        private readonly bool[] _occupied;
        private readonly float[] _fitness;
        private readonly float[] _parameters;
        private readonly float[] _descriptors;
        private readonly int[] _occupiedCells;

        private int _occupiedCount;
        private float _qdScore;
        private bool _disposed;

        public GridEliteArchive(
            int parameterCount,
            ReadOnlySpan<int> binsPerDimension,
            ReadOnlySpan<float> descriptorMin,
            ReadOnlySpan<float> descriptorMax)
        {
            if (parameterCount <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(parameterCount));
            }

            if (binsPerDimension.Length == 0)
            {
                throw new ArgumentException("At least one descriptor dimension is required.", nameof(binsPerDimension));
            }

            if (descriptorMin.Length != binsPerDimension.Length)
            {
                throw new ArgumentException("descriptorMin length must match binsPerDimension length.", nameof(descriptorMin));
            }

            if (descriptorMax.Length != binsPerDimension.Length)
            {
                throw new ArgumentException("descriptorMax length must match binsPerDimension length.", nameof(descriptorMax));
            }

            _parameterCount = parameterCount;
            DescriptorDimensions = binsPerDimension.Length;

            _binsPerDimension = binsPerDimension.ToArray();
            _cellStrides = new int[DescriptorDimensions];
            _descriptorMin = descriptorMin.ToArray();
            _descriptorMax = descriptorMax.ToArray();
            _descriptorInvRange = new float[DescriptorDimensions];

            var cellCount = 1;
            for (var d = DescriptorDimensions - 1; d >= 0; d--)
            {
                var bins = _binsPerDimension[d];
                if (bins <= 0)
                {
                    throw new ArgumentOutOfRangeException(nameof(binsPerDimension), "All bin counts must be positive.");
                }

                var min = _descriptorMin[d];
                var max = _descriptorMax[d];
                if (!(max > min))
                {
                    throw new ArgumentException("Each descriptor dimension must have max > min.");
                }

                _descriptorInvRange[d] = 1f / (max - min);
                _cellStrides[d] = cellCount;
                cellCount = checked(cellCount * bins);
            }

            CellCount = cellCount;

            _occupied = new bool[cellCount];
            _fitness = new float[cellCount];
            Array.Fill(_fitness, float.NegativeInfinity);

            _parameters = new float[cellCount * _parameterCount];
            _descriptors = new float[cellCount * DescriptorDimensions];
            _occupiedCells = new int[cellCount];

            _occupiedCount = 0;
            _qdScore = 0f;
        }

        public int DescriptorDimensions { get; }

        public int ParameterCount => _parameterCount;

        public int CellCount { get; }

        public int OccupiedCount
        {
            get
            {
                ThrowIfDisposed();
                return _occupiedCount;
            }
        }

        public float Coverage
        {
            get
            {
                ThrowIfDisposed();
                return CellCount == 0 ? 0f : (float)_occupiedCount / CellCount;
            }
        }

        public float QdScore
        {
            get
            {
                ThrowIfDisposed();
                return _qdScore;
            }
        }

        public bool TryInsert(ReadOnlySpan<float> parameters, float fitness, ReadOnlySpan<float> descriptor)
        {
            return Insert(parameters, fitness, descriptor) is EliteInsertStatus.InsertedNewCell
                or EliteInsertStatus.ReplacedExistingCell;
        }

        public EliteInsertStatus Insert(
            ReadOnlySpan<float> parameters,
            float fitness,
            ReadOnlySpan<float> descriptor)
        {
            ThrowIfDisposed();

            if (parameters.Length != _parameterCount)
            {
                throw new ArgumentException(
                    $"parameters length must be {_parameterCount}.",
                    nameof(parameters));
            }

            if (descriptor.Length != DescriptorDimensions)
            {
                throw new ArgumentException(
                    $"descriptor length must be {DescriptorDimensions}.",
                    nameof(descriptor));
            }

            if (!TryGetCellIndex(descriptor, out var cellIndex))
            {
                return EliteInsertStatus.OutOfBounds;
            }

            if (!_occupied[cellIndex])
            {
                WriteCell(cellIndex, parameters, fitness, descriptor);
                _occupied[cellIndex] = true;
                _occupiedCells[_occupiedCount++] = cellIndex;
                _qdScore += fitness;
                return EliteInsertStatus.InsertedNewCell;
            }

            var existingFitness = _fitness[cellIndex];
            if (fitness <= existingFitness)
            {
                return EliteInsertStatus.Rejected;
            }

            _qdScore += fitness - existingFitness;
            WriteCell(cellIndex, parameters, fitness, descriptor);
            return EliteInsertStatus.ReplacedExistingCell;
        }

        public bool TryGetCellIndex(ReadOnlySpan<float> descriptor, out int cellIndex)
        {
            ThrowIfDisposed();

            if (descriptor.Length != DescriptorDimensions)
            {
                throw new ArgumentException(
                    $"descriptor length must be {DescriptorDimensions}.",
                    nameof(descriptor));
            }

            var index = 0;

            for (var d = 0; d < DescriptorDimensions; d++)
            {
                var value = descriptor[d];
                var min = _descriptorMin[d];
                var max = _descriptorMax[d];

                if (value < min || value > max)
                {
                    cellIndex = -1;
                    return false;
                }

                int bin;
                if (value == max)
                {
                    bin = _binsPerDimension[d] - 1;
                }
                else
                {
                    var normalized = (value - min) * _descriptorInvRange[d];
                    bin = (int)(normalized * _binsPerDimension[d]);

                    if (bin < 0)
                    {
                        bin = 0;
                    }
                    else if (bin >= _binsPerDimension[d])
                    {
                        bin = _binsPerDimension[d] - 1;
                    }
                }

                index += bin * _cellStrides[d];
            }

            cellIndex = index;
            return true;
        }

        public bool TrySampleOccupiedCell(ref uint rngState, out int cellIndex)
        {
            ThrowIfDisposed();

            if (_occupiedCount == 0)
            {
                cellIndex = -1;
                return false;
            }

            cellIndex = _occupiedCells[(int)NextUInt32Below(ref rngState, (uint)_occupiedCount)];
            return true;
        }

        public bool IsOccupied(int cellIndex)
        {
            ThrowIfDisposed();
            ValidateCellIndex(cellIndex);
            return _occupied[cellIndex];
        }

        public float GetFitness(int cellIndex)
        {
            ThrowIfDisposed();
            ValidateCellIndex(cellIndex);
            return _fitness[cellIndex];
        }

        public ReadOnlySpan<float> GetParameters(int cellIndex)
        {
            ThrowIfDisposed();
            ValidateCellIndex(cellIndex);
            return _parameters.AsSpan(cellIndex * _parameterCount, _parameterCount);
        }

        public ReadOnlySpan<float> GetDescriptor(int cellIndex)
        {
            ThrowIfDisposed();
            ValidateCellIndex(cellIndex);
            return _descriptors.AsSpan(cellIndex * DescriptorDimensions, DescriptorDimensions);
        }

        public void Dispose()
        {
            _disposed = true;
        }

        private void WriteCell(
            int cellIndex,
            ReadOnlySpan<float> parameters,
            float fitness,
            ReadOnlySpan<float> descriptor)
        {
            var parameterOffset = cellIndex * _parameterCount;
            var descriptorOffset = cellIndex * DescriptorDimensions;

            parameters.CopyTo(_parameters.AsSpan(parameterOffset, _parameterCount));
            descriptor.CopyTo(_descriptors.AsSpan(descriptorOffset, DescriptorDimensions));
            _fitness[cellIndex] = fitness;
        }

        private void ValidateCellIndex(int cellIndex)
        {
            if ((uint)cellIndex >= (uint)CellCount)
            {
                throw new ArgumentOutOfRangeException(nameof(cellIndex));
            }
        }

        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
        }

        private static uint NormalizeSeed(uint seed)
        {
            return seed == 0u ? DefaultNonZeroSeed : seed;
        }

        private static uint NextUInt32(ref uint state)
        {
            var x = NormalizeSeed(state);
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            state = NormalizeSeed(x);
            return state;
        }

        private static uint NextUInt32Below(ref uint state, uint maxExclusive)
        {
            if (maxExclusive == 0u)
            {
                throw new ArgumentOutOfRangeException(nameof(maxExclusive));
            }

            var product = (ulong)NextUInt32(ref state) * maxExclusive;
            var low = (uint)product;

            if (low < maxExclusive)
            {
                var threshold = unchecked((uint)(0 - maxExclusive)) % maxExclusive;
                while (low < threshold)
                {
                    product = (ulong)NextUInt32(ref state) * maxExclusive;
                    low = (uint)product;
                }
            }

            return (uint)(product >> 32);
        }
    }
}