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

        // ====================================================================
        // Persistence
        // ====================================================================

        /// <summary>
        ///     Magic header for serialised archive blobs. Spells "MEAD" (MAP-Elites
        ///     Archive Data) in little-endian ASCII. Used to detect mis-routed files
        ///     before reading further fields.
        /// </summary>
        private const uint SaveMagic = 0x4D454144u;

        /// <summary>
        ///     Schema version of the on-disk format. Bump when adding/removing/reordering
        ///     fields. Old files are rejected on Load with InvalidDataException — there
        ///     is no migration path; users are expected to retrain on schema change.
        /// </summary>
        private const int SaveSchemaVersion = 1;

        /// <summary>
        ///     Writes the archive's full state to <paramref name="writer"/> in a
        ///     versioned binary format. Includes shape (parameter count, descriptor
        ///     dimensions, bins per dimension, descriptor bounds) so that Load can
        ///     verify the receiving archive is compatible. The on-disk size is
        ///     dominated by the dense per-cell arrays (parameters and descriptors)
        ///     even when most cells are unoccupied — this keeps the format simple
        ///     and Load fast at the cost of file size for sparse archives.
        /// </summary>
        public void Save(BinaryWriter writer)
        {
            ThrowIfDisposed();
            ArgumentNullException.ThrowIfNull(writer);

            writer.Write(SaveMagic);
            writer.Write(SaveSchemaVersion);

            writer.Write(_parameterCount);
            writer.Write(DescriptorDimensions);

            for (var d = 0; d < DescriptorDimensions; d++)
            {
                writer.Write(_binsPerDimension[d]);
            }

            for (var d = 0; d < DescriptorDimensions; d++)
            {
                writer.Write(_descriptorMin[d]);
            }

            for (var d = 0; d < DescriptorDimensions; d++)
            {
                writer.Write(_descriptorMax[d]);
            }

            writer.Write(CellCount);
            writer.Write(_qdScore);
            writer.Write(_occupiedCount);

            for (var i = 0; i < _occupiedCount; i++)
            {
                writer.Write(_occupiedCells[i]);
            }

            // Dense per-cell arrays. Unoccupied cells contribute their initial values
            // (NegativeInfinity for fitness, zeros for parameters/descriptors), which
            // are wasted bytes but make Save/Load trivially correct: no need to walk
            // the occupancy mask while serialising.
            for (var i = 0; i < CellCount; i++)
            {
                writer.Write(_fitness[i]);
            }

            var paramsLen = CellCount * _parameterCount;
            for (var i = 0; i < paramsLen; i++)
            {
                writer.Write(_parameters[i]);
            }

            var descLen = CellCount * DescriptorDimensions;
            for (var i = 0; i < descLen; i++)
            {
                writer.Write(_descriptors[i]);
            }

            for (var i = 0; i < CellCount; i++)
            {
                writer.Write(_occupied[i]);
            }
        }

        /// <summary>
        ///     Loads an archive snapshot previously written by <see cref="Save"/>.
        ///     The receiving archive's shape (parameter count, descriptor dimensions,
        ///     bins per dimension, descriptor bounds) must exactly match the saved
        ///     blob — otherwise an <see cref="InvalidDataException"/> is thrown.
        ///     Any prior contents of the archive are discarded.
        /// </summary>
        public void Load(BinaryReader reader)
        {
            ThrowIfDisposed();
            ArgumentNullException.ThrowIfNull(reader);

            var magic = reader.ReadUInt32();
            if (magic != SaveMagic)
            {
                throw new InvalidDataException(
                    $"Not a GridEliteArchive snapshot — magic header 0x{magic:X8} does not match expected 0x{SaveMagic:X8}.");
            }

            var schemaVersion = reader.ReadInt32();
            if (schemaVersion != SaveSchemaVersion)
            {
                throw new InvalidDataException(
                    $"GridEliteArchive snapshot schema version {schemaVersion} is not supported by this build (expected {SaveSchemaVersion}). There is no automatic migration; retrain to produce a fresh snapshot.");
            }

            var parameterCount = reader.ReadInt32();
            if (parameterCount != _parameterCount)
            {
                throw new InvalidDataException(
                    $"Snapshot parameterCount={parameterCount} does not match this archive's parameterCount={_parameterCount}.");
            }

            var descriptorDimensions = reader.ReadInt32();
            if (descriptorDimensions != DescriptorDimensions)
            {
                throw new InvalidDataException(
                    $"Snapshot descriptorDimensions={descriptorDimensions} does not match this archive's DescriptorDimensions={DescriptorDimensions}.");
            }

            for (var d = 0; d < descriptorDimensions; d++)
            {
                var bins = reader.ReadInt32();
                if (bins != _binsPerDimension[d])
                {
                    throw new InvalidDataException(
                        $"Snapshot bins[{d}]={bins} does not match archive bins[{d}]={_binsPerDimension[d]}.");
                }
            }

            for (var d = 0; d < descriptorDimensions; d++)
            {
                var min = reader.ReadSingle();
                if (min != _descriptorMin[d])
                {
                    throw new InvalidDataException(
                        $"Snapshot descriptorMin[{d}]={min} does not match archive descriptorMin[{d}]={_descriptorMin[d]}.");
                }
            }

            for (var d = 0; d < descriptorDimensions; d++)
            {
                var max = reader.ReadSingle();
                if (max != _descriptorMax[d])
                {
                    throw new InvalidDataException(
                        $"Snapshot descriptorMax[{d}]={max} does not match archive descriptorMax[{d}]={_descriptorMax[d]}.");
                }
            }

            var cellCount = reader.ReadInt32();
            if (cellCount != CellCount)
            {
                throw new InvalidDataException(
                    $"Snapshot cellCount={cellCount} does not match archive CellCount={CellCount}.");
            }

            // Reset existing state before populating from the stream. Anything that
            // was in the archive prior to Load is discarded.
            Clear();

            _qdScore = reader.ReadSingle();
            _occupiedCount = reader.ReadInt32();

            if ((uint)_occupiedCount > (uint)CellCount)
            {
                throw new InvalidDataException(
                    $"Snapshot occupiedCount={_occupiedCount} exceeds cellCount={CellCount}.");
            }

            for (var i = 0; i < _occupiedCount; i++)
            {
                var cellIdx = reader.ReadInt32();
                if ((uint)cellIdx >= (uint)CellCount)
                {
                    throw new InvalidDataException(
                        $"Snapshot occupiedCells[{i}]={cellIdx} is out of range for cellCount={CellCount}.");
                }
                _occupiedCells[i] = cellIdx;
            }

            for (var i = 0; i < CellCount; i++)
            {
                _fitness[i] = reader.ReadSingle();
            }

            var paramsLen = CellCount * _parameterCount;
            for (var i = 0; i < paramsLen; i++)
            {
                _parameters[i] = reader.ReadSingle();
            }

            var descLen = CellCount * DescriptorDimensions;
            for (var i = 0; i < descLen; i++)
            {
                _descriptors[i] = reader.ReadSingle();
            }

            for (var i = 0; i < CellCount; i++)
            {
                _occupied[i] = reader.ReadBoolean();
            }
        }

        /// <summary>
        ///     Empties the archive: resets occupancy, qd-score, and fitness back to
        ///     <see cref="float.NegativeInfinity"/>. Parameter and descriptor buffers
        ///     are not cleared (they're considered garbage when the corresponding cell
        ///     is unoccupied). The archive's grid shape (bins, bounds, parameter count)
        ///     stays intact — re-use the same instance for a fresh QD run.
        /// </summary>
        public void Clear()
        {
            ThrowIfDisposed();

            Array.Clear(_occupied, 0, _occupied.Length);
            Array.Fill(_fitness, float.NegativeInfinity);
            Array.Clear(_occupiedCells, 0, _occupiedCells.Length);
            _occupiedCount = 0;
            _qdScore = 0f;
        }

        public bool TryInsert(ReadOnlySpan<float> parameters, float fitness, ReadOnlySpan<float> descriptor)
        {
            return Insert(parameters, fitness, descriptor) is EliteInsertStatus.InsertedNewCell
                or EliteInsertStatus.ReplacedExistingCell;
        }

        /// <summary>
        ///     Convenience overload that discards the cell index. Use the
        ///     <see cref="Insert(ReadOnlySpan{float}, float, ReadOnlySpan{float}, out int)"/>
        ///     overload when the caller needs to track which cell currently holds the
        ///     archive's strongest elite (e.g. MAP-Elites'
        ///     <c>BestEliteFitness</c> pointer).
        /// </summary>
        public EliteInsertStatus Insert(
            ReadOnlySpan<float> parameters,
            float fitness,
            ReadOnlySpan<float> descriptor)
        {
            return Insert(parameters, fitness, descriptor, out _);
        }

        /// <summary>
        ///     Inserts a candidate and reports which cell received it. <paramref name="cellIndex"/>
        ///     is set to a valid index for <see cref="EliteInsertStatus.InsertedNewCell"/> and
        ///     <see cref="EliteInsertStatus.ReplacedExistingCell"/>; for all other outcomes
        ///     it is set to <c>-1</c>. <see cref="EliteInsertStatus.Rejected"/> still resolves
        ///     <paramref name="cellIndex"/> to the matched cell so the caller can inspect the
        ///     existing elite if needed.
        /// </summary>
        public EliteInsertStatus Insert(
            ReadOnlySpan<float> parameters,
            float fitness,
            ReadOnlySpan<float> descriptor,
            out int cellIndex)
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

            // Reject NaN and ±∞ before touching the archive. Without this guard, a NaN
            // fitness would be stored as the cell's elite and then poison the run:
            //   - _qdScore += NaN turns the global QD score into NaN for the rest of the run
            //   - "fitness <= existingFitness" with NaN always returns false, so any later
            //     valid candidate would replace the NaN cell (good by accident) but the
            //     QD score is already corrupted and cannot be recovered.
            // Surfacing this as a distinct status — instead of silently dropping under
            // EliteInsertStatus.Rejected — gives evaluators a chance to detect that they
            // produced an invalid fitness for one of their candidates.
            if (!float.IsFinite(fitness))
            {
                cellIndex = -1;
                return EliteInsertStatus.InvalidFitness;
            }

            if (!TryGetCellIndex(descriptor, out cellIndex))
            {
                cellIndex = -1;
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