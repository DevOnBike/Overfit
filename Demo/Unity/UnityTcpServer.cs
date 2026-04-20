// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;
using System.Net;
using System.Net.Sockets;
using System.Numerics;

namespace DevOnBike.Overfit.Demo.Unity.Server
{
    /// <summary>
    ///     Minimal TCP bridge to the Unity client. The server is authoritative over bot
    ///     positions — Unity sends only target and predator coordinates each frame, and the
    ///     server returns the full position table for rendering.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Protocol (all floats little-endian):
    ///         <list type="bullet">
    ///             <item>Unity → Server: <c>[targetX, targetZ, predatorX, predatorZ]</c> (16 B)</item>
    ///             <item>Server → Unity: <c>[posX, posZ] × SwarmSize</c> (8 B × <c>SwarmSize</c>)</item>
    ///         </list>
    ///     </para>
    ///     <para>
    ///         The listener accepts one client at a time and blocks the caller's thread for
    ///         the duration of the session. This matches the demo's usage pattern — one
    ///         Unity instance connecting at a time — and avoids the complexity of concurrent
    ///         clients for a showcase whose focus is the Overfit library rather than
    ///         networking.
    ///     </para>
    /// </remarks>
    public sealed class UnityTcpServer : IDisposable
    {
        private const int CommandSize = 4 * sizeof(float);

        private readonly TcpListener _listener;
        private readonly int _swarmSize;
        private readonly byte[] _commandBuffer;
        private readonly byte[] _positionBuffer;
        private readonly Vector2[] _positionsScratch;

        public UnityTcpServer(int port, int swarmSize)
        {
            if (swarmSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(swarmSize));
            }

            _listener = new TcpListener(IPAddress.Loopback, port);
            _swarmSize = swarmSize;
            _commandBuffer = new byte[CommandSize];
            _positionBuffer = new byte[swarmSize * 2 * sizeof(float)];

            // Copy scratch so the per-frame serialisation can be parallelised without
            // capturing Span<Vector2> (spans are ref structs, not allowed in lambdas).
            // 800 KB at SwarmSize=100k — allocated once, reused every frame.
            _positionsScratch = new Vector2[swarmSize];
        }

        public int Port => ((IPEndPoint)_listener.LocalEndpoint).Port;

        public void Start()
        {
            _listener.Start();
        }

        /// <summary>
        ///     Blocks until a Unity client connects, then invokes <paramref name="onFrame"/>
        ///     for every incoming command frame. The callback receives the deserialised
        ///     target + predator, runs one simulation step on those inputs and returns a
        ///     read-only view of the updated bot positions; the server serialises those to
        ///     the client before awaiting the next command.
        /// </summary>
        /// <remarks>
        ///     Throws <see cref="IOException"/> if the client disconnects mid-frame; callers
        ///     typically catch that and loop back into <see cref="AcceptAndServe"/> to accept
        ///     a reconnection.
        /// </remarks>
        public void AcceptAndServe(
            Func<Vector2, Vector2, ReadOnlySpan<Vector2>> onFrame,
            CancellationToken cancellation)
        {
            using var client = _listener.AcceptTcpClient();
            client.NoDelay = true;
            using var stream = client.GetStream();

            Console.WriteLine("[CONNECT] Unity client attached.");

            while (!cancellation.IsCancellationRequested)
            {
                ReadExactly(stream, _commandBuffer);

                var target = new Vector2(
                    BinaryPrimitives.ReadSingleLittleEndian(_commandBuffer.AsSpan(0)),
                    BinaryPrimitives.ReadSingleLittleEndian(_commandBuffer.AsSpan(4)));

                var predator = new Vector2(
                    BinaryPrimitives.ReadSingleLittleEndian(_commandBuffer.AsSpan(8)),
                    BinaryPrimitives.ReadSingleLittleEndian(_commandBuffer.AsSpan(12)));

                var positions = onFrame(target, predator);

                if (positions.Length != _swarmSize)
                {
                    throw new InvalidOperationException(
                        $"onFrame returned {positions.Length} positions, expected {_swarmSize}.");
                }

                // Copy into pre-allocated scratch so WritePositions can fan out across
                // threads without capturing the incoming span.
                positions.CopyTo(_positionsScratch);
                WritePositions(_positionsScratch, _positionBuffer);
                stream.Write(_positionBuffer);
            }
        }

        private static void ReadExactly(NetworkStream stream, byte[] buffer)
        {
            var total = 0;

            while (total < buffer.Length)
            {
                var read = stream.Read(buffer, total, buffer.Length - total);

                if (read == 0)
                {
                    throw new IOException("Stream closed mid-frame.");
                }

                total += read;
            }
        }

        private static void WritePositions(Vector2[] positions, byte[] buffer)
        {
            // Parallel fill — per-bot work is trivial, but SwarmSize can be 100k and this
            // runs every frame. Pointer trick avoids Span capture in the lambda.
            unsafe
            {
                fixed (byte* pBuf = buffer)
                {
                    var ptr = (IntPtr)pBuf;

                    Parallel.For(0, positions.Length, i =>
                    {
                        var floats = (float*)ptr;
                        floats[(i * 2) + 0] = positions[i].X;
                        floats[(i * 2) + 1] = positions[i].Y;
                    });
                }
            }
        }

        public void Dispose()
        {
            _listener.Stop();
        }
    }
}
