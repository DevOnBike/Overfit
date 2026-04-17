using System.Net;
using System.Net.Sockets;

namespace DevOnBike.Overfit.Demo.Unity.Server
{
    internal static class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("=== Overfit AI: Swarm Engine (Production Build) ===");

            // Production server defaults to Inference mode
            RunInference();
        }

        /// <summary>
        /// Production mode: Executes swarm logic using optimized weights.
        /// </summary>
        static void RunInference()
        {
            Console.WriteLine("\n[MODE: DEMO] Activating stable swarm instinct...");

            const int swarmSize = 100_000;

            // Proprietary optimized weights (Result of evolutionary simulation)
            var brain = new float[10];
            brain[0] = 0.8f;   // Attraction to Target X
            brain[1] = -1.2f;  // Tangential Swirl X (Tornado effect)
            brain[2] = 1.5f;   // Predator Avoidance X
            brain[3] = 0.0f;
            brain[4] = 1.2f;   // Tangential Swirl Z (Tornado effect)
            brain[5] = 0.8f;   // Attraction to Target Z
            brain[6] = 0.0f;
            brain[7] = 1.5f;   // Predator Avoidance Z
            brain[8] = 0.0f;   // Bias X
            brain[9] = 0.0f;   // Bias Z

            var listener = new TcpListener(IPAddress.Loopback, 5000);
            listener.Start();
            Console.WriteLine("Server listening on port 5000. Awaiting Unity connection...");

            while (true)
            {
                try
                {
                    using var client = listener.AcceptTcpClient();
                    client.NoDelay = true; // Minimize latency
                    using var stream = client.GetStream();
                    Console.WriteLine("Unity connected. Streaming inference data...");

                    int inputSize = 4, outputSize = 2;
                    var recvBuf = new byte[swarmSize * inputSize * 4];
                    var sendBuf = new byte[swarmSize * outputSize * 4];
                    var inputs = new float[swarmSize * inputSize];
                    var outputs = new float[swarmSize * outputSize];

                    while (true)
                    {
                        // 1. Receive relative vectors from the game engine
                        ReadExactly(stream, recvBuf, recvBuf.Length);
                        Buffer.BlockCopy(recvBuf, 0, inputs, 0, recvBuf.Length);

                        // 2. High-performance inference loop
                        for (var i = 0; i < swarmSize; i++)
                        {
                            var inIdx = i * 4;
                            var outIdx = i * 2;

                            // Tornado mathematical engine
                            outputs[outIdx + 0] = inputs[inIdx + 0] * brain[0] + inputs[inIdx + 1] * brain[1] + inputs[inIdx + 2] * brain[2];
                            outputs[outIdx + 1] = inputs[inIdx + 0] * brain[4] + inputs[inIdx + 1] * brain[5] + inputs[inIdx + 3] * brain[7];
                        }

                        // 3. Send calculated steering forces back to Unity
                        Buffer.BlockCopy(outputs, 0, sendBuf, 0, sendBuf.Length);
                        stream.Write(sendBuf, 0, sendBuf.Length);
                    }
                }
                catch (Exception e)
                {
                    Console.WriteLine($"Connection lost: {e.Message}. Re-initializing listener...");
                }
            }
        }

        /// <summary>
        /// Utility to ensure the full byte buffer is read from the stream.
        /// </summary>
        static void ReadExactly(NetworkStream stream, byte[] buffer, int amount)
        {
            var total = 0;
            while (total < amount)
            {
                var read = stream.Read(buffer, total, amount - total);
                if (read == 0)
                {
                    throw new Exception("Network link terminated by client.");
                }
                total += read;
            }
        }
    }
}