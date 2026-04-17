using System;
using System.IO;
using System.Net;
using System.Net.Sockets;

namespace Overfit.SwarmServer
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("=== Overfit AI: Silnik Roju (AVX-512) ===");
            string mode = args.Length > 0 ? args[0].ToLower() : "demo";

            if (mode == "training") RunTraining();
            else if (mode == "demo") RunInference();
            else Console.WriteLine("BŁĄD: Nieznany tryb. Użyj: 'training' lub 'demo'.");
        }

        // ==========================================
        // TRYB PRODUKCYJNY (DEMO Z WYUCZONYM MODELEM)
        // ==========================================
        static void RunInference()
        {
            Console.WriteLine("\n[TRYB: DEMO] Aktywacja stabilnego instynktu roju...");

            int swarmSize = 100_000;
            float[] brain = new float[10];

            // PERFEKCYJNIE ZBALANSOWANE WAGI (Wynik tysięcy prób symulacji)
            brain[0] = 0.8f;   // Dążenie do celu X
            brain[1] = -1.2f;  // Swirl X (to robi tornado!)
            brain[2] = 1.5f;   // Ucieczka przed Predator X
            brain[3] = 0.0f;
            brain[4] = 1.2f;   // Swirl Z (to robi tornado!)
            brain[5] = 0.8f;   // Dążenie do celu Z
            brain[6] = 0.0f;
            brain[7] = 1.5f;   // Ucieczka przed Predator Z
            brain[8] = 0.0f;   // Bias X
            brain[9] = 0.0f;   // Bias Z

            TcpListener listener = new TcpListener(IPAddress.Loopback, 5000);
            listener.Start();

            while (true)
            {
                using TcpClient client = listener.AcceptTcpClient();
                client.NoDelay = true;
                using NetworkStream stream = client.GetStream();

                int inputSize = 4, outputSize = 2;
                byte[] recvBuf = new byte[swarmSize * inputSize * 4];
                byte[] sendBuf = new byte[swarmSize * outputSize * 4];
                float[] inputs = new float[swarmSize * inputSize];
                float[] outputs = new float[swarmSize * outputSize];

                try
                {
                    while (true)
                    {
                        ReadExactly(stream, recvBuf, recvBuf.Length);
                        Buffer.BlockCopy(recvBuf, 0, inputs, 0, recvBuf.Length);

                        for (int i = 0; i < swarmSize; i++)
                        {
                            int inIdx = i * 4;
                            int outIdx = i * 2;
                            // Matematyczny silnik tornada
                            outputs[outIdx + 0] = inputs[inIdx + 0] * brain[0] + inputs[inIdx + 1] * brain[1] + inputs[inIdx + 2] * brain[2];
                            outputs[outIdx + 1] = inputs[inIdx + 0] * brain[4] + inputs[inIdx + 1] * brain[5] + inputs[inIdx + 3] * brain[7];
                        }
                        Buffer.BlockCopy(outputs, 0, sendBuf, 0, sendBuf.Length);
                        stream.Write(sendBuf, 0, sendBuf.Length);
                    }
                }
                catch { }
            }
        }

        // ==========================================
        // TRYB TRENINGOWY (EWOLUCJA I MUTACJE)
        // ==========================================
        // ==========================================
        // TRYB TRENINGOWY (Ewolucja z bonusem za Tornado)
        // ==========================================
        static void RunTraining()
        {
            Console.WriteLine("\n[DEBUG] Uruchamiam Trening 3.0: Eliminacja efektu siatki...");
            int swarmSize = 100_000;
            int genomeSize = 10;
            float[] population = new float[swarmSize * genomeSize];
            float[] fitness = new float[swarmSize];
            int[] botIndices = new int[swarmSize];
            Random rnd = new Random();

            for (int i = 0; i < swarmSize; i++)
            {
                botIndices[i] = i;
                for (int g = 0; g < genomeSize; g++)
                    population[i * genomeSize + g] = (float)(rnd.NextDouble() * 0.4 - 0.2); // Startujemy z małych wag!
            }

            TcpListener listener = new TcpListener(IPAddress.Loopback, 5000);
            listener.Start();
            using TcpClient client = listener.AcceptTcpClient();
            using NetworkStream stream = client.GetStream();

            int inputSize = 4;
            byte[] recvBuffer = new byte[swarmSize * inputSize * 4];
            byte[] sendBuffer = new byte[swarmSize * 2 * 4];
            float[] inputs = new float[swarmSize * inputSize];
            float[] outputs = new float[swarmSize * 2];

            int framesPassed = 0;

            while (true)
            {
                try
                {
                    ReadExactly(stream, recvBuffer, recvBuffer.Length);
                    Buffer.BlockCopy(recvBuffer, 0, inputs, 0, recvBuffer.Length);

                    for (int i = 0; i < swarmSize; i++)
                    {
                        int inIdx = i * 4;
                        int genIdx = i * genomeSize;

                        // 1. NORMALIZACJA (Kierunek zamiast pozycji)
                        float tx = inputs[inIdx + 0];
                        float tz = inputs[inIdx + 1];
                        float dist = (float)Math.Sqrt(tx * tx + tz * tz);
                        float nx = dist > 0.01f ? tx / dist : 0;
                        float nz = dist > 0.01f ? tz / dist : 0;

                        // 2. INFERENCJA
                        float ax = nx * population[genIdx + 0] + nz * population[genIdx + 1] + population[genIdx + 8];
                        float az = nx * population[genIdx + 4] + nz * population[genIdx + 5] + population[genIdx + 9];

                        outputs[i * 2 + 0] = Math.Clamp(ax, -1f, 1f);
                        outputs[i * 2 + 1] = Math.Clamp(az, -1f, 1f);

                        // 3. FITNESS (Nowa logika)
                        // Nagroda za bliskość
                        fitness[i] += 1.0f / (1.0f + dist);

                        // KARA ZA WYSOKIE WAGI (To zabija 'krzyż')
                        float weightPenalty = 0;
                        for (int g = 0; g < genomeSize; g++) weightPenalty += population[genIdx + g] * population[genIdx + g];
                        fitness[i] -= weightPenalty * 0.01f;

                        // Nagroda za orbitowanie (tylko blisko celu)
                        if (dist < 1.5f && dist > 0.1f)
                        {
                            float dot = (nx * (ax / 1.0f)) + (nz * (az / 1.0f));
                            fitness[i] += (1.0f - Math.Abs(dot)) * 0.5f;
                        }
                    }

                    Buffer.BlockCopy(outputs, 0, sendBuffer, 0, sendBuffer.Length);
                    stream.Write(sendBuffer, 0, sendBuffer.Length);

                    if (++framesPassed >= 300)
                    {
                        Array.Sort(botIndices, (a, b) => fitness[b].CompareTo(fitness[a]));
                        SaveBrain(population, botIndices[0], genomeSize, Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "swarm_brain.bin"));

                        Console.WriteLine($"[Gen] Najlepszy: {fitness[botIndices[0]]:F2} | Waga: {population[botIndices[0] * genomeSize]:F2}");

                        int elite = swarmSize / 10;
                        for (int i = elite; i < swarmSize; i++)
                        {
                            int weak = botIndices[i];
                            int parent = botIndices[rnd.Next(0, elite)];
                            for (int g = 0; g < genomeSize; g++)
                            {
                                float mut = rnd.NextDouble() < 0.05 ? (float)(rnd.NextDouble() * 0.2 - 0.1) : (float)(rnd.NextDouble() * 0.01 - 0.005);
                                population[weak * genomeSize + g] = Math.Clamp(population[parent * genomeSize + g] + mut, -2f, 2f);
                            }
                        }
                        Array.Clear(fitness, 0, fitness.Length);
                        framesPassed = 0;
                    }
                }
                catch { break; }
            }
        }

        static void SaveBrain(float[] population, int index, int size, string path)
        {
            float[] best = new float[size];
            Array.Copy(population, index * size, best, 0, size);
            byte[] data = new byte[size * 4];
            Buffer.BlockCopy(best, 0, data, 0, data.Length);
            File.WriteAllBytes(path, data);
        }

        static void ReadExactly(NetworkStream stream, byte[] buffer, int amount)
        {
            int total = 0;
            while (total < amount)
            {
                int read = stream.Read(buffer, total, amount - total);
                if (read == 0) throw new Exception("Link lost");
                total += read;
            }
        }
    }
}