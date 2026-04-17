// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Net;
using System.Net.Sockets;
using DevOnBike.Overfit.DeepLearning;

namespace DevOnBike.Overfit.Demo.Unity.Server
{
    internal static class Program
    {
        private static void Main(string[] args)
        {
            Console.WriteLine("=== DevOnBike Overfit AI Server (AVX-512) ===");

            var swarmSize = 100_000;
            var inputSize = 4; // Wektory odległości do celu i zagrożenia
            var outputSize = 2; // Wektory przyspieszenia (X, Y)

            // 1. Inicjalizacja sieci Overfit
            var brain = new LinearLayer(inputSize, outputSize);
            brain.Eval();

            // 2. Pre-alokacja potężnych buforów (Zero GC podczas działania!)
            var bytesToRead = swarmSize * inputSize * sizeof(float); // ~1.6 MB
            var bytesToSend = swarmSize * outputSize * sizeof(float); // ~0.8 MB

            var receiveBuffer = new byte[bytesToRead];
            var sendBuffer = new byte[bytesToSend];

            var inputs = new float[swarmSize * inputSize];
            var outputs = new float[swarmSize * outputSize];

            // 3. Konfiguracja szybkiego gniazda TCP
            var listener = new TcpListener(IPAddress.Loopback, 5000);
            listener.Start();
            Console.WriteLine($"Nasłuchiwanie na porcie TCP 5000. Oczekuję na połączenie od Unity...");

            using var client = listener.AcceptTcpClient();
            client.NoDelay = true; // Wyłącza algorytm Nagle'a (kluczowe dla mikrosekundowych opóźnień!)
            using var stream = client.GetStream();

            Console.WriteLine("Unity połączone! Rozpoczynam inferencję z prędkością 144 Hz...");

            // Główna pętla serwera AI
            while (true)
            {
                try
                {
                    // 1. ODBIÓR DANYCH (Czekamy, aż Unity wyśle całe 1.6 MB)
                    ReadExactly(stream, receiveBuffer, bytesToRead);

                    // 2. DESERIALIZACJA (Ekstremalnie szybkie kopiowanie bitów)
                    Buffer.BlockCopy(receiveBuffer, 0, inputs, 0, bytesToRead);

                    // 3. INFERENCJA OVERFIT (AVX-512 wkracza do akcji!)
                    brain.ForwardInference(inputs.AsSpan(), outputs.AsSpan());

                    // 4. SERIALIZACJA I WYSYŁKA (Wysyłamy 0.8 MB decyzji z powrotem)
                    Buffer.BlockCopy(outputs, 0, sendBuffer, 0, bytesToSend);
                    stream.Write(sendBuffer, 0, bytesToSend);
                }
                catch (Exception)
                {
                    Console.WriteLine("Połączenie przerwane. Unity wyłączone.");
                    break;
                }
            }
        }

        private static void ReadExactly(NetworkStream stream, byte[] buffer, int amount)
        {
            var totalRead = 0;

            while (totalRead < amount)
            {
                var read = stream.Read(buffer, totalRead, amount - totalRead);
                
                if (read == 0)
                {
                    throw new Exception("Socket closed");
                }
                
                totalRead += read;
            }
        }
    }
}