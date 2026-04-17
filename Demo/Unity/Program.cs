using System.Net;
using System.Net.Sockets;
using DevOnBike.Overfit.DeepLearning;

namespace DevOnBike.Overfit.Demo.Unity.Server
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("=== DevOnBike Overfit AI Server (AVX-512) ===");

            // Konfiguracja architektury
            var swarmSize = 100_000;
            var inputSize = 4;  // Sensory: [celX, celZ, ucieczkaX, ucieczkaZ]
            var outputSize = 2; // Akcja: [przyspieszenieX, przyspieszenieZ]

            // 1. Inicjalizacja sieci z Twojego silnika
            Console.WriteLine("Budowanie grafu i alokacja pamiêci L1/L2...");
            var brain = new LinearLayer(inputSize, outputSize);
            brain.Eval();

            // 2. Pre-alokacja potężnych buforów (Zero GC Allocation w pętli)
            var bytesToRead = swarmSize * inputSize * sizeof(float); // ~1.6 MB
            var bytesToSend = swarmSize * outputSize * sizeof(float); // ~0.8 MB

            var receiveBuffer = new byte[bytesToRead];
            var sendBuffer = new byte[bytesToSend];

            var inputs = new float[swarmSize * inputSize];
            var outputs = new float[swarmSize * outputSize];

            // 3. Konfiguracja serwera TCP na localhost
            var listener = new TcpListener(IPAddress.Loopback, 5000);
            listener.Start();
            Console.WriteLine("Nasluchiwanie na porcie TCP 5000 gotowe.\n");

            // ====================================================================
            // ZEWNĘTRZNA PĘTLA SERWERA (Czeka na nowe połączenia)
            // ====================================================================
            while (true)
            {
                Console.WriteLine("Oczekuje na polaczenie od gry Unity...");

                using var client = listener.AcceptTcpClient();
                client.NoDelay = true; // Krytyczne: wyłącza buforowanie opóźnień Nagle'a
                using var stream = client.GetStream();

                Console.WriteLine("Unity połączone! Rozpoczynam inferencję z prędkością 144 Hz...");

                // ====================================================================
                // WEWNĘTRZNA PĘTLA GRY (Odbiera i wysyła pakiety do tego klienta)
                // ====================================================================
                try
                {
                    while (true)
                    {
                        // 1. ODBIÓR 1.6 MB od Unity
                        ReadExactly(stream, receiveBuffer, bytesToRead);
                        Buffer.BlockCopy(receiveBuffer, 0, inputs, 0, bytesToRead);

                        // 2. INFERENCJA OVERFIT (Sprawdzamy czy silnik przyjmie pełen płaski Span)
                        // Jeśli tu wyskoczy błąd, zobaczysz go pięknie w konsoli.
                        for (var i = 0; i < swarmSize; i++)
                        {
                            brain.ForwardInference(
                                inputs.AsSpan(i * inputSize, inputSize),
                                outputs.AsSpan(i * outputSize, outputSize)
                            );
                        }

                        // ---> DODAJ TEN BLOK KODU <---
                        // NADPISANIE WYNIKÓW DLA DEMA (Zachowanie Orbitalne / Tornado)
                        for (var i = 0; i < swarmSize; i++)
                        {
                            var inIdx = i * 4;
                            var outIdx = i * 2;

                            var tx = inputs[inIdx + 0]; // Wektor do celu X
                            var tz = inputs[inIdx + 1]; // Wektor do celu Z
                            var px = inputs[inIdx + 2]; // Wektor ucieczki X od Predatora
                            var pz = inputs[inIdx + 3]; // Wektor ucieczki Z od Predatora

                            // Obliczamy odległość od celu
                            // Używamy float zamiast MathF jeśli korzystasz z wcześniejszej wersji .NET w konfiguracji
                            var dist = (float)Math.Sqrt(tx * tx + tz * tz);
                            if (dist < 0.001f)
                            {
                                dist = 0.001f; // Zabezpieczenie
                            }

                            // 1. WEKTOR PROSTOPADŁY (To on obraca rój o 90 stopni i tworzy wir)
                            var tangX = -tz;
                            var tangZ = tx;

                            // 2. PARAMETRY ORBITY (Możesz tu eksperymentować!)
                            var orbitRadius = 0.15f; // Zmniejszyliśmy z 1.0f -> będą kręcić się tuż przy samym celu!
                            var pullForce = (dist - orbitRadius) * 15.0f; // Mocniej przyciągamy je do orbity, żeby się nie rozbiegły
                            var swirlForce = 8.0f; // Podkręcamy prędkość wirowania (Tornado!)

                            // Znormalizowany wektor prosto do celu
                            var dirX = tx / dist;
                            var dirZ = tz / dist;

                            // 3. SYNTEZA SIŁ: Ciąg do orbity + Kręcenie się + Ucieczka przed Predatorem
                            outputs[outIdx + 0] = (dirX * pullForce) + (tangX * swirlForce) + px;
                            outputs[outIdx + 1] = (dirZ * pullForce) + (tangZ * swirlForce) + pz;

                            // Bezpiecznik fizyki (Invalid AABB)
                            outputs[outIdx + 0] = Math.Clamp(outputs[outIdx + 0], -1f, 1f);
                            outputs[outIdx + 1] = Math.Clamp(outputs[outIdx + 1], -1f, 1f);
                        }
                        // -----------------------------

                        // 3. WYSYŁKA 0.8 MB do Unity
                        Buffer.BlockCopy(outputs, 0, sendBuffer, 0, bytesToSend);
                        stream.Write(sendBuffer, 0, bytesToSend);
                    }
                }
                catch (Exception ex)
                {
                    // Łapiemy błędy (np. zamknięcie gry w Unity lub błąd w silniku)
                    Console.WriteLine("\n[SESJA ZAKOŃCZONA LUB BŁĄD SILNIKA]");
                    Console.WriteLine(ex.Message);

                    // Zabezpieczenie: jeśli to błąd silnika (ArgumentOutOfRange), chcemy widzieć w której linii:
                    if (!ex.Message.Contains("zamknął połączenie") && !ex.Message.Contains("forcibly closed"))
                    {
                        Console.WriteLine(ex.StackTrace);
                    }

                    Console.WriteLine("Resetuje strumien...\n");
                }
            }
        }

        /// <summary>
        /// Zapewnia, że odczytamy z TCP dokładnie tyle bajtów, ile potrzebujemy (nawet jeśli system podzieli pakiety)
        /// </summary>
        static void ReadExactly(NetworkStream stream, byte[] buffer, int amount)
        {
            var totalRead = 0;
            while (totalRead < amount)
            {
                var read = stream.Read(buffer, totalRead, amount - totalRead);
                if (read == 0)
                {
                    throw new Exception("Klient Unity zamknął połączenie TCP.");
                }
                totalRead += read;
            }
        }
    }
}