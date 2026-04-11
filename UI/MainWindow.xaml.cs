// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace DevOnBike.Overfit.UI
{
    public partial class MainWindow : Window
    {
        private MnistPredictor _predictor;

        public MainWindow()
        {
            InitializeComponent();
            LoadModel();
        }

        private void LoadModel()
        {
            try
            {
                TxtStatus.Text = "Ładowanie wag GigaBestii...";

                // PODAJ ŚCIEŻKĘ DO SWOJEGO ZAPISANEGO MODELU
                // Pamiętaj, że prefiks musi zgadzać się z tym, co podałeś w MnistPredictor!
                var modelPrefix = @"d:\ml\bestia.bin";

                _predictor = new MnistPredictor(modelPrefix);
                TxtStatus.Text = "Bestia gotowa! Rysuj!";
            }
            catch (Exception ex)
            {
                TxtStatus.Text = "Błąd: " + ex.Message;
                TxtStatus.Foreground = Brushes.Red;
            }
        }

        // Zdarzenie wyzwalane za każdym razem, gdy użytkownik odrywa myszkę/rysik od płótna
        private void DrawingCanvas_MouseUp(object sender, MouseButtonEventArgs e)
        {
            if (_predictor == null || DrawingCanvas.Strokes.Count == 0) return;

            PredictDigit();
        }

        private void BtnClear_Click(object sender, RoutedEventArgs e)
        {
            DrawingCanvas.Strokes.Clear();
            TxtResult.Text = "?";
            TxtStatus.Text = "Bestia gotowa! Rysuj!";
        }

        private void PredictDigit()
        {
            try
            {
                var pixelData = GetMnistPixelsFromCanvas();

                // --- INFERENCJA (Nasza 48-sekundowa bestia w akcji) ---
                var stopwatch = Stopwatch.StartNew();
                var prediction = _predictor.Predict(pixelData);
                stopwatch.Stop();

                TxtResult.Text = prediction.ToString();
                TxtStatus.Text = $"Rozpoznano w {stopwatch.ElapsedMilliseconds} ms!";
            }
            catch (Exception ex)
            {
                TxtStatus.Text = "Błąd predykcji: " + ex.Message;
            }
        }

        // Magia konwersji WPF (280x280 InkCanvas) -> MNIST (28x28 double[])
        private float[] GetMnistPixelsFromCanvas()
        {
            // 1. Renderujemy Canvas do bitmapy
            var rtb = new RenderTargetBitmap(
            (int)DrawingCanvas.Width, (int)DrawingCanvas.Height,
            96d, 96d, PixelFormats.Default);
            rtb.Render(DrawingCanvas);

            // 2. Skalujemy w dół do 28x28 używając wbudowanego wysokiej jakości algorytmu
            var scaled = new TransformedBitmap(rtb, new ScaleTransform(
            28.0 / DrawingCanvas.Width,
            28.0 / DrawingCanvas.Height));

            // 3. Pobieramy surowe piksele
            var stride = 28 * 4; // 4 bajty na piksel (BGRA)
            var pixels = new byte[28 * stride];
            scaled.CopyPixels(pixels, stride, 0);

            // 4. Konwertujemy na format, który przyjmuje nasza sieć (tablica 784 x double, 0.0 - 1.0)
            var mnistData = new float[784];
            for (var y = 0; y < 28; y++)
            {
                for (var x = 0; x < 28; x++)
                {
                    var idx = y * stride + x * 4;

                    // W MNIST interesuje nas tylko jasność (bierzemy kanał Red, bo tło to czarny, a rysik biały)
                    var brightness = pixels[idx + 2];

                    // Normalizacja do 0.0 - 1.0
                    mnistData[y * 28 + x] = brightness / 255.0f;
                }
            }

            return mnistData;
        }

        protected override void OnClosed(EventArgs e)
        {
            _predictor?.Dispose(); // Pamiętaj o sprzątaniu!
            base.OnClosed(e);
        }
    }
}