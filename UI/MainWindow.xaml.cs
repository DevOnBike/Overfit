// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using DevOnBike.Overfit.Diagnostics;

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
                TxtStatus.Text = "Loading Beast weights...";

                // PROVIDE THE PATH TO YOUR SAVED MODEL
                // Make sure the prefix matches what you passed to MnistPredictor!
                var modelPrefix = @"d:\ml\bestia.bin";

                _predictor = new MnistPredictor(modelPrefix);
                TxtStatus.Text = "Beast ready! Draw!";
            }
            catch (Exception ex)
            {
                TxtStatus.Text = "Error: " + ex.Message;
                TxtStatus.Foreground = Brushes.Red;
            }
        }

        // Event fired every time the user lifts the mouse button or stylus from the canvas
        private void DrawingCanvas_MouseUp(object sender, MouseButtonEventArgs e)
        {
            if (_predictor == null || DrawingCanvas.Strokes.Count == 0)
            {
                return;
            }

            PredictDigit();
        }

        private void BtnClear_Click(object sender, RoutedEventArgs e)
        {
            DrawingCanvas.Strokes.Clear();
            TxtResult.Text = "?";
            TxtStatus.Text = "Beast ready! Draw!";
        }

        private void PredictDigit()
        {
            try
            {
                var pixelData = GetMnistPixelsFromCanvas();

                // --- INFERENCE (our 48-second beast in action) ---
                var stopwatch = ValueStopwatch.StartNew();
                var prediction = _predictor.Predict(pixelData);
                var elapsed = stopwatch.GetElapsedTime();

                TxtResult.Text = prediction.ToString();
                TxtStatus.Text = $"Recognized in {elapsed.TotalMilliseconds} ms!";
            }
            catch (Exception ex)
            {
                TxtStatus.Text = "Prediction error: " + ex.Message;
            }
        }

        // WPF conversion magic: (280x280 InkCanvas) -> MNIST (28x28 double[])
        private float[] GetMnistPixelsFromCanvas()
        {
            // 1. Render the canvas to a bitmap
            var rtb = new RenderTargetBitmap(
            (int)DrawingCanvas.Width, (int)DrawingCanvas.Height,
            96d, 96d, PixelFormats.Default);
            rtb.Render(DrawingCanvas);

            // 2. Scale down to 28x28 using the built-in high-quality scaling algorithm
            var scaled = new TransformedBitmap(rtb, new ScaleTransform(
            28.0 / DrawingCanvas.Width,
            28.0 / DrawingCanvas.Height));

            // 3. Extract raw pixels
            var stride = 28 * 4; // 4 bytes per pixel (BGRA)
            var pixels = new byte[28 * stride];
            scaled.CopyPixels(pixels, stride, 0);

            // 4. Convert to the format expected by the network (784-element float array, 0.0 - 1.0)
            var mnistData = new float[784];
            for (var y = 0; y < 28; y++)
            {
                for (var x = 0; x < 28; x++)
                {
                    var idx = y * stride + x * 4;

                    // In MNIST we only care about brightness (we take the Red channel because background is black and the stylus is white)
                    var brightness = pixels[idx + 2];

                    // Normalize to 0.0 - 1.0
                    mnistData[y * 28 + x] = brightness / 255.0f;
                }
            }

            return mnistData;
        }

        protected override void OnClosed(EventArgs e)
        {
            _predictor?.Dispose(); // Remember to clean up!
            base.OnClosed(e);
        }
    }
}