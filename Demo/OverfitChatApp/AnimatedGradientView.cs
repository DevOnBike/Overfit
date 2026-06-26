// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using Android.Animation;
using Android.Content;
using Android.Graphics;
using Android.OS;
using Android.Views;
using Android.Views.Animations;

namespace DevOnBike.OverfitChat
{
    /// <summary>
    /// Animated "mesh gradient" backdrop — a dark base with soft colour blobs drifting on wandering
    /// (Lissajous) paths. Driven by an accumulated monotonic clock, NOT a looping 0→1 fraction, so the
    /// motion never jumps/"restarts" at a loop boundary. The bottom blob roams up to mid-screen. Pauses
    /// while the model generates so it never steals CPU from decode.
    /// </summary>
    public sealed class AnimatedGradientView : View
    {
        private readonly Paint _blobPaint = new() { AntiAlias = true, Dither = true };
        private readonly Paint _basePaint = new();
        private ValueAnimator? _animator;
        private float _t;          // continuously accumulated seconds (never resets)
        private long _lastMs;

        // color, baseX, baseY, radiusFactor, ampX, ampY, freqX, freqY, phase  (amp/base are fractions of w/h;
        // freqX != freqY → the path is a non-repeating wander, not a circle).
        private static readonly (Color Color, float Fx, float Fy, float Fr, float AmpX, float AmpY, float FreqX, float FreqY, float Phase)[] Blobs =
        {
            (Color.ParseColor("#7C3AED"), 0.22f, 0.18f, 0.95f, 0.12f, 0.10f, 0.13f, 0.17f, 0.0f), // violet
            (Color.ParseColor("#2563EB"), 0.82f, 0.30f, 1.05f, 0.10f, 0.12f, 0.11f, 0.19f, 2.1f), // blue
            (Color.ParseColor("#EC4899"), 0.50f, 0.85f, 1.00f, 0.24f, 0.38f, 0.15f, 0.23f, 4.2f), // bottom: wide wander up to ~half screen
        };

        public AnimatedGradientView(Context context) : base(context)
        {
            _basePaint.Color = Color.ParseColor("#070A1A");

            _animator = ValueAnimator.OfFloat(0f, 1f);
            _animator.SetDuration(10000);
            _animator.RepeatCount = ValueAnimator.Infinite;
            _animator.SetInterpolator(new LinearInterpolator());
            _animator.Update += (_, _) =>
            {
                var now = SystemClock.UptimeMillis();
                if (_lastMs == 0)
                {
                    _lastMs = now;
                }
                _t += (now - _lastMs) / 1000f;   // advance by real elapsed time → seamless, no loop reset
                _lastMs = now;
                Invalidate();
            };
            _animator.Start();
        }

        public void Pause() => _animator?.Pause();

        public void Resume()
        {
            _lastMs = 0;          // drop the stale delta so resuming doesn't jump _t forward
            _animator?.Resume();
        }

        protected override void OnDraw(Canvas canvas)
        {
            int w = Width, h = Height;
            if (w == 0 || h == 0)
            {
                return;
            }

            canvas.DrawRect(0, 0, w, h, _basePaint);

            var minDim = Math.Min(w, h);
            foreach (var b in Blobs)
            {
                var cx = w * (b.Fx + (b.AmpX * (float)Math.Cos((_t * b.FreqX) + b.Phase)));
                var cy = h * (b.Fy + (b.AmpY * (float)Math.Sin((_t * b.FreqY) + b.Phase)));
                var radius = minDim * b.Fr;

                var colors = new[]
                {
                    Color.Argb(165, b.Color.R, b.Color.G, b.Color.B).ToArgb(),
                    Color.Argb(0, b.Color.R, b.Color.G, b.Color.B).ToArgb(),
                };

                _blobPaint.SetShader(new RadialGradient(
                    cx, cy, radius, colors, new[] { 0f, 1f }, Shader.TileMode.Clamp!));
                canvas.DrawRect(0, 0, w, h, _blobPaint);
            }
        }

        protected override void OnDetachedFromWindow()
        {
            _animator?.Cancel();
            _animator = null;
            base.OnDetachedFromWindow();
        }
    }
}
