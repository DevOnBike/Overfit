// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using Android.Animation;
using Android.Content;
using Android.Graphics;
using Android.Views;
using Android.Views.Animations;

namespace DevOnBike.OverfitChat
{
    /// <summary>
    /// Animated "mesh gradient" backdrop — a dark base with a few soft color blobs drifting in slow
    /// circles (the trendy AI-app look). Pure Canvas, no image assets. Pauses while the model generates
    /// so it never steals CPU from decode.
    /// </summary>
    public sealed class AnimatedGradientView : View
    {
        private readonly Paint _blobPaint = new() { AntiAlias = true, Dither = true };
        private readonly Paint _basePaint = new();
        private ValueAnimator? _animator;
        private float _t;

        // color, baseX, baseY, radiusFactor, orbitSpeed, phase
        private static readonly (Color Color, float Fx, float Fy, float Fr, float Speed, float Phase)[] Blobs =
        {
            (Color.ParseColor("#7C3AED"), 0.22f, 0.18f, 0.95f, 1.00f, 0.0f),  // violet
            (Color.ParseColor("#2563EB"), 0.82f, 0.30f, 1.05f, 0.80f, 2.1f),  // blue
            (Color.ParseColor("#EC4899"), 0.60f, 0.88f, 1.00f, 1.20f, 4.2f),  // pink
        };

        public AnimatedGradientView(Context context) : base(context)
        {
            _basePaint.Color = Color.ParseColor("#070A1A");

            _animator = ValueAnimator.OfFloat(0f, 1f);
            _animator.SetDuration(18000);
            _animator.RepeatCount = ValueAnimator.Infinite;
            _animator.SetInterpolator(new LinearInterpolator());
            _animator.Update += (_, _) =>
            {
                _t = (_animator?.AnimatedFraction ?? 0f) * (float)(Math.PI * 2.0);
                Invalidate();
            };
            _animator.Start();
        }

        public void Pause() => _animator?.Pause();

        public void Resume() => _animator?.Resume();

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
                var cx = w * (b.Fx + 0.13f * (float)Math.Cos((_t * b.Speed) + b.Phase));
                var cy = h * (b.Fy + 0.13f * (float)Math.Sin((_t * b.Speed) + b.Phase));
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
