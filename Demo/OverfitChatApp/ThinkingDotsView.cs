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
    /// Three dots pulsing in a wave — the "model is thinking" indicator shown in the assistant bubble
    /// until the first token streams in. Pure Canvas + a looping animator.
    /// </summary>
    public sealed class ThinkingDotsView : View
    {
        private readonly Paint _paint = new() { AntiAlias = true, Color = Color.White };
        private ValueAnimator? _animator;
        private float _phase;

        public ThinkingDotsView(Context context) : base(context)
        {
            _animator = ValueAnimator.OfFloat(0f, 1f);
            _animator.SetDuration(1100);
            _animator.RepeatCount = ValueAnimator.Infinite;
            _animator.SetInterpolator(new LinearInterpolator());
            _animator.Update += (_, _) =>
            {
                _phase = (_animator?.AnimatedFraction ?? 0f) * (float)(Math.PI * 2.0);
                Invalidate();
            };
            _animator.Start();
        }

        protected override void OnMeasure(int widthMeasureSpec, int heightMeasureSpec)
        {
            var d = Resources?.DisplayMetrics?.Density ?? 2f;
            SetMeasuredDimension((int)(36 * d), (int)(18 * d));
        }

        protected override void OnDraw(Canvas canvas)
        {
            var d = Resources?.DisplayMetrics?.Density ?? 2f;
            var radius = 3.3f * d;
            var gap = 11f * d;
            var cy = Height / 2f;
            var x0 = radius + (3f * d);

            for (var i = 0; i < 3; i++)
            {
                var s = 0.5f + (0.5f * (float)Math.Sin(_phase - (i * 0.9f)));
                _paint.Alpha = (int)(110 + (145 * s));
                canvas.DrawCircle(x0 + (i * gap), cy, radius * (0.65f + (0.55f * s)), _paint);
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
