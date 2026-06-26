// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using Android.Animation;
using Android.Content;
using Android.Graphics;
using Android.Views.Animations;
using Android.Widget;

namespace DevOnBike.OverfitChat
{
    /// <summary>
    /// A TextView whose glyphs are filled with the brand gradient, animated so the colours flow across the
    /// word — a living "aurora text" headline. The drop-shadow glow set by the caller still applies.
    /// </summary>
    public sealed class ShimmerTextView : TextView
    {
        private static readonly int[] GradientColors =
        {
            Color.ParseColor("#A78BFA").ToArgb(), // violet
            Color.ParseColor("#EC4899").ToArgb(), // pink
            Color.ParseColor("#22D3EE").ToArgb(), // cyan
            Color.ParseColor("#A78BFA").ToArgb(), // back to violet → seamless loop
        };

        private readonly Matrix _matrix = new();
        private LinearGradient? _shader;
        private ValueAnimator? _animator;
        private int _width;

        public ShimmerTextView(Context context) : base(context)
        {
        }

        protected override void OnSizeChanged(int w, int h, int oldw, int oldh)
        {
            base.OnSizeChanged(w, h, oldw, oldh);
            if (w == 0)
            {
                return;
            }

            _width = w;
            _shader = new LinearGradient(0, 0, w, 0, GradientColors, null, Shader.TileMode.Repeat!);
            Paint!.SetShader(_shader);
            StartAnimation();
        }

        private void StartAnimation()
        {
            if (_animator != null)
            {
                return;
            }

            _animator = ValueAnimator.OfFloat(0f, 1f);
            _animator.SetDuration(3600);
            _animator.RepeatCount = ValueAnimator.Infinite;
            _animator.SetInterpolator(new LinearInterpolator());
            _animator.Update += (_, _) =>
            {
                var fraction = _animator?.AnimatedFraction ?? 0f;
                _matrix.SetTranslate(-fraction * _width, 0); // one full period per loop → seamless flow
                _shader?.SetLocalMatrix(_matrix);
                Invalidate();
            };
            _animator.Start();
        }

        protected override void OnDetachedFromWindow()
        {
            _animator?.Cancel();
            _animator = null;
            base.OnDetachedFromWindow();
        }
    }
}
