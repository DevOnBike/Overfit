// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using Android.App;
using Android.Content;
using Android.Content.PM;
using Android.Graphics;
using Android.Graphics.Drawables;
using Android.OS;
using Android.Views;
using Android.Views.Animations;
using Android.Widget;
using DevOnBike.Overfit.LanguageModels;

namespace DevOnBike.OverfitChat
{
    /// <summary>
    /// Overfit Chat — pick a GGUF, then chat with streaming tokens over an animated mesh-gradient backdrop.
    /// Pure .NET-for-Android, code-built UI (no XML), inference via the in-process <see cref="OverfitClient"/>.
    /// </summary>
    [Activity(
        Label = "Overfit",
        MainLauncher = true,
        // AdjustResize: shrink the layout when the IME shows so the input row rides up above the keyboard.
        WindowSoftInputMode = SoftInput.AdjustResize,
        ConfigurationChanges = ConfigChanges.Orientation | ConfigChanges.ScreenSize | ConfigChanges.KeyboardHidden)]
    public class MainActivity : Activity
    {
        private const int RequestPickModel = 42;

        private AnimatedGradientView _gradient = null!;
        private LinearLayout _chatContainer = null!;
        private ScrollView _scroll = null!;
        private EditText _input = null!;
        private TextView _status = null!;

        private OverfitClient? _client;
        private volatile bool _busy;

        protected override void OnCreate(Bundle? savedInstanceState)
        {
            base.OnCreate(savedInstanceState);

            // Edge-to-edge: the gradient draws behind transparent system bars; we inset the CONTENT and
            // pad for the IME via the insets listener below — the reliable way to keep the input row above
            // the keyboard on modern Android (plain AdjustResize is flaky once edge-to-edge is enforced).
            Window!.SetDecorFitsSystemWindows(false);
            Window!.SetStatusBarColor(Color.Transparent);
            Window!.SetNavigationBarColor(Color.Transparent);

            var root = new FrameLayout(this);

            _gradient = new AnimatedGradientView(this);
            root.AddView(_gradient, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MatchParent, ViewGroup.LayoutParams.MatchParent));

            var column = new LinearLayout(this) { Orientation = Orientation.Vertical };
            // System bars are now opaque (window fits them), so only a small inner pad is needed.
            column.SetPadding(0, Dp(6), 0, Dp(6));
            root.AddView(column, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MatchParent, ViewGroup.LayoutParams.MatchParent));

            // Pad the content for the status bar (top) and for the nav bar OR the keyboard (bottom,
            // whichever is taller) — so when the IME opens, the input row lifts above it.
            root.SetOnApplyWindowInsetsListener(new InsetsListener((_, insets) =>
            {
                var bars = insets.GetInsets(WindowInsets.Type.SystemBars());
                var ime = insets.GetInsets(WindowInsets.Type.Ime());
                column.SetPadding(0, bars.Top + Dp(6), 0, Math.Max(bars.Bottom, ime.Bottom) + Dp(6));
                return insets;
            }));

            column.AddView(BuildHeader());
            column.AddView(BuildChatScroll(), new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MatchParent, 0, 1f));
            column.AddView(BuildInputRow());

            SetContentView(root);
            root.RequestApplyInsets();

            AddBubble("Tap “Load model”, pick a .gguf, then say hi. Tokens stream as they generate.", isUser: false);

            // Auto-load a previously picked model so you don't re-pick every launch.
            var cached = System.IO.Path.Combine(GetExternalFilesDir(null)!.AbsolutePath, "model.gguf");
            if (System.IO.File.Exists(cached))
            {
                LoadModelFromPath(cached, announce: true);
            }
        }

        // ─────────────────────────── UI builders ───────────────────────────

        private View BuildHeader()
        {
            var bar = new LinearLayout(this) { Orientation = Orientation.Horizontal };
            bar.SetGravity(GravityFlags.CenterVertical);
            bar.SetPadding(Dp(18), Dp(6), Dp(14), Dp(10));

            var title = new TextView(this) { Text = "Overfit" };
            title.SetTextColor(Color.White);
            title.TextSize = 24f;
            title.SetTypeface(Typeface.DefaultBold, TypefaceStyle.Bold);
            title.SetShadowLayer(Dp(18), 0, 0, Color.ParseColor("#7C3AED"));
            bar.AddView(title, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WrapContent, ViewGroup.LayoutParams.WrapContent));

            _status = new TextView(this) { Text = "no model" };
            _status.SetTextColor(Color.ParseColor("#CBD5E1"));
            _status.TextSize = 12f;
            _status.SetPadding(Dp(12), Dp(6), Dp(12), Dp(6));
            _status.Background = RoundedRect(Color.Argb(46, 255, 255, 255), 14f);
            var statusLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WrapContent, ViewGroup.LayoutParams.WrapContent) { Weight = 1f };
            statusLp.SetMargins(Dp(12), 0, Dp(12), 0);
            statusLp.Gravity = GravityFlags.CenterVertical;
            bar.AddView(_status, statusLp);

            var load = new Button(this) { Text = "Load model" };
            load.SetTextColor(Color.White);
            load.SetAllCaps(false);
            load.TextSize = 13f;
            load.Background = GradientPill(18f);
            load.SetPadding(Dp(16), Dp(8), Dp(16), Dp(8));
            load.Click += (_, _) => PickModel();
            bar.AddView(load, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WrapContent, ViewGroup.LayoutParams.WrapContent));

            return bar;
        }

        private View BuildChatScroll()
        {
            _scroll = new ScrollView(this);
            _scroll.FillViewport = true;
            _chatContainer = new LinearLayout(this) { Orientation = Orientation.Vertical };
            _chatContainer.SetPadding(Dp(4), Dp(6), Dp(4), Dp(6));
            _scroll.AddView(_chatContainer, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MatchParent, ViewGroup.LayoutParams.WrapContent));
            return _scroll;
        }

        private View BuildInputRow()
        {
            var row = new LinearLayout(this) { Orientation = Orientation.Horizontal };
            row.SetGravity(GravityFlags.CenterVertical);
            row.SetPadding(Dp(12), Dp(6), Dp(12), Dp(10));

            _input = new EditText(this) { Hint = "Message…" };
            _input.SetTextColor(Color.White);
            _input.SetHintTextColor(Color.ParseColor("#94A3B8"));
            _input.Background = RoundedRect(Color.Argb(54, 255, 255, 255), 22f);
            _input.SetPadding(Dp(18), Dp(12), Dp(18), Dp(12));
            _input.SetMaxLines(4);
            var inputLp = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WrapContent, 1f);
            row.AddView(_input, inputLp);

            var send = new Button(this) { Text = "➤" };
            send.SetTextColor(Color.White);
            send.TextSize = 18f;
            send.Background = GradientCircle();
            var sendLp = new LinearLayout.LayoutParams(Dp(52), Dp(52));
            sendLp.SetMargins(Dp(10), 0, 0, 0);
            send.Click += (_, _) =>
            {
                send.Animate()!.ScaleX(0.86f).ScaleY(0.86f).SetDuration(80)!
                    .WithEndAction(new Java.Lang.Runnable(() => send.Animate()!.ScaleX(1f).ScaleY(1f).SetDuration(120)!.Start()))!
                    .Start();
                OnSend();
            };
            row.AddView(send, sendLp);

            return row;
        }

        // ─────────────────────────── chat ───────────────────────────

        private TextView AddBubble(string text, bool isUser)
        {
            var tv = new TextView(this) { Text = text };
            tv.SetTextColor(Color.White);
            tv.TextSize = 16f;
            tv.SetPadding(Dp(15), Dp(11), Dp(15), Dp(11));
            tv.Background = isUser
                ? new GradientDrawable(GradientDrawable.Orientation.TlBr,
                        new[] { Color.ParseColor("#7C3AED").ToArgb(), Color.ParseColor("#2563EB").ToArgb() })
                    .Also(d => d.SetCornerRadius(Dp(20)))
                : RoundedRect(Color.Argb(38, 255, 255, 255), 20f);

            var lp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WrapContent, ViewGroup.LayoutParams.WrapContent);
            lp.SetMargins(isUser ? Dp(64) : Dp(12), Dp(6), isUser ? Dp(12) : Dp(64), Dp(6));
            lp.Gravity = isUser ? GravityFlags.End : GravityFlags.Start;
            tv.LayoutParameters = lp;

            _chatContainer.AddView(tv);

            tv.Alpha = 0f;
            tv.TranslationY = Dp(18);
            tv.Animate()!.Alpha(1f).TranslationY(0).SetDuration(260)
                .SetInterpolator(new DecelerateInterpolator())!.Start();

            ScrollToBottom();
            return tv;
        }

        private void OnSend()
        {
            var text = _input.Text?.Trim();
            if (string.IsNullOrEmpty(text))
            {
                return;
            }
            if (_client is null)
            {
                Toast.MakeText(this, "Load a model first.", ToastLength.Short)!.Show();
                return;
            }
            if (_busy)
            {
                return;
            }

            _busy = true;
            _gradient.Pause();
            _input.Text = string.Empty;

            AddBubble(text, isUser: true);
            var assistant = AddBubble("▌", isUser: false);
            var sb = new StringBuilder();
            var client = _client;

            System.Threading.Tasks.Task.Run(() =>
            {
                try
                {
                    client.Send(text, onText: token =>
                    {
                        sb.Append(token);
                        RunOnUiThread(() =>
                        {
                            assistant.Text = sb.ToString() + "▌";
                            ScrollToBottom();
                        });
                    });
                    RunOnUiThread(() => assistant.Text = sb.Length > 0 ? sb.ToString() : "(no output)");
                }
                catch (Exception ex)
                {
                    RunOnUiThread(() => assistant.Text = "⚠ " + ex.Message);
                }
                finally
                {
                    RunOnUiThread(() =>
                    {
                        _busy = false;
                        _gradient.Resume();
                    });
                }
            });
        }

        private void ScrollToBottom() => _scroll.Post(() => _scroll.FullScroll(FocusSearchDirection.Down));

        // ─────────────────────────── model loading ───────────────────────────

        private void PickModel()
        {
            var intent = new Intent(Intent.ActionOpenDocument);
            intent.AddCategory(Intent.CategoryOpenable!);
            intent.SetType("*/*");
            StartActivityForResult(Intent.CreateChooser(intent, "Pick a .gguf model"), RequestPickModel);
        }

        protected override void OnActivityResult(int requestCode, Result resultCode, Intent? data)
        {
            base.OnActivityResult(requestCode, resultCode, data);
            if (requestCode == RequestPickModel && resultCode == Result.Ok && data?.Data is { } uri)
            {
                LoadModelFromUri(uri);
            }
        }

        private void LoadModelFromUri(Android.Net.Uri uri)
        {
            SetStatus("copying…");
            System.Threading.Tasks.Task.Run(() =>
            {
                try
                {
                    var dir = GetExternalFilesDir(null)!.AbsolutePath;
                    var dest = System.IO.Path.Combine(dir, "model.gguf");
                    using (var input = ContentResolver!.OpenInputStream(uri))
                    using (var output = System.IO.File.Create(dest))
                    {
                        input!.CopyTo(output);
                    }

                    LoadModelFromPath(dest, announce: true);
                }
                catch (Exception ex)
                {
                    SetStatus("load failed");
                    RunOnUiThread(() => AddBubble("⚠ Couldn't load model: " + ex.Message, isUser: false));
                }
            });
        }

        // Loads a model already on local disk (the SAF copy, or a previous session's copy on startup).
        private void LoadModelFromPath(string path, bool announce)
        {
            SetStatus("loading…");
            System.Threading.Tasks.Task.Run(() =>
            {
                try
                {
                    // quantize:false keeps Q4_K resident (measured ~4× faster on-device than the Q8 requant path).
                    var client = OverfitClient.LoadGguf(path, mmap: true, quantize: false);
                    RunOnUiThread(() =>
                    {
                        _client?.Dispose();
                        _client = client;
                        SetStatus("ready ✓");
                        if (announce)
                        {
                            AddBubble("Model loaded. Ask me anything.", isUser: false);
                        }
                    });
                }
                catch (Exception ex)
                {
                    SetStatus("load failed");
                    RunOnUiThread(() => AddBubble("⚠ Couldn't load model: " + ex.Message, isUser: false));
                }
            });
        }

        private void SetStatus(string text) => RunOnUiThread(() => _status.Text = text);

        protected override void OnDestroy()
        {
            _client?.Dispose();
            _client = null;
            base.OnDestroy();
        }

        // Bridges a lambda to the platform inset callback (avoids an AndroidX dependency).
        private sealed class InsetsListener : Java.Lang.Object, View.IOnApplyWindowInsetsListener
        {
            private readonly Func<View, WindowInsets, WindowInsets> _handler;

            public InsetsListener(Func<View, WindowInsets, WindowInsets> handler) => _handler = handler;

            public WindowInsets OnApplyWindowInsets(View v, WindowInsets insets) => _handler(v, insets);
        }

        // ─────────────────────────── helpers ───────────────────────────

        private int Dp(float v) => (int)((v * (Resources?.DisplayMetrics?.Density ?? 2f)) + 0.5f);

        private GradientDrawable RoundedRect(Color fill, float radiusDp)
        {
            var d = new GradientDrawable();
            d.SetColor(fill);
            d.SetCornerRadius(Dp(radiusDp));
            return d;
        }

        private GradientDrawable GradientPill(float radiusDp)
        {
            var d = new GradientDrawable(GradientDrawable.Orientation.LeftRight,
                new[] { Color.ParseColor("#7C3AED").ToArgb(), Color.ParseColor("#EC4899").ToArgb() });
            d.SetCornerRadius(Dp(radiusDp));
            return d;
        }

        private GradientDrawable GradientCircle()
        {
            var d = new GradientDrawable(GradientDrawable.Orientation.TlBr,
                new[] { Color.ParseColor("#7C3AED").ToArgb(), Color.ParseColor("#2563EB").ToArgb() });
            d.SetShape(ShapeType.Oval);
            return d;
        }
    }

    internal static class FluentExtensions
    {
        // Tiny "configure inline" helper so a GradientDrawable can be built + tweaked in an expression.
        public static T Also<T>(this T self, Action<T> configure)
        {
            configure(self);
            return self;
        }
    }
}
