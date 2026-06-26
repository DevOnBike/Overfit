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
using DevOnBike.Overfit.LanguageModels.Loading;

namespace DevOnBike.OverfitChat
{
    /// <summary>
    /// Overfit Chat — a first-run onboarding (pick a GGUF) → streaming chat, over an animated mesh-gradient
    /// backdrop. Single-Activity with screen swapping; the chosen model is remembered so returning users go
    /// straight to chat. NoActionBar theme (the app draws its own header). Inference via <see cref="OverfitClient"/>.
    /// </summary>
    [Activity(
        Label = "OverThink",
        Theme = "@android:style/Theme.Material.NoActionBar",
        MainLauncher = true,
        WindowSoftInputMode = SoftInput.AdjustResize,
        ConfigurationChanges = ConfigChanges.Orientation | ConfigChanges.ScreenSize | ConfigChanges.KeyboardHidden)]
    public class MainActivity : Activity
    {
        private const int RequestPickModel = 42;
        private const int MaxMessageChars = 200;

        private AnimatedGradientView _gradient = null!;
        private FrameLayout _screenRoot = null!;

        // Chat-screen views (rebuilt by ShowChat).
        private LinearLayout _chatContainer = null!;
        private ScrollView _scroll = null!;
        private EditText _input = null!;
        private Button _send = null!;
        private TextView _counter = null!;
        private TextView _subtitle = null!;

        // Welcome-screen controls — the model loads HERE; we only enter chat once it's ready.
        private LinearLayout _modelSelectField = null!;
        private TextView _modelSelectLabel = null!;
        private TextView _welcomeStatus = null!;
        private TextView _welcomeAddLink = null!;
        private ProgressBar _welcomeSpinner = null!;

        private OverfitClient? _client;
        private ModelInfo? _modelInfo;
        private volatile bool _busy;

        // After this much idle time the model is unloaded (frees ~RAM) and the user returns to model select.
        private const int IdleUnloadMs = 30_000;
        private readonly Android.OS.Handler _idleHandler = new(Android.OS.Looper.MainLooper!);

        // Models live as individual *.gguf files here (the bundled SmolLM2 + any the user added).
        private string ModelsDir => System.IO.Path.Combine(GetExternalFilesDir(null)!.AbsolutePath, "models");
        private const string BuiltInAsset = "smollm2-135m.gguf";
        private const string BuiltInName = "SmolLM2-135M (built-in)";

        private readonly System.Collections.Generic.List<string> _modelPaths = new();

        private ISharedPreferences Prefs => GetSharedPreferences("overfit", FileCreationMode.Private)!;

        protected override void OnCreate(Bundle? savedInstanceState)
        {
            base.OnCreate(savedInstanceState);

            AppLog.Init(GetExternalFilesDir(null)!.AbsolutePath);
            Android.Runtime.AndroidEnvironment.UnhandledExceptionRaiser += (_, e) =>
                AppLog.Write("Unhandled exception", e.Exception);
            AppDomain.CurrentDomain.UnhandledException += (_, e) =>
                AppLog.Write("Unhandled (domain)", e.ExceptionObject as Exception);

            // Edge-to-edge: gradient draws behind transparent system bars; the insets listener pads content
            // for the bars and the keyboard so the input row rides above the IME (AdjustResize alone is flaky).
            Window!.SetDecorFitsSystemWindows(false);
            Window!.SetStatusBarColor(Color.Transparent);
            Window!.SetNavigationBarColor(Color.Transparent);

            var root = new FrameLayout(this);

            _gradient = new AnimatedGradientView(this);
            root.AddView(_gradient, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MatchParent, ViewGroup.LayoutParams.MatchParent));

            _screenRoot = new FrameLayout(this);
            root.AddView(_screenRoot, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MatchParent, ViewGroup.LayoutParams.MatchParent));

            root.SetOnApplyWindowInsetsListener(new InsetsListener((_, insets) =>
            {
                var bars = insets.GetInsets(WindowInsets.Type.SystemBars());
                var ime = insets.GetInsets(WindowInsets.Type.Ime());
                _screenRoot.SetPadding(0, bars.Top + Dp(6), 0, Math.Max(bars.Bottom, ime.Bottom) + Dp(6));
                return insets;
            }));

            SetContentView(root);
            root.RequestApplyInsets();

            EnsureModelsReady();

            // Start on the welcome screen, then auto-load the last-used model (first run: the bundled one)
            // so it works out of the box; we only enter chat once it's ready.
            ShowWelcome();
            var last = Prefs.GetString("last_model_path", null);
            if (last is null || !System.IO.File.Exists(last))
            {
                last = _modelPaths.Count > 0 ? _modelPaths[0] : null;
            }
            if (last is not null)
            {
                var path = last;
                SetWelcomeLoading("loading…");
                System.Threading.Tasks.Task.Run(() => LoadAndEnter(path, DisplayName(path)));
            }
        }

        // Extracts the bundled model on first run and migrates any legacy single-file model into ModelsDir.
        private void EnsureModelsReady()
        {
            try
            {
                System.IO.Directory.CreateDirectory(ModelsDir);

                var builtIn = System.IO.Path.Combine(ModelsDir, "SmolLM2-135M.gguf");
                if (!System.IO.File.Exists(builtIn))
                {
                    using var asset = Assets!.Open(BuiltInAsset);
                    using var output = System.IO.File.Create(builtIn);
                    asset.CopyTo(output);
                    AppLog.Write("Extracted built-in model.");
                }

                // Migrate the old single-file location (pre-models-dir builds) if present.
                var legacy = System.IO.Path.Combine(GetExternalFilesDir(null)!.AbsolutePath, "model.gguf");
                if (System.IO.File.Exists(legacy))
                {
                    var dest = System.IO.Path.Combine(ModelsDir, "Imported.gguf");
                    if (!System.IO.File.Exists(dest))
                    {
                        System.IO.File.Move(legacy, dest);
                    }
                }
            }
            catch (Exception ex)
            {
                AppLog.Write("EnsureModelsReady failed", ex);
            }
        }

        private System.Collections.Generic.List<string> ListModelPaths()
        {
            var list = new System.Collections.Generic.List<string>();
            try
            {
                if (System.IO.Directory.Exists(ModelsDir))
                {
                    foreach (var f in System.IO.Directory.GetFiles(ModelsDir, "*.gguf"))
                    {
                        list.Add(f);
                    }
                }
            }
            catch (Exception ex)
            {
                AppLog.Write("ListModelPaths failed", ex);
            }

            list.Sort();
            return list;
        }

        private static string DisplayName(string path)
        {
            var name = System.IO.Path.GetFileNameWithoutExtension(path);
            return name.Equals("SmolLM2-135M", StringComparison.OrdinalIgnoreCase) ? BuiltInName : name;
        }

        // ─────────────────────────── screens ───────────────────────────

        private void ShowWelcome()
        {
            var col = new LinearLayout(this) { Orientation = Orientation.Vertical };
            col.SetGravity(GravityFlags.Center);
            col.SetPadding(Dp(36), 0, Dp(36), 0);

            var logo = new ShimmerTextView(this) { Text = "OverThink" };
            logo.SetTextColor(Color.White);
            logo.TextSize = 46f;
            logo.SetTypeface(Typeface.DefaultBold, TypefaceStyle.Bold);
            logo.SetShadowLayer(Dp(26), 0, 0, Color.ParseColor("#7C3AED"));
            logo.Gravity = GravityFlags.Center;
            col.AddView(logo);

            var poweredBy = new TextView(this) { Text = "powered by DevOnBike’s Overfit" };
            poweredBy.SetTextColor(Color.ParseColor("#94A3B8"));
            poweredBy.TextSize = 15f;
            poweredBy.LetterSpacing = 0.03f;
            poweredBy.Gravity = GravityFlags.Center;
            var poweredLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WrapContent, ViewGroup.LayoutParams.WrapContent);
            poweredLp.SetMargins(0, Dp(4), 0, 0);
            col.AddView(poweredBy, poweredLp);

            var tagline = new TextView(this) { Text = "On-device AI chat — no servers, no cloud." };
            tagline.SetTextColor(Color.ParseColor("#CBD5E1"));
            tagline.TextSize = 15f;
            tagline.Gravity = GravityFlags.Center;
            var taglineLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WrapContent, ViewGroup.LayoutParams.WrapContent);
            taglineLp.SetMargins(0, Dp(10), 0, Dp(40));
            col.AddView(tagline, taglineLp);

            _modelPaths.Clear();
            _modelPaths.AddRange(ListModelPaths());

            // A styled "select field": the model name on the left, a chevron pinned to the right. Tapping
            // opens a picker list, and choosing an item loads it immediately (no separate Load button).
            _modelSelectField = new LinearLayout(this) { Orientation = Orientation.Horizontal };
            _modelSelectField.SetGravity(GravityFlags.CenterVertical);
            _modelSelectField.SetPadding(Dp(18), Dp(15), Dp(16), Dp(15));
            _modelSelectField.Background = RoundedRect(Color.Argb(46, 255, 255, 255), 16f);
            _modelSelectField.Clickable = true;
            _modelSelectField.Click += (_, _) => ShowModelPicker();

            _modelSelectLabel = new TextView(this);
            _modelSelectLabel.SetTextColor(Color.White);
            _modelSelectLabel.TextSize = 15f;
            _modelSelectField.AddView(_modelSelectLabel, new LinearLayout.LayoutParams(
                0, ViewGroup.LayoutParams.WrapContent, 1f));

            var chevron = new TextView(this) { Text = "▾" };
            chevron.SetTextColor(Color.ParseColor("#CBD5E1"));
            chevron.TextSize = 17f;
            _modelSelectField.AddView(chevron, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WrapContent, ViewGroup.LayoutParams.WrapContent));

            UpdateSelectFieldText();
            var selLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MatchParent, ViewGroup.LayoutParams.WrapContent);
            selLp.SetMargins(Dp(6), 0, Dp(6), Dp(14));
            col.AddView(_modelSelectField, selLp);

            _welcomeSpinner = new ProgressBar(this) { Visibility = ViewStates.Gone };
            col.AddView(_welcomeSpinner, new LinearLayout.LayoutParams(Dp(42), Dp(42)));

            _welcomeStatus = new TextView(this) { Text = "Tap to choose a model" };
            _welcomeStatus.SetTextColor(Color.ParseColor("#94A3B8"));
            _welcomeStatus.TextSize = 12f;
            _welcomeStatus.Gravity = GravityFlags.Center;
            var stLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WrapContent, ViewGroup.LayoutParams.WrapContent);
            stLp.SetMargins(Dp(24), Dp(14), Dp(24), 0);
            col.AddView(_welcomeStatus, stLp);

            _welcomeAddLink = new TextView(this) { Text = "＋  Add a model from file" };
            _welcomeAddLink.SetTextColor(Color.ParseColor("#A78BFA"));
            _welcomeAddLink.TextSize = 14f;
            _welcomeAddLink.Gravity = GravityFlags.Center;
            _welcomeAddLink.SetPadding(Dp(12), Dp(12), Dp(12), Dp(12));
            _welcomeAddLink.Clickable = true;
            _welcomeAddLink.Click += (_, _) => PickModel();
            var addLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WrapContent, ViewGroup.LayoutParams.WrapContent);
            addLp.SetMargins(0, Dp(14), 0, 0);
            col.AddView(_welcomeAddLink, addLp);

            var about = new TextView(this) { Text = "About  ·  GitHub" };
            about.SetTextColor(Color.ParseColor("#A78BFA"));
            about.TextSize = 13f;
            about.Gravity = GravityFlags.Center;
            about.SetPadding(Dp(12), Dp(10), Dp(12), Dp(10));
            about.Clickable = true;
            about.Click += (_, _) => ShowAbout();
            var aboutLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WrapContent, ViewGroup.LayoutParams.WrapContent);
            aboutLp.SetMargins(0, Dp(30), 0, 0);
            col.AddView(about, aboutLp);

            var version = new TextView(this) { Text = AppVersionLabel() };
            version.SetTextColor(Color.ParseColor("#5B6478"));
            version.TextSize = 11f;
            version.Gravity = GravityFlags.Center;
            var verLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WrapContent, ViewGroup.LayoutParams.WrapContent);
            verLp.SetMargins(0, Dp(20), 0, 0);
            col.AddView(version, verLp);

            _screenRoot.RemoveAllViews();
            _screenRoot.AddView(col, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MatchParent, ViewGroup.LayoutParams.MatchParent));
        }

        private void ShowChat()
        {
            var column = new LinearLayout(this) { Orientation = Orientation.Vertical };

            column.AddView(BuildChatHeader());
            column.AddView(BuildChatScroll(), new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MatchParent, 0, 1f));
            column.AddView(BuildInputRow());

            _screenRoot.RemoveAllViews();
            _screenRoot.AddView(column, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MatchParent, ViewGroup.LayoutParams.MatchParent));

            var hint = AddAssistantBubble("Say hi — tokens stream as the model generates.");
            hint.PostDelayed(() =>
            {
                hint.Animate()!.Alpha(0f).TranslationY(-Dp(8)).SetDuration(400)!
                    .WithEndAction(new Java.Lang.Runnable(() => _chatContainer.RemoveView(hint)))!
                    .Start();
            }, 5000);
            ScheduleIdleUnload();
        }

        private View BuildChatHeader()
        {
            var bar = new LinearLayout(this) { Orientation = Orientation.Horizontal };
            bar.SetGravity(GravityFlags.CenterVertical);
            bar.SetPadding(Dp(18), Dp(4), Dp(12), Dp(8));

            var titles = new LinearLayout(this) { Orientation = Orientation.Vertical };
            var title = new TextView(this) { Text = "OverThink" };
            title.SetTextColor(Color.White);
            title.TextSize = 22f;
            title.SetTypeface(Typeface.DefaultBold, TypefaceStyle.Bold);
            title.SetShadowLayer(Dp(14), 0, 0, Color.ParseColor("#7C3AED"));
            titles.AddView(title);

            _subtitle = new TextView(this) { Text = _modelInfo is { } mi ? mi.Name : "ready" };
            _subtitle.SetTextColor(Color.ParseColor("#94A3B8"));
            _subtitle.TextSize = 11f;
            titles.AddView(_subtitle);

            bar.AddView(titles, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WrapContent, 1f));

            bar.AddView(IconButton("?", ShowAbout));
            bar.AddView(IconButton("ⓘ", ShowModelInfo));
            bar.AddView(IconButton("⟳", () => ShowWelcome()));

            return bar;
        }

        private View BuildChatScroll()
        {
            _scroll = new ScrollView(this) { FillViewport = true };
            _chatContainer = new LinearLayout(this) { Orientation = Orientation.Vertical };
            _chatContainer.SetPadding(Dp(4), Dp(6), Dp(4), Dp(6));
            _scroll.AddView(_chatContainer, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MatchParent, ViewGroup.LayoutParams.WrapContent));
            return _scroll;
        }

        private View BuildInputRow()
        {
            var container = new LinearLayout(this) { Orientation = Orientation.Vertical };
            container.SetPadding(Dp(12), Dp(2), Dp(12), Dp(8));

            var row = new LinearLayout(this) { Orientation = Orientation.Horizontal };
            row.SetGravity(GravityFlags.CenterVertical);

            _input = new EditText(this) { Hint = "Message…" };
            _input.SetTextColor(Color.White);
            _input.SetHintTextColor(Color.ParseColor("#94A3B8"));
            _input.Background = RoundedRect(Color.Argb(54, 255, 255, 255), 22f);
            _input.SetPadding(Dp(18), Dp(12), Dp(18), Dp(12));
            _input.SetMaxLines(4);
            _input.SetFilters(new Android.Text.IInputFilter[] { new Android.Text.InputFilterLengthFilter(MaxMessageChars) });
            _input.TextChanged += (_, _) =>
            {
                UpdateCounter();
                ScheduleIdleUnload();
            };
            row.AddView(_input, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WrapContent, 1f));

            _send = new Button(this) { Text = "➤" };
            _send.SetTextColor(Color.White);
            _send.TextSize = 18f;
            _send.Background = GradientCircle();
            var sendLp = new LinearLayout.LayoutParams(Dp(52), Dp(52));
            sendLp.SetMargins(Dp(10), 0, 0, 0);
            _send.Click += (_, _) =>
            {
                _send.Animate()!.ScaleX(0.86f).ScaleY(0.86f).SetDuration(80)!
                    .WithEndAction(new Java.Lang.Runnable(() =>
                        _send.Animate()!.ScaleX(1f).ScaleY(1f).SetDuration(120)!.Start()))!
                    .Start();
                OnSend();
            };
            row.AddView(_send, sendLp);

            container.AddView(row);

            _counter = new TextView(this) { Text = $"0/{MaxMessageChars}" };
            _counter.SetTextColor(Color.ParseColor("#7C8499"));
            _counter.TextSize = 11f;
            _counter.Gravity = GravityFlags.End;
            var counterLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MatchParent, ViewGroup.LayoutParams.WrapContent);
            counterLp.SetMargins(0, Dp(3), Dp(8), 0);
            container.AddView(_counter, counterLp);

            return container;
        }

        private void UpdateCounter()
        {
            var len = _input.Text?.Length ?? 0;
            _counter.Text = $"{len}/{MaxMessageChars}";
            _counter.SetTextColor(Color.ParseColor(len >= MaxMessageChars ? "#F87171" : "#7C8499"));
        }

        // ─────────────────────────── chat ───────────────────────────

        private void OnSend()
        {
            var text = _input.Text?.Trim();
            if (string.IsNullOrEmpty(text))
            {
                return;
            }
            if (_client is null)
            {
                Toast.MakeText(this, "Model is still loading…", ToastLength.Short)!.Show();
                return;
            }
            if (_busy)
            {
                return;
            }

            _busy = true;
            _gradient.Pause();
            _input.Text = string.Empty;
            SetComposerEnabled(false);
            _idleHandler.RemoveCallbacksAndMessages(null);

            AddUserBubble(text);
            var update = AddStreamingAssistantBubble();
            var sb = new StringBuilder();
            var client = _client;

            System.Threading.Tasks.Task.Run(() =>
            {
                try
                {
                    client.Send(text, onText: token =>
                    {
                        sb.Append(token);
                        RunOnUiThread(() => update(sb.ToString()));
                    });
                    RunOnUiThread(() => update(sb.Length > 0 ? sb.ToString() : "(no output)"));
                }
                catch (Exception ex)
                {
                    AppLog.Write("Generation failed", ex);
                    RunOnUiThread(() => update("⚠ " + ex.Message));
                }
                finally
                {
                    RunOnUiThread(() =>
                    {
                        _busy = false;
                        _gradient.Resume();
                        SetComposerEnabled(true);
                        ScheduleIdleUnload();
                    });
                }
            });
        }

        // Disable the input + send button while the model is generating (and dim them), re-enable after.
        private void SetComposerEnabled(bool enabled)
        {
            _input.Enabled = enabled;
            _input.Alpha = enabled ? 1f : 0.55f;
            _send.Enabled = enabled;
            _send.Alpha = enabled ? 1f : 0.45f;
        }

        private void AddUserBubble(string text)
        {
            var tv = NewBubbleText(text);
            tv.Background = new GradientDrawable(GradientDrawable.Orientation.TlBr,
                    new[] { Color.ParseColor("#7C3AED").ToArgb(), Color.ParseColor("#2563EB").ToArgb() })
                .Also(d => d.SetCornerRadius(Dp(20)));
            AttachBubble(tv, isUser: true);
        }

        private View AddAssistantBubble(string text)
        {
            var tv = NewBubbleText(text);
            tv.Background = RoundedRect(Color.Argb(38, 255, 255, 255), 20f);
            AttachBubble(tv, isUser: false);
            return tv;
        }

        // Assistant bubble that starts as thinking-dots and swaps to streamed text on the first update.
        private Action<string> AddStreamingAssistantBubble()
        {
            var bubble = new FrameLayout(this);
            bubble.SetPadding(Dp(15), Dp(11), Dp(15), Dp(11));
            bubble.Background = RoundedRect(Color.Argb(38, 255, 255, 255), 20f);

            var dots = new ThinkingDotsView(this);
            bubble.AddView(dots);

            var tv = new TextView(this) { Visibility = ViewStates.Gone };
            tv.SetTextColor(Color.White);
            tv.TextSize = 16f;
            bubble.AddView(tv);

            AttachBubble(bubble, isUser: false);

            var swapped = false;
            return text =>
            {
                if (!swapped)
                {
                    swapped = true;
                    bubble.RemoveView(dots);
                    tv.Visibility = ViewStates.Visible;
                }
                tv.Text = text;
                ScrollToBottom();
            };
        }

        private TextView NewBubbleText(string text)
        {
            var tv = new TextView(this) { Text = text };
            tv.SetTextColor(Color.White);
            tv.TextSize = 16f;
            tv.SetPadding(Dp(15), Dp(11), Dp(15), Dp(11));
            return tv;
        }

        private void AttachBubble(View bubble, bool isUser)
        {
            var lp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WrapContent, ViewGroup.LayoutParams.WrapContent);
            lp.SetMargins(isUser ? Dp(64) : Dp(12), Dp(6), isUser ? Dp(12) : Dp(64), Dp(6));
            lp.Gravity = isUser ? GravityFlags.End : GravityFlags.Start;
            bubble.LayoutParameters = lp;

            _chatContainer.AddView(bubble);

            bubble.Alpha = 0f;
            bubble.TranslationY = Dp(18);
            bubble.Animate()!.Alpha(1f).TranslationY(0).SetDuration(260)
                .SetInterpolator(new DecelerateInterpolator())!.Start();

            ScrollToBottom();
        }

        private void ScrollToBottom() => _scroll.Post(() => _scroll.FullScroll(FocusSearchDirection.Down));

        // ─────────────────────────── model loading + info ───────────────────────────

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
                var name = GetDisplayName(uri) ?? "model.gguf";
                var dest = ModelDestPath(name);
                SetWelcomeLoading("copying…");
                System.Threading.Tasks.Task.Run(() =>
                {
                    try
                    {
                        using (var input = ContentResolver!.OpenInputStream(uri))
                        using (var output = System.IO.File.Create(dest))
                        {
                            input!.CopyTo(output);
                        }
                        RunOnUiThread(() => SetWelcomeLoading("loading…"));
                        LoadAndEnter(dest, DisplayName(dest));
                    }
                    catch (Exception ex)
                    {
                        AppLog.Write("Copy/load failed", ex);
                        RunOnUiThread(() => SetWelcomeError(ex.Message));
                    }
                });
            }
        }

        // Runs on a background thread: load the model at <paramref name="path"/>; only ENTER the chat screen
        // on success, otherwise stay on the welcome screen and show the error.
        private void LoadAndEnter(string path, string displayName)
        {
            try
            {
                // quantize:false keeps Q4_K resident (measured ~4× faster on-device than the Q8 requant path);
                // 4096 context + sliding window so long multi-turn chats don't error on "context full".
                var client = OverfitClient.LoadGguf(
                    path, maxContextLength: 4096, mmap: true, quantize: false, maxNewTokens: 160, slidingWindow: true);
                var info = BuildInfo(path, client, displayName);
                RunOnUiThread(() =>
                {
                    _client?.Dispose();
                    _client = client;
                    _modelInfo = info;
                    Prefs.Edit()!.PutString("last_model_path", path)!.Apply();
                    ShowChat();
                });
            }
            catch (Exception ex)
            {
                AppLog.Write("Model load failed", ex);
                RunOnUiThread(() => SetWelcomeError(ex.Message));
            }
        }

        private string ModelDestPath(string displayName)
        {
            var safe = string.Join("_", displayName.Split(System.IO.Path.GetInvalidFileNameChars()));
            if (!safe.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
            {
                safe += ".gguf";
            }
            System.IO.Directory.CreateDirectory(ModelsDir);
            return System.IO.Path.Combine(ModelsDir, safe);
        }

        private void SetWelcomeLoading(string status)
        {
            if (_modelSelectField is null)
            {
                return;
            }
            _modelSelectField.Visibility = ViewStates.Gone;
            _welcomeAddLink.Visibility = ViewStates.Gone;
            _welcomeSpinner.Visibility = ViewStates.Visible;
            _welcomeStatus.SetTextColor(Color.ParseColor("#94A3B8"));
            _welcomeStatus.Text = status;
        }

        private void SetWelcomeError(string message)
        {
            if (_modelSelectField is null)
            {
                return;
            }
            _welcomeSpinner.Visibility = ViewStates.Gone;
            _modelSelectField.Visibility = ViewStates.Visible;
            _welcomeAddLink.Visibility = ViewStates.Visible;
            _welcomeStatus.SetTextColor(Color.ParseColor("#F87171"));
            _welcomeStatus.Text = "⚠ " + message;
        }

        private void UpdateSelectFieldText()
        {
            string label;
            var last = Prefs.GetString("last_model_path", null);
            if (last != null && _modelPaths.Contains(last))
            {
                label = DisplayName(last);
            }
            else if (_modelPaths.Count > 0)
            {
                label = DisplayName(_modelPaths[0]);
            }
            else
            {
                label = "No models";
            }

            _modelSelectLabel.Text = label;
        }

        // Opens the model list; choosing an item loads it immediately (no separate Load button).
        private void ShowModelPicker()
        {
            if (_modelPaths.Count == 0)
            {
                PickModel();
                return;
            }

            var names = new string[_modelPaths.Count];
            for (var i = 0; i < _modelPaths.Count; i++)
            {
                names[i] = DisplayName(_modelPaths[i]);
            }

            new AlertDialog.Builder(this)!
                .SetTitle("Choose a model")!
                .SetItems(names, (_, e) =>
                {
                    var path = _modelPaths[e.Which];
                    Prefs.Edit()!.PutString("last_model_path", path)!.Apply();
                    SetWelcomeLoading("loading…");
                    System.Threading.Tasks.Task.Run(() => LoadAndEnter(path, DisplayName(path)));
                })!
                .Show();
        }

        private void ScheduleIdleUnload()
        {
            _idleHandler.RemoveCallbacksAndMessages(null);
            _idleHandler.PostDelayed(UnloadIfIdle, IdleUnloadMs);
        }

        // Fired 30s after the last activity: free the model and send the user back to model select.
        private void UnloadIfIdle()
        {
            if (_busy || _client is null)
            {
                return;
            }

            AppLog.Write("Model unloaded after 30s idle.");
            _client.Dispose();
            _client = null;
            _modelInfo = null;
            ShowWelcome();
        }

        private static ModelInfo BuildInfo(string path, OverfitClient client, string fallbackName)
        {
            var name = fallbackName;
            var arch = "?";
            int layers = 0, ctx = 0, emb = 0;
            try
            {
                using var reader = new GgufReader(path);
                arch = reader.GetMeta<string>("general.architecture", "?");
                name = reader.GetMeta<string>("general.name", fallbackName);
                layers = reader.GetMeta<int>($"{arch}.block_count", 0);
                ctx = reader.GetMeta<int>($"{arch}.context_length", 0);
                emb = reader.GetMeta<int>($"{arch}.embedding_length", 0);
            }
            catch
            {
                // metadata best-effort
            }

            return new ModelInfo(name, arch, layers, ctx, emb, client.Tokenizer.VocabularySize,
                new System.IO.FileInfo(path).Length);
        }

        private const string GitHubUrl = "https://github.com/DevOnBike/Overfit";

        private void ShowAbout()
        {
            const string msg =
                "OverThink runs a language model entirely on your phone — no servers, no cloud, nothing "
                + "leaves the device.\n\n"
                + "It's powered by Overfit: an open-source, pure-C# / .NET deep-learning & inference engine "
                + "(zero-allocation CPU inference, GGUF models, no Python runtime).\n\n"
                + "Pick any GGUF model and chat with streaming tokens, fully offline.";

            new AlertDialog.Builder(this)!
                .SetTitle("About OverThink")!
                .SetMessage(msg)!
                .SetPositiveButton("Close", (_, _) => { })!
                .SetNeutralButton("Overfit on GitHub", (_, _) => OpenUrl(GitHubUrl))!
                .SetNegativeButton("Report a problem", (_, _) => ShareLogs())!
                .Show();
        }

        // Emails the recent error log to the developer so a user can report a problem without adb.
        private void ShareLogs()
        {
            var body = "Describe what happened:\n\n\n----- diagnostic log -----\n" + AppLog.ReadTail();
            var intent = new Intent(Intent.ActionSend);
            intent.SetType("message/rfc822");
            intent.PutExtra(Intent.ExtraEmail, new[] { "devonbike@gmail.com" });
            intent.PutExtra(Intent.ExtraSubject, "OverThink — problem report");
            intent.PutExtra(Intent.ExtraText, body);
            try
            {
                StartActivity(Intent.CreateChooser(intent, "Report a problem"));
            }
            catch (Exception ex)
            {
                Toast.MakeText(this, "No email app found: " + ex.Message, ToastLength.Short)!.Show();
            }
        }

        private void OpenUrl(string url)
        {
            try
            {
                var intent = new Intent(Intent.ActionView, Android.Net.Uri.Parse(url));
                intent.AddFlags(ActivityFlags.NewTask);
                StartActivity(intent);
            }
            catch (Exception ex)
            {
                Toast.MakeText(this, "Couldn't open link: " + ex.Message, ToastLength.Short)!.Show();
            }
        }

        private void ShowModelInfo()
        {
            if (_modelInfo is not { } info)
            {
                Toast.MakeText(this, "No model loaded yet.", ToastLength.Short)!.Show();
                return;
            }

            var sb = new StringBuilder();
            sb.AppendLine($"Name:  {info.Name}");
            sb.AppendLine($"Architecture:  {info.Arch}");
            if (info.Layers > 0)
            {
                sb.AppendLine($"Layers:  {info.Layers}");
            }
            if (info.Context > 0)
            {
                sb.AppendLine($"Context length:  {info.Context:N0}");
            }
            if (info.Embedding > 0)
            {
                sb.AppendLine($"Embedding dim:  {info.Embedding:N0}");
            }
            sb.AppendLine($"Vocabulary:  {info.Vocab:N0}");
            sb.AppendLine($"File size:  {info.FileSizeBytes / (1024.0 * 1024.0):F0} MB");

            new AlertDialog.Builder(this)!
                .SetTitle("Model")!
                .SetMessage(sb.ToString())!
                .SetPositiveButton("OK", (_, _) => { })!
                .SetNeutralButton("Change model", (_, _) => ShowWelcome())!
                .Show();
        }

        private string? GetDisplayName(Android.Net.Uri uri)
        {
            try
            {
                using var cursor = ContentResolver!.Query(uri, null, null, null, null);
                if (cursor != null && cursor.MoveToFirst())
                {
                    var idx = cursor.GetColumnIndex(Android.Provider.IOpenableColumns.DisplayName);
                    if (idx >= 0)
                    {
                        return cursor.GetString(idx);
                    }
                }
            }
            catch
            {
                // fall through
            }

            return uri.LastPathSegment;
        }

        protected override void OnDestroy()
        {
            _idleHandler.RemoveCallbacksAndMessages(null);
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

        private sealed record ModelInfo(
            string Name, string Arch, int Layers, int Context, int Embedding, int Vocab, long FileSizeBytes);

        // ─────────────────────────── helpers ───────────────────────────

        private int Dp(float v) => (int)((v * (Resources?.DisplayMetrics?.Density ?? 2f)) + 0.5f);

        private string AppVersionLabel()
        {
            try
            {
                var info = PackageManager!.GetPackageInfo(PackageName!, 0);
                return $"v{info!.VersionName}";
            }
            catch
            {
                return "v1.0";
            }
        }

        private View IconButton(string glyph, Action onClick)
        {
            var b = new Button(this) { Text = glyph };
            b.SetTextColor(Color.White);
            b.TextSize = 17f;
            b.SetAllCaps(false);
            b.Background = RoundedRect(Color.Argb(46, 255, 255, 255), 20f);
            b.Click += (_, _) => onClick();
            var lp = new LinearLayout.LayoutParams(Dp(42), Dp(42));
            lp.SetMargins(Dp(6), 0, 0, 0);
            b.LayoutParameters = lp;
            return b;
        }

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
        public static T Also<T>(this T self, Action<T> configure)
        {
            configure(self);
            return self;
        }
    }
}
