# Overfit Chat (Android)

Native .NET-for-Android chat app: pick a GGUF model, chat with **streaming tokens**, over an animated
mesh-gradient background. Pure in-process inference via `OverfitClient` (no server, no Python).

- App id: `com.devonbike.overfitchat` (shows as **Overfit** in the launcher)
- Runtime: **Mono** (stable; Play-Store-appropriate). Dev builds use JIT; release builds add AOT.
- Model is **not bundled** (too big for a base APK) — load it via the in-app picker or pre-push it.

---

## Just use it (no PC needed)

It's already installed on the phone with a model seeded, so:

1. Open the **Overfit** app from the launcher.
2. It auto-loads the model it has (status pill → `ready ✓`).
3. Type a message → tap ➤ → tokens stream in.

To use a **different** model: tap **Load model** → pick a `.gguf` from Downloads.

---

## Rebuild & redeploy later (after code changes)

**Shortcut:** `deploy.cmd` in this folder does build + install + launch in one go:

```bat
Demo\OverfitChatApp\deploy.cmd                       :: device already connected (USB / prior wireless)
Demo\OverfitChatApp\deploy.cmd 192.168.1.174:PORT    :: connect wireless first (PORT from the phone)
Demo\OverfitChatApp\deploy.cmd 192.168.1.174:PORT aot :: + AOT (slower build, faster decode)
```

Or the manual steps below. Run these from the repo root (`D:\Overfit`) in **PowerShell**. `adb` lives at
`%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe` — alias it for convenience:

```powershell
$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
$sdk = "$env:LOCALAPPDATA\Android\Sdk"
```

### 1. Build the APK

```powershell
# Dev build (fast, JIT):
dotnet build Demo/OverfitChatApp/OverfitChatApp.csproj -c Release -f net10.0-android `
  -p:AndroidSdkDirectory="$sdk" -p:AcceptAndroidSDKLicenses=true

# Faster on-device decode (slower build) — flip AOT on:
#   add  -p:RunAOTCompilation=true
```

APK: `Demo/OverfitChatApp/bin/Release/net10.0-android/android-arm64/com.devonbike.overfitchat-Signed.apk`

### 2. Connect the phone

**USB:** plug in, accept the "Allow USB debugging" prompt, then `& $adb devices` should list it.

**Wireless** (same Wi-Fi): on the phone, Developer options → **Wireless debugging**.
First time, "Pair device with pairing code":

```powershell
& $adb pair 192.168.1.x:<pairPort> <6-digit-code>     # first time only
& $adb connect 192.168.1.x:<connectPort>              # port from the main Wireless-debugging screen
& $adb devices                                         # confirm "device"
```

> The wireless port **rotates** whenever the adb server restarts — if `connect` is refused, reopen the
> Wireless-debugging screen, read the new port, and `connect` again.

### 3. Install

```powershell
& $adb install -r Demo/OverfitChatApp/bin/Release/net10.0-android/android-arm64/com.devonbike.overfitchat-Signed.apk
```

### 4. (Optional) Pre-load a model so the app auto-loads it

```powershell
& $adb push C:\path\to\model.gguf /sdcard/Android/data/com.devonbike.overfitchat/files/model.gguf
```

Otherwise use the in-app **Load model** picker. A small Q4_K_M GGUF (e.g. Qwen2.5-0.5B ~0.46 GB) is a good
fit for a 12 GB phone.

### 5. Launch

Open **Overfit** from the launcher, or:

```powershell
& $adb shell monkey -p com.devonbike.overfitchat -c android.intent.category.LAUNCHER 1
```

---

## Toolchain (one-time, already set up on this machine)

- .NET 10 SDK + `android` workload, **JDK 17**.
- Writable Android SDK at `%LOCALAPPDATA%\Android\Sdk` with **android-36** platform, **build-tools 36**,
  the **NDK** (bundled by the workload), and **platform-tools** (`adb`). Built with
  `-p:AndroidSdkDirectory=$sdk -p:AcceptAndroidSDKLicenses=true`.

## Notes / known characteristics

- On-device decode of a 0.5B Q4_K is ~**3.8 tok/s** on a Snapdragon 7s Gen 2 (pure-managed mobile;
  it's ~5–10× off a tuned C++ engine like llama.cpp — that's the managed-on-mobile tax, not a bug).
- Loads with `quantize:false` (Q4_K-resident) — measured ~4× faster on-device than the Q8 requant path,
  at the cost of more RAM (F32 lm-head). Fine for a 0.5B on 12 GB.
- For a Play Store release: build an **AAB** (`-p:AndroidPackageFormat=aab`), turn on
  `RunAOTCompilation=true`, and sign with a real upload key (the debug `-Signed.apk` here is dev-signed).
