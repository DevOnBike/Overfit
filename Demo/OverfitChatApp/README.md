# Overfit Chat (Android)

Native .NET-for-Android chat app: pick a GGUF model, chat with **streaming tokens**, over an animated
mesh-gradient background. Pure in-process inference via `OverfitClient` (no server, no Python).

- App id: `com.devonbike.overthink` (shows as **OverThink** in the launcher)
- Runtime: **Mono** (stable; Play-Store-appropriate). Dev builds use JIT; release builds add AOT.
- A starter model **is** bundled (`Assets/smollm2-135m.gguf`, SmolLM2-135M Q4_K, ~101 MB) and extracted on
  first run; bigger models come from the in-app picker or a pre-push. The asset is git-ignored — over
  GitHub's 100 MB file limit — and the build scripts fetch it if it is missing.

> **Read [`docs/measured-baselines.md`](../../docs/measured-baselines.md) before optimising anything here.**
> Mobile behaves unlike the desktop in ways that are counter-intuitive and already measured: the scheduler
> parks decode on the little cores, the shipped parallel-work threshold is wrong for a 4-core cluster, and
> **no ARM SIMD intrinsic is available to this runtime at all**.

---

## Just use it (no PC needed)

It's already installed on the phone with a model seeded, so:

1. Open the **Overfit** app from the launcher.
2. It auto-loads the model it has (status pill → `ready ✓`).
3. Type a message → tap ➤ → tokens stream in.

To use a **different** model: tap **Load model** → pick a `.gguf` from Downloads.

---

## Which command, and when

Two scripts cover everything. Pick by what you are doing, not by which is newer:

| You are… | Run | Takes | Produces |
|---|---|---|---|
| iterating on code | `.\build-dev.ps1` | ~20 s | dev-signed APK, installed + launched |
| on a new wireless session | `.\build-dev.ps1 -Device <ip:port>` | ~20 s | same, connects first |
| watching what the app logs | `.\build-dev.ps1 -Log` | ~20 s + tail | same, then follows `overthink.log` |
| measuring **startup**, or testing anything intrinsics-dependent | `.\build-dev.ps1 -Aot` | minutes | dev-signed **AOT** APK |
| handing the app to someone (sideload, GitHub Release) | `.\build-release.ps1` | minutes | **signed APK**, upload key |
| shipping to Google Play | `.\build-release.ps1 -Format aab -VersionCode N` | minutes | **signed AAB** |
| setting up signing, once ever | `.\generate-upload-key.ps1` | seconds | `overthink-upload.keystore` |

```powershell
cd Demo\OverfitChatApp
.\build-dev.ps1                                   # the one you will use 95% of the time
.\build-release.ps1 -Install                      # release APK, and flash it to the phone too
```

**Do not reach for `-Aot` during normal development.** It costs minutes per iteration and buys **nothing on
decode** (8.5 vs 8.6 tok/s, inside noise) — only cold start (~400 ms against ~640). Use it when the launch
time is the subject, or before handing the build to anyone.

### Cutting a release, in order

1. **Decide the version.** `<Version>` lives in `Directory.Build.props` and flows into the app's displayed
   version. For an AAB, also pick a `versionCode` **strictly higher than every bundle already uploaded to
   any Play track** — internal, closed and production share one number space.
2. **Build it clean.** `build-release.ps1` runs `dotnet clean` first on purpose: an incremental tree can
   carry native libraries from a previous non-AOT build, and then nobody can vouch for what is inside the
   package.
3. **Check the arm actually shipped.** Install it (`-Install`) and read the line the script prints:
   `aotLibs=24 (7364 kB)`. **`aotLibs=0` in a release means AOT silently did not happen.** A number in the
   low tens of kB means `AndroidEnableProfiledAot` was left at its default and only the startup profile got
   compiled — 16 kB instead of 1155 kB for the engine, i.e. AOT in name only.
4. **Smoke-test on the phone**: launch, send one message, confirm tokens stream. Discard the first ~3 cold
   starts if you are timing anything — they run ~680-710 ms while the system settles the new package.
5. **Then distribute**: attach the APK to a GitHub Release, or upload the AAB in Play Console → Internal
   testing first. Both flows are written out further down.

Everything below this line is detail: manual equivalents of the scripts, the wireless-pairing dance, and
the two distribution paths.

Both **install over** the app rather than uninstalling it: `adb uninstall` wipes app data, which deletes
`overthink.log` and every model added by hand. Both also print which build is actually running, read from
the app's own probe of its installed `libaot-*.so` files — a build flag says what was *requested*,
`aotLibs=` says what is executing.

**`-Aot` is not only for release.** It is the only way to test anything that depends on hardware
intrinsics — although, as measured below, on this platform the answer is the same either way.

These two replaced `deploy.cmd`, `make-apk.ps1` and `make-aab.ps1`, which were deleted on 2026-08-14. Three
scripts had grown to cover overlapping cases, and the two `make-*` ones shipped AOT **without**
`AndroidEnableProfiledAot=false` — "AOT" in name only, see below. (The unrelated `k8s\overfit\deploy.cmd`,
which brings up the inference-server lab, is a different file and still exists.)

## Doing it by hand (what the scripts run)

Useful when something in a script fails and you need to see which step. Run these from the repo root
(`D:\Overfit`) in **PowerShell**. `adb` lives at `%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe` —
alias it for convenience:

```powershell
$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
$sdk = "$env:LOCALAPPDATA\Android\Sdk"
```

### 1. Build the APK

```powershell
# Dev build (fast, JIT):
dotnet build Demo/OverfitChatApp/OverfitChatApp.csproj -c Release -f net10.0-android `
  -p:AndroidSdkDirectory="$sdk" -p:AcceptAndroidSDKLicenses=true

# Faster COLD START (slower build) — AOT. Both flags, or only the startup profile is compiled:
#   add  -p:RunAOTCompilation=true -p:AndroidEnableProfiledAot=false
# It does NOT speed up decode — measured 8.5 vs 8.6 tok/s, inside noise.
```

APK: `Demo/OverfitChatApp/bin/Release/net10.0-android/android-arm64/com.devonbike.overthink-Signed.apk`

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
& $adb install -r Demo/OverfitChatApp/bin/Release/net10.0-android/android-arm64/com.devonbike.overthink-Signed.apk
```

### 4. (Optional) Pre-load a model so the app auto-loads it

```powershell
& $adb push C:\path\to\model.gguf /sdcard/Android/data/com.devonbike.overthink/files/model.gguf
```

Otherwise use the in-app **Load model** picker. A small Q4_K_M GGUF (e.g. Qwen2.5-0.5B ~0.46 GB) is a good
fit for a 12 GB phone.

### 5. Launch

Open **Overfit** from the launcher, or:

```powershell
& $adb shell monkey -p com.devonbike.overthink -c android.intent.category.LAUNCHER 1
```

---

## Share it directly (sideload APK — no Google Play)

The fastest way to get OverThink to people without the Play review/testing gates: build a **signed APK** and hand
it out (a GitHub Release, a Drive/WeTransfer link, email). Recipients install it directly. No auto-updates, and
they must allow "install unknown apps", but it's immediate.

### 1. Build the shareable APK

```powershell
cd Demo/OverfitChatApp
.\build-release.ps1            # signed APK, full AOT, ~minutes
```

(One script for both formats — pass `-Format aab` for Play.)

Enter the keystore password (Enter on the key prompt if you used one password). It prints the path, e.g.
`bin\Release\net10.0-android\android-arm64\com.devonbike.overthink-Signed.apk` (~104 MB — the bundled model is
inside). Needs the upload keystore first (`generate-upload-key.ps1`). Signing with the upload key is fine for
sideloading (Play App Signing only matters for Play). Rename it to something friendly, e.g.
`OverThink-v10.0.28.apk`.

### 2. Attach it to a GitHub Release

1. **github.com/DevOnBike/Overfit → Releases → Draft a new release.**
2. **Choose a tag** → a new one, e.g. `overthink-v10.0.28` → *Create new tag on publish*, target `main`.
3. **Title:** `OverThink v10.0.28 (Android APK)`.
4. **Description** (example):
   > On-device AI chat — offline, pure .NET. **Android APK, sideload.**
   > 1. Download `OverThink-v10.0.28.apk`. 2. On the phone tap the file → allow "Install unknown apps" for your
   > browser/file manager → Install. The built-in mini-model works out of the box; add a bigger GGUF in-app.
5. **Drag the `.apk` into the "Attach binaries" box** (upload ~104 MB — GitHub allows up to 2 GB per release
   asset, unlike the 100 MB *repo-file* limit).
6. **Publish release**, then share the release URL (or the direct asset link).

### Caveats

- **No auto-update** — a new version means a new APK; bump `versionCode`/the name each time.
- **Don't commit the APK to the repo** (over the 100 MB file limit) — it lives only as a release asset. Keystore
  and passwords stay out of the repo too (a password manager + `.gitignore`).
- **arm64 Android only.** No iOS build (no native .NET-iOS port of this app).

## Publishing to Google Play

High-level path (one-time setup, then repeat steps 4–6 per release):

### 1. Developer account
- Create a **Google Play Developer account** (one-time **$25**) at <https://play.google.com/console>.

### 2. Upload keystore (sign your releases)
Generate an **upload key** once and keep it safe (losing it = can't update the app):
```powershell
keytool -genkeypair -v -keystore overfit-upload.keystore -alias overfit `
  -keyalg RSA -keysize 2048 -validity 10000
```
Use **Play App Signing** (default & recommended): you sign uploads with this upload key, Google holds the
real app-signing key.

### 3. Pre-flight code changes (needed before a real release)
- **minSdk vs edge-to-edge:** the app calls `SetDecorFitsSystemWindows` / `WindowInsets.Ime()` (API 30+) but
  `SupportedOSPlatformVersion` is 24 → either **raise it to `30`** (`<SupportedOSPlatformVersion>30</…>`, drops
  the CA1416 warnings) or guard those calls with `if (OperatingSystem.IsAndroidVersionAtLeast(30))`.
- **targetSdk:** Play requires a recent target — keep building against **android-36** (already are).
- **App icon:** add a real launcher icon (`Resources/mipmap-*/appicon.png` + `<application android:icon>`); a
  512×512 PNG is also needed for the store listing.
- **ABIs:** this project builds **arm64 only** (`RuntimeIdentifier=android-arm64`) — covers ~all modern phones.
  For wider reach add `<RuntimeIdentifiers>android-arm64;android-arm</RuntimeIdentifiers>` (the AAB splits per ABI).
- **Model for reviewers — solved.** SmolLM2-135M Q4_K (~101 MB) is bundled as an asset and extracted on first
  run, so a reviewer has something to chat with immediately. It is the reason the APK is ~104 MB. Bigger models
  still come from the in-app picker. If the bundle size ever becomes the problem, **Play Asset Delivery** or an
  in-app "download a starter model" button are the alternatives.

### 4. Build a signed **release AAB** (not APK)
Google Play takes an **Android App Bundle**. Bump `ApplicationVersion` (the integer **versionCode** — must
increase every upload) in the csproj, then:
```powershell
dotnet publish Demo/OverfitChatApp/OverfitChatApp.csproj -c Release -f net10.0-android `
  -p:AndroidPackageFormat=aab `
  -p:RunAOTCompilation=true `
  -p:AndroidKeyStore=true `
  -p:AndroidSigningKeyStore=overfit-upload.keystore `
  -p:AndroidSigningKeyAlias=overfit `
  -p:AndroidSigningStorePass=YOUR_STORE_PASS `
  -p:AndroidSigningKeyPass=YOUR_KEY_PASS
```
Output: `Demo/OverfitChatApp/bin/Release/net10.0-android/com.devonbike.overthink-Signed.aab`.

### 5. Play Console listing (one-time, then edits)
- Create the app, fill the **store listing**: title, short + full description, **phone screenshots**, feature
  graphic (1024×500), **512×512 icon**.
- **Privacy policy URL** (required). **Data safety** form — declare **no data collected / all on-device**
  (a genuine selling point here). Content rating questionnaire. Target audience. Ads: none.

### 6. Release tracks
Upload the AAB to **Internal testing** first (installs via Play on your own devices) → fix issues →
**Closed/Open testing** → **Production**. Review is typically hours–days; first submissions take longer.

### Notes
- The debug `-Signed.apk` from `build-dev.ps1` is **dev-signed** — fine for sideloading/testing, **not** for Play.
- Keep `RunAOTCompilation=true` **plus `AndroidEnableProfiledAot=false`** for release. It buys **cold start**
  (~400 ms against ~640), not decode — measured, see "AOT buys startup, not tokens" below. `build-release.ps1`
  passes both.

---

## Toolchain (one-time, already set up on this machine)

- .NET 10 SDK + `android` workload, **JDK 17**.
- Writable Android SDK at `%LOCALAPPDATA%\Android\Sdk` with **android-36** platform, **build-tools 36**,
  the **NDK** (bundled by the workload), and **platform-tools** (`adb`). Built with
  `-p:AndroidSdkDirectory=$sdk -p:AcceptAndroidSDKLicenses=true`.

## Notes / known characteristics

All figures below are from a Motorola Edge 50 Fusion (Snapdragon 7s Gen 2: 4x Cortex-A78 up to 2.4 GHz on
cpu4-7, 4x A55 up to 1.96 GHz on cpu0-3), SmolLM2-135M, same prompt, ~300 generated tokens. Full conditions
and the refuted alternatives are in [`docs/measured-baselines.md`](../../docs/measured-baselines.md).

### Three things this app does that look odd and are not

Each was measured; removing any of them costs throughput.

**1. It pins its threads to the fast CPU cluster** (`BigCoreAffinity`). Without it, all four big cores sat
at their 691 MHz **idle floor** for an entire generation while the little cores ran near their ceiling —
the model was executing on the A55s. The process is not confined by the system (cpuset `top-app`, cpus
0-7); it is placed badly, because each worker runs in short bursts and parks on a semaphore, so no thread
ever accumulates the utilisation that earns a big core. Costs energy per token; earns 4.1 → 6.3 tok/s.

**2. It sizes the worker pool to the fast cores, not `ProcessorCount`.** Eight thin workers over four big
cores is 2x oversubscription and the governor never holds the clock; four workers, one per core, keep all
four at 2.4 GHz for the whole run. 6.3 → 8.5 tok/s.

**3. It lowers `SingleTokenProjectionKernel.ParallelWorkThresholdOverride` to 100,000.** The library
default is 1,000,000 elements, and this model's FFN matmuls are 576x1536 = **884,736** — 12% below it — so
every one of them took the sequential path and ran on the calling thread, which was 46% of decode wall
time. 8.5 → ~10 tok/s. **Do not "fix" this in the library**: the same lowering measured 1.2x to 2.9x
*slower* on a 32-core desktop, so it is set per-app on purpose.

Together: **4.1 → 9.9 tok/s, 2.4x**, and the collapse users saw — ~10 tok/s for the first few tokens then
a settle to ~3.6 — is gone. The remaining ~20% decline within a long answer is filed as `PB-11`.

### AOT buys startup, not tokens

| | decode | cold start (steady state) |
|---|---|---|
| JIT | 8.6 tok/s | 627-665 ms |
| Full AOT | 8.5 tok/s | **~400-440 ms** |

1.2% on decode is inside run-to-run noise; 240 ms on launch is not. So **AOT for release, JIT for the
edit-build-run loop** — the old advice here ("keep AOT for faster on-device decode") was wrong.

Two traps: pass `AndroidEnableProfiledAot=false` with it, or only the startup profile is compiled and the
engine library comes out at **16 kB instead of 1155 kB** — AOT in name only. And **discard the first ~3
launches after an install**; they run ~680-710 ms while the system settles the new package, which is
enough to make AOT look slower than JIT.

### Quantised weights are slower here, and the reason is not what it looks like

The app loads models under ~550 MB with `quantize:false` (F32-resident). Forcing `quantize:true` measured
**639 ms/token against 100 ms** — 6.4x slower while reading **5.4x fewer bytes** (0.16 GB/s against F32's
5.5 GB/s), so it is nowhere near bandwidth-bound. RSS did drop as intended, 1035 → 652 MB.

The cause is the last note below, and it is the single most important thing on this page.

### .NET-for-Android exposes NO ARM hardware intrinsics

Logged by the app at startup, in **both** JIT and full-AOT builds:

```text
simd: Dp=False  AdvSimd=False  AdvSimd64=False  V128hw=True  forceScalar=False
```

`AdvSimd.IsSupported` false on an **arm64** device is base NEON reported as absent, and the hardware is not
the reason — `/proc/cpuinfo` lists `asimd asimdrdm asimdhp asimddp`. **Every kernel written against
`System.Runtime.Intrinsics.Arm.*` therefore runs its scalar fallback on Android, silently**, because the
fallback is a correctness path and the only symptom is speed.

Consequences worth knowing before you plan work here:

- The repository's recorded result *"ARM NEON SDOT: correct and pointless — decode is dequant-bound"*
  measured code that **never executed**. The port was unreachable, not pointless.
- Keeping mobile on F32 is right, but not because "F32 beats the quantised kernels under Mono's codegen" —
  it is because the quantised kernels have **no vector path at all** here.
- The way to get SIMD on this platform is the portable `Vector128<T>` API, which the runtime *does*
  accelerate (`V128hw=True`). Porting the quantised GEMV kernels to it — validated bit-identical against
  the existing scalar oracle — is the open lever for closing the gap to llama.cpp on mobile, and its size
  is known: 5.4x fewer bytes read, currently 6.4x slower.

### Play Store

Build an **AAB** (`.\build-release.ps1 -Format aab -VersionCode N`) and sign with a real upload key — the
`-Signed.apk` from `build-dev.ps1` is dev-signed.
