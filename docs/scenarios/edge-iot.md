# Overfit for Edge and IoT

**For embedded engineers deploying ML on Raspberry Pi, industrial controllers, and field devices.**

---

## The Problem

You need to run inference at the edge. Maybe it's defect detection on a factory line. Maybe it's predictive maintenance on a wind turbine. Maybe it's sensor fusion on a drone.

The constraints are real:

- **Hardware is small.** Raspberry Pi 4 has 2-8 GB RAM, an ARM CPU, no GPU.
- **Network is unreliable.** Cloud round-trips aren't an option. Inference must happen locally.
- **Deployment must be simple.** Field technicians are installing these. They don't run `conda activate`.
- **Boot time matters.** Devices restart frequently. 30-second cold starts are unacceptable.
- **Binary size matters.** OTA updates are expensive. 500 MB containers don't fly.

**Overfit was built for this.** Zero runtime dependencies, Native AOT compilation, single-file deployment.

---

## The Deployment Story

```bash
# On your dev machine
dotnet publish -c Release -r linux-arm64 /p:PublishAot=true

# Result: one file
ls -la bin/Release/net10.0/linux-arm64/publish/
# -rwxr-xr-x  edge-inference   42M
```

Copy that single file to your device:

```bash
scp edge-inference pi@raspberry-pi:/opt/myapp/
ssh pi@raspberry-pi "systemctl restart myapp"
```

That's the whole deployment. No Python environment. No Docker. No dependency graph to reconcile.

### Typical binary sizes

| Application type | Binary size | Cold start |
|------------------|------------:|-----------:|
| Pure inference service | 20-30 MB | < 50 ms |
| Inference + HTTP API | 35-45 MB | < 100 ms |
| Inference + MQTT + logging | 45-60 MB | < 150 ms |

Compare to the Python equivalent: ~300 MB (Python runtime + NumPy + PyTorch + your code), 2-5 second cold start, plus whatever base image you're using.

---

## Architecture Examples

### Example 1: Factory Line Defect Detection

Camera captures frames. CNN classifies each frame. Anomalies trigger an alert via MQTT.

```csharp
public class DefectInspector
{
    private readonly Sequential _model;
    private readonly FastTensor<float> _inputBuffer;

    public DefectInspector(string modelPath)
    {
        _model = new Sequential(
            new ConvLayer(inChannels: 3, outChannels: 16, h: 64, w: 64, kSize: 3),
            new ReluActivation(),
            new MaxPool2D(poolSize: 2),
            new ConvLayer(inChannels: 16, outChannels: 32, h: 32, w: 32, kSize: 3),
            new ReluActivation(),
            new GlobalAveragePool2D(),
            new LinearLayer(32, 2) // binary: defect / no defect
        );
        _model.Load(modelPath);
        _model.Eval();

        _inputBuffer = new FastTensor<float>(1, 3, 64, 64, clearMemory: false);
    }

    public bool IsDefective(ReadOnlySpan<byte> rgbFrame)
    {
        NormalizeToTensor(rgbFrame, _inputBuffer.GetView().AsSpan());

        using var input = new AutogradNode(_inputBuffer, requiresGrad: false);
        var output = _model.Forward(null, input).DataView.AsReadOnlySpan();

        return output[1] > output[0]; // class 1 = defect
    }
}
```

**Runs at camera framerate** on a Raspberry Pi 4 with zero heap allocations after initialization.

### Example 2: Predictive Maintenance on Sensor Streams

LSTM consumes rolling window of vibration/temperature/current sensors, predicts failure probability.

```csharp
public class FailurePredictor
{
    private readonly Sequential _model;
    private readonly float[] _rollingWindow;
    private int _windowHead;

    // ... construct model, load weights ...

    public float FailureProbability(float[] latestReading)
    {
        UpdateRollingWindow(latestReading);
        var prediction = _model.Forward(null, _windowTensor);
        return prediction.DataView.AsReadOnlySpan()[0];
    }
}
```

### Example 3: Drone / Robot Control Loop

Hard real-time loop. Must complete in < 10 ms. Zero GC pauses allowed.

```csharp
while (running)
{
    var sensors = ReadIMU();
    using var obs = BuildObservationTensor(sensors);

    // Inference: hundreds of nanoseconds
    var action = _policyModel.Forward(null, obs).DataView.AsReadOnlySpan();

    WriteMotorCommands(action);

    Thread.SpinWait(loopBudgetRemaining);
}
```

---

## Hardware Targets

### Raspberry Pi 4 / 5

Verified: Overfit runs cleanly on Raspberry Pi OS (ARM64). Use `linux-arm64` runtime identifier for publishing.

Typical performance (Pi 4, 4GB):
- Small MLP inference (100→10): ~5 μs
- Medium CNN inference (3×64×64): ~15-30 ms
- Concurrent load: scales linearly with cores (Pi 4 has 4 cores)

### Intel NUC / Industrial PCs

Typical mid-range industrial controllers run x86-64 Linux or Windows IoT. Overfit runs natively. Performance scales with CPU — a modern i5 NUC rivals server-grade performance for inference.

### ARM-based SBCs

Jetson Nano, Orange Pi, Radxa Rock, etc. — if they run .NET 10 AOT, they run Overfit. Use `linux-arm64` (or `linux-arm` for 32-bit).

### Windows IoT Core

For industrial deployments on Windows. Publish with `win-arm64` or `win-x64` and deploy as a single executable.

### Bare-metal / RTOS

Not supported. Overfit requires .NET 10 runtime minimum. For microcontroller-class deployment (Cortex-M, ESP32), look at TensorFlow Lite Micro or similar.

---

## OTA Update Story

Over-the-air model updates are straightforward:

1. Model weights are just a binary file (`model.bin`).
2. Host the file on S3, HTTPS server, or internal OTA service.
3. Device downloads the new file, replaces local copy, reloads the model.

```csharp
public class ModelUpdater
{
    private Sequential _currentModel;

    public async Task CheckForUpdates()
    {
        var remoteVersion = await FetchRemoteModelVersion();
        if (remoteVersion > _localVersion)
        {
            await DownloadToTempFile("/tmp/model.new");
            File.Move("/tmp/model.new", "/opt/app/model.bin", overwrite: true);

            var newModel = new Sequential(/* architecture */);
            newModel.Load("/opt/app/model.bin");
            newModel.Eval();

            Interlocked.Exchange(ref _currentModel, newModel);
        }
    }
}
```

Model binaries are typically **small** (kilobytes to low megabytes) — OTA-friendly even on constrained connections.

---

## Resource Footprint

### Memory

Loaded Overfit model + runtime:
- MLP (784→10): ~30 KB for weights + ~2 MB for .NET AOT runtime
- CNN (few layers): ~500 KB - 2 MB for weights + runtime

Total RSS for a typical edge inference service: **30-80 MB**. Python equivalent: 300-800 MB.

### CPU

Overfit is single-threaded by default (configurable). For edge devices with 2-4 cores, reserve 1 core for inference and leave the rest for OS/network/logging.

### Power

Lower memory + less CPU time = less power draw. On battery-powered IoT, this translates to measurable battery life improvements over Python-based inference.

---

## Common Pitfalls

### Don't forget `clearMemory: false`

```csharp
// BAD: zeros out memory unnecessarily
var tensor = new FastTensor<float>(1000);

// GOOD: skip the zero-fill when you'll overwrite everything anyway
var tensor = new FastTensor<float>(1000, clearMemory: false);
```

### Use `Eval()` mode

Training mode builds autograd graphs (= allocations). For inference-only scenarios:

```csharp
model.Eval();
```

Do this once after loading.

### Mind the trim warnings

When publishing with AOT + trimming, watch for warnings about reflection or dynamic code. Overfit itself is clean, but your application code might pull in libraries that aren't AOT-safe. Add `TrimmerRootAssembly` entries for any problematic dependencies.

### Profile before optimizing

Most edge deployments are I/O-bound (camera, sensors, network), not CPU-bound. Measure where the time actually goes before optimizing inference.

---

## Further Reading

- [Main README](../../README.md) — project overview and benchmarks
- [ROADMAP](../../ROADMAP.md) — planned features (ONNX import, quantization)
- [ASP.NET scenario](aspnet-microservice.md) — if you also run a cloud backend
- [.NET AOT documentation](https://learn.microsoft.com/en-us/dotnet/core/deploying/native-aot/) — for Native AOT deployment details