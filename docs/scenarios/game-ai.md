# Overfit for Game AI

**For Unity developers, game engine programmers, and anyone running neural networks at frame-rate.**

---

## The Problem

You want to put neural networks into your game. Maybe it's behavior trees for enemy AI. Maybe it's a physics-based agent trained via reinforcement learning. Maybe it's a flocking swarm of 100,000 entities.

The traditional options are all painful:

- **Unity ML-Agents** — fine for research, but deploys via ONNX Runtime. ONNX adds ~3 μs overhead per call and allocates on every inference. At 60 FPS with 1,000 agents, that's **1.8 MB/sec of GC pressure** — hello frame drops.
- **TensorFlow.NET / TorchSharp** — drag megabytes of native binaries into your build. Your game's download size doubles. Mobile deployment becomes a nightmare.
- **Hand-rolled matrix math** — you end up reinventing what Overfit already does, with bugs.

**Overfit is built for this.** Zero allocations in the hot path means no GC hitches. Pure managed C# means it compiles into your game, not alongside it.

---

## Showcase: 100,000 Bots at 60 FPS

We built this to prove the point:

- **100,000 independent neural networks** (each bot has its own 10-parameter brain)
- **Running at 60 FPS** in Unity's Update loop
- **Orbital swarm behavior** — bots learn to circle a target while avoiding a predator
- **Training**: 400 generations of genetic algorithm in **36 seconds** on a Ryzen 9 9950X3D

The full code is in [`Demo/Unity/`](../../Demo/Unity). It's two files: a console server that runs training/inference, and a Unity MonoBehaviour that renders the swarm.

### Key numbers

| Metric | Value |
|--------|------:|
| Inference cost per bot | ~17 ns |
| Total swarm inference (100k bots) | ~1.7 ms |
| Frame budget used at 60 FPS | ~10% |
| GC allocations per frame | **0 bytes** |
| Training time (400 generations) | 36 seconds |

For comparison, running the same network through ONNX Runtime per-bot would take ~300 ms per frame — unplayable.

---

## Integration Patterns

### Pattern 1: Shared Brain (all agents same policy)

Good for: swarms, crowds, NPCs with similar behavior.

```csharp
// One brain, applied to all agents
var brain = new float[10];
File.ReadAllBytes("brain.bin").CopyTo(brain.AsSpan().AsBytes());

for (var i = 0; i < agents.Length; i++)
{
    var input = BuildAgentInput(agents[i], target, predator);
    var output = ApplyBrain(brain, input); // Linear 4→2 + tanh
    agents[i].velocity += output * deltaTime;
}
```

### Pattern 2: Per-Agent Brain (genetic algorithm style)

Good for: evolutionary training, emergent diversity.

```csharp
// Each agent has its own brain - stored in a single FastTensor
using var population = new FastTensor<float>(agentCount, genomeSize, clearMemory: false);

// Per-frame: each bot inferences with its own weights
Parallel.For(0, agentCount, i =>
{
    var genome = population.GetView().AsReadOnlySpan().Slice(i * genomeSize, genomeSize);
    var output = InferPerAgent(genome, agents[i], target);
    agents[i].ApplyAcceleration(output);
});
```

### Pattern 3: Trained Policy Network (deep RL)

Good for: enemy AI trained via PPO/DQN, behavior cloning.

```csharp
public class AgentPolicy : MonoBehaviour
{
    private Sequential _model;

    void Start()
    {
        _model = new Sequential(
            new LinearLayer(observationSize, 64),
            new ReluActivation(),
            new LinearLayer(64, actionSize)
        );
        _model.Load("policy.bin");
        _model.Eval();
    }

    void FixedUpdate()
    {
        using var obs = BuildObservation();
        using var action = _model.Forward(null, obs);
        ApplyAction(action);
    }
}
```

---

## Why This Works in Unity

### No GC Pressure

Unity's garbage collector causes frame hitches. The traditional rules: pool everything, reuse buffers, avoid `new` in Update. Overfit respects these rules by design — the entire forward pass through a loaded model allocates zero bytes.

```csharp
// This runs every frame. Zero allocations.
var result = model.Forward(null, input).DataView.AsReadOnlySpan();
```

### Burst-Compatible Workflows

While Overfit itself uses managed SIMD (not Burst), it interoperates cleanly with Unity's Job System and Burst-compiled code. Build your observation tensors in a Burst job, hand them off to Overfit for inference, apply results back in another Burst job.

### IL2CPP Ready

Overfit compiles under Unity's IL2CPP backend. No reflection-heavy patterns, no `Reflection.Emit`, no runtime type generation. Your iOS and Android builds work the same as your desktop builds.

### Mobile Friendly

Overfit's core has no native dependencies. On mobile, this means:

- No `.so` / `.dylib` / `.dll` to ship per platform
- No AOT bootstrap issues
- Your APK/IPA stays small

---

## Training in the Editor

A common workflow: train a model while testing in Play mode, save the weights, redeploy.

The Unity Swarm demo does exactly this:

1. **Training server** runs as a console app, communicates with Unity via TCP
2. **Unity client** sends observations (target position, predator position) and receives actions
3. **Genetic algorithm** evolves 100k brains in parallel, saves the best one to disk
4. **Switch to demo mode** — the server loads the trained brain and applies it

You can also train **offline** without Unity for 100× faster iteration:

```bash
dotnet run -- offline 300   # 300 generations in ~30 seconds
dotnet run -- demo          # Serve the trained brain to Unity
```

---

## Common Gotchas

### Don't allocate tensors per frame

Anti-pattern:
```csharp
void Update()
{
    // BAD: allocates every frame
    using var input = new FastTensor<float>(1, obsSize);
    // ...
}
```

Pattern:
```csharp
FastTensor<float> _inputBuffer;

void Start()
{
    _inputBuffer = new FastTensor<float>(1, obsSize, clearMemory: false);
}

void Update()
{
    // GOOD: reuse the buffer
    FillObservation(_inputBuffer.GetView().AsSpan());
    // ...
}
```

### Use `Eval()` mode for inference

If you forget `model.Eval()`, the model runs in training mode, building an autograd graph every call. This allocates and is slower.

### Watch the thread affinity

Unity's main thread is where `MonoBehaviour` lives. You can inference from the main thread (it's fast enough for most cases), or offload to a background thread for very large swarms. If you offload, be careful — `Transform.position` etc. are main-thread only.

---

## Engine Support

While the Unity demo is the showcase, Overfit works in any C# game engine:

- **Unity** — verified, IL2CPP compatible
- **Godot (C#)** — should work out of the box (not tested yet, contributions welcome)
- **Stride / MonoGame / FNA** — standard .NET, no issues expected
- **Custom engines** — if you run on .NET 10, you run Overfit

---

## Further Reading

- [Main README](../../README.md) — project overview and benchmarks
- [Unity Swarm demo](../../Demo/Unity) — full source code of the 100k bots demo
- [ROADMAP](../../ROADMAP.md) — what's planned (multi-agent environments, more examples)
- [ASP.NET scenario](aspnet-microservice.md) — if you also host a training backend