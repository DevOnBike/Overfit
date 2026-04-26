# Overfit Architecture Refactor Plan

## Cel refaktoru

Celem refaktoru nie jest jednorazowe przepisanie biblioteki. Celem jest rozdzielenie odpowiedzialności tak, aby dalszy rozwój treningu, inferencji, optymalizatorów i kernelów był prostszy, bezpieczniejszy i łatwiejszy do mierzenia.

Najważniejsza decyzja architektoniczna:

```text
TrainingEngine   = fasada workflow treningu
ComputationGraph = mózg autograd / training runtime
Layers           = parametry + kompozycja operacji grafu
Kernels          = czysta matematyka na Spanach
InferenceEngine  = osobna fasada inferencji
```

Obecny największy problem architektoniczny to mieszanie ról:

```text
AutogradNode = wartość grafu + parametr + temporary + owner pamięci + owner gradientu
ComputationGraph = tape + allocator + backward executor + op facade + workspace manager
TensorMath = nazwa sugerująca czystą matematykę, ale obecnie zawiera graph-aware operacje
```

Docelowo `ComputationGraph` ma być centralnym API treningowym. Metody, które nagrywają tape, powinny być metodami grafu:

```csharp
var prediction = model.Forward(graph, input);
var loss = graph.SoftmaxCrossEntropy(prediction, target);

graph.Backward(loss);
```

Zamiast obecnego stylu:

```csharp
TensorMath.SoftmaxCrossEntropy(graph, prediction, target);
```

`TensorMath` powinno oznaczać czystą matematykę albo zostać zastąpione przez `Kernels`.

---

## Docelowy podział odpowiedzialności

### 1. `TensorStorage<T>`

**Odpowiedzialność:** fizyczna pamięć.

Robi:

```text
- posiada bufor
- zwraca Span<T> / ReadOnlySpan<T>
- zna Length / Size
- Dispose / ReturnToPool
```

Nie robi:

```text
- nie zna shape
- nie zna gradientów
- nie zna autograd
- nie zna graph
- nie zna layerów
- nie zna optimizerów
```

To jest najniższy poziom pamięci.

---

### 2. `TensorShape`

**Odpowiedzialność:** opis wymiarów tensora.

Robi:

```text
- D0/D1/D2/D3
- Rank
- ElementCount
- walidacja shape
```

Nie robi:

```text
- nie zna storage
- nie zna danych
- nie zna gradientów
- nie zna graph
```

---

### 3. `TensorView<T>` / `TensorSpan<T>`

**Odpowiedzialność:** widok na pamięć + shape, bez ownership.

Przykładowy kierunek:

```csharp
public readonly struct TensorView<T>
{
    public TensorStorage<T> Storage { get; }
    public TensorShape Shape { get; }
    public int Offset { get; }
}
```

Albo dla hot path:

```csharp
public readonly ref struct TensorSpan<T>
{
    public Span<T> Data { get; }
    public TensorShape Shape { get; }
}
```

Robi:

```text
- interpretuje pamięć jako tensor
- może mieć offset/stride w przyszłości
```

Nie robi:

```text
- nie posiada pamięci
- nie dispose'uje
- nie zna gradientów
- nie zna grafu
```

---

### 4. `Parameter`

**Odpowiedzialność:** długowieczny trenowalny tensor modelu.

Parametr nie powinien być zwykłym `AutogradNode`, bo ma inny lifecycle niż temporary node w grafie.

Przykładowa abstrakcja:

```csharp
public sealed class Parameter : IDisposable
{
    public TensorStorage<float> Data { get; }
    public TensorStorage<float> Grad { get; }
    public TensorShape Shape { get; }
    public bool RequiresGrad { get; }

    public void ZeroGrad()
    {
        Grad.AsSpan().Clear();
    }

    public AutogradNode AsNode()
    {
        // Lightweight graph-visible view over parameter.
        // Na początku może zwracać wrapper zgodny ze starym API.
    }
}
```

Robi:

```text
- posiada Data
- posiada Grad
- żyje razem z layerem/modellem
- jest widoczny dla optimizerów
- umie wyzerować Grad
- może umieć Save/Load danych
```

Nie robi:

```text
- nie jest tape op
- nie jest temporary node
- nie zna ComputationGraph
- nie wykonuje backward
```

Docelowo optymalizatory powinny przyjmować:

```csharp
IEnumerable<Parameter>
```

zamiast:

```csharp
IEnumerable<AutogradNode>
```

---

### 5. `AutogradNode`

**Odpowiedzialność:** uchwyt wartości biorącej udział w grafie autograd.

Docelowo powinien być raczej graph value handle, a nie owner wszystkiego.

Robi:

```text
- wskazuje na DataView
- wskazuje na GradView, jeśli istnieje
- ma Shape
- ma RequiresGrad
- ma Ownership metadata
- może mieć NodeId/generation do debugowania
```

Nie robi:

```text
- nie powinien sam decydować skąd alokować pamięć
- nie powinien sam tworzyć grad storage bez kontekstu
- nie powinien być jednocześnie parametrem i temporary
- nie zna optimizerów
- nie wykonuje backward
```

Minimalne ownership metadata:

```csharp
public enum AutogradNodeOwnership
{
    Unknown = 0,
    Parameter = 1,
    GraphTemporary = 2,
    GraphAuxiliary = 3,
    ExternalBorrowed = 4,
    View = 5
}
```

Docelowe właściwości:

```csharp
public AutogradNodeOwnership Ownership { get; }
public bool OwnsDataStorage { get; }
public bool OwnsGradStorage { get; }
public bool IsDisposed { get; }
```

To pozwala debugować i egzekwować lifecycle.

---

### 6. `ComputationGraph`

**Odpowiedzialność:** mózg treningu/autograd.

Graf powinien spinać logikę treningową na poziomie operacji autograd.

Robi:

```text
- tworzy temporary nodes
- tworzy external/input nodes
- tworzy parameter views
- recorduje TapeOp
- wykonuje Backward
- resetuje graph temporaries
- udostępnia graph-aware ops: Linear, Conv2D, ReLU, Losses
```

Nie robi:

```text
- nie jest TrainingEngine
- nie jest optimizerem
- nie zawiera ręcznych pętli SIMD
- nie powinien być storage poole'm ogólnego przeznaczenia
- nie posiada parametrów modelu
```

Przykładowe API:

```csharp
public partial class ComputationGraph
{
    public AutogradNode Linear(
        AutogradNode input,
        AutogradNode weights,
        AutogradNode bias)
    {
        return TensorMath.Linear(this, input, weights, bias); // etap przejściowy
    }

    public AutogradNode Relu(AutogradNode input)
    {
        return TensorMath.ReLU(this, input); // etap przejściowy
    }

    public AutogradNode SoftmaxCrossEntropy(
        AutogradNode prediction,
        AutogradNode target)
    {
        return TensorMath.SoftmaxCrossEntropy(this, prediction, target);
    }
}
```

Docelowo implementacja tych metod powinna wyglądać tak:

```text
graph.Linear(...)
    -> validate shape
    -> graph.CreateTemporary(...)
    -> LinearKernels.Forward(...)
    -> graph.Record(TapeOp.Linear)
    -> return output
```

Ważne: `ComputationGraph` może mieć operacje jako metody, ale implementacja musi być rozbita przez `partial` pliki, żeby główny plik nie stał się monolitem.

Proponowany folder:

```text
Autograd/
  ComputationGraph.cs
  ComputationGraph.Linear.cs
  ComputationGraph.Conv2D.cs
  ComputationGraph.Activation.cs
  ComputationGraph.Pooling.cs
  ComputationGraph.Losses.cs
  ComputationGraph.Backward.cs
  TapeOp.cs
  OpCode.cs
  AutogradNode.cs
  AutogradNodeOwnership.cs
```

---

### 7. Graph allocator

**Odpowiedzialność:** tworzenie node'ów o właściwym lifecycle.

Może być internal.

Przykładowa abstrakcja:

```csharp
internal interface IGraphTensorAllocator
{
    AutogradNode CreateTemporary(
        TensorShape shape,
        bool requiresGrad,
        string? debugName = null);

    AutogradNode CreateExternalBorrowed(
        TensorStorage<float> data,
        TensorShape shape,
        bool requiresGrad,
        string? debugName = null);

    AutogradNode CreateParameterView(
        Parameter parameter,
        string? debugName = null);
}
```

To zabiera decyzje alokacyjne z `AutogradNode` i przenosi je do grafu.

---

### 8. `TapeOp`

**Odpowiedzialność:** zapis pojedynczej operacji do backward tape.

Robi:

```text
- OpCode
- referencje do input/output nodes
- małe context slots
- shape/context inty
```

Nie robi:

```text
- nie wykonuje forward
- nie alokuje
- nie dispose'uje sam
```

Dla performance lepszy kierunek to:

```text
struct TapeOp + enum OpCode + switch w BackwardExecutor
```

Nie polecam robić podstawowego systemu przez `IOpBackward` per op, bo może to dołożyć dispatch i komplikacje. Custom ops można dodać później jako osobną furtkę.

---

### 9. `BackwardExecutor`

**Odpowiedzialność:** wykonanie backward po tape.

Może być osobną klasą lub częścią `ComputationGraph.Backward.cs`.

Robi:

```text
- iteruje tape od końca
- switch po OpCode
- wywołuje backward kernels
- akumuluje gradienty
```

Nie robi:

```text
- nie buduje forward graph
- nie zna optimizerów
- nie zna layerów
```

Przykładowo:

```csharp
internal sealed class BackwardExecutor
{
    public void Execute(
        ReadOnlySpan<TapeOp> tape,
        AutogradNode loss)
    {
        for (var i = tape.Length - 1; i >= 0; i--)
        {
            var op = tape[i];

            switch (op.OpCode)
            {
                case OpCode.Linear:
                    // Linear backward
                    break;

                case OpCode.Relu:
                    // ReLU backward
                    break;
            }
        }
    }
}
```

---

### 10. `TensorMath` i `Kernels`

Decyzja nazewnicza:

```text
TensorMath = czysta matematyka albo facade nad kernels
Kernels    = niskopoziomowe zoptymalizowane implementacje Span-only
```

Obecne graph-aware `TensorMath.*` powinno docelowo zniknąć z layerów.

Docelowe czyste API:

```csharp
TensorMath.LinearForward(...);
TensorMath.Conv2DValidNchw(...);
TensorMath.Relu(...);
TensorMath.SoftmaxCrossEntropyForward(...);
```

Albo bez pośrednika:

```csharp
LinearKernels.Forward(...);
Conv2DKernels.ForwardValidNchw(...);
ActivationKernels.Relu(...);
LossKernels.SoftmaxCrossEntropyForward(...);
```

`Kernels` robią:

```text
- Span<float> in/out
- int shape params
- SIMD/scalar dispatch
- zero wiedzy o AutogradNode
- zero wiedzy o ComputationGraph
- brak alokacji
```

Nie robią:

```text
- nie recordują tape
- nie zarządzają ownership
- nie znają parametrów
```

---

### 11. Layers

**Odpowiedzialność:** parametry + kompozycja operacji.

Layer powinien robić:

```text
- posiadać Parameter
- znać input/output shape
- Forward training przez graph ops
- ForwardInference przez Kernels/TensorMath
- Train/Eval
- PrepareInference cache
- Save/Load parametrów
```

Layer nie powinien robić:

```text
- nie implementować długich SIMD pętli
- nie recordować TapeOp ręcznie
- nie zarządzać graph temporary lifecycle
- nie znać optimizerów
```

Docelowy styl:

```csharp
public sealed class LinearLayer : IModule
{
    public Parameter Weights { get; }
    public Parameter Bias { get; }

    public AutogradNode Forward(
        ComputationGraph graph,
        AutogradNode input)
    {
        return graph.Linear(input, Weights, Bias);
    }

    public void ForwardInference(
        ReadOnlySpan<float> input,
        Span<float> output)
    {
        LinearKernels.Forward(
            input,
            Weights.Data.AsReadOnlySpan(),
            _weightsTransposed.AsReadOnlySpan(),
            Bias.Data.AsReadOnlySpan(),
            output,
            _inputSize,
            _outputSize);
    }

    public IEnumerable<Parameter> Parameters()
    {
        yield return Weights;
        yield return Bias;
    }
}
```

---

### 12. Module interfaces

Obecne `IModule` można rozbić na role, a `IModule` zostawić jako zbiorczą fasadę.

Przykładowe interfejsy:

```csharp
public interface IParameterProvider
{
    IEnumerable<Parameter> Parameters();
}

public interface ITrainableModule
{
    AutogradNode Forward(
        ComputationGraph graph,
        AutogradNode input);
}

public interface IInferenceModule
{
    void ForwardInference(
        ReadOnlySpan<float> input,
        Span<float> output);
}

public interface IModelModeAware
{
    bool IsTraining { get; }

    void Train();

    void Eval();

    void PrepareInference();
}

public interface IModule :
    IParameterProvider,
    ITrainableModule,
    IInferenceModule,
    IModelModeAware,
    IDisposable
{
}
```

W praktyce nie trzeba tego robić od razu. To jest docelowy kierunek, gdy zacznie przeszkadzać wielka powierzchnia `IModule`.

---

### 13. `Sequential`

**Odpowiedzialność:** kompozycja modułów.

Robi:

```text
- przechowuje listę IModule
- Forward przez moduły
- ForwardInference przez moduły
- zarządza workspace inference między warstwami
- propaguje Train/Eval
- agreguje Parameters()
- Save/Load przez moduły
```

Nie robi:

```text
- nie zna matematyki Linear/Conv/ReLU
- nie zna optimizerów
- nie wykonuje backward
```

---

### 14. Losses

Loss może być częścią graph ops, bo loss nagrywa tape:

```csharp
var loss = graph.SoftmaxCrossEntropy(prediction, target);
```

Można mieć też publiczną abstrakcję dla `TrainingEngine`:

```csharp
public interface ITrainingLoss
{
    AutogradNode Forward(
        ComputationGraph graph,
        AutogradNode prediction,
        AutogradNode target);

    void Backward(
        ComputationGraph graph,
        AutogradNode loss);

    float ReadScalar(AutogradNode loss);
}
```

Docelowo konkretna implementacja:

```csharp
public sealed class SoftmaxCrossEntropyLoss : ITrainingLoss
{
    public AutogradNode Forward(
        ComputationGraph graph,
        AutogradNode prediction,
        AutogradNode target)
    {
        return graph.SoftmaxCrossEntropy(prediction, target);
    }

    public void Backward(
        ComputationGraph graph,
        AutogradNode loss)
    {
        graph.Backward(loss);
    }

    public float ReadScalar(AutogradNode loss)
    {
        return loss.DataView.AsReadOnlySpan()[0];
    }
}
```

---

### 15. Optimizers

**Odpowiedzialność:** aktualizacja parametrów.

Optimizer robi:

```text
- trzyma optimizer state
- ZeroGrad
- Step
- aktualizuje Parameter.Data na podstawie Parameter.Grad
```

Nie robi:

```text
- nie zna ComputationGraph
- nie robi forward
- nie liczy loss
- nie zna batcha
```

Docelowo:

```csharp
public interface IOptimizer
{
    void ZeroGrad();
    void Step();
}
```

Ale pod spodem powinien być zbudowany z:

```csharp
IEnumerable<Parameter>
```

nie z temporary autograd nodes.

---

### 16. `TrainingEngine`

**Odpowiedzialność:** workflow jednego kroku treningowego.

Robi:

```text
- przyjmuje batch input/target
- tworzy/wypełnia input/target nodes
- optimizer.ZeroGrad()
- model.Forward(graph, input)
- loss.Forward(graph, prediction, target)
- graph.Backward(loss)
- optimizer.Step()
- graph.Reset()
```

Nie robi:

```text
- nie implementuje matematyki
- nie zna SIMD
- nie jest ComputationGraph
- nie jest optimizerem
```

To już masz jako fasadę. W tym modelu `TrainingEngine` zostaje orkiestratorem, a `ComputationGraph` jest mózgiem autograd.

---

### 17. `InferenceEngine`

**Odpowiedzialność:** workflow inferencji.

Robi:

```text
- model.Eval()
- model.PrepareInference()
- warmup
- walidacja input/output
- preallocated output buffer
- Run/Predict
```

Nie robi:

```text
- nie zna AutogradNode
- nie zna ComputationGraph
- nie zna gradientów
- nie zna optimizerów
```

Inference path powinien być całkowicie oddzielony od autograd.

---

## Docelowy flow treningu

```text
TrainingEngine.TrainBatch(input, target)
    -> inputNode = graph.CreateExternalBorrowed(...) albo preallocated input node
    -> targetNode = graph.CreateExternalBorrowed(...) albo preallocated target node
    -> optimizer.ZeroGrad()
    -> prediction = model.Forward(graph, inputNode)
        -> LinearLayer.Forward
            -> graph.Linear(input, Weights, Bias)
                -> graph.CreateTemporary(...)
                -> LinearKernels.Forward(...)
                -> graph.Record(TapeOp.Linear)
                -> return outputNode
    -> lossNode = graph.SoftmaxCrossEntropy(prediction, targetNode)
    -> graph.Backward(lossNode)
        -> BackwardExecutor
            -> Linear backward kernels
            -> Conv backward kernels
    -> optimizer.Step()
    -> graph.Reset()
```

---

## Docelowy flow inferencji

```text
InferenceEngine.Predict(input)
    -> Sequential.ForwardInference(input, output)
        -> LinearLayer.ForwardInference
            -> LinearKernels.Forward
        -> ReluActivation.ForwardInference
            -> ActivationKernels.Relu
        -> ConvLayer.ForwardInference
            -> Conv2DKernels.ForwardValidNchw
```

Inference nie powinno dotykać:

```text
AutogradNode
ComputationGraph
TapeOp
Grad
Optimizer
Loss
```

---

## Plan refaktoru etapami

### Etap 0 — zamrozić aktualny milestone inference

Status:

```text
- inference zero-alloc działa
- MLP/CNN szybsze niż ONNX Runtime w małych workloadach
- kernels częściowo wyciągnięte z layerów
- TrainingEngine istnieje jako fasada treningu
```

Działanie:

```text
- commit aktualnego stabilnego stanu
- nie mieszać kolejnego refaktoru z inference optimization
```

---

### Etap 1 — Graph operation facade

Cel: layer ma wołać `graph.*`, nie `TensorMath.*`.

Zakres:

```text
- dodać partial ComputationGraph.* wrappery
- nie przenosić jeszcze implementacji TensorMath
- zmienić layery, żeby wołały graph.Linear / graph.Conv2D / graph.Relu itd.
```

Pliki:

```text
Sources/Main/Autograd/ComputationGraph.Linear.cs
Sources/Main/Autograd/ComputationGraph.Conv2D.cs
Sources/Main/Autograd/ComputationGraph.Activation.cs
Sources/Main/Autograd/ComputationGraph.Pooling.cs
Sources/Main/Autograd/ComputationGraph.Losses.cs
```

Przykład:

```csharp
public partial class ComputationGraph
{
    public AutogradNode Linear(
        AutogradNode input,
        AutogradNode weights,
        AutogradNode bias)
    {
        return TensorMath.Linear(this, input, weights, bias);
    }
}
```

Layer po zmianie:

```csharp
public AutogradNode Forward(
    ComputationGraph graph,
    AutogradNode input)
{
    return graph.Linear(input, Weights, Bias);
}
```

Ryzyko: niskie. To głównie zmiana API/wywołań.

---

### Etap 2 — `AutogradNodeOwnership`

Cel: nazwać lifecycle node'ów bez zmiany zachowania.

Zakres:

```text
- dodać AutogradNodeOwnership enum
- dodać Ownership do AutogradNode
- dodać debug/testy ownership
- nie zmieniać jeszcze Reset/Dispose semantyki
```

Pliki:

```text
Sources/Main/Autograd/AutogradNodeOwnership.cs
Sources/Main/Autograd/AutogradNode.cs
Tests/AutogradOwnershipTests.cs
```

Przykład:

```csharp
public enum AutogradNodeOwnership
{
    Unknown = 0,
    Parameter = 1,
    GraphTemporary = 2,
    GraphAuxiliary = 3,
    ExternalBorrowed = 4,
    View = 5
}
```

Ryzyko: niskie, jeśli tylko dodajemy metadata.

---

### Etap 3 — Graph factory methods

Cel: graf zaczyna jawnie tworzyć node'y według ownership.

Zakres:

```text
- graph.CreateTemporary(...)
- graph.CreateExternalBorrowed(...)
- graph.CreateAuxiliary(...)
- graph.CreateParameterView(...)
```

Przykład:

```csharp
public AutogradNode CreateTemporary(
    TensorShape shape,
    bool requiresGrad,
    string? debugName = null)
{
    // allocate storage from graph arena/pool
    // create node with Ownership = GraphTemporary
}
```

Ryzyko: średnie. Jeszcze bez przepinania całego TensorMath.

---

### Etap 4 — `Parameter` jako osobny typ

Cel: oddzielić długowieczne parametry modelu od temporary nodes.

Zakres:

```text
- dodać Parameter
- dodać ParameterCollection
- dodać ParameterFactory opcjonalnie
- nie przepinać jeszcze wszystkich layerów naraz
```

Pliki:

```text
Sources/Main/Parameters/Parameter.cs
Sources/Main/Parameters/ParameterCollection.cs
Sources/Main/Parameters/ParameterFactory.cs
Tests/ParameterTests.cs
```

Przykład:

```csharp
public sealed class Parameter : IDisposable
{
    public TensorStorage<float> Data { get; }
    public TensorStorage<float> Grad { get; }
    public TensorShape Shape { get; }
    public bool RequiresGrad { get; }

    public void ZeroGrad()
    {
        Grad.AsSpan().Clear();
    }
}
```

Ryzyko: niskie, jeśli typ tylko dodajemy.

---

### Etap 5 — przepiąć `LinearLayer` na `Parameter`

Cel: pierwszy layer używa nowego modelu parametrów.

Zakres:

```text
- LinearLayer.Weights: Parameter
- LinearLayer.Bias: Parameter
- graph.Linear overload na Parameter
- Parameters() zwraca Parameter albo adapter przejściowy
```

Przykład:

```csharp
public AutogradNode Forward(
    ComputationGraph graph,
    AutogradNode input)
{
    return graph.Linear(input, Weights, Bias);
}
```

Ryzyko: średnie. Tylko jeden layer.

---

### Etap 6 — optimizer adapter / Adam na `Parameter`

Cel: optimizer nie widzi temporary `AutogradNode`.

Zakres:

```text
- Adam(IEnumerable<Parameter>)
- ewentualnie adapter dla starego API
- ZeroGrad/Step po Parameter.Data/Grad
```

Przykład:

```csharp
public sealed class Adam : IOptimizer
{
    private readonly Parameter[] _parameters;

    public Adam(IEnumerable<Parameter> parameters, float learningRate)
    {
        _parameters = parameters.ToArray();
    }

    public void ZeroGrad()
    {
        foreach (var p in _parameters)
        {
            p.ZeroGrad();
        }
    }

    public void Step()
    {
        // update p.Data using p.Grad
    }
}
```

Ryzyko: średnie/wysokie, bo dotyka treningu.

---

### Etap 7 — przenieść implementacje graph-aware z `TensorMath` do `ComputationGraph.*`

Cel: `TensorMath` przestaje być graph-aware.

Zakres:

```text
- graph.Linear zawiera autograd wrapper
- graph.Conv2D zawiera autograd wrapper
- graph.Relu zawiera autograd wrapper
- graph.SoftmaxCrossEntropy zawiera autograd wrapper
- TensorMath zostaje jako pure math albo znika
```

Przykład docelowy:

```csharp
public AutogradNode Linear(
    AutogradNode input,
    Parameter weights,
    Parameter bias)
{
    var output = CreateTemporary(
        new TensorShape(input.Shape.D0, weights.Shape.D1),
        requiresGrad: input.RequiresGrad || weights.RequiresGrad || bias.RequiresGrad);

    LinearKernels.Forward(
        input.DataView.AsReadOnlySpan(),
        weights.Data.AsReadOnlySpan(),
        weights.TransposedCache.AsReadOnlySpan(),
        bias.Data.AsReadOnlySpan(),
        output.DataView.AsSpan(),
        weights.Shape.D0,
        weights.Shape.D1);

    RecordLinear(input, weights, bias, output);

    return output;
}
```

Ryzyko: średnie/wysokie. Robić operacja po operacji.

---

### Etap 8 — Graph Reset cleanup

Cel: graf sprząta według ownership, nie według ręcznych wyjątków.

Zakres:

```text
- graph.Reset dispose'uje tylko GraphTemporary / GraphAuxiliary
- nie dotyka Parameter / ExternalBorrowed
- debug assert: node dispose exactly once
```

Przykład:

```csharp
private static void DisposeIfGraphOwned(AutogradNode? node)
{
    if (node is null)
    {
        return;
    }

    if (node.Ownership is AutogradNodeOwnership.GraphTemporary or
        AutogradNodeOwnership.GraphAuxiliary)
    {
        node.Dispose();
    }
}
```

Ryzyko: wysokie. Robić dopiero gdy ownership jest pewny.

---

### Etap 9 — Backward kernels cleanup

Cel: backward używa czystych kernelów, analogicznie do inference forward.

Zakres:

```text
- LinearKernels.BackwardInput
- LinearKernels.AccumulateWeightGrad
- Conv2DKernels.BackwardInput
- Conv2DKernels.AccumulateKernelGrad
- LossKernels backward helpers
```

Ryzyko: średnie/wysokie. To jest etap performance.

---

### Etap 10 — Training performance work

Dopiero po ownership cleanup.

Zakres:

```text
- profilowanie ZeroGrad / Forward / Loss / Backward / Step / Reset
- redukcja alokacji, jeśli mają znaczenie
- optimizer kernels
- graph arena improvements
- custom scheduler, jeśli nadal potrzebny
```

Na tym etapie alokacje treningowe mogą istnieć. Liczy się przede wszystkim performance i poprawny lifecycle.

---

## Zasady refaktoru

### 1. Nie mieszać refaktoru architektury z optymalizacją

Jeden PR = jeden cel.

Dobre:

```text
PR: Add graph operation facade
PR: Add AutogradNodeOwnership metadata
PR: Introduce Parameter type
```

Złe:

```text
PR: Add Parameter + rewrite Adam + optimize Conv backward + change benchmarks
```

---

### 2. Najpierw adaptery, potem migracja

Najpierw dodać nowe API jako wrapper nad starym.

Dopiero potem przepinać layery i implementacje.

---

### 3. Inference zostawić oddzielone od graph

Inference path jest już dobry i zero-alloc. Nie podłączać inferencji z powrotem pod `ComputationGraph`.

---

### 4. Kernels nie znają autograd

Jeśli metoda przyjmuje `AutogradNode`, to nie jest kernel.

---

### 5. Metoda z `ComputationGraph` w sygnaturze powinna być metodą grafu

Docelowo nie chcemy:

```csharp
TensorMath.Linear(graph, input, weights, bias);
```

Chcemy:

```csharp
graph.Linear(input, weights, bias);
```

---

## Minimalny pierwszy PR

Najbezpieczniejszy pierwszy krok:

```text
PR: Add graph operation facade
```

Zakres:

```text
1. Dodać partial ComputationGraph.* pliki.
2. Każda metoda deleguje do obecnego TensorMath.
3. Przepiąć layery na graph.*.
4. Testy.
5. Zero zmian w algorytmach.
```

Przykładowe pliki:

```text
Sources/Main/Autograd/ComputationGraph.Linear.cs
Sources/Main/Autograd/ComputationGraph.Conv2D.cs
Sources/Main/Autograd/ComputationGraph.Activation.cs
Sources/Main/Autograd/ComputationGraph.Pooling.cs
Sources/Main/Autograd/ComputationGraph.Losses.cs
```

Przykładowe wrappery:

```csharp
public partial class ComputationGraph
{
    public AutogradNode Linear(
        AutogradNode input,
        AutogradNode weights,
        AutogradNode bias)
    {
        return TensorMath.Linear(this, input, weights, bias);
    }

    public AutogradNode Conv2D(
        AutogradNode input,
        AutogradNode kernels,
        int inChannels,
        int outChannels,
        int inputH,
        int inputW,
        int kernelSize)
    {
        return TensorMath.Conv2D(
            this,
            input,
            kernels,
            inChannels,
            outChannels,
            inputH,
            inputW,
            kernelSize);
    }

    public AutogradNode Relu(AutogradNode input)
    {
        return TensorMath.ReLU(this, input);
    }

    public AutogradNode SoftmaxCrossEntropy(
        AutogradNode prediction,
        AutogradNode target)
    {
        return TensorMath.SoftmaxCrossEntropy(this, prediction, target);
    }
}
```

Potem layer:

```csharp
public AutogradNode Forward(
    ComputationGraph graph,
    AutogradNode input)
{
    return graph.Linear(input, Weights, Bias);
}
```

---

## Docelowy stan po refaktorze

Po pełnej migracji:

```text
TrainingEngine
    orchestrates workflow

ComputationGraph
    owns autograd runtime and graph-aware operations

Parameter
    owns model trainable state

AutogradNode
    is graph-visible value handle

Kernels/TensorMath
    pure math only

Layers
    compose graph ops and own parameters

Optimizers
    update Parameters

InferenceEngine
    separate zero-alloc inference workflow
```

Najważniejszy efekt:

```text
AutogradNode przestaje być wszystkim naraz.
ComputationGraph jest mózgiem treningu, ale nie mięśniami matematyki.
TensorMath/Kernels są czystą matematyką.
Layer jest prosty i czytelny.
Optimizer nie widzi temporary graph nodes.
```
