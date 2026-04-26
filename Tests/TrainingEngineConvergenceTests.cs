using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Licensing;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Optimizers;
using DevOnBike.Overfit.Training;

namespace DevOnBike.Overfit.Tests
{
    public sealed class TrainingEngineConvergenceTests
    {
        [Fact]
        public void TrainingEngine_SingleLinear_SoftmaxCrossEntropy_LossDecreases()
        {
            OverfitLicense.SuppressNotice = true;

            const int batchSize = 4;
            const int inputSize = 2;
            const int classCount = 2;
            const int steps = 80;

            var input =
                new float[]
                {
                    1f, 0f,
                    0f, 1f,
                    1f, 0f,
                    0f, 1f
                };

            var target =
                new float[]
                {
                    1f, 0f,
                    0f, 1f,
                    1f, 0f,
                    0f, 1f
                };

            using var model = new Sequential(
                new LinearLayer(inputSize, classCount));

            ZeroParameters(model);

            var adam = new Adam(
                model.Parameters(),
                learningRate: 0.05f);

            var optimizer = new DelegateTrainingOptimizer(
                zeroGrad: adam.ZeroGrad,
                step: adam.Step);

            var loss = CreateSoftmaxCrossEntropyLoss();

            using var trainer = TrainingEngine.FromBackend(
                new SequentialTrainingBackend(
                    model,
                    optimizer,
                    loss,
                    batchSize,
                    inputSize,
                    classCount,
                    new TrainingEngineOptions
                    {
                        ResetGraphAfterStep = true,
                        ValidateFiniteInput = true,
                        ValidateFiniteTarget = true
                    }));

            var first = trainer.TrainBatch(input, target).Loss;
            var last = first;

            for (var i = 1; i < steps; i++)
            {
                last = trainer.TrainBatch(input, target).Loss;
            }

            Assert.True(
                last < first,
                $"Expected loss to decrease. First={first}, Last={last}");

            Assert.True(
                last < first * 0.75f,
                $"Expected meaningful loss decrease. First={first}, Last={last}");
        }

        [Fact]
        public void TrainingEngine_Mlp_SoftmaxCrossEntropy_LossDecreases()
        {
            OverfitLicense.SuppressNotice = true;

            const int batchSize = 4;
            const int inputSize = 2;
            const int hiddenSize = 8;
            const int classCount = 2;
            const int steps = 120;

            var input =
                new float[]
                {
                    1f, 0f,
                    0f, 1f,
                    1f, 0f,
                    0f, 1f
                };

            var target =
                new float[]
                {
                    1f, 0f,
                    0f, 1f,
                    1f, 0f,
                    0f, 1f
                };

            using var model = new Sequential(
                new LinearLayer(inputSize, hiddenSize),
                new ReluActivation(),
                new LinearLayer(hiddenSize, classCount));

            ZeroParameters(model);

            // Avoid dead ReLU at exact zero for the first layer.
            InitializeFirstLayerForIdentityLikeSignal(model);

            var adam = new Adam(
                model.Parameters(),
                learningRate: 0.03f);

            var optimizer = new DelegateTrainingOptimizer(
                zeroGrad: adam.ZeroGrad,
                step: adam.Step);

            var loss = CreateSoftmaxCrossEntropyLoss();

            using var trainer = TrainingEngine.FromBackend(
                new SequentialTrainingBackend(
                    model,
                    optimizer,
                    loss,
                    batchSize,
                    inputSize,
                    classCount,
                    new TrainingEngineOptions
                    {
                        ResetGraphAfterStep = true,
                        ValidateFiniteInput = true,
                        ValidateFiniteTarget = true
                    }));

            var first = trainer.TrainBatch(input, target).Loss;
            var last = first;

            for (var i = 1; i < steps; i++)
            {
                last = trainer.TrainBatch(input, target).Loss;
            }

            Assert.True(
                last < first,
                $"Expected loss to decrease. First={first}, Last={last}");

            Assert.True(
                last < first * 0.75f,
                $"Expected meaningful loss decrease. First={first}, Last={last}");
        }

        [Fact]
        public void TrainingEngine_TrainBatch_RejectsWrongInputLength()
        {
            OverfitLicense.SuppressNotice = true;

            const int batchSize = 4;
            const int inputSize = 2;
            const int classCount = 2;

            using var model = new Sequential(
                new LinearLayer(inputSize, classCount));

            var adam = new Adam(
                model.Parameters(),
                learningRate: 0.01f);

            var optimizer = new DelegateTrainingOptimizer(
                zeroGrad: adam.ZeroGrad,
                step: adam.Step);

            var loss = CreateSoftmaxCrossEntropyLoss();

            using var trainer = TrainingEngine.FromBackend(
                new SequentialTrainingBackend(
                    model,
                    optimizer,
                    loss,
                    batchSize,
                    inputSize,
                    classCount));

            var badInput = new float[batchSize * inputSize - 1];
            var target = new float[batchSize * classCount];

            Assert.Throws<ArgumentException>(
                () => trainer.TrainBatch(badInput, target));
        }

        [Fact]
        public void TrainingEngine_TrainBatch_RejectsWrongTargetLength()
        {
            OverfitLicense.SuppressNotice = true;

            const int batchSize = 4;
            const int inputSize = 2;
            const int classCount = 2;

            using var model = new Sequential(
                new LinearLayer(inputSize, classCount));

            var adam = new Adam(
                model.Parameters(),
                learningRate: 0.01f);

            var optimizer = new DelegateTrainingOptimizer(
                zeroGrad: adam.ZeroGrad,
                step: adam.Step);

            var loss = CreateSoftmaxCrossEntropyLoss();

            using var trainer = TrainingEngine.FromBackend(
                new SequentialTrainingBackend(
                    model,
                    optimizer,
                    loss,
                    batchSize,
                    inputSize,
                    classCount));

            var input = new float[batchSize * inputSize];
            var badTarget = new float[batchSize * classCount - 1];

            Assert.Throws<ArgumentException>(
                () => trainer.TrainBatch(input, badTarget));
        }

        private static DelegateTrainingLoss CreateSoftmaxCrossEntropyLoss()
        {
            return new DelegateTrainingLoss(
                forward: (graph, prediction, target) =>
                    TensorMath.SoftmaxCrossEntropy(
                        graph,
                        prediction,
                        target),

                backward: (graph, lossNode) =>
                    graph.Backward(lossNode));
        }

        private static void ZeroParameters(
            Sequential model)
        {
            foreach (var parameter in model.Parameters())
            {
                parameter.DataView.AsSpan().Clear();
            }

            model.InvalidateParameterCaches();
        }

        private static void InitializeFirstLayerForIdentityLikeSignal(
            Sequential model)
        {
            var parameters = model.Parameters().ToArray();

            if (parameters.Length < 2)
            {
                return;
            }

            var firstWeights = parameters[0].DataView.AsSpan();

            if (firstWeights.Length < 16)
            {
                return;
            }

            // Linear(2, 8), weights layout: [input, output].
            // Make some hidden units initially active for both input dimensions.
            firstWeights[0 * 8 + 0] = 1f;
            firstWeights[0 * 8 + 1] = 0.5f;
            firstWeights[0 * 8 + 2] = 0.25f;
            firstWeights[0 * 8 + 3] = 0.1f;

            firstWeights[1 * 8 + 4] = 1f;
            firstWeights[1 * 8 + 5] = 0.5f;
            firstWeights[1 * 8 + 6] = 0.25f;
            firstWeights[1 * 8 + 7] = 0.1f;

            model.InvalidateParameterCaches();
        }
    }
}