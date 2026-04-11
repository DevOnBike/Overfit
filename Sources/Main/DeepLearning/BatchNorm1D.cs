// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    ///     Implements 1D Batch Normalization as described by Ioffe and Szegedy.
    ///     Normalizes activations to improve training stability and speed up convergence.
    /// </summary>
    public sealed class BatchNorm1D : IModule
    {

        public BatchNorm1D(int numFeatures)
        {
            Gamma = new AutogradNode(new FastTensor<float>(numFeatures));
            Beta = new AutogradNode(new FastTensor<float>(numFeatures));

            Gamma.Data.AsSpan().Fill(1f);
            Beta.Data.AsSpan().Fill(0f);

            RunningMean = new FastTensor<float>(numFeatures);
            RunningVar = new FastTensor<float>(numFeatures);

            RunningMean.AsSpan().Fill(0f);
            RunningVar.AsSpan().Fill(1f);
        }
        /// <summary> Learned scale parameter. </summary>
        public AutogradNode Gamma { get; }

        /// <summary> Learned bias/shift parameter. </summary>
        public AutogradNode Beta { get; }

        /// <summary> Cumulative moving average of the input mean. </summary>
        public FastTensor<float> RunningMean { get; }

        /// <summary> Cumulative moving average of the input variance. </summary>
        public FastTensor<float> RunningVar { get; }

        /// <summary> Exponential moving average factor for running statistics. </summary>
        public float Momentum { get; set; } = 0.1f;

        /// <summary> Small constant added to the variance for numerical stability (avoiding division by zero). </summary>
        public float Eps { get; set; } = 1e-5f;

        public bool IsTraining { get; private set; } = true;

        public void Train()
        {
            IsTraining = true;
        }
        public void Eval()
        {
            IsTraining = false;
        }

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            return TensorMath.BatchNorm1D(graph, input, Gamma, Beta, RunningMean, RunningVar, Momentum, Eps, IsTraining);
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            yield return Gamma;
            yield return Beta;
        }

        /// <summary>
        ///     Saves parameters and running statistics to a binary stream.
        /// </summary>
        public void Save(BinaryWriter bw)
        {
            var len = Gamma.Data.Size;
            bw.Write(len);

            foreach (var val in Gamma.Data.AsSpan())
            {
                bw.Write(val);
            }
            foreach (var val in Beta.Data.AsSpan())
            {
                bw.Write(val);
            }
            foreach (var val in RunningMean.AsSpan())
            {
                bw.Write(val);
            }
            foreach (var val in RunningVar.AsSpan())
            {
                bw.Write(val);
            }
        }

        /// <summary>
        ///     Loads parameters and running statistics from a binary stream.
        ///     Performs a dimension check to ensure architectural consistency.
        /// </summary>
        public void Load(BinaryReader br)
        {
            var len = br.ReadInt32();

            if (len != Gamma.Data.Size)
            {
                throw new Exception($"Weight dimensions in file ({len}) do not match the layer architecture ({Gamma.Data.Size}).");
            }

            var gSpan = Gamma.Data.AsSpan();
            for (var i = 0; i < len; i++)
            {
                gSpan[i] = br.ReadSingle();
            }

            var bSpan = Beta.Data.AsSpan();
            for (var i = 0; i < len; i++)
            {
                bSpan[i] = br.ReadSingle();
            }

            var rmSpan = RunningMean.AsSpan();
            for (var i = 0; i < len; i++)
            {
                rmSpan[i] = br.ReadSingle();
            }

            var rvSpan = RunningVar.AsSpan();
            for (var i = 0; i < len; i++)
            {
                rvSpan[i] = br.ReadSingle();
            }
        }

        public void Dispose()
        {
            Gamma?.Dispose();
            Beta?.Dispose();
            RunningMean?.Dispose();
            RunningVar?.Dispose();
        }
    }
}