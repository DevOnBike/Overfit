// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.

using System.Numerics.Tensors;
using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    public sealed class BatchNorm1D : IModule
    {
        public AutogradNode Gamma { get; }
        public AutogradNode Beta { get; }
        public FastTensor<float> RunningMean { get; }
        public FastTensor<float> RunningVar { get; }
        public float Momentum { get; set; } = 0.1f;
        public float Eps { get; set; } = 1e-5f;
        public bool IsTraining { get; private set; } = true;

        private FastTensor<float> _fusedScale;
        private FastTensor<float> _fusedShift;

        public BatchNorm1D(int numFeatures)
        {
            Gamma = new AutogradNode(new FastTensor<float>(numFeatures, clearMemory: false), requiresGrad: true);
            Beta = new AutogradNode(new FastTensor<float>(numFeatures, clearMemory: false), requiresGrad: true);

            Gamma.DataView.AsSpan().Fill(1f);
            Beta.DataView.AsSpan().Fill(0f);

            RunningMean = new FastTensor<float>(numFeatures, clearMemory: false);
            RunningVar = new FastTensor<float>(numFeatures, clearMemory: false);
            
            RunningVar.GetView().AsSpan().Fill(1f);
        }

        public void Train()
        {
            IsTraining = true;
        }

        public void Eval()
        {
            IsTraining = false;

            // Fuzja parametrów - liczymy Scale i Shift TYLKO RAZ podczas przełączania w tryb Eval
            var C = Gamma.DataView.Size;
            
            if (_fusedScale == null)
            {
                _fusedScale = new FastTensor<float>(C, clearMemory: false);
                _fusedShift = new FastTensor<float>(C, clearMemory: false);
            }

            var scaleS = _fusedScale.GetView().AsSpan();
            var shiftS = _fusedShift.GetView().AsSpan();
            var gammaS = Gamma.DataView.AsReadOnlySpan();
            var betaS = Beta.DataView.AsReadOnlySpan();
            var meanS = RunningMean.GetView().AsReadOnlySpan();
            var varS = RunningVar.GetView().AsReadOnlySpan();

            for (var i = 0; i < C; i++)
            {
                scaleS[i] = gammaS[i] / MathF.Sqrt(varS[i] + Eps);
                shiftS[i] = betaS[i] - meanS[i] * scaleS[i];
            }
        }

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
        {
            if (IsTraining)
            {
                throw new InvalidOperationException("Layer must be in Eval mode.");
            }

            // MISTRZOSTWO AVX-512: Cały BatchNorm wykonany jako jedna operacja y = (x * Scale) + Shift!
            TensorPrimitives.MultiplyAdd(
                input,
                _fusedScale.GetView().AsReadOnlySpan(),
                _fusedShift.GetView().AsReadOnlySpan(),
                output
            );
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

        public void Save(BinaryWriter bw)
        {
            var len = Gamma.DataView.Size;
            bw.Write(len);

            foreach (var val in Gamma.DataView.AsSpan())
            {
                bw.Write(val);
            }
            foreach (var val in Beta.DataView.AsSpan())
            {
                bw.Write(val);
            }
            foreach (var val in RunningMean.GetView().AsSpan())
            {
                bw.Write(val);
            }
            foreach (var val in RunningVar.GetView().AsSpan())
            {
                bw.Write(val);
            }
        }

        public void Load(BinaryReader br)
        {
            var len = br.ReadInt32();
            if (len != Gamma.DataView.Size)
            {
                throw new Exception($"Weight dimensions mismatch: {len} vs {Gamma.DataView.Size}");
            }

            var gSpan = Gamma.DataView.AsSpan();
            for (var i = 0; i < len; i++)
            {
                gSpan[i] = br.ReadSingle();
            }

            var bSpan = Beta.DataView.AsSpan();
            for (var i = 0; i < len; i++)
            {
                bSpan[i] = br.ReadSingle();
            }

            var rmSpan = RunningMean.GetView().AsSpan();
            for (var i = 0; i < len; i++)
            {
                rmSpan[i] = br.ReadSingle();
            }

            var rvSpan = RunningVar.GetView().AsSpan();
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