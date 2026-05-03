// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
//
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Adapter that exposes the current GPT1Model through the generic SLM model contract.
    ///
    /// This type does not own the model. The caller is responsible for disposing
    /// the wrapped GPT1Model unless another owner is explicitly introduced later.
    /// </summary>
    public sealed class Gpt1SlmModelAdapter : ISlmModel
    {
        private readonly GPT1Model _model;

        public Gpt1SlmModelAdapter(GPT1Model model)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
        }

        public GPT1Model InnerModel => _model;

        public int VocabularySize => _model.Config.VocabSize;

        public int ContextLength => _model.Config.ContextLength;

        public int LayerCount => _model.Config.NLayers;

        public int HeadCount => _model.Config.NHeads;

        public int HeadDimension => _model.Config.DModel / _model.Config.NHeads;

        public int EmbeddingDimension => _model.Config.DModel;

        public long ParameterCount => _model.Config.ParameterCount;

        public string ArchitectureName => "GPT-1";

        public bool IsTraining => _model.IsTraining;

        public void Train()
        {
            _model.Train();
        }

        public void Eval()
        {
            _model.Eval();
        }

        public void Dispose()
        {
            // Adapter does not own the wrapped model.
            // Ownership stays with the caller or the future engine factory.
        }
    }
}
