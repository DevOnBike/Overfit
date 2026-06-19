// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Contracts
{
    public interface ISlmModel : IDisposable
    {
        int VocabularySize
        {
            get;
        }

        int ContextLength
        {
            get;
        }

        int LayerCount
        {
            get;
        }

        int HeadCount
        {
            get;
        }

        int HeadDimension
        {
            get;
        }

        int EmbeddingDimension
        {
            get;
        }

        long ParameterCount
        {
            get;
        }

        string ArchitectureName
        {
            get;
        }

        bool IsTraining
        {
            get;
        }

        void Train();

        void Eval();
    }
}
