// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.

namespace DevOnBike.Overfit.Tensors.Core
{
    /// <summary>
    /// PURE DOD: Oddzielona logika tworzenia, klonowania i materializacji pamięci.
    /// </summary>
    public static class TensorFactory
    {
        public static TensorStorage<T> CloneStorage<T>(TensorStorage<T> template, bool clearMemory = true) where T : unmanaged
        {
            TensorStorage<T> result;

            if (template._isBorrowedMemory && template._arena != null)
            {
                result = new TensorStorage<T>(template._arena, template.Length);
            }
            else
            {
                result = new TensorStorage<T>(template.Length, false);
            }

            if (clearMemory)
            {
                result.AsSpan().Clear();
            }

            return result;
        }

        /// <summary>
        /// Tworzy ciągły bufor pamięci na podstawie (nawet poszatkowanego) widoku.
        /// </summary>
        public static TensorStorage<T> Materialize<T>(TensorView<T> view) where T : unmanaged
        {
            var storage = new TensorStorage<T>(view.Size, false);
            var targetSpan = storage.AsSpan();

            if (view.IsContiguous)
            {
                view.AsReadOnlySpan().CopyTo(targetSpan);
                return storage;
            }

            // Implementacja dla nieciągłych widoków 2D (np. po Transpose)
            if (view.Rank == 2)
            {
                var index = 0;
                
                for (var i = 0; i < view.GetDim(0); i++)
                {
                    for (var j = 0; j < view.GetDim(1); j++)
                    {
                        targetSpan[index++] = view[i, j];
                    }
                }
                
                return storage;
            }

            throw new NotImplementedException("Kopiowanie nieciągłych widoków > 2D nie jest zaimplementowane.");
        }
    }
}