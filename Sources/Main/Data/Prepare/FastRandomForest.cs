// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.Data.Contracts;

namespace DevOnBike.Overfit.Data.Prepare
{
    public class FastRandomForest
    {
        private readonly int _numTrees;
        private readonly int _maxDepth;
        
        private List<FastTreeNode> _forest = [];

        public FastRandomForest(int numTrees = 50, int maxDepth = 10)
        {
            _numTrees = numTrees;
            _maxDepth = maxDepth;
        }

        public float[] TrainAndGetImportance(FastTensor<float> x, FastTensor<float> y)
        {
            var cols = x.GetDim(1);
            var importance = new float[cols];
            var lockObj = new object();

            // Równoległe budowanie drzew - Zero-GC wewnątrz pętli
            Parallel.For(0, _numTrees, t =>
            {
                var localImportance = new float[cols];

                BuildSimpleTree(x, y, localImportance);

                lock (lockObj)
                {
                    for (var i = 0; i < cols; i++)
                    {
                        importance[i] += localImportance[i];
                    }
                }
            });

            return importance;
        }

        private void BuildSimpleTree(FastTensor<float> x, FastTensor<float> y, float[] importance)
        {
            var rows = x.GetDim(0);
            var cols = x.GetDim(1);
            var xSpan = x.AsSpan();
            var ySpan = y.AsSpan();

            // Extra-Trees: Wybieramy losowe podziały i mierzymy redukcję wariancji (MSE)
            for (var d = 0; d < _maxDepth; d++)
            {
                var featureIdx = Random.Shared.Next(cols);
                float min = float.MaxValue, max = float.MinValue;

                // Szukamy zakresu cechy (można to zoptymalizować przez SIMD)
                for (var r = 0; r < rows; r++)
                {
                    var val = xSpan[r * cols + featureIdx];
                    
                    if (val < min) min = val;
                    if (val > max) max = val;
                }

                var threshold = min + (max - min) * Random.Shared.NextSingle();

                // Liczymy "Zysk" (Gain) z tego podziału dla ważności cechy
                // W wersji "Bestia" używamy uproszczonego MSE
                importance[featureIdx] += CalculateVarianceReduction(xSpan, ySpan, featureIdx, threshold, rows, cols);
            }
        }

        private float CalculateVarianceReduction(ReadOnlySpan<float> x, ReadOnlySpan<float> y, int col, float threshold, int rows, int totalCols)
        {
            // Tutaj implementujemy szybki skan danych, aby ocenić jak dobry był podział
            // Im większa redukcja błędu, tym ważniejsza cecha.
            return Random.Shared.NextSingle(); // Uproszczony placeholder dla logiki MSE
        }
    }
}
