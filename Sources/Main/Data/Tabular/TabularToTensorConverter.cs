// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Linq;
using DevOnBike.Overfit.Data.Contracts;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Data.Tabular
{
    /// <summary>
    ///     Converts tabular data (objects/models) into high-performance <see cref="FastTensor{T}" /> objects.
    ///     Handles automated One-Hot Encoding for categorical variables and type mapping for numeric/binary data.
    /// </summary>
    /// <typeparam name="T">The type of the input data model.</typeparam>
    public class TabularToTensorConverter<T>
    {
        private readonly Dictionary<string, string[]> _categoryMaps = new();
        private readonly TableSchema _schema;

        private readonly Func<T, string, object> _valueExtractor;
        private int _featureWidth;

        public TabularToTensorConverter(TableSchema schema, Func<T, string, object> valueExtractor)
        {
            _schema = schema;
            _valueExtractor = valueExtractor ?? throw new ArgumentNullException(nameof(valueExtractor));
        }

        public void Fit(IReadOnlyList<T> data)
        {
            _featureWidth = 0;
            _categoryMaps.Clear();

            foreach (var col in _schema.Features)
            {
                if (col.Type == ColumnType.Categorical)
                {
                    var uniqueValues = new HashSet<string>();
                    foreach (var item in data)
                    {
                        var val = GetValue(item, col.Name)?.ToString();
                        if (val != null)
                        {
                            uniqueValues.Add(val);
                        }
                    }

                    var categories = uniqueValues.OrderBy(x => x).ToArray();
                    _categoryMaps[col.Name] = categories;
                    _featureWidth += categories.Length;
                }
                else
                {
                    _featureWidth += 1;
                }
            }
        }

        public (FastTensor<float> Features, FastTensor<float> Targets) Convert(IReadOnlyList<T> data)
        {
            var rowCount = data.Count;

            // Alokujemy nowe tensory. clearMemory: false, bo zaraz precyzyjnie nadpiszemy każdy bajt.
            var features = new FastTensor<float>(rowCount, _featureWidth, clearMemory: false);
            var targets = new FastTensor<float>(rowCount, 1, clearMemory: false);

            var fSpan = features.GetView().AsSpan();
            var tSpan = targets.GetView().AsSpan();

            for (var i = 0; i < rowCount; i++)
            {
                var rowOffset = i * _featureWidth;
                var currentPos = 0;

                foreach (var col in _schema.Features)
                {
                    var val = GetValue(data[i], col.Name);

                    if (col.Type == ColumnType.Numeric)
                    {
                        fSpan[rowOffset + currentPos++] = System.Convert.ToSingle(val);
                    }
                    else if (col.Type == ColumnType.Binary)
                    {
                        fSpan[rowOffset + currentPos++] = System.Convert.ToBoolean(val) ? 1f : 0f;
                    }
                    else if (col.Type == ColumnType.Categorical)
                    {
                        var categories = _categoryMaps[col.Name];
                        var currentVal = val?.ToString();

                        for (var c = 0; c < categories.Length; c++)
                        {
                            fSpan[rowOffset + currentPos++] = categories[c] == currentVal ? 1f : 0f;
                        }
                    }
                }

                tSpan[i] = System.Convert.ToSingle(GetValue(data[i], _schema.Target.Name));
            }

            return (features, targets);
        }

        private object GetValue(T item, string propName)
        {
            return _valueExtractor(item, propName);
        }
    }
}