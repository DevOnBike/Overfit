// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.Data.Contracts;

namespace DevOnBike.Overfit.Data.Prepare
{
    /// <summary>
    /// Converts tabular data (objects/models) into high-performance <see cref="FastTensor{float}"/> objects.
    /// Handles automated One-Hot Encoding for categorical variables and type mapping for numeric/binary data.
    /// </summary>
    /// <typeparam name="T">The type of the input data model.</typeparam>
    public class TabularToTensorConverter<T>
    {
        private readonly TableSchema _schema;
        private readonly Dictionary<string, string[]> _categoryMaps = new();

        private readonly Func<T, string, object> _valueExtractor;
        private int _featureWidth;

        /// <param name="schema">Defines column types and their roles (Features vs Target).</param>
        /// <param name="valueExtractor">Delegate to extract a property value by name from the input object.</param>
        public TabularToTensorConverter(TableSchema schema, Func<T, string, object> valueExtractor)
        {
            _schema = schema;
            _valueExtractor = valueExtractor ?? throw new ArgumentNullException(nameof(valueExtractor));
        }

        /// <summary>
        /// Analyzes the dataset to calculate feature width and builds category maps for One-Hot Encoding.
        /// </summary>
        public void Fit(IEnumerable<T> data)
        {
            _featureWidth = 0;

            foreach (var col in _schema.Features)
            {
                if (col.Type == ColumnType.Categorical)
                {
                    var uniqueSet = new HashSet<string>();

                    foreach (var item in data)
                    {
                        var valStr = GetValue(item, col.Name)?.ToString();

                        if (valStr != null)
                        {
                            uniqueSet.Add(valStr);
                        }
                    }

                    var uniqueValues = new string[uniqueSet.Count];
                    uniqueSet.CopyTo(uniqueValues);

                    Array.Sort(uniqueValues);

                    _categoryMaps[col.Name] = uniqueValues;
                    _featureWidth += uniqueValues.Length; // One-Hot Encoding adds multiple columns.
                }
                else
                {
                    _featureWidth += 1;
                }
            }
        }

        /// <summary>
        /// Transforms the input list into a feature tensor and a target tensor.
        /// Requires <see cref="Fit"/> to be called beforehand.
        /// </summary>
        public (FastTensor<float> Features, FastTensor<float> Targets) Transform(List<T> data)
        {
            if (_featureWidth == 0 && _schema.Features.Count > 0)
            {
                throw new InvalidOperationException("Call Fit() before calling Transform().");
            }

            var rowCount = data.Count;
            var features = new FastTensor<float>(rowCount, _featureWidth);
            var targets = new FastTensor<float>(rowCount, 1);

            var fSpan = features.AsSpan();
            var tSpan = targets.AsSpan();

            for (var i = 0; i < rowCount; i++)
            {
                var rowOffset = i * _featureWidth;
                var currentPos = 0;

                foreach (var col in _schema.Features)
                {
                    var val = GetValue(data[i], col.Name);

                    if (col.Type == ColumnType.Numeric)
                    {
                        fSpan[rowOffset + currentPos++] = Convert.ToSingle(val);
                    }
                    else if (col.Type == ColumnType.Binary)
                    {
                        fSpan[rowOffset + currentPos++] = Convert.ToBoolean(val) ? 1f : 0f;
                    }
                    else if (col.Type == ColumnType.Categorical)
                    {
                        var categories = _categoryMaps[col.Name];
                        var currentVal = val?.ToString();

                        for (var c = 0; c < categories.Length; c++)
                        {
                            fSpan[rowOffset + currentPos++] = (categories[c] == currentVal) ? 1f : 0f;
                        }
                    }
                }

                tSpan[i] = Convert.ToSingle(GetValue(data[i], _schema.Target.Name));
            }

            return (features, targets);
        }

        private object GetValue(T item, string propName)
            => _valueExtractor(item, propName);
    }
}