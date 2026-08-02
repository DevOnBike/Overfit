// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
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

                    var categoriesList = new List<string>(uniqueValues);
                    categoriesList.Sort();
                    var categories = categoriesList.ToArray();
                    _categoryMaps[col.Name] = categories;
                    _featureWidth += categories.Length;

                    continue;
                }

                _featureWidth += 1;
            }
        }

        /// <summary>Total one-hot-expanded feature width, or zero before <see cref="Fit"/>.</summary>
        public int FeatureWidth => _featureWidth;

        /// <summary>
        /// The fitted category ordering, as text, so it can be stored beside the weights.
        ///
        /// <para><b>Without this, a model could be reloaded and fed scrambled inputs with nothing objecting.</b>
        /// The ordering is decided by whatever data <see cref="Fit"/> happened to see, so re-fitting at
        /// inference on a different set produces one-hot columns of the <i>same width</i> in a
        /// <i>different order</i>. <c>ModelSerializer</c> validates tensor shape and not schema, so every shape
        /// check passes, nothing throws, and the model reads the wrong column for every category.</para>
        ///
        /// <para>Format: one line per categorical column, <c>name</c> then its categories, tab-separated.
        /// Deliberately plain text - this is a compatibility record, and a record nobody can read by eye is
        /// one nobody checks.</para>
        /// </summary>
        public string WriteCategories()
        {
            var text = new StringBuilder();

            foreach (var (name, categories) in _categoryMaps)
            {
                text.Append(name);

                for (var i = 0; i < categories.Length; i++)
                {
                    text.Append('\t').Append(categories[i]);
                }

                text.Append('\n');
            }

            return text.ToString();
        }

        /// <summary>
        /// Restores an ordering produced by <see cref="WriteCategories"/>, in place of re-fitting.
        ///
        /// <para>Throws when the restored columns do not match the schema, rather than accepting them: a
        /// mismatch here means the saved model and this schema disagree about what the inputs are, and the
        /// symptom of continuing is a plausible-looking prediction from scrambled columns.</para>
        /// </summary>
        public void ReadCategories(string state)
        {
            ArgumentNullException.ThrowIfNull(state);

            _categoryMaps.Clear();

            var lines = state.Split('\n', StringSplitOptions.RemoveEmptyEntries);

            for (var i = 0; i < lines.Length; i++)
            {
                var parts = lines[i].Split('\t');

                if (parts.Length < 1 || parts[0].Length == 0)
                {
                    continue;
                }

                var categories = new string[parts.Length - 1];

                for (var c = 0; c < categories.Length; c++)
                {
                    categories[c] = parts[c + 1];
                }

                _categoryMaps[parts[0]] = categories;
            }

            _featureWidth = 0;

            foreach (var col in _schema.Features)
            {
                if (col.Type != ColumnType.Categorical)
                {
                    _featureWidth += 1;

                    continue;
                }

                if (!_categoryMaps.TryGetValue(col.Name, out var categories))
                {
                    throw new OverfitFormatException(
                        $"The restored category map has no entry for categorical column '{col.Name}'. The "
                        + "saved model and this schema disagree about the inputs; re-fitting instead would "
                        + "produce columns of the right width in the wrong order.");
                }

                _featureWidth += categories.Length;
            }
        }

        public (FastTensor<float> Features, FastTensor<float> Targets) Convert(IReadOnlyList<T> data)
        {
            ArgumentNullException.ThrowIfNull(data);

            if (_featureWidth == 0)
            {
                throw new OverfitRuntimeException(
                    $"{nameof(TabularToTensorConverter<T>)} has no fitted schema — call Fit, or ReadCategories "
                    + "with the ordering saved alongside the model. Converting without one would invent a "
                    + "column order.");
            }

            var rowCount = data.Count;

            // Allocate new tensors. clearMemory: false because we are about to overwrite every byte precisely.
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

                    switch (col.Type)
                    {
                        case ColumnType.Numeric:
                            fSpan[rowOffset + currentPos++] = System.Convert.ToSingle(val);
                            break;

                        case ColumnType.Binary:
                            fSpan[rowOffset + currentPos++] = System.Convert.ToBoolean(val) ? 1f : 0f;
                            break;

                        case ColumnType.Categorical:
                            var categories = _categoryMaps[col.Name];
                            var currentVal = val?.ToString();

                            for (var c = 0; c < categories.Length; c++)
                            {
                                fSpan[rowOffset + currentPos++] = categories[c] == currentVal ? 1f : 0f;
                            }
                            break;
                    }
                }

                tSpan[i] = System.Convert.ToSingle(GetValue(data[i], (_schema.Target ?? throw new OverfitRuntimeException(
                        "TableSchema.Target is not set — a target column is required to build label tensors.")).Name));
            }

            return (features, targets);
        }

        private object GetValue(T item, string propName)
        {
            return _valueExtractor(item, propName);
        }
    }
}