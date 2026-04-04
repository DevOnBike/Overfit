using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.Data.Contracts;

namespace DevOnBike.Overfit.Data.Prepare
{
    public class TabularToTensorConverter<T>
    {
        private readonly TableSchema _schema;
        private readonly Dictionary<string, string[]> _categoryMaps = new();

        // DODANO: Delegat zastępujący Refleksję. Działa z natywną prędkością (AOT-Safe).
        private readonly Func<T, string, object> _valueExtractor;

        private int _featureWidth;

        // ZMIANA: Konstruktor wymaga teraz dostarczenia metody wyciągającej dane
        public TabularToTensorConverter(TableSchema schema, Func<T, string, object> valueExtractor)
        {
            _schema = schema;
            _valueExtractor = valueExtractor ?? throw new ArgumentNullException(nameof(valueExtractor));
        }

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
                    _featureWidth += uniqueValues.Length;
                }
                else
                {
                    _featureWidth += 1;
                }
            }
        }

        public (FastTensor<float> Features, FastTensor<float> Targets) Transform(List<T> data)
        {
            if (_featureWidth == 0 && _schema.Features.Count > 0)
                throw new InvalidOperationException("Wywołaj Fit() przed Transform().");

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
                        fSpan[rowOffset + currentPos++] = Convert.ToSingle(val);
                    else if (col.Type == ColumnType.Binary)
                        fSpan[rowOffset + currentPos++] = Convert.ToBoolean(val) ? 1f : 0f;
                    else if (col.Type == ColumnType.Categorical)
                    {
                        var categories = _categoryMaps[col.Name];
                        var currentVal = val?.ToString();
                        for (var c = 0; c < categories.Length; c++)
                            fSpan[rowOffset + currentPos++] = (categories[c] == currentVal) ? 1f : 0f;
                    }
                }
                tSpan[i] = Convert.ToSingle(GetValue(data[i], _schema.Target.Name));
            }

            return (features, targets);
        }

        // ZMIANA: Wykorzystujemy przekazany delegat zamiast powolnej Refleksji
        private object GetValue(T item, string propName)
            => _valueExtractor(item, propName);
    }
}