// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Text;
using DevOnBike.Overfit.Anomalies.Contracts;

namespace DevOnBike.Overfit.Anomalies.Monitoring
{
    /// <summary>
    /// Which of the known features this deployment can actually supply, and from what.
    ///
    /// <para><b>The onboarding step, made answerable.</b> Configuring the guard used to mean writing full
    /// PromQL per feature — the part nobody gets right first time, and where a mistake returns an empty
    /// result that Prometheus reports as <c>success</c>. Here the client names their metric and its shape,
    /// and the query is built for them.</para>
    ///
    /// <para><b>What is absent is as much of an answer as what is present.</b> A feature with no binding is
    /// not a gap to paper over: <see cref="Unmapped"/> lists it, the query for it is never issued, and the
    /// guard counts it as blind every cycle. A cluster with no CPU-limit set has no CFS throttling counters
    /// and saying so is correct; pretending otherwise produces a query that matches nothing and reads as
    /// health.</para>
    /// </summary>
    public sealed class MetricMap
    {
        private readonly MetricBinding[] _bindings = new MetricBinding[(int)MetricIndex.Count];
        private readonly bool[] _bound = new bool[(int)MetricIndex.Count];
        private readonly List<CustomMetricBinding> _custom = [];

        /// <param name="bindings">
        /// One per feature this deployment can supply. A later binding for the same feature replaces an
        /// earlier one, so a caller can start from a preset and override individual entries.
        /// </param>
        /// <param name="custom">
        /// Metrics this project does not model. They are evaluated by the rules, peer and trend families and
        /// <b>not</b> by the learned one — see <see cref="CustomMetricBinding"/> for why the enum cannot grow
        /// per deployment.
        /// </param>
        public MetricMap(
            IEnumerable<MetricBinding> bindings,
            IEnumerable<CustomMetricBinding>? custom = null)
        {
            ArgumentNullException.ThrowIfNull(bindings);

            foreach (var binding in bindings)
            {
                var index = (int)binding.Target;

                if ((uint)index >= (uint)MetricIndex.Count || !binding.IsUsable)
                {
                    continue;
                }

                _bindings[index] = binding;
                _bound[index] = true;
            }

            if (custom is null)
            {
                return;
            }

            foreach (var binding in custom)
            {
                if (!binding.IsUsable)
                {
                    continue;
                }

                // A later binding for the same name replaces an earlier one, matching the known-metric rule.
                for (var i = _custom.Count - 1; i >= 0; i--)
                {
                    if (string.Equals(_custom[i].Name, binding.Name, StringComparison.Ordinal))
                    {
                        _custom.RemoveAt(i);
                    }
                }

                _custom.Add(binding);
            }
        }

        /// <summary>Metrics outside the modelled set that this deployment wants watched.</summary>
        public IReadOnlyList<CustomMetricBinding> Custom => _custom;

        /// <summary>
        /// Names of the custom channels whose observations must not become a floor — the form
        /// <c>AnomalyGuardOptions.NonCalibratedCustomChannels</c> takes, because <c>FloorCalibrator</c> is
        /// keyed by name and never sees a binding.
        /// </summary>
        public IReadOnlyList<string> NonCalibratedChannels
        {
            get
            {
                var names = new List<string>();

                for (var i = 0; i < _custom.Count; i++)
                {
                    if (!_custom[i].Calibrated)
                    {
                        names.Add(_custom[i].Name);
                    }
                }

                return names;
            }
        }

        /// <summary>
        /// PromQL for each custom metric, keyed by its reported name. Same wrapping rules as the known set,
        /// so a gauge is summed by pod and a histogram keeps the pod label through the quantile — and, since
        /// 2026-08-10, the same verbatim-wins-over-kind escape hatch.
        ///
        /// <para><b>The precedence is not implemented here.</b> <see cref="Build"/> already resolves it for
        /// the built-in path, so the custom binding's own <see cref="CustomMetricBinding.Query"/> is handed to
        /// the same method rather than branched on twice. Two copies of a precedence rule is how the built-in
        /// path came to have an escape hatch the custom path did not.</para>
        /// </summary>
        public Dictionary<string, string> CustomQueries(
            string? selectorToken = null,
            TimeSpan? rateWindow = null)
        {
            var token = selectorToken ?? PromqlCatalog.SelectorToken;
            var range = FormatRange(rateWindow ?? TimeSpan.FromMinutes(2));
            var queries = new Dictionary<string, string>(_custom.Count, StringComparer.Ordinal);

            for (var i = 0; i < _custom.Count; i++)
            {
                var binding = _custom[i];

                queries[binding.Name] = Build(
                    new MetricBinding(MetricIndex.RequestsPerSecond, binding.Source, binding.Kind,
                        binding.Quantile > 0.0 ? binding.Quantile : 0.95, binding.Query),
                    token,
                    range);
            }

            return queries;
        }

        /// <summary>How many known features this deployment supplies.</summary>
        public int MappedCount
        {
            get
            {
                var count = 0;

                for (var i = 0; i < _bound.Length; i++)
                {
                    count += _bound[i] ? 1 : 0;
                }

                return count;
            }
        }

        /// <summary>Whether this deployment supplies <paramref name="metric"/> at all.</summary>
        public bool IsMapped(MetricIndex metric)
        {
            var index = (int)metric;

            return (uint)index < (uint)MetricIndex.Count && _bound[index];
        }

        /// <summary>
        /// Features with no binding — the honest list of what this guard will be blind to, available
        /// <b>before</b> anything is deployed rather than discovered from a cycle that saw nothing.
        /// </summary>
        public IReadOnlyList<MetricIndex> Unmapped
        {
            get
            {
                var missing = new List<MetricIndex>();

                for (var i = 0; i < _bound.Length; i++)
                {
                    if (!_bound[i])
                    {
                        missing.Add((MetricIndex)i);
                    }
                }

                return missing;
            }
        }

        /// <summary>
        /// Builds the <c>QueryOverrides</c> table the Prometheus sources take.
        /// </summary>
        /// <param name="selectorToken">
        /// Placeholder the sources replace with the namespace and pod matchers. Defaults to
        /// <see cref="PromqlCatalog.SelectorToken"/>, which is what they look for.
        /// </param>
        /// <param name="rateWindow">
        /// Range used inside <c>rate()</c> and <c>increase()</c>. Must span several scrapes or the result is
        /// full of gaps; must not be so long that it smooths away what the detectors look for.
        /// </param>
        public Dictionary<MetricIndex, string> ToQueryOverrides(
            string? selectorToken = null,
            TimeSpan? rateWindow = null)
        {
            var token = selectorToken ?? PromqlCatalog.SelectorToken;
            var range = FormatRange(rateWindow ?? TimeSpan.FromMinutes(2));
            var overrides = new Dictionary<MetricIndex, string>();

            for (var i = 0; i < _bound.Length; i++)
            {
                overrides[(MetricIndex)i] = _bound[i]
                    ? Build(_bindings[i], token, range)

                    // An empty template is how the sources are told not to issue a query at all. Issuing one
                    // built from a metric this cluster does not have returns an empty result that cannot be
                    // told apart from a real one.
                    : string.Empty;
            }

            return overrides;
        }

        /// <summary>A short report of what is mapped and what is not, for a pre-flight check.</summary>
        public string Describe()
        {
            var report = new StringBuilder();

            report.Append(CultureInfo.InvariantCulture, $"{MappedCount} of {(int)MetricIndex.Count} features mapped\n");

            for (var i = 0; i < _bound.Length; i++)
            {
                var metric = (MetricIndex)i;

                report.Append(_bound[i]
                    ? $"  {metric,-24} <- {_bindings[i].SourceMetric} ({_bindings[i].Kind})\n"
                    : $"  {metric,-24} -- not available in this deployment\n");
            }

            for (var i = 0; i < _custom.Count; i++)
            {
                report.Append($"  {_custom[i].Name,-24} <- {_custom[i].Source} ({_custom[i].Kind}, custom, ")
                    .Append($"{_custom[i].SignalKind}, {_custom[i].Class})")
                    .Append('\n');
            }

            return report.ToString();
        }

        private static string Build(MetricBinding binding, string token, string range)
        {
            var name = binding.SourceMetric;

            // Verbatim PromQL wins over name-plus-kind — see MetricBinding.Query for the channel that could
            // not be expressed any other way. The selector token is still substituted downstream, which is
            // why the reader refuses a query that does not contain one.
            if (!string.IsNullOrWhiteSpace(binding.Query))
            {
                return binding.Query;
            }

            return binding.Kind switch
            {
                MetricSourceKind.Gauge => $"sum by (pod) ({name}{{{token}}})",
                MetricSourceKind.Counter => $"sum by (pod) (rate({name}{{{token}}}[{range}]))",
                MetricSourceKind.EventCount => $"sum by (pod) (increase({name}{{{token}}}[{range}]))",
                MetricSourceKind.Ratio => $"{name}{{{token}}}",
                MetricSourceKind.HistogramSeconds => Histogram(binding, token, range),
                _ => string.Empty,
            };
        }

        /// <summary>
        /// <c>sum by (pod, le)</c>, and the <c>le</c> is load-bearing: without the pod label surviving the
        /// quantile every sample is discarded during parsing, which loses the feature entirely and silently.
        /// Multiplied to milliseconds because that is what the feature is named in.
        /// </summary>
        private static string Histogram(MetricBinding binding, string token, string range)
        {
            var quantile = binding.Quantile > 0.0 ? binding.Quantile : DefaultQuantile(binding.Target);

            return $"histogram_quantile({quantile.ToString("0.##", CultureInfo.InvariantCulture)},"
                   + $" sum by (pod, le) (rate({binding.SourceMetric}_bucket{{{token}}}[{range}]))) * 1000";
        }

        private static double DefaultQuantile(MetricIndex metric)
        {
            return metric switch
            {
                MetricIndex.LatencyP50Ms => 0.50,
                MetricIndex.LatencyP95Ms => 0.95,
                MetricIndex.LatencyP99Ms => 0.99,
                _ => 0.95,
            };
        }

        private static string FormatRange(TimeSpan window)
        {
            var seconds = (int)window.TotalSeconds;

            return seconds % 60 == 0
                ? $"{seconds / 60}m"
                : $"{seconds}s";
        }
    }
}
