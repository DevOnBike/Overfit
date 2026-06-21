// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Globalization;
using System.Text.Json;

namespace DevOnBike.Overfit.Trees
{
    /// <summary>
    /// Loads an XGBoost model saved as JSON (<c>booster.save_model("model.json")</c>) into a
    /// <see cref="BoostedTreeModel"/> for pure-managed, zero-allocation inference. Read-only and
    /// dependency-free — consistent with Overfit's one-directional loading story (external → Overfit).
    ///
    /// Supported objectives: <c>binary:logistic</c>, <c>reg:squarederror</c> / <c>reg:linear</c>,
    /// <c>multi:softprob</c> / <c>multi:softmax</c>. Numerical splits only — categorical splits
    /// (<c>split_type == 1</c>) throw a clear <see cref="OverfitRuntimeException"/>.
    /// </summary>
    public static class XgboostModelLoader
    {
        public static BoostedTreeModel Load(string path)
        {
            ArgumentException.ThrowIfNullOrEmpty(path);

            using var stream = File.OpenRead(path);
            return Load(stream);
        }

        public static BoostedTreeModel Load(Stream stream)
        {
            ArgumentNullException.ThrowIfNull(stream);

            using var doc = JsonDocument.Parse(stream);
            return Build(doc.RootElement);
        }

        private static BoostedTreeModel Build(JsonElement root)
        {
            var learner = root.GetProperty("learner");
            var modelParam = learner.GetProperty("learner_model_param");

            var numFeature = ReadInt(modelParam.GetProperty("num_feature"));
            var numClass = ReadInt(modelParam.GetProperty("num_class"));
            var numGroups = numClass > 0 ? numClass : 1;

            var objectiveName = learner.GetProperty("objective").GetProperty("name").GetString();
            var objective = MapObjective(objectiveName);

            // base_score is a JSON-array string in *probability* space, one entry per output group
            // (e.g. "[5.2375E-1]" for binary, "[-2.97E-2,2.5E-1,-2.2E-1]" for 3-class). The per-group
            // base margin is the objective's prob→margin link applied to each entry.
            var baseMargin = ParseBaseScore(modelParam.GetProperty("base_score").GetString(), numGroups, objective);

            var gbModel = learner.GetProperty("gradient_booster").GetProperty("model");
            var trees = gbModel.GetProperty("trees");

            var numTrees = trees.GetArrayLength();
            var totalNodes = 0;

            foreach (var tree in trees.EnumerateArray())
            {
                totalNodes += tree.GetProperty("split_indices").GetArrayLength();
            }

            var featureIndex = new int[totalNodes];
            var threshold = new float[totalNodes];
            var left = new int[totalNodes];
            var right = new int[totalNodes];
            var defaultLeft = new byte[totalNodes];
            var treeRoot = new int[numTrees];
            var treeGroup = new int[numTrees];

            ReadTreeGroups(gbModel, treeGroup);

            var offset = 0;
            var t = 0;

            foreach (var tree in trees.EnumerateArray())
            {
                // XGBoost's root is always local node 0, so the tree's root is its global offset.
                treeRoot[t] = offset;

                var nodeCount = FillTree(tree, offset, featureIndex, threshold, left, right, defaultLeft);

                offset += nodeCount;
                t++;
            }

            return new BoostedTreeModel(
                numFeature,
                numGroups,
                objective,
                baseMargin,
                featureIndex,
                threshold,
                left,
                right,
                defaultLeft,
                treeRoot,
                treeGroup);
        }

        private static int FillTree(
            JsonElement tree,
            int offset,
            int[] featureIndex,
            float[] threshold,
            int[] left,
            int[] right,
            byte[] defaultLeft)
        {
            var leftChildren = tree.GetProperty("left_children");
            var rightChildren = tree.GetProperty("right_children");
            var splitIndices = tree.GetProperty("split_indices");
            var splitConditions = tree.GetProperty("split_conditions");
            var defaults = tree.GetProperty("default_left");
            var splitType = tree.GetProperty("split_type");

            var nodeCount = splitIndices.GetArrayLength();

            // Each column is enumerated once (O(nodes)); JsonElement indexing would be O(nodes²).
            var i = 0;

            foreach (var child in leftChildren.EnumerateArray())
            {
                var local = child.GetInt32();
                left[offset + i] = local < 0 ? -1 : offset + local;
                i++;
            }

            i = 0;

            foreach (var child in rightChildren.EnumerateArray())
            {
                var local = child.GetInt32();
                right[offset + i] = local < 0 ? -1 : offset + local;
                i++;
            }

            i = 0;

            foreach (var feature in splitIndices.EnumerateArray())
            {
                featureIndex[offset + i] = feature.GetInt32();
                i++;
            }

            i = 0;

            foreach (var condition in splitConditions.EnumerateArray())
            {
                threshold[offset + i] = condition.GetSingle();
                i++;
            }

            i = 0;

            foreach (var flag in defaults.EnumerateArray())
            {
                defaultLeft[offset + i] = (byte)(flag.GetInt32() != 0 ? 1 : 0);
                i++;
            }

            foreach (var type in splitType.EnumerateArray())
            {
                if (type.GetInt32() != 0)
                {
                    throw new OverfitRuntimeException(
                        "XGBoost categorical splits (split_type == 1) are not supported; " +
                        "only numerical splits can be loaded.");
                }
            }

            return nodeCount;
        }

        private static void ReadTreeGroups(JsonElement gbModel, int[] treeGroup)
        {
            // tree_info[t] = output group of tree t (all zeros for binary / regression, 0..num_class-1
            // for multiclass). Absent ⇒ single group ⇒ leave zero-initialized.
            if (!gbModel.TryGetProperty("tree_info", out var treeInfo))
            {
                return;
            }

            var t = 0;

            foreach (var group in treeInfo.EnumerateArray())
            {
                if (t >= treeGroup.Length)
                {
                    break;
                }

                treeGroup[t] = group.GetInt32();
                t++;
            }
        }

        private static TreeObjective MapObjective(string? name)
        {
            return name switch
            {
                "binary:logistic" => TreeObjective.Logistic,
                "reg:squarederror" or "reg:linear" or "reg:pseudohubererror" => TreeObjective.Identity,
                "multi:softprob" or "multi:softmax" => TreeObjective.Softmax,
                _ => throw new OverfitRuntimeException(
                    $"XGBoost objective '{name}' is not supported. " +
                    "Supported: binary:logistic, reg:squarederror/reg:linear, multi:softprob/multi:softmax.")
            };
        }

        private static float[] ParseBaseScore(string? raw, int numGroups, TreeObjective objective)
        {
            if (string.IsNullOrWhiteSpace(raw))
            {
                throw new OverfitRuntimeException("XGBoost model is missing learner_model_param.base_score.");
            }

            var trimmed = raw.Trim();

            if (trimmed.StartsWith('[') && trimmed.EndsWith(']'))
            {
                trimmed = trimmed.Substring(1, trimmed.Length - 2);
            }

            var parts = trimmed.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

            if (parts.Length != numGroups)
            {
                throw new OverfitRuntimeException(
                    $"base_score has {parts.Length} entries but the model has {numGroups} output groups.");
            }

            var result = new float[numGroups];

            for (var g = 0; g < numGroups; g++)
            {
                var probability = float.Parse(parts[g], CultureInfo.InvariantCulture);
                result[g] = ProbabilityToMargin(probability, objective);
            }

            return result;
        }

        private static float ProbabilityToMargin(float probability, TreeObjective objective)
        {
            // XGBoost stores base_score in probability space and applies the objective's prob→margin
            // link at load time. Logistic ⇒ logit; identity / softmax ⇒ the value passes through.
            if (objective == TreeObjective.Logistic)
            {
                return MathF.Log(probability / (1f - probability));
            }

            return probability;
        }

        private static int ReadInt(JsonElement element)
        {
            // learner_model_param values are JSON strings ("12", "0"); tolerate numbers too.
            if (element.ValueKind == JsonValueKind.String)
            {
                return int.Parse(element.GetString()!, CultureInfo.InvariantCulture);
            }

            return element.GetInt32();
        }
    }
}
