// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Data.Contracts
{
    /// <summary>
    /// Defines the structure of a data table, specifying the mapping for input features and the prediction target.
    /// </summary>
    public class TableSchema
    {
        /// <summary>
        /// The collection of column definitions used as input features for the model.
        /// </summary>
        public List<ColumnDefinition> Features { get; set; } = [];

        /// <summary>
        /// The column definition representing the target variable (label) the model aims to predict.
        /// </summary>
        public ColumnDefinition Target { get; set; }
    }
}