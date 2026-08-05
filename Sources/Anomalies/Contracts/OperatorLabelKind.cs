// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Anomalies.Contracts
{
    /// <summary>
    /// What an operator said about an incident.
    ///
    /// <para><b>Two signs, and the second one is what makes the first safe.</b> Every mechanism that responds
    /// to "this was noise" makes the guard quieter, and nothing in a one-signed feedback loop ever makes it
    /// louder again: a hundred honest dismissals produce a detector that reports nothing, arriving gradually
    /// enough that nobody notices the day it stopped working. That is the failure this whole subsystem exists
    /// to remove, arriving through the feature meant to build trust in it.</para>
    ///
    /// <para><see cref="Real"/> is therefore not a nicety. It pins an observation the guard must go on being
    /// able to make, and every proposed threshold is checked against it.</para>
    /// </summary>
    public enum OperatorLabelKind
    {
        /// <summary>The operator judged the finding not worth reporting.</summary>
        Noise = 0,

        /// <summary>The operator judged the finding correct. No proposed floor may silence it.</summary>
        Real = 1,
    }
}
