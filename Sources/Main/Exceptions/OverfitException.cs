// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Exceptions
{
    /// <summary>
    /// Base type for every error that originates in the Overfit runtime — a malformed model file
    /// (<see cref="OverfitFormatException"/>) or an invalid / unsupported runtime operation
    /// (<see cref="OverfitRuntimeException"/>). Catch this to handle any Overfit-domain failure distinctly
    /// from system exceptions (IO, out-of-memory) and from programming errors. Bad arguments are NOT Overfit
    /// exceptions — they stay as the standard <see cref="System.ArgumentException"/> family
    /// (<see cref="System.ArgumentNullException"/> / <see cref="System.ArgumentOutOfRangeException"/>), and a
    /// disposed object still throws <see cref="System.ObjectDisposedException"/>.
    /// </summary>
    public class OverfitException : Exception
    {
        public OverfitException()
        {
        }

        public OverfitException(string message)
            : base(message)
        {
        }

        public OverfitException(string message, Exception innerException)
            : base(message, innerException)
        {
        }
    }
}
