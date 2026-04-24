using System.Text;

namespace DevOnBike.Overfit.Autograd
{
    public static class BackwardProfileFormatter
    {
        public static string Format(BackwardProfileSnapshot snapshot, int top = 12)
        {
            ArgumentNullException.ThrowIfNull(snapshot);

            var sb = new StringBuilder(1024);
            sb.AppendLine("=== BACKWARD BREAKDOWN ===");
            sb.AppendLine($"total backward.ms:      {snapshot.TotalElapsedMs:F1}");
            sb.AppendLine($"total backward.alloc:   {snapshot.TotalAllocatedMb:F2} MB");

            if (snapshot.IsEmpty)
            {
                sb.AppendLine("no backward profile data");
                return sb.ToString();
            }

            var items = snapshot.Profiles.ToArray();
            Array.Sort(items, static (x, y) =>
            {
                var allocCmp = y.AllocatedBytes.CompareTo(x.AllocatedBytes);
                if (allocCmp != 0)
                {
                    return allocCmp;
                }

                return y.ElapsedMs.CompareTo(x.ElapsedMs);
            });

            sb.AppendLine("top.backward.ops:");
            var count = Math.Min(top, items.Length);
            for (var i = 0; i < count; i++)
            {
                var item = items[i];
                sb.Append("  ");
                sb.Append(item.Code.ToString().PadRight(26));
                sb.Append(" | count ");
                sb.Append(item.Count.ToString().PadLeft(6));
                sb.Append(" | ms ");
                sb.Append(item.ElapsedMs.ToString("F1").PadLeft(10));
                sb.Append(" | alloc ");
                sb.Append(item.AllocatedMb.ToString("F2").PadLeft(10));
                sb.AppendLine(" MB");
            }

            return sb.ToString();
        }
    }
}