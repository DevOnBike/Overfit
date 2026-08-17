---
name: python-mtime-vs-dotnet-ticks
description: A Python-recorded os.stat mtime and .NET's LastWriteTimeUtc differ in the last digits for the SAME untouched file — compare with a tolerance, never for equality.
metadata:
  type: reference
---

Measured 2026-08-15 on `C:\qwen3b\qwen.bin` while adding a model-identity assertion to
`QwenLayer0CompareTests.LoadTwoTokenOracle`:

- Python `os.stat().st_mtime` is a **float64** (~0.2 us resolution at this epoch) and `isoformat()`
  truncates to microseconds: recorded `2026-08-07T12:29:37.968373Z`.
- .NET `FileInfo.LastWriteTimeUtc` reads **100 ns ticks**: `2026-08-07T12:29:37.9683732Z`.
- True value from `st_mtime_ns` = `...968373200`.

So an exact string or `DateTime ==` compare goes red on a file nobody touched — a 200 ns gap. Assert the
**byte length exactly** (integer, lossless on both sides) and the mtime with a tolerance; 1 s is ample,
because a re-conversion moves the mtime by hours.

`JsonElement.GetDateTimeOffset()` parses the `...Z` ISO form natively, so no `System.Globalization`
using is needed in the test.

Both halves were mutation-proved red (fixture `model_bytes` +1, and mtime +1 h), mutating the **build
output** copy under `Tests/bin/Release/net10.0/test_fixtures/` and re-running with `--no-build` — see
[[test-output-and-anchors]].
