---
name: mutation-theory-test-names
description: In the mutation harness, a [Theory] victim comes back from dotnet test as "Name(param: value)" so an exact-match membership check reports a false "survived" — match by prefix
metadata:
  type: reference
---

The harness in [[overfit-mutate]] extracts failed tests with
`re.findall(r"(?:Niepowodzenie|Failed) (DevOnBike\S+)", text)` and then asks `EXPECTED in failed`. For a
`[Fact]` that works. For a `[Theory]` the name comes back with the inline data attached and **truncated at
the first space** by `\S+`, e.g.

```
TheBuiltInGapChangeFloorReachesTheGuardOptions(declared:
```

so an exact-equality membership test reports the victim as **survived** while it was in fact red. Measured
2026-08-12 on `AN-D1` blocker 2: two of four mutations printed a false survival for the same Theory.

Use `any(f.startswith(expected) for f in failed)`, not `expected in failed`. The direction of the error is
the dangerous one — a green mutation is supposed to be a finding, so a harness that manufactures fake green
results burns a round trip on investigating nothing.
