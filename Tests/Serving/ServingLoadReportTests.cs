// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Generic;
using DevOnBike.Overfit.Serving;

namespace DevOnBike.Overfit.Tests.Serving
{
    public sealed class ServingLoadReportTests
    {
        [Fact]
        public void Percentile_LinearInterpolation_MatchesKnownValues()
        {
            var sorted = new double[] { 10, 20, 30, 40, 50 };

            Assert.Equal(10.0, ServingLoadReport.Percentile(sorted, 0.0), 6);
            Assert.Equal(30.0, ServingLoadReport.Percentile(sorted, 0.50), 6);
            Assert.Equal(50.0, ServingLoadReport.Percentile(sorted, 1.0), 6);
            // rank = 0.95 * 4 = 3.8 → 40 + 0.8 * (50 − 40) = 48
            Assert.Equal(48.0, ServingLoadReport.Percentile(sorted, 0.95), 6);
        }

        [Fact]
        public void Percentile_EmptyIsZero_SingleIsItself()
        {
            Assert.Equal(0.0, ServingLoadReport.Percentile([], 0.5));
            Assert.Equal(7.0, ServingLoadReport.Percentile([7.0], 0.99));
        }

        [Fact]
        public void Sample_InterTokenLatency_IsDecodeGapPerStep()
        {
            // Fewer than two tokens → no gap to measure.
            Assert.Equal(0.0, ServingRequestSample.Success(10, 10, 1).InterTokenLatencyMs);
            // ttft 10, e2e 110, 11 tokens → (110 − 10) / (11 − 1) = 10 ms/token
            Assert.Equal(10.0, ServingRequestSample.Success(10, 110, 11).InterTokenLatencyMs, 6);
        }

        [Fact]
        public void From_CountsErrors_AndComputesGoodput()
        {
            var samples = new List<ServingRequestSample>
            {
                ServingRequestSample.Success(10, 110, 11),
                ServingRequestSample.Success(10, 110, 11),
                ServingRequestSample.Success(10, 110, 11),
                ServingRequestSample.Failure(),
            };

            var report = ServingLoadReport.From(samples, concurrency: 4, wallClockSeconds: 1.0);

            Assert.Equal(4, report.TotalRequests);
            Assert.Equal(3, report.SuccessfulRequests);
            Assert.Equal(1, report.FailedRequests);
            Assert.Equal(0.25, report.ErrorRate, 6);
            // 33 output tokens over 1.0 s.
            Assert.Equal(33.0, report.ThroughputTokensPerSecond, 6);
            // goodput = throughput × (1 − error_rate) = 33 × 0.75.
            Assert.Equal(24.75, report.Goodput, 6);
        }

        [Fact]
        public void From_Score_EqualsHolisticFormula()
        {
            // 10 identical requests: TTFT 20 ms, E2E 220 ms, 11 tokens → ITL = (220 − 20) / 10 = 20 ms.
            var samples = new List<ServingRequestSample>();
            for (var i = 0; i < 10; i++)
            {
                samples.Add(ServingRequestSample.Success(20, 220, 11));
            }

            var report = ServingLoadReport.From(samples, concurrency: 5, wallClockSeconds: 2.0, costUnits: 1.0);

            Assert.Equal(0.0, report.ErrorRate);
            Assert.Equal(20.0, report.TimeToFirstTokenP95Ms, 6);
            Assert.Equal(20.0, report.InterTokenLatencyP95Ms, 6);
            // throughput = 110 tokens / 2 s = 55; goodput = 55 (no errors).
            Assert.Equal(55.0, report.ThroughputTokensPerSecond, 6);
            // score = (goodput × users) / (TTFT_p95_s × ITL_p95_s × cost) = (55 × 5) / (0.02 × 0.02 × 1) = 687,500.
            Assert.Equal(687500.0, report.Score, 1);
        }

        [Fact]
        public void From_EmptyOrAllFailed_ScoresZero_WithoutThrowing()
        {
            var empty = ServingLoadReport.From([], concurrency: 4, wallClockSeconds: 1.0);
            Assert.Equal(0, empty.SuccessfulRequests);
            Assert.Equal(0.0, empty.Score);

            var failed = new List<ServingRequestSample> { ServingRequestSample.Failure(), ServingRequestSample.Failure() };
            var report = ServingLoadReport.From(failed, concurrency: 4, wallClockSeconds: 1.0);

            Assert.Equal(1.0, report.ErrorRate);
            Assert.Equal(0.0, report.Score);
        }

        [Fact]
        public void From_CostUnits_DivideOutBruteForce()
        {
            var samples = new List<ServingRequestSample>();
            for (var i = 0; i < 8; i++)
            {
                samples.Add(ServingRequestSample.Success(25, 225, 11));
            }

            var oneUnit = ServingLoadReport.From(samples, concurrency: 4, wallClockSeconds: 1.0, costUnits: 1.0);
            var fourUnits = ServingLoadReport.From(samples, concurrency: 4, wallClockSeconds: 1.0, costUnits: 4.0);

            // Quadrupling the cost units quarters the score — efficiency, not brute force, is rewarded.
            Assert.Equal(oneUnit.Score / 4.0, fourUnits.Score, 3);
        }
    }
}
