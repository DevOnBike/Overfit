// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Redaction;

namespace DevOnBike.Overfit.Tests.Redaction
{
    /// <summary>
    /// The PL-identifier checksum validators and their use as the precision gate: a genuine PESEL/NIP/IBAN is
    /// redacted, a look-alike (right shape, wrong checksum) is not — which is the whole point of the validator.
    /// </summary>
    public sealed class RedactionValidatorsTests
    {
        [Theory]
        [InlineData("44051401359", true)]   // valid PESEL
        [InlineData("44051401350", false)]  // right shape, wrong control digit
        [InlineData("12345678901", false)]  // arbitrary 11 digits
        public void Pesel(string value, bool valid) => Assert.Equal(valid, RedactionValidators.Pesel(value));

        [Theory]
        [InlineData("1234563218", true)]
        [InlineData("123-456-32-18", true)]  // dashed form
        [InlineData("1234563210", false)]
        public void Nip(string value, bool valid) => Assert.Equal(valid, RedactionValidators.Nip(value));

        [Theory]
        [InlineData("123456785", true)]
        [InlineData("123456789", false)]
        public void Regon(string value, bool valid) => Assert.Equal(valid, RedactionValidators.Regon(value));

        [Theory]
        [InlineData("4111111111111111", true)]   // Visa test number
        [InlineData("4111111111111112", false)]
        public void Luhn(string value, bool valid) => Assert.Equal(valid, RedactionValidators.Luhn(value));

        [Theory]
        [InlineData("PL61109010140000071219812874", true)]
        [InlineData("PL61 1090 1014 0000 0712 1981 2874", true)]  // grouped form
        [InlineData("PL61109010140000071219812875", false)]
        public void IbanPl(string value, bool valid) => Assert.Equal(valid, RedactionValidators.IbanPl(value));

        [Fact]
        public void Redactor_RedactsValidPesel_ButNotALookAlike()
        {
            var redactor = new Redactor(PolishRedactionRules.All());

            var real = redactor.Redact("Mój PESEL to 44051401359, proszę.");
            Assert.True(real.HasRedactions);
            Assert.Contains("[REDACTED_PESEL_0]", real.Text);
            Assert.DoesNotContain("44051401359", real.Text);

            // Same shape, invalid checksum → the validator rejects it, so nothing is redacted.
            var lookAlike = redactor.Redact("Numer 12345678901 nie jest peselem.");
            Assert.False(lookAlike.HasRedactions);
            Assert.Contains("12345678901", lookAlike.Text);
        }
    }
}
