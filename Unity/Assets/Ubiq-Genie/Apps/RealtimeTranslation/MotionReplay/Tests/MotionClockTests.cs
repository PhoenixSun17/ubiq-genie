using NUnit.Framework;

namespace Ubiq.Genie.Samples.RealtimeTranslationMotion.Tests
{
    public sealed class MotionClockTests
    {
        [Test]
        public void MapsIndependentMonotonicClocksAndRejectsExpiredEstimates()
        {
            var clock = new MotionClock();
            Assert.IsFalse(clock.TryMap(5000, 1000, out _));
            Assert.IsTrue(clock.Observe(1000, 5010, 5012, 1022)); // +4000 ms, 20 ms RTT
            Assert.IsTrue(clock.TryMap(9000, 2000, out var mapped));
            Assert.AreEqual(5, mapped, 0.00001);
            Assert.IsFalse(clock.TryMap(9000, 16023, out _));
            clock.Clear();
            Assert.IsFalse(clock.TryMap(9000, 2000, out _));
        }

        [Test]
        public void ChoosesLowestRoundTripAndEventuallyReplacesOldSamples()
        {
            var clock = new MotionClock();
            clock.Observe(1000, 5010, 5010, 1020);
            clock.Observe(1100, 5210, 5210, 1300); // Slower asymmetric route: do not change mapping.
            clock.TryMap(9000, 1400, out var mapped);
            Assert.AreEqual(5, mapped, 0.00001);
            for (int i = 0; i < 8; ++i)
                clock.Observe(2000 + i * 100, 7005 + i * 100, 7005 + i * 100, 2010 + i * 100);
            clock.TryMap(9000, 3000, out mapped);
            Assert.AreEqual(4, mapped, 0.00001);
        }

        [TestCase(double.NaN, 10, 10, 20)]
        [TestCase(20, 10, 10, 0)]
        [TestCase(0, 20, 10, 30)]
        [TestCase(0, 10, 1000, 20)]
        [TestCase(0, 10, 10, 20000)]
        public void RejectsInvalidObservations(double t0, double s1, double s2, double t3)
        {
            Assert.IsFalse(new MotionClock().Observe(t0, s1, s2, t3));
        }

        [Test]
        public void MetadataRequiresMatchingSessionAndExactPcmCount()
        {
            var message = new MotionMetadata
            {
                version = 1, backendUuid = "backend", sessionId = "session",
                requestId = "request", speakerUuid = "speaker", sampleRate = 16000,
                sampleCount = 16000, audioLength = 32000, sourceStartMs = 100, sourceEndMs = 1100
            };
            Assert.IsTrue(message.Matches("backend", "session"));
            Assert.IsFalse(message.Matches("backend", "previous-session"));
            Assert.IsTrue(message.ValidAudio);
            Assert.IsTrue(message.ValidInterval);
            message.audioLength--;
            Assert.IsFalse(message.ValidAudio);
            message.sourceEndMs = double.PositiveInfinity;
            Assert.IsFalse(message.ValidInterval);
        }
    }
}
