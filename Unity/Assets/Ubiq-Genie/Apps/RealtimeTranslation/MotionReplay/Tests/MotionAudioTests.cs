using System;
using System.Collections.Generic;
using System.Text;
using NUnit.Framework;

namespace Ubiq.Genie.Samples.RealtimeTranslationMotion.Tests
{
    public sealed class MotionAudioTests
    {
        private static byte[] Header(int samples)
        {
            return Encoding.UTF8.GetBytes("{\"type\":\"AudioInfo\",\"targetPeer\":\"\",\"audioLength\":\"" + samples * 2 + "\"}");
        }

        private static byte[] Pcm(int samples, short value = 16384)
        {
            var bytes = new byte[samples * 2];
            for (int index = 0; index < bytes.Length; index += 2)
            {
                bytes[index] = unchecked((byte)value);
                bytes[index + 1] = unchecked((byte)(value >> 8));
            }
            return bytes;
        }

        private static MotionAudioIdentity Identity(string id, int count, string speaker = "speaker")
        {
            return new MotionAudioIdentity { RequestId = id, SpeakerUuid = speaker, SampleCount = count, SampleRate = 16000 };
        }

        private static bool Receive(MotionPcmQueue queue, byte[] bytes, MotionAudioIdentity identity = null)
        {
            return queue.Receive(bytes, 0, bytes.Length, identity, out _);
        }

        [Test]
        public void PrebufferDoesNotStartMotionBeforeScheduledAudio()
        {
            var queue = new MotionPcmQueue(16000);
            Assert.IsTrue(Receive(queue, Header(6400), Identity("a", 6400)));
            Assert.IsTrue(Receive(queue, Pcm(1600)));
            Assert.IsFalse(queue.TrySchedule(0, 0.05, new float[3200], out _));
            Assert.IsTrue(Receive(queue, Pcm(1600)));
            Assert.IsTrue(queue.TrySchedule(0, 0.05, new float[3200], out var chunk));
            Assert.AreEqual(0.05, chunk.DspStart, 0.00001);
            Assert.IsFalse(queue.TryGetPlayback(0.049, out _));
            Assert.IsTrue(queue.TryGetPlayback(0.15, out var state));
            Assert.AreEqual("a", state.RequestId);
            Assert.AreEqual(0.25f, state.Progress, 0.001);
        }

        [Test]
        public void FullShortSentencePlaysItsExactTailWithoutPadding()
        {
            var queue = new MotionPcmQueue(16000);
            Receive(queue, Header(100), Identity("short", 100));
            Receive(queue, Pcm(100, -16384));
            var buffer = new float[3200];
            Assert.IsTrue(queue.TrySchedule(0, 0.05, buffer, out var chunk));
            Assert.AreEqual(100, chunk.SampleCount);
            Assert.AreEqual(0.00625, chunk.DspEnd - chunk.DspStart, 0.000001);
            Assert.AreEqual(-0.5f, buffer[99]);
            Assert.AreEqual(0, buffer[100]);
        }

        [Test]
        public void UnderrunHoldsMotionAndResumesWithoutSkippingSamples()
        {
            var queue = new MotionPcmQueue(16000);
            Receive(queue, Header(9600), Identity("a", 9600));
            Receive(queue, Pcm(3200));
            Assert.IsTrue(queue.TrySchedule(0, 0.05, new float[3200], out _));
            Assert.IsTrue(queue.TryGetPlayback(0.5, out var held));
            Assert.AreEqual(1f / 3, held.Progress, 0.0001);
            Assert.IsTrue(queue.TryGetPlayback(5, out var stillHeld));
            Assert.AreEqual(held.Progress, stillHeld.Progress);
            Receive(queue, Pcm(1600));
            Assert.IsFalse(queue.TrySchedule(5, 5.05, new float[3200], out _));
            Receive(queue, Pcm(1600));
            Assert.IsTrue(queue.TrySchedule(5, 5.05, new float[3200], out var resumed));
            Assert.AreEqual(5.05, resumed.DspStart, 0.00001);
            Assert.IsTrue(queue.TryGetPlayback(5.049, out stillHeld));
            Assert.AreEqual(held.Progress, stillHeld.Progress);
            Assert.IsTrue(queue.TryGetPlayback(5.15, out var advancing));
            Assert.AreEqual(0.5f, advancing.Progress, 0.001);
        }

        [Test]
        public void NewSpeakerHeaderDoesNotReplaceThePreviousUnplayedTail()
        {
            var queue = new MotionPcmQueue(16000);
            var finished = new List<string>();
            queue.Finished += finished.Add;
            Receive(queue, Header(3200), Identity("a", 3200, "first"));
            Receive(queue, Pcm(3200));
            Receive(queue, Header(3200), Identity("b", 3200, "second"));
            Receive(queue, Pcm(3200));
            Assert.IsTrue(queue.TrySchedule(0, 0.05, new float[3200], out var first));
            Assert.IsTrue(queue.TrySchedule(0, 0.05, new float[3200], out var second));
            Assert.AreEqual(first.DspEnd, second.DspStart, 0.00001);
            Assert.IsTrue(queue.TryGetPlayback(0.15, out var state));
            Assert.AreEqual("first", state.SpeakerUuid);
            Assert.IsEmpty(finished);
            Assert.IsTrue(queue.TryGetPlayback(0.35, out state));
            Assert.AreEqual("second", state.SpeakerUuid);
            CollectionAssert.AreEqual(new[] { "a" }, finished);
            queue.Advance(1);
            CollectionAssert.AreEqual(new[] { "a", "b" }, finished);
            Assert.AreEqual(0, queue.BufferedSamples);
        }

        [Test]
        public void CapacityIncludesAlreadyScheduledButUnplayedSamples()
        {
            var queue = new MotionPcmQueue(16000, 200, 200);
            Receive(queue, Header(6400), Identity("a", 6400));
            Receive(queue, Pcm(3200));
            queue.TrySchedule(0, 0.05, new float[3200], out _);
            Assert.AreEqual(3200, queue.BufferedSamples);
            Assert.IsFalse(Receive(queue, Pcm(100)));
            Assert.AreEqual(0, queue.BufferedSamples);
        }

        [Test]
        public void PlayingAudioReleasesCapacityBeforeItsChunkEnds()
        {
            var queue = new MotionPcmQueue(16000, 200, 200);
            Receive(queue, Header(6400), Identity("a", 6400));
            Receive(queue, Pcm(3200));
            queue.TrySchedule(0, 0.05, new float[3200], out _);
            queue.Advance(0.151);
            Assert.Less(queue.BufferedSamples, 1600);
            Assert.IsTrue(Receive(queue, Pcm(1600)));
            Assert.LessOrEqual(queue.BufferedSamples, 3200);
        }

        [Test]
        public void CancelRemovesScheduledAndQueuedAudioExactlyOnce()
        {
            var queue = new MotionPcmQueue(16000);
            var canceled = new List<string>();
            queue.Aborted += (id, reason) => canceled.Add(id);
            Receive(queue, Header(6400), Identity("a", 6400));
            Receive(queue, Pcm(6400));
            queue.TrySchedule(0, 0.05, new float[3200], out _);
            queue.Advance(0.15);
            queue.Cancel("a");
            queue.Cancel("a");
            Assert.AreEqual(0, queue.BufferedSamples);
            Assert.IsFalse(queue.TryGetPlayback(10, out _));
            Assert.IsFalse(queue.TrySchedule(10, 10.05, new float[3200], out _));
            CollectionAssert.AreEqual(new[] { "a" }, canceled);
        }

        [Test]
        public void CancelingOneSentenceKeepsAnotherSpeaker()
        {
            var queue = new MotionPcmQueue(16000);
            Receive(queue, Header(3200), Identity("a", 3200));
            Receive(queue, Pcm(3200));
            Receive(queue, Header(3200), Identity("b", 3200));
            Receive(queue, Pcm(3200));
            queue.Cancel("a");
            Assert.AreEqual(3200, queue.BufferedSamples);
            Assert.IsTrue(queue.TrySchedule(0, 0.05, new float[3200], out var chunk));
            Assert.AreEqual("b", chunk.RequestId);
        }

        [TestCase(3201, 16000)]
        [TestCase(3200, 48000)]
        public void MismatchedMetadataFallsBackToAnonymousAudio(int count, int rate)
        {
            var queue = new MotionPcmQueue(16000);
            var canceled = new List<string>();
            queue.Aborted += (id, reason) => canceled.Add(id);
            var identity = Identity("a", count);
            identity.SampleRate = rate;
            Assert.IsTrue(Receive(queue, Header(3200), identity));
            Assert.IsTrue(Receive(queue, Pcm(3200)));
            Assert.IsTrue(queue.TrySchedule(0, 0.05, new float[3200], out var chunk));
            Assert.IsNull(chunk.SpeakerUuid);
            Assert.IsTrue(chunk.RequestId.StartsWith("anonymous:", StringComparison.Ordinal));
            CollectionAssert.AreEqual(new[] { "a" }, canceled);
        }

        [Test]
        public void RepeatedIdentityCannotCancelTheEarlierSentence()
        {
            var queue = new MotionPcmQueue(16000);
            var identity = Identity("a", 3200);
            Receive(queue, Header(3200), identity);
            Receive(queue, Pcm(3200));
            Assert.IsTrue(Receive(queue, Header(3200), identity));
            Assert.IsTrue(Receive(queue, Pcm(3200)));
            Assert.AreEqual(6400, queue.BufferedSamples);
            Assert.IsTrue(queue.TrySchedule(0, 0.05, new float[3200], out var first));
            Assert.IsTrue(queue.TrySchedule(0, 0.05, new float[3200], out var second));
            Assert.AreEqual("a", first.RequestId);
            Assert.IsNull(second.SpeakerUuid);
            Assert.AreNotEqual(first.RequestId, second.RequestId);
        }

        [Test]
        public void InvalidMetadataDoesNotDiscardAnEarlierScheduledTail()
        {
            var queue = new MotionPcmQueue(16000);
            Receive(queue, Header(3200), Identity("a", 3200));
            Receive(queue, Pcm(3200));
            queue.TrySchedule(0, 0.05, new float[3200], out var first);
            Assert.IsTrue(Receive(queue, Header(3200), Identity("b", 100)));
            Assert.IsTrue(Receive(queue, Pcm(3200)));
            Assert.IsTrue(queue.TryGetPlayback(0.15, out var state));
            Assert.AreEqual("a", state.RequestId);
            Assert.IsTrue(queue.TrySchedule(0.15, 0.20, new float[3200], out var second));
            Assert.AreEqual(first.DspEnd, second.DspStart, 0.00001);
            Assert.IsNull(second.SpeakerUuid);
        }

        [Test]
        public void InterruptedHeaderCancelsOnlyTheIncompleteSentence()
        {
            var queue = new MotionPcmQueue(16000);
            var canceled = new List<string>();
            queue.Aborted += (id, reason) => canceled.Add(id);
            Receive(queue, Header(3200), Identity("complete", 3200));
            Receive(queue, Pcm(3200));
            Receive(queue, Header(6400), Identity("incomplete", 6400));
            Receive(queue, Pcm(3200));
            Assert.IsTrue(Receive(queue, Header(3200), Identity("next", 3200)));
            CollectionAssert.AreEqual(new[] { "incomplete" }, canceled);
            Assert.AreEqual(3200, queue.BufferedSamples);
            Assert.IsTrue(queue.TrySchedule(0, 0.05, new float[3200], out var chunk));
            Assert.AreEqual("complete", chunk.RequestId);
        }

        [Test]
        public void AnonymousLegacyAudioStillPlaysWithoutSpeakerIdentity()
        {
            var queue = new MotionPcmQueue(16000);
            Receive(queue, Header(100));
            Receive(queue, Pcm(100));
            Assert.IsTrue(queue.TrySchedule(0, 0.05, new float[3200], out var chunk));
            Assert.IsTrue(chunk.RequestId.StartsWith("anonymous:", StringComparison.Ordinal));
            Assert.IsNull(chunk.SpeakerUuid);
        }

        [Test]
        public void ExactNetworkMessageSliceIsRespected()
        {
            var queue = new MotionPcmQueue(16000);
            byte[] header = Header(100);
            var bytes = new byte[header.Length + 20];
            Array.Copy(header, 0, bytes, 7, header.Length);
            Assert.IsTrue(queue.Receive(bytes, 7, header.Length, Identity("a", 100), out _));
            byte[] pcm = Pcm(100, -32768);
            bytes = new byte[pcm.Length + 20];
            Array.Copy(pcm, 0, bytes, 9, pcm.Length);
            Assert.IsTrue(queue.Receive(bytes, 9, pcm.Length, null, out _));
            var output = new float[3200];
            queue.TrySchedule(0, 0.05, output, out _);
            Assert.AreEqual(-1, output[0]);
            Assert.AreEqual(-1, output[99]);
        }

        [TestCase("{\"type\":\"AudioInfo\",\"targetPeer\":\"\",\"audioLength\":200}")]
        [TestCase("{\"type\":\"AudioInfo\",\"targetPeer\":\"\",\"audioLength\":\"200\",\"x\":\"x\"}")]
        [TestCase("{\"type\":\"AudioInfo\",\"targetPeer\":\"\",\"audioLength\":\"201\"}")]
        [TestCase("{\"type\":\"AudioInfo\",\"targetPeer\":\"\",\"audioLength\":\"200\",}")]
        [TestCase("{\"type\":\"AudioInfo\",\"targetPeer\":\"\",\"audioLength\":\"200\",\"type\":\"AudioInfo\"}")]
        public void StrictHeaderRejectsMalformedOrExtendedLegacyProtocol(string text)
        {
            Assert.IsFalse(Receive(new MotionPcmQueue(16000), Encoding.UTF8.GetBytes(text)));
        }

        [Test]
        public void InvalidUtf8AndInvalidSlicesAreRejected()
        {
            var queue = new MotionPcmQueue(16000);
            Assert.IsFalse(Receive(queue, new byte[] { 0x7b, 0xc3, 0x28, 0x7d }));
            Assert.IsFalse(queue.Receive(new byte[200], int.MaxValue, 200, null, out _));
            Assert.IsFalse(queue.Receive(null, 0, 200, null, out _));
        }

        [TestCase(16000)]
        [TestCase(48000)]
        public void SourceSampleRateDeterminesDspDuration(int sourceRate)
        {
            var queue = new MotionPcmQueue(sourceRate);
            int samples = sourceRate / 10;
            var identity = Identity("a", samples);
            identity.SampleRate = sourceRate;
            Receive(queue, Header(samples), identity);
            Receive(queue, Pcm(samples));
            Assert.IsTrue(queue.TrySchedule(0, 0.05, new float[sourceRate / 5], out var chunk));
            Assert.AreEqual(0.1, chunk.DspEnd - chunk.DspStart, 0.000001);
        }

        [Test]
        public void ResetRequiresNewHeaderAndCannotFinishCanceledAudio()
        {
            var queue = new MotionPcmQueue(16000);
            bool finished = false;
            queue.Finished += id => finished = true;
            Receive(queue, Header(3200), Identity("a", 3200));
            Receive(queue, Pcm(3200));
            queue.TrySchedule(0, 0.05, new float[3200], out _);
            queue.Clear();
            queue.Advance(5);
            Assert.IsFalse(finished);
            Assert.IsFalse(Receive(queue, Pcm(100)));
            Assert.IsTrue(Receive(queue, Header(100), Identity("b", 100)));
        }

        [Test]
        public void RequestOwnershipContinuesThroughScheduledTailAndEndsOnPlaybackOrCancellation()
        {
            var queue = new MotionPcmQueue(16000);
            Assert.IsFalse(queue.ContainsRequest(null));
            Assert.IsFalse(queue.ContainsRequest("a"));
            Receive(queue, Header(3200), Identity("a", 3200));
            Assert.IsTrue(queue.ContainsRequest("a"));
            Receive(queue, Pcm(3200));
            queue.TrySchedule(0, 0.05, new float[3200], out _);
            Assert.IsTrue(queue.ContainsRequest("a"));
            queue.Advance(0.15);
            Assert.IsTrue(queue.ContainsRequest("a"));
            queue.Advance(0.3);
            Assert.IsFalse(queue.ContainsRequest("a"));
            Receive(queue, Header(3200), Identity("b", 3200));
            Assert.IsTrue(queue.ContainsRequest("b"));
            queue.Cancel("b");
            Assert.IsFalse(queue.ContainsRequest("b"));
        }
    }
}
