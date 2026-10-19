using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using NUnit.Framework;

namespace UbiqGenie.RealtimeTranslation.Tests
{
    public class SoVITSPcmStreamTests
    {
        private static byte[] Header(long byteCount)
        {
            return Encoding.UTF8.GetBytes("{\"type\":\"AudioInfo\",\"targetPeer\":\"\",\"audioLength\":\"" + byteCount + "\"}");
        }

        private static byte[] Pcm(int sampleCount, short value)
        {
            var bytes = new byte[sampleCount * 2];
            for (int i = 0; i < bytes.Length; i += 2)
            {
                bytes[i] = unchecked((byte)value);
                bytes[i + 1] = unchecked((byte)(value >> 8));
            }
            return bytes;
        }

        private static SoVITSPcmStream.ReceiveResult Send(SoVITSPcmStream stream, byte[] bytes)
        {
            return stream.Receive(bytes, 0, bytes.Length);
        }

        [Test]
        public void HeaderAndPcmRespectMessageSliceOffsets()
        {
            var stream = new SoVITSPcmStream(16000);
            byte[] header = Header(200);
            var wrappedHeader = new byte[header.Length + 17];
            Array.Copy(header, 0, wrappedHeader, 7, header.Length);
            Assert.IsTrue(stream.Receive(wrappedHeader, 7, header.Length).Accepted);
            byte[] pcm = Pcm(100, -16384);
            var wrappedPcm = new byte[pcm.Length + 29];
            Array.Copy(pcm, 0, wrappedPcm, 11, pcm.Length);
            Assert.IsTrue(stream.Receive(wrappedPcm, 11, pcm.Length).Accepted);
            var output = new float[100];
            stream.Read(output);
            Assert.That(output, Is.All.EqualTo(-0.5f));
        }

        [TestCase(0, 0)]
        [TestCase(-1, 200)]
        [TestCase(0, -1)]
        [TestCase(1, 200)]
        [TestCase(0, 201)]
        [TestCase(int.MaxValue, 1)]
        public void InvalidSlicesAreRejectedAndReset(int start, int length)
        {
            var stream = new SoVITSPcmStream(16000);
            Send(stream, Header(200));
            Send(stream, Pcm(100, 123));
            var result = stream.Receive(new byte[200], start, length);
            Assert.IsFalse(result.Accepted);
            Assert.IsTrue(result.ResetPlayback);
            Assert.IsNotEmpty(result.Warning);
            Assert.AreEqual(0, stream.BufferedSamples);
        }

        [Test]
        public void NullMessageIsRejected()
        {
            Assert.IsFalse(new SoVITSPcmStream(16000).Receive(null, 0, 200).Accepted);
        }

        [TestCase("")]
        [TestCase("{}")]
        [TestCase("{\"type\":\"AudioInfo\",\"audioLength\":\"200\"}")]
        [TestCase("{\"type\":\"Other\",\"targetPeer\":\"\",\"audioLength\":\"200\"}")]
        [TestCase("{\"type\":\"AudioInfo\",\"targetPeer\":\"\",\"audioLength\":200}")]
        [TestCase("{\"type\":\"AudioInfo\",\"targetPeer\":null,\"audioLength\":\"200\"}")]
        [TestCase("{\"type\":\"AudioInfo\",\"targetPeer\":\"\",\"audioLength\":\"200\",\"extra\":\"x\"}")]
        [TestCase("{\"type\":\"AudioInfo\",\"targetPeer\":\"\",\"audioLength\":\"200\",\"type\":\"AudioInfo\"}")]
        [TestCase("{\"type\":\"AudioInfo\",\"targetPeer\":\"\",\"audioLength\":\"200\",}")]
        [TestCase("{\"type\":\"AudioInfo\",\"targetPeer\":\"\",\"audioLength\":\"200\"}x")]
        [TestCase("{\"type\":\"AudioInfo\",\"targetPeer\":\"\",\"audioLength\":\" 200\"}")]
        [TestCase("{\"type\":\"AudioInfo\",\"targetPeer\":\"\\q\",\"audioLength\":\"200\"}")]
        public void MalformedHeadersRequireFreshValidHeader(string json)
        {
            var stream = new SoVITSPcmStream(16000);
            var result = Send(stream, Encoding.UTF8.GetBytes(json));
            Assert.IsFalse(result.Accepted);
            Assert.IsTrue(result.ResetPlayback);
            Assert.IsFalse(Send(stream, Pcm(100, 1)).Accepted);
            Assert.IsTrue(Send(stream, Header(200)).Accepted);
            Assert.IsTrue(Send(stream, Pcm(100, 1)).Accepted);
        }

        [Test]
        public void InvalidUtf8HeaderIsRejected()
        {
            Assert.IsFalse(Send(new SoVITSPcmStream(16000), new byte[] { 0x7b, 0xc3, 0x28, 0x7d }).Accepted);
        }

        [Test]
        public void HeaderAcceptsFieldOrderWhitespaceAndStringEscapes()
        {
            var stream = new SoVITSPcmStream(16000);
            byte[] bytes = Encoding.UTF8.GetBytes(" { \"audioLength\":\"200\",\"targetPeer\":\"peer\\u0031\",\"type\":\"AudioInfo\" } ");
            Assert.IsTrue(Send(stream, bytes).Accepted);
        }

        [TestCase(198, false)]
        [TestCase(200, true)]
        [TestCase(201, false)]
        [TestCase(3840000, true)]
        [TestCase(3840002, false)]
        [TestCase(-200, false)]
        public void HeaderLengthIsEvenBoundedAndAtLeastOnePacket(long length, bool accepted)
        {
            Assert.AreEqual(accepted, Send(new SoVITSPcmStream(16000), Header(length)).Accepted);
        }

        [TestCase(100)]
        [TestCase(199)]
        [TestCase(201)]
        [TestCase(16001)]
        [TestCase(16002)]
        public void InvalidPcmPacketSizesAreRejected(int length)
        {
            var stream = new SoVITSPcmStream(16000);
            Send(stream, Header(20000));
            Assert.IsFalse(Send(stream, new byte[length]).Accepted);
            Assert.IsFalse(Send(stream, Pcm(100, 1)).Accepted);
        }

        [Test]
        public void MaximumPcmPacketIsAccepted()
        {
            var stream = new SoVITSPcmStream(16000);
            Send(stream, Header(16000));
            Assert.IsTrue(Send(stream, Pcm(8000, short.MaxValue)).Accepted);
            var output = new float[8000];
            stream.Read(output);
            Assert.That(output, Is.All.EqualTo(short.MaxValue / 32768.0f));
        }

        [Test]
        public void DeclaredLengthCannotBeOverrun()
        {
            var stream = new SoVITSPcmStream(16000);
            Send(stream, Header(200));
            Assert.IsFalse(Send(stream, Pcm(200, 1)).Accepted);
            Assert.AreEqual(0, stream.BufferedSamples);
        }

        [Test]
        public void PcmRequiresHeaderIncludingAfterCompletedSentence()
        {
            var stream = new SoVITSPcmStream(16000);
            Assert.IsFalse(Send(stream, Pcm(100, 1)).Accepted);
            Send(stream, Header(200));
            Send(stream, Pcm(100, 1));
            Assert.IsFalse(Send(stream, Pcm(100, 2)).Accepted);
            Assert.AreEqual(0, stream.BufferedSamples);
        }

        [Test]
        public void PrebufferWaitsWithoutConsumingSamplesThenBeginsPlayback()
        {
            var stream = new SoVITSPcmStream(1000);
            Send(stream, Header(600));
            Send(stream, Pcm(100, 8192));
            var output = new float[100];
            stream.Read(output);
            Assert.That(output, Is.All.Zero);
            Assert.AreEqual(100, stream.BufferedSamples);
            Send(stream, Pcm(100, 16384));
            stream.Read(output);
            Assert.That(output, Is.All.EqualTo(0.25f));
            stream.Read(output);
            Assert.That(output, Is.All.EqualTo(0.5f));
        }

        [Test]
        public void ShortCompletedSentenceDrainsBeforePrebufferThreshold()
        {
            var stream = new SoVITSPcmStream(16000);
            Send(stream, Header(200));
            Send(stream, Pcm(100, short.MinValue));
            var output = new float[120];
            stream.Read(output);
            for (int i = 0; i < 100; ++i) Assert.AreEqual(-1.0f, output[i]);
            for (int i = 100; i < 120; ++i) Assert.AreEqual(0.0f, output[i]);
            Assert.IsFalse(stream.HasPendingAudio);
        }

        [Test]
        public void TwoHundredByteFinalPacketIsRetained()
        {
            var stream = new SoVITSPcmStream(16000);
            Send(stream, Header(3400));
            Send(stream, Pcm(1600, 8192));
            Assert.IsTrue(Send(stream, Pcm(100, 16384)).Accepted);
            var output = new float[1700];
            stream.Read(output);
            for (int i = 0; i < 1600; ++i) Assert.AreEqual(0.25f, output[i]);
            for (int i = 1600; i < 1700; ++i) Assert.AreEqual(0.5f, output[i]);
        }

        [Test]
        public void UnderflowOutputsSilenceAndRetainsLaterNetworkSamples()
        {
            var stream = new SoVITSPcmStream(1000, 0);
            Send(stream, Header(400));
            Send(stream, Pcm(100, 8192));
            var output = new float[150];
            stream.Read(output);
            for (int i = 0; i < 100; ++i) Assert.AreEqual(0.25f, output[i]);
            for (int i = 100; i < 150; ++i) Assert.AreEqual(0.0f, output[i]);
            stream.Read(output);
            Assert.That(output, Is.All.Zero);
            Send(stream, Pcm(100, 16384));
            stream.Read(output);
            for (int i = 0; i < 100; ++i) Assert.AreEqual(0.5f, output[i]);
            for (int i = 100; i < 150; ++i) Assert.AreEqual(0.0f, output[i]);
        }

        [Test]
        public void CompletedSentenceTailPrecedesNextSentenceWithoutReset()
        {
            var stream = new SoVITSPcmStream(16000);
            Send(stream, Header(200));
            Send(stream, Pcm(100, 8192));
            stream.Read(new float[60]);
            var result = Send(stream, Header(200));
            Assert.IsTrue(result.Accepted);
            Assert.IsFalse(result.ResetPlayback);
            Assert.IsNull(result.Warning);
            Send(stream, Pcm(100, 16384));
            var output = new float[140];
            stream.Read(output);
            for (int i = 0; i < 40; ++i) Assert.AreEqual(0.25f, output[i]);
            for (int i = 40; i < 140; ++i) Assert.AreEqual(0.5f, output[i]);
        }

        [Test]
        public void InterruptingHeaderDiscardsIncompleteSentenceAndStartsFresh()
        {
            var stream = new SoVITSPcmStream(16000);
            Send(stream, Header(400));
            Send(stream, Pcm(100, 8192));
            var result = Send(stream, Header(200));
            Assert.IsTrue(result.Accepted);
            Assert.IsTrue(result.ResetPlayback);
            Assert.IsNotEmpty(result.Warning);
            Assert.AreEqual(0, stream.BufferedSamples);
            Send(stream, Pcm(100, 16384));
            var output = new float[100];
            stream.Read(output);
            Assert.That(output, Is.All.EqualTo(0.5f));
        }

        [Test]
        public void OverflowClearsAndDiscardsUntilNextHeader()
        {
            var stream = new SoVITSPcmStream(1000, 100, 200);
            Send(stream, Header(600));
            Send(stream, Pcm(100, 1));
            Send(stream, Pcm(100, 1));
            var result = Send(stream, Pcm(100, 1));
            Assert.IsFalse(result.Accepted);
            Assert.IsTrue(result.ResetPlayback);
            Assert.IsNotEmpty(result.Warning);
            Assert.AreEqual(0, stream.BufferedSamples);
            Assert.IsFalse(Send(stream, Pcm(100, 1)).Accepted);
            Assert.IsTrue(Send(stream, Header(200)).Accepted);
            Assert.IsTrue(Send(stream, Pcm(100, 2)).Accepted);
        }

        [Test]
        public void RingWrapPreservesSampleOrder()
        {
            var stream = new SoVITSPcmStream(1000, 0, 300);
            Send(stream, Header(1000));
            var output = new float[100];
            for (short sample = 1; sample <= 5; ++sample)
            {
                Assert.IsTrue(Send(stream, Pcm(100, sample)).Accepted);
                stream.Read(output);
                Assert.That(output, Is.All.EqualTo(sample / 32768.0f));
            }
        }

        [Test]
        public void PendingAudioRequiresBufferedSamplesAndClearResetsEverything()
        {
            var stream = new SoVITSPcmStream(16000);
            Assert.AreEqual(16000, stream.SourceSampleRate);
            Send(stream, Header(400));
            Assert.IsFalse(stream.HasPendingAudio);
            Send(stream, Pcm(100, 1));
            Assert.IsTrue(stream.HasPendingAudio);
            stream.Clear();
            Assert.IsFalse(stream.HasPendingAudio);
            Assert.AreEqual(0, stream.BufferedSamples);
            var output = new float[] { 1, 2, 3 };
            stream.Read(output);
            Assert.That(output, Is.All.Zero);
            Assert.IsFalse(Send(stream, Pcm(100, 1)).Accepted);
            Send(stream, Header(200));
            Assert.IsTrue(Send(stream, Pcm(100, 2)).Accepted);
        }

        [Test]
        public void ConcurrentReceiveAndReadPreserveEverySampleInOrder()
        {
            const int chunks = 20;
            var stream = new SoVITSPcmStream(1000, 0, 2000);
            Send(stream, Header(chunks * 200));
            var captured = new List<float>();
            using (var producerFinished = new ManualResetEventSlim(false))
            {
                var producer = Task.Run(() =>
                {
                    try
                    {
                        for (short sample = 1; sample <= chunks; ++sample)
                        {
                            Assert.IsTrue(Send(stream, Pcm(100, sample)).Accepted);
                            Thread.Yield();
                        }
                    }
                    finally { producerFinished.Set(); }
                });
                var consumer = Task.Run(() =>
                {
                    var output = new float[37];
                    while (!producerFinished.IsSet || stream.HasPendingAudio)
                    {
                        stream.Read(output);
                        foreach (float value in output)
                            if (value != 0) captured.Add(value);
                        Thread.Yield();
                    }
                });
                Assert.IsTrue(Task.WaitAll(new[] { producer, consumer }, 5000), "Concurrent workers timed out.");
            }
            Assert.AreEqual(chunks * 100, captured.Count);
            for (int i = 0; i < captured.Count; ++i)
                Assert.AreEqual((i / 100 + 1) / 32768.0f, captured[i]);
        }

        [TestCase(0, 200, 2000)]
        [TestCase(-1, 200, 2000)]
        [TestCase(16000, -1, 2000)]
        [TestCase(16000, 2001, 2000)]
        [TestCase(16000, 0, 0)]
        [TestCase(int.MaxValue, 200, int.MaxValue)]
        public void InvalidConfigurationIsRejected(int rate, int prebuffer, int capacity)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new SoVITSPcmStream(rate, prebuffer, capacity));
        }
    }
}
