using System;
using System.Globalization;
using System.Text;

namespace UbiqGenie.RealtimeTranslation
{
    /// <summary>
    /// Receives the Node AudioInfo / mono PCM16LE protocol and supplies a bounded
    /// float FIFO to Unity's audio thread. No Unity API is used on either thread.
    /// </summary>
    public sealed class SoVITSPcmStream
    {
        public readonly struct ReceiveResult
        {
            public bool Accepted { get; }
            public bool ResetPlayback { get; }
            public string Warning { get; }

            internal ReceiveResult(bool accepted, bool resetPlayback, string warning)
            {
                Accepted = accepted;
                ResetPlayback = resetPlayback;
                Warning = warning;
            }
        }

        private static readonly UTF8Encoding StrictUtf8 = new UTF8Encoding(false, true);
        private readonly object gate = new object();
        private readonly float[] samples;
        private readonly int prebufferSamples;
        private readonly long maximumUtteranceBytes;
        private int readPosition;
        private int writePosition;
        private int bufferedSamples;
        private long remainingBytes;
        private bool receivingUtterance;
        private bool playbackReady;

        public int SourceSampleRate { get; }

        public int BufferedSamples
        {
            get { lock (gate) { return bufferedSamples; } }
        }

        // A header waiting for network data alone must not animate the assistant.
        public bool HasPendingAudio
        {
            get { lock (gate) { return bufferedSamples > 0; } }
        }

        public SoVITSPcmStream(int sourceSampleRate, int prebufferMilliseconds = 200,
            int maxBufferMilliseconds = 2000)
        {
            if (sourceSampleRate <= 0)
                throw new ArgumentOutOfRangeException(nameof(sourceSampleRate));
            if (maxBufferMilliseconds <= 0)
                throw new ArgumentOutOfRangeException(nameof(maxBufferMilliseconds));
            if (prebufferMilliseconds < 0 || prebufferMilliseconds > maxBufferMilliseconds)
                throw new ArgumentOutOfRangeException(nameof(prebufferMilliseconds));

            long capacity = ((long)sourceSampleRate * maxBufferMilliseconds + 999) / 1000;
            if (capacity > int.MaxValue)
                throw new ArgumentOutOfRangeException(nameof(maxBufferMilliseconds));

            SourceSampleRate = sourceSampleRate;
            samples = new float[(int)capacity];
            prebufferSamples = (int)(((long)sourceSampleRate * prebufferMilliseconds + 999) / 1000);
            maximumUtteranceBytes = (long)sourceSampleRate * 2 * 120;
        }

        /// <summary>Call on the network/main thread using the message's valid byte slice.</summary>
        public ReceiveResult Receive(byte[] bytes, int start, int length)
        {
            lock (gate)
            {
                if (bytes == null || start < 0 || length <= 0 || length > bytes.Length ||
                    start > bytes.Length - length)
                    return Reject("Invalid audio message slice.");

                if (length < 100)
                {
                    long declaredBytes;
                    if (!TryReadHeader(bytes, start, length, out declaredBytes))
                        return Reject("Invalid AudioInfo header; waiting for the next valid header.");

                    bool interrupted = receivingUtterance;
                    if (interrupted)
                        ClearLocked();

                    // A normally completed sentence may still have a tail in the FIFO.
                    // Retain it so consecutive sentences never lose their final samples.
                    remainingBytes = declaredBytes;
                    receivingUtterance = true;
                    return new ReceiveResult(true, interrupted, interrupted
                        ? "New AudioInfo interrupted an incomplete sentence; its buffered audio was cleared."
                        : null);
                }

                if (length < 200 || length > 16000 || (length & 1) != 0)
                    return Reject("PCM packets must contain 200–16000 bytes and an even byte count.");
                if (!receivingUtterance)
                    return Reject("PCM arrived without an active AudioInfo header.");
                if (length > remainingBytes)
                    return Reject("PCM exceeds the audioLength declared by AudioInfo.");

                int incomingSamples = length / 2;
                if (incomingSamples > samples.Length - bufferedSamples)
                    return Reject("PCM playback buffer overflow; sentence discarded until the next header.");

                int end = start + length;
                for (int offset = start; offset < end; offset += 2)
                {
                    short pcm = unchecked((short)(bytes[offset] | (bytes[offset + 1] << 8)));
                    samples[writePosition] = pcm / 32768.0f;
                    writePosition = (writePosition + 1) % samples.Length;
                }
                bufferedSamples += incomingSamples;
                remainingBytes -= length;
                receivingUtterance = remainingBytes != 0;
                if (bufferedSamples >= prebufferSamples || !receivingUtterance)
                    playbackReady = true;
                return new ReceiveResult(true, false, null);
            }
        }

        /// <summary>
        /// Audio-thread callback: consumes available samples once prebuffered, then
        /// fills any remainder with silence. It never allocates, logs, or calls Unity.
        /// </summary>
        public void Read(float[] destination)
        {
            if (destination == null)
                return;

            lock (gate)
            {
                int count = playbackReady ? Math.Min(bufferedSamples, destination.Length) : 0;
                for (int index = 0; index < count; ++index)
                {
                    destination[index] = samples[readPosition];
                    readPosition = (readPosition + 1) % samples.Length;
                }
                Array.Clear(destination, count, destination.Length - count);
                bufferedSamples -= count;
                if (bufferedSamples == 0)
                    playbackReady = false;
            }
        }

        public void Clear()
        {
            lock (gate) { ClearLocked(); }
        }

        private void ClearLocked()
        {
            readPosition = 0;
            writePosition = 0;
            bufferedSamples = 0;
            remainingBytes = 0;
            receivingUtterance = false;
            playbackReady = false;
        }

        private ReceiveResult Reject(string warning)
        {
            ClearLocked();
            return new ReceiveResult(false, true, warning);
        }

        // A tiny flat-object parser keeps the protocol strict without a dependency
        // on JSON packages or Unity's permissive field conversion. All values must
        // be strings; in particular audioLength is a numeric STRING on the wire.
        private bool TryReadHeader(byte[] bytes, int start, int length, out long declaredBytes)
        {
            declaredBytes = 0;
            string json;
            try { json = StrictUtf8.GetString(bytes, start, length); }
            catch (DecoderFallbackException) { return false; }

            int position = 0;
            if (!Take(json, ref position, '{'))
                return false;

            string type = null;
            string target = null;
            string audioLength = null;
            int seen = 0;
            while (true)
            {
                string key;
                string value;
                if (!ReadString(json, ref position, out key) ||
                    !Take(json, ref position, ':') ||
                    !ReadString(json, ref position, out value))
                    return false;

                int bit;
                switch (key)
                {
                    case "type": bit = 1; type = value; break;
                    case "targetPeer": bit = 2; target = value; break;
                    case "audioLength": bit = 4; audioLength = value; break;
                    default: return false;
                }
                if ((seen & bit) != 0)
                    return false;
                seen |= bit;

                if (Take(json, ref position, '}'))
                    break;
                if (!Take(json, ref position, ','))
                    return false;
            }

            SkipWhitespace(json, ref position);
            return position == json.Length && seen == 7 && type == "AudioInfo" &&
                target != null && long.TryParse(audioLength, NumberStyles.None,
                    CultureInfo.InvariantCulture, out declaredBytes) &&
                declaredBytes >= 200 && declaredBytes <= maximumUtteranceBytes &&
                (declaredBytes & 1) == 0;
        }

        private static void SkipWhitespace(string text, ref int position)
        {
            while (position < text.Length && (text[position] == ' ' || text[position] == '\t' ||
                text[position] == '\r' || text[position] == '\n'))
                ++position;
        }

        private static bool Take(string text, ref int position, char expected)
        {
            SkipWhitespace(text, ref position);
            if (position >= text.Length || text[position] != expected)
                return false;
            ++position;
            return true;
        }

        private static bool ReadString(string text, ref int position, out string value)
        {
            value = null;
            if (!Take(text, ref position, '"'))
                return false;

            var result = new StringBuilder();
            while (position < text.Length)
            {
                char c = text[position++];
                if (c == '"')
                {
                    value = result.ToString();
                    return true;
                }
                if (c < 0x20)
                    return false;
                if (c != '\\')
                {
                    result.Append(c);
                    continue;
                }
                if (position >= text.Length)
                    return false;

                switch (text[position++])
                {
                    case '"': result.Append('"'); break;
                    case '\\': result.Append('\\'); break;
                    case '/': result.Append('/'); break;
                    case 'b': result.Append('\b'); break;
                    case 'f': result.Append('\f'); break;
                    case 'n': result.Append('\n'); break;
                    case 'r': result.Append('\r'); break;
                    case 't': result.Append('\t'); break;
                    case 'u':
                        if (position + 4 > text.Length)
                            return false;
                        ushort code;
                        if (!ushort.TryParse(text.Substring(position, 4), NumberStyles.AllowHexSpecifier,
                            CultureInfo.InvariantCulture, out code))
                            return false;
                        result.Append((char)code);
                        position += 4;
                        break;
                    default: return false;
                }
            }
            return false;
        }
    }
}
