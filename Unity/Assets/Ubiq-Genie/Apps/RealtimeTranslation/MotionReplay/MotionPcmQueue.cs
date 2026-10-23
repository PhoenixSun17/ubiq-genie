using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace Ubiq.Genie.Samples.RealtimeTranslationMotion
{
    /// <summary>Sideband identity for the following unchanged channel 95 AudioInfo header.</summary>
    public sealed class MotionAudioIdentity
    {
        public string RequestId;
        public string SpeakerUuid;
        public int SampleCount;
        public int SampleRate;
    }

    public readonly struct MotionPlaybackPosition
    {
        public string RequestId { get; }
        public string SpeakerUuid { get; }
        public float Progress { get; }

        public MotionPlaybackPosition(string requestId, string speakerUuid, float progress)
        {
            RequestId = requestId;
            SpeakerUuid = speakerUuid;
            Progress = progress;
        }
    }

    public readonly struct MotionScheduledChunk
    {
        public string RequestId { get; }
        public string SpeakerUuid { get; }
        public int SampleCount { get; }
        public double DspStart { get; }
        public double DspEnd { get; }

        internal MotionScheduledChunk(string requestId, string speakerUuid, int count, double start, double end)
        {
            RequestId = requestId;
            SpeakerUuid = speakerUuid;
            SampleCount = count;
            DspStart = start;
            DspEnd = end;
        }
    }

    /// <summary>
    /// Main-thread PCM framing and a deterministic DSP playback timeline. Capacity includes
    /// queued samples and scheduled samples that have not played. No Unity APIs are used.
    /// </summary>
    public sealed class MotionPcmQueue
    {
        private sealed class Sentence
        {
            public string Id;
            public string Speaker;
            public int Total;
            public int Received;
            public int Scheduled;
            public int Played;
            public bool Started;
            public readonly Queue<float> Samples = new Queue<float>();
        }

        private sealed class Span
        {
            public Sentence Sentence;
            public int First;
            public int Count;
            public double Start;
            public double End;
        }

        private static readonly UTF8Encoding StrictUtf8 = new UTF8Encoding(false, true);
        private readonly List<Sentence> sentences = new List<Sentence>();
        private readonly List<Span> spans = new List<Span>();
        private readonly int prebufferSamples;
        private readonly int capacity;
        private Sentence receiving;
        private long anonymousSequence;

        public int SampleRate { get; }
        public int BufferedSamples { get; private set; }
        public event Action<string> Finished;
        public event Action<string, string> Aborted;

        public MotionPcmQueue(int sampleRate, int prebufferMilliseconds = 200, int maxBufferMilliseconds = 2000)
        {
            if (sampleRate < 8000 || sampleRate > 96000) throw new ArgumentOutOfRangeException(nameof(sampleRate));
            if (maxBufferMilliseconds <= 0 || maxBufferMilliseconds > 10000)
                throw new ArgumentOutOfRangeException(nameof(maxBufferMilliseconds));
            if (prebufferMilliseconds < 0 || prebufferMilliseconds > maxBufferMilliseconds)
                throw new ArgumentOutOfRangeException(nameof(prebufferMilliseconds));
            SampleRate = sampleRate;
            capacity = (int)(((long)sampleRate * maxBufferMilliseconds + 999) / 1000);
            prebufferSamples = (int)(((long)sampleRate * prebufferMilliseconds + 999) / 1000);
        }

        public bool Receive(byte[] bytes, int start, int length, MotionAudioIdentity identity, out string warning)
        {
            warning = null;
            if (bytes == null || start < 0 || length <= 0 || length > bytes.Length || start > bytes.Length - length)
                return Reject("Invalid audio message slice.", out warning);

            if (length < 100)
            {
                if (!TryReadHeader(bytes, start, length, out int total))
                    return Reject("Invalid AudioInfo header; waiting for the next valid header.", out warning);
                if (identity != null && (string.IsNullOrEmpty(identity.RequestId) ||
                    string.IsNullOrEmpty(identity.SpeakerUuid) || identity.RequestId.Length > 128 ||
                    identity.SpeakerUuid.Length > 128 || identity.SampleRate != SampleRate || identity.SampleCount != total ||
                    sentences.Exists(item => item.Id == identity.RequestId)))
                {
                    warning = "Translation metadata is invalid, repeated, or mismatched; playing anonymous audio only.";
                    // An unusable association must not discard valid PCM or an earlier sentence's tail.
                    // Never let a repeated ID cancel the earlier playback that legitimately owns it.
                    if (!string.IsNullOrEmpty(identity.RequestId) && !sentences.Exists(item => item.Id == identity.RequestId))
                        Aborted?.Invoke(identity.RequestId, warning);
                    identity = null;
                }
                string id = identity?.RequestId ?? "anonymous:" + (++anonymousSequence).ToString(CultureInfo.InvariantCulture);
                if (receiving != null)
                {
                    warning = "New AudioInfo interrupted an incomplete sentence.";
                    Cancel(receiving.Id, warning);
                }
                // Minimum PCM packet length bounds the normal count; also bound empty/malicious headers.
                if (sentences.Count >= 128) return Reject("Too many pending audio sentences.", out warning);
                receiving = new Sentence { Id = id, Speaker = identity?.SpeakerUuid, Total = total };
                sentences.Add(receiving);
                return true;
            }

            if (length < 200 || length > 16000 || (length & 1) != 0)
                return Reject("PCM packets must contain 200–16000 bytes and an even byte count.", out warning);
            if (receiving == null) return Reject("PCM arrived without an active AudioInfo header.", out warning);
            int count = length / 2;
            if (count > receiving.Total - receiving.Received)
                return Reject("PCM exceeds the declared audio length.", out warning);
            if (count > capacity - BufferedSamples)
                return Reject("PCM playback buffer overflow; waiting for a new header.", out warning);
            for (int offset = start; offset < start + length; offset += 2)
            {
                short pcm = unchecked((short)(bytes[offset] | (bytes[offset + 1] << 8)));
                receiving.Samples.Enqueue(pcm / 32768.0f);
            }
            receiving.Received += count;
            BufferedSamples += count;
            if (receiving.Received == receiving.Total) receiving = null;
            return true;
        }

        /// <summary>Copy one real-audio chunk into the caller's reusable buffer and reserve its DSP span.</summary>
        public bool TrySchedule(double now, double earliestStart, float[] destination, out MotionScheduledChunk chunk)
        {
            chunk = default;
            if (destination == null || destination.Length == 0 || double.IsNaN(now) || double.IsInfinity(now) ||
                double.IsNaN(earliestStart) || double.IsInfinity(earliestStart))
                return false;
            Advance(now);
            Sentence sentence = null;
            foreach (var candidate in sentences)
            {
                if (candidate.Scheduled < candidate.Total) { sentence = candidate; break; }
            }
            if (sentence == null || sentence.Samples.Count == 0) return false;
            bool complete = sentence.Received == sentence.Total;
            double previousEnd = spans.Count > 0 ? spans[spans.Count - 1].End : now;
            bool restarting = previousEnd <= now;
            if ((!sentence.Started || restarting) && !complete && sentence.Samples.Count < prebufferSamples)
                return false;
            // Coalesce transport packets into full chunks except the exact final tail.
            if (sentence.Samples.Count < destination.Length && !complete) return false;
            int count = Math.Min(destination.Length, sentence.Samples.Count);
            for (int index = 0; index < count; ++index) destination[index] = sentence.Samples.Dequeue();
            Array.Clear(destination, count, destination.Length - count);
            double start = Math.Max(Math.Max(now, earliestStart), previousEnd);
            double end = start + (double)count / SampleRate;
            spans.Add(new Span { Sentence = sentence, First = sentence.Scheduled, Count = count, Start = start, End = end });
            sentence.Scheduled += count;
            chunk = new MotionScheduledChunk(sentence.Id, sentence.Speaker, count, start, end);
            return true;
        }

        /// <summary>Advance only through scheduled real audio. Silence during underrun leaves progress unchanged.</summary>
        public void Advance(double now)
        {
            for (int index = 0; index < spans.Count;)
            {
                var span = spans[index];
                if (now < span.Start) break;
                var sentence = span.Sentence;
                sentence.Started = true;
                int within = now >= span.End ? span.Count :
                    Math.Min(span.Count, Math.Max(0, (int)Math.Floor((now - span.Start) * SampleRate)));
                int played = Math.Max(sentence.Played, span.First + within);
                BufferedSamples -= played - sentence.Played;
                sentence.Played = played;
                if (now < span.End) break;
                spans.RemoveAt(index);
            }
            for (int index = 0; index < sentences.Count;)
            {
                var sentence = sentences[index];
                if (sentence.Played < sentence.Total) { ++index; continue; }
                sentences.RemoveAt(index);
                Finished?.Invoke(sentence.Id);
            }
        }

        public bool TryGetPlayback(double now, out MotionPlaybackPosition position)
        {
            Advance(now);
            foreach (var sentence in sentences)
            {
                if (!sentence.Started) continue;
                position = new MotionPlaybackPosition(sentence.Id, sentence.Speaker,
                    (float)sentence.Played / sentence.Total);
                return true;
            }
            position = default;
            return false;
        }

        public bool ContainsRequest(string requestId)
        {
            if (string.IsNullOrEmpty(requestId)) return false;
            foreach (var sentence in sentences)
                if (sentence.Id == requestId) return true;
            return false;
        }

        public void Cancel(string requestId, string reason = "Translation canceled.")
        {
            var sentence = sentences.Find(item => item.Id == requestId);
            if (sentence == null) return;
            if (receiving == sentence) receiving = null;
            BufferedSamples -= sentence.Received - sentence.Played;
            spans.RemoveAll(item => item.Sentence == sentence);
            sentences.Remove(sentence);
            Aborted?.Invoke(sentence.Id, reason);
        }

        public void Clear(string reason = "Audio playback reset.")
        {
            while (sentences.Count > 0) Cancel(sentences[0].Id, reason);
            receiving = null;
            spans.Clear();
            BufferedSamples = 0;
        }

        private bool Reject(string message, out string warning)
        {
            warning = message;
            Clear(message);
            return false;
        }

        // Preserve the strict legacy header grammar without changing the existing sample's parser.
        private bool TryReadHeader(byte[] bytes, int start, int length, out int samples)
        {
            samples = 0;
            string json;
            try { json = StrictUtf8.GetString(bytes, start, length); }
            catch (DecoderFallbackException) { return false; }
            int position = 0;
            if (!Take(json, ref position, '{')) return false;
            string type = null, target = null, audioLength = null;
            int seen = 0;
            while (true)
            {
                if (!ReadString(json, ref position, out string key) || !Take(json, ref position, ':') ||
                    !ReadString(json, ref position, out string value)) return false;
                int bit;
                switch (key)
                {
                    case "type": bit = 1; type = value; break;
                    case "targetPeer": bit = 2; target = value; break;
                    case "audioLength": bit = 4; audioLength = value; break;
                    default: return false;
                }
                if ((seen & bit) != 0) return false;
                seen |= bit;
                if (Take(json, ref position, '}')) break;
                if (!Take(json, ref position, ',')) return false;
            }
            SkipWhitespace(json, ref position);
            if (position != json.Length || seen != 7 || type != "AudioInfo" || target == null ||
                !int.TryParse(audioLength, NumberStyles.None, CultureInfo.InvariantCulture, out int bytesCount) ||
                bytesCount < 200 || bytesCount > (long)SampleRate * 2 * 120 || (bytesCount & 1) != 0)
                return false;
            samples = bytesCount / 2;
            return true;
        }

        private static void SkipWhitespace(string text, ref int position)
        {
            while (position < text.Length && (text[position] == ' ' || text[position] == '\t' ||
                text[position] == '\r' || text[position] == '\n')) ++position;
        }

        private static bool Take(string text, ref int position, char expected)
        {
            SkipWhitespace(text, ref position);
            if (position >= text.Length || text[position] != expected) return false;
            ++position;
            return true;
        }

        private static bool ReadString(string text, ref int position, out string value)
        {
            value = null;
            if (!Take(text, ref position, '"')) return false;
            var result = new StringBuilder();
            while (position < text.Length)
            {
                char c = text[position++];
                if (c == '"') { value = result.ToString(); return true; }
                if (c < 0x20) return false;
                if (c != '\\') { result.Append(c); continue; }
                if (position >= text.Length) return false;
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
                        if (position + 4 > text.Length || !ushort.TryParse(text.Substring(position, 4),
                            NumberStyles.AllowHexSpecifier, CultureInfo.InvariantCulture, out ushort code)) return false;
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
