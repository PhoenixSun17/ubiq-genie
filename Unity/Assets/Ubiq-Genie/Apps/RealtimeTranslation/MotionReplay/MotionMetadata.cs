using System;

namespace Ubiq.Genie.Samples.RealtimeTranslationMotion
{
    /// <summary>Version 1 JSON on channel 101. Channel 95 retains the legacy AudioInfo/PCM format.</summary>
    [Serializable]
    public sealed class MotionMetadata
    {
        public const string BackendProperty = "ubiq.genie.sovits.motion";
        public int version;
        public string type;
        public string backendUuid;
        public string sessionId;
        public string requestId;
        public string speakerUuid;
        public double sourceStartMs;
        public double sourceEndMs;
        public int audioLength;
        public int sampleRate;
        public int sampleCount;
        public string status;
        public string peerUuid;
        public string nonce;
        public double t0;
        public double s1;
        public double s2;

        public bool Matches(string backend, string session) =>
            version == 1 && ValidId(backendUuid) && ValidId(sessionId) &&
            backendUuid == backend && sessionId == session;

        public bool ValidUtterance => ValidId(requestId) && ValidId(speakerUuid);

        public bool ValidInterval => ValidUtterance && MotionClock.Finite(sourceStartMs) &&
            MotionClock.Finite(sourceEndMs) && sourceEndMs > sourceStartMs &&
            sourceEndMs - sourceStartMs <= 30000;

        public bool ValidAudio => ValidUtterance && sampleRate >= 8000 && sampleRate <= 96000 &&
            sampleCount >= 100 && sampleCount <= sampleRate * 120 &&
            (long)sampleCount * 2 == audioLength;

        public static bool ValidId(string value) => !string.IsNullOrEmpty(value) && value.Length <= 128;
    }
}
