using System;
using UnityEngine;

namespace Ubiq.Genie.Samples.RealtimeTranslationMotion
{
    /// <summary>Plays tagged translation PCM on a DSP timeline shared with remote motion replay.</summary>
    [DisallowMultipleComponent]
    [DefaultExecutionOrder(-100)]
    public sealed class MotionAudioPlayer : MonoBehaviour
    {
        [Tooltip("Must match Node translation.outputSampleRate. Re-enable after changing audio settings.")]
        public int sourceSampleRate = 16000;
        [Min(0)] public int prebufferMilliseconds = 200;
        [Min(1)] public int maxBufferMilliseconds = 2000;

        public Func<string, Pose?> HeadPose;
        public Func<string, bool> IsLocalSpeaker;
        public event Action<string> PlaybackFinished;
        public event Action<string, string> PlaybackAborted;
        public bool IsReady => isActiveAndEnabled && queue != null;

        private sealed class Slot
        {
            public AudioSource Source;
            public AudioClip Clip;
            public float[] Samples;
            public float[] Tail;
            public MotionScheduledChunk Chunk;
            public bool Busy;
        }

        private readonly Slot[] slots = new Slot[3];
        private MotionPcmQueue queue;
        private double leadSeconds;

        private void OnEnable()
        {
            try
            {
                queue = new MotionPcmQueue(sourceSampleRate, prebufferMilliseconds, maxBufferMilliseconds);
                queue.Finished += OnFinished;
                queue.Aborted += OnAborted;
                int chunkSamples = Math.Max(1, sourceSampleRate / 5);
                for (int index = 0; index < slots.Length; ++index)
                {
                    var child = new GameObject("Translation audio " + index);
                    child.transform.SetParent(transform, false);
                    var output = child.AddComponent<AudioSource>();
                    output.playOnAwake = false;
                    output.loop = false;
                    output.pitch = 1;
                    output.dopplerLevel = 0;
                    output.rolloffMode = AudioRolloffMode.Logarithmic;
                    output.minDistance = 1;
                    output.maxDistance = 20;
                    slots[index] = new Slot { Source = output, Samples = new float[chunkSamples] };
                }
                RefreshLead();
                AudioSettings.OnAudioConfigurationChanged += OnAudioConfigurationChanged;
            }
            catch (Exception error)
            {
                Debug.LogError("[MotionReplay] Cannot initialise translated audio: " + error.Message, this);
                enabled = false;
            }
        }

        public bool Receive(byte[] bytes, int start, int length, MotionAudioIdentity identityForHeader, out string warning)
        {
            warning = null;
            if (!IsReady) { warning = "Motion audio player is not ready."; return false; }
            queue.Advance(AudioSettings.dspTime);
            return queue.Receive(bytes, start, length, identityForHeader, out warning);
        }

        public bool TryGetPlayback(out MotionPlaybackPosition state)
        {
            state = default;
            return IsReady && queue.TryGetPlayback(AudioSettings.dspTime, out state);
        }

        public bool ContainsRequest(string requestId)
        {
            return IsReady && queue.ContainsRequest(requestId);
        }

        public void Cancel(string requestId)
        {
            queue?.Cancel(requestId);
        }

        public void ClearPlayback()
        {
            queue?.Clear();
            StopSlots(null);
        }

        /// <summary>Call after applying the remote replay pose in LateUpdate to follow its current head.</summary>
        public void RefreshSpatialPositions()
        {
            foreach (var slot in slots)
            {
                if (slot != null && slot.Busy) FollowSpeaker(slot);
            }
        }

        private void Update()
        {
            if (!IsReady) return;
            double now = AudioSettings.dspTime;
            queue.Advance(now);
            foreach (var slot in slots)
            {
                if (slot.Busy && now >= slot.Chunk.DspEnd) slot.Busy = false;
            }
            foreach (var slot in slots)
            {
                if (slot.Busy) continue;
                if (!queue.TrySchedule(now, now + leadSeconds, slot.Samples, out var chunk)) break;
                slot.Chunk = chunk;
                slot.Busy = true;
                PrepareClip(slot, chunk.SampleCount);
                FollowSpeaker(slot);
                slot.Source.PlayScheduled(chunk.DspStart);
            }
        }

        private void PrepareClip(Slot slot, int count)
        {
            // Only the final partial chunk needs a smaller clip; normal chunks reuse pooled storage.
            if (!slot.Clip || slot.Clip.samples != count)
            {
                slot.Source.Stop();
                slot.Source.clip = null;
                if (slot.Clip) Destroy(slot.Clip);
                slot.Clip = AudioClip.Create("Translation PCM", count, 1, sourceSampleRate, false);
            }
            if (count == slot.Samples.Length)
            {
                slot.Clip.SetData(slot.Samples, 0);
            }
            else
            {
                if (slot.Tail == null || slot.Tail.Length != count) slot.Tail = new float[count];
                Array.Copy(slot.Samples, slot.Tail, count);
                slot.Clip.SetData(slot.Tail, 0);
            }
            slot.Source.clip = slot.Clip;
        }

        private void FollowSpeaker(Slot slot)
        {
            string speaker = slot.Chunk.SpeakerUuid;
            bool local = !string.IsNullOrEmpty(speaker) && IsLocalSpeaker != null && IsLocalSpeaker(speaker);
            Pose? pose = !local && !string.IsNullOrEmpty(speaker) && HeadPose != null ? HeadPose(speaker) : null;
            slot.Source.spatialBlend = pose.HasValue ? 1 : 0;
            if (pose.HasValue)
                slot.Source.transform.SetPositionAndRotation(pose.Value.position, pose.Value.rotation);
        }

        private void OnFinished(string requestId)
        {
            PlaybackFinished?.Invoke(requestId);
        }

        private void OnAborted(string requestId, string reason)
        {
            StopSlots(requestId);
            PlaybackAborted?.Invoke(requestId, reason);
        }

        private void StopSlots(string requestId)
        {
            foreach (var slot in slots)
            {
                if (slot == null || (requestId != null && slot.Chunk.RequestId != requestId)) continue;
                if (slot.Source) slot.Source.Stop();
                slot.Busy = false;
            }
        }

        private void RefreshLead()
        {
            AudioSettings.GetDSPBufferSize(out int length, out _);
            leadSeconds = Math.Max(0.05, 2.0 * length / Math.Max(1, AudioSettings.outputSampleRate));
        }

        private void OnAudioConfigurationChanged(bool changed)
        {
            ClearPlayback();
            RefreshLead();
        }

        private void OnDisable()
        {
            AudioSettings.OnAudioConfigurationChanged -= OnAudioConfigurationChanged;
            ClearPlayback();
            if (queue != null)
            {
                queue.Finished -= OnFinished;
                queue.Aborted -= OnAborted;
                queue = null;
            }
            for (int index = 0; index < slots.Length; ++index)
            {
                var slot = slots[index];
                if (slot == null) continue;
                if (slot.Clip) Destroy(slot.Clip);
                if (slot.Source) Destroy(slot.Source.gameObject);
                slots[index] = null;
            }
        }
    }
}
