using System.Collections.Generic;
using Ubiq.Voip;
using UnityEngine;

namespace Ubiq.Genie.Samples.RealtimeTranslationMotion
{
    /// <summary>Controls only this listener's original VoIP receive sources, never microphone capture.</summary>
    public sealed class OriginalVoiceToggle : MonoBehaviour
    {
        [Tooltip("Hear original remote speech as well as translated speech on this client.")]
        public bool playOriginalVoice = true;

        private readonly Dictionary<AudioSource, bool> sources = new Dictionary<AudioSource, bool>();
        private readonly List<AudioSource> removedSources = new List<AudioSource>();
        private VoipPeerConnectionManager manager;
        private bool subscribed;
        private bool previousValue;

        public void Initialize(VoipPeerConnectionManager connectionManager)
        {
            Unsubscribe();
            manager = connectionManager;
            Subscribe();
        }

        private void OnEnable() => Subscribe();

        private void Subscribe()
        {
            if (subscribed || !manager || !isActiveAndEnabled) return;
            subscribed = true;
            previousValue = playOriginalVoice;
            manager.OnPeerConnection.AddListener(OnPeerConnection);
        }

        private void OnPeerConnection(VoipPeerConnection connection)
        {
            if (!connection) return;
            removedSources.Clear();
            foreach (var entry in sources)
                if (!entry.Key) removedSources.Add(entry.Key);
            foreach (var removed in removedSources) sources.Remove(removed);
            // The current Unity WebRTC implementation puts its receive source directly on this object.
            // Capture is on PeerConnectionMicrophone, which is intentionally never searched here.
            var source = connection.GetComponent<AudioSource>();
            if (!source)
            {
                Debug.LogWarning("[Translation motion] No supported original-voice receive source; leaving it unchanged.", this);
                return;
            }
            if (!sources.ContainsKey(source)) sources.Add(source, source.mute);
            source.mute = !playOriginalVoice || sources[source];
        }

        private void Update()
        {
            if (previousValue == playOriginalVoice) return;
            previousValue = playOriginalVoice;
            foreach (var entry in sources)
                if (entry.Key) entry.Key.mute = !playOriginalVoice || entry.Value;
        }

        private void Unsubscribe()
        {
            if (subscribed && manager) manager.OnPeerConnection.RemoveListener(OnPeerConnection);
            subscribed = false;
            foreach (var entry in sources)
                if (entry.Key) entry.Key.mute = entry.Value;
            sources.Clear();
        }

        private void OnDisable() => Unsubscribe();
        private void OnDestroy() => Unsubscribe();
    }
}
