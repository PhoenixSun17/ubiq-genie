using System;
using System.Collections.Generic;
using System.Text;
using Ubiq.Avatars;
using Ubiq.Messaging;
using Ubiq.Rooms;
using Ubiq.Voip;
using UnityEngine;

namespace Ubiq.Genie.Samples.RealtimeTranslationMotion
{
    /// <summary>Associates translation PCM with remote presentation without writing local tracking state.</summary>
    [DisallowMultipleComponent]
    [DefaultExecutionOrder(11000)]
    [RequireComponent(typeof(RemoteAvatarReplay), typeof(MotionAudioPlayer), typeof(OriginalVoiceToggle))]
    public sealed class MotionTranslationReceiver : MonoBehaviour
    {
        private static readonly NetworkId AudioId = new NetworkId(95);
        private static readonly NetworkId MetadataId = new NetworkId(101);
        private static readonly UTF8Encoding StrictUtf8 = new UTF8Encoding(false, true);

        [SerializeField] private RoomClient roomClient;
        private NetworkScene networkScene;
        private RemoteAvatarReplay replay;
        private MotionAudioPlayer player;
        private OriginalVoiceToggle originalVoice;
        private readonly MotionClock clock = new MotionClock();
        private readonly Dictionary<string, string> speakers = new Dictionary<string, string>();
        private readonly Dictionary<string, double> pings = new Dictionary<string, double>();
        private readonly List<string> scratch = new List<string>();
        private MotionAudioIdentity pendingPlay;
        private string backendUuid;
        private string sessionId;
        private string currentPlayback;
        private bool started;
        private bool registered;
        private int initialPings;
        private double nextPing;
        private double nextWarning;
        private string status = "Waiting for a translation backend.";

        public string Status => status;

        private void Start()
        {
            started = true;
            Register();
        }

        private void OnEnable()
        {
            if (started) Register();
        }

        private void Register()
        {
            if (registered) return;
            networkScene = NetworkScene.Find(this);
            if (!roomClient && networkScene) roomClient = networkScene.GetComponent<RoomClient>();
            if (!networkScene || !roomClient)
            {
                Debug.LogError("[Translation Motion] A NetworkScene and RoomClient are required.", this);
                enabled = false;
                return;
            }
            foreach (var processor in networkScene.GetProcessors())
            {
                if (processor.Value.Equals(AudioId) || processor.Value.Equals(MetadataId))
                {
                    Debug.LogError("[Translation Motion] Another receiver owns channel 95 or 101.", this);
                    enabled = false;
                    return;
                }
            }
            replay = GetComponent<RemoteAvatarReplay>();
            player = GetComponent<MotionAudioPlayer>();
            originalVoice = GetComponent<OriginalVoiceToggle>();
            replay.Initialize(networkScene.GetComponentInChildren<AvatarManager>(), roomClient);
            originalVoice.Initialize(networkScene.GetComponentInChildren<VoipPeerConnectionManager>());
            player.HeadPose = HeadPose;
            player.IsLocalSpeaker = IsLocal;
            player.PlaybackFinished += OnFinished;
            player.PlaybackAborted += OnAborted;
            networkScene.AddProcessor(AudioId, ReceiveAudio);
            networkScene.AddProcessor(MetadataId, ReceiveMetadata);
            networkScene.OnConnectionsModified.AddListener(OnConnectionsModified);
            roomClient.OnJoinedRoom.AddListener(OnJoinedRoom);
            roomClient.OnPeerAdded.AddListener(OnPeerChanged);
            roomClient.OnPeerUpdated.AddListener(OnPeerChanged);
            roomClient.OnPeerRemoved.AddListener(OnPeerRemoved);
            AudioSettings.OnAudioConfigurationChanged += OnAudioConfigurationChanged;
            registered = true;
            RefreshBackend();
        }

        private bool IsLocal(string uuid) => roomClient && roomClient.Me != null && uuid == roomClient.Me.uuid;
        private Pose? HeadPose(string uuid) => replay && replay.TryGetHead(uuid, out var pose) ? pose : (Pose?)null;

        private void Update()
        {
            if (!registered) return;
            double now = Time.realtimeSinceStartupAsDouble;
            if (!string.IsNullOrEmpty(backendUuid) && now >= nextPing) SendPing(now);
            if (player.TryGetPlayback(out var position))
            {
                if (currentPlayback != null && currentPlayback != position.RequestId)
                    replay.Finish(currentPlayback);
                currentPlayback = position.RequestId;
                if (!IsLocal(position.SpeakerUuid))
                    replay.Present(position.RequestId, position.SpeakerUuid, position.Progress);
            }
            else if (currentPlayback != null)
            {
                replay.Finish(currentPlayback);
                currentPlayback = null;
            }
        }

        private void ReceiveAudio(ReferenceCountedSceneGraphMessage message)
        {
            if (!registered || !isActiveAndEnabled) return;
            // The backend writes play, AudioInfo, then PCM to one ordered TCP connection.
            // Consume the association on the VERY NEXT audio message, including a malformed one.
            var identity = pendingPlay;
            pendingPlay = null;
            if (identity != null && message.length >= 100)
            {
                OnAborted(identity.RequestId, "Play metadata was not followed by AudioInfo.");
                identity = null;
            }
            if (!player.Receive(message.bytes, message.start, message.length, identity, out var warning) &&
                identity != null) OnAborted(identity.RequestId, warning ?? "Audio header rejected.");
            if (!string.IsNullOrEmpty(warning)) Warn(warning);
        }

        private void LateUpdate()
        {
            // RemoteAvatarReplay applies at order 10000; follow that frame's displayed head.
            if (registered) player.RefreshSpatialPositions();
        }

        private void ReceiveMetadata(ReferenceCountedSceneGraphMessage message)
        {
            if (!registered || !isActiveAndEnabled || message.length <= 0 || message.length > 4096) return;
            MotionMetadata metadata;
            try
            {
                metadata = JsonUtility.FromJson<MotionMetadata>(
                    StrictUtf8.GetString(message.bytes, message.start, message.length));
            }
            catch (Exception error) when (error is ArgumentException || error is DecoderFallbackException)
            {
                Warn("Invalid motion metadata.");
                return;
            }
            if (metadata == null || !metadata.Matches(backendUuid, sessionId)) return;
            switch (metadata.type)
            {
                case "pong":
                    if (IsLocal(metadata.peerUuid) && metadata.nonce != null &&
                        pings.TryGetValue(metadata.nonce, out var t0) && t0 == metadata.t0)
                    {
                        pings.Remove(metadata.nonce);
                        clock.Observe(t0, metadata.s1, metadata.s2, Time.realtimeSinceStartupAsDouble * 1000);
                    }
                    break;
                case "accepted":
                    Accept(metadata);
                    break;
                case "play":
                    PreparePlayback(metadata);
                    break;
                case "terminal":
                    if (!metadata.ValidUtterance || !speakers.TryGetValue(metadata.requestId, out var speaker) ||
                        speaker != metadata.speakerUuid) return;
                    if (metadata.status == "sent")
                    {
                        // Anonymous fallback or a missing header cannot produce a completion for this ID.
                        // Keep only requests actually queued/scheduled locally (including their audible tail).
                        if ((pendingPlay == null || pendingPlay.RequestId != metadata.requestId) &&
                            !player.ContainsRequest(metadata.requestId)) OnFinished(metadata.requestId);
                        return;
                    }
                    if (metadata.status == "cancelled" || metadata.status == "skipped" || metadata.status == "failed")
                    {
                        player.Cancel(metadata.requestId);
                        OnFinished(metadata.requestId);
                    }
                    break;
            }
        }

        private void Accept(MotionMetadata metadata)
        {
            if (!metadata.ValidInterval || speakers.ContainsKey(metadata.requestId)) return;
            if (speakers.Count >= 64)
            {
                Warn("Pending metadata limit reached; new translation will use audio only.");
                return;
            }
            speakers.Add(metadata.requestId, metadata.speakerUuid);
            if (IsLocal(metadata.speakerUuid)) return;
            double now = Time.realtimeSinceStartupAsDouble * 1000;
            if (!clock.TryMap(metadata.sourceStartMs, now, out var start) ||
                !clock.TryMap(metadata.sourceEndMs, now, out var end))
            {
                Warn($"Audio only for {metadata.requestId}: clock synchronization is unavailable.");
                return;
            }
            if (!replay.Pin(metadata.requestId, metadata.speakerUuid, start, end, out var reason))
                Warn($"Audio only for {metadata.requestId}: {reason}");
        }

        private void PreparePlayback(MotionMetadata metadata)
        {
            if (pendingPlay != null)
            {
                OnAborted(pendingPlay.RequestId, "A new play event replaced an unmatched audio header.");
                pendingPlay = null;
            }
            if (!metadata.ValidAudio || metadata.sampleRate != player.sourceSampleRate)
            {
                if (MotionMetadata.ValidId(metadata.requestId)) OnFinished(metadata.requestId);
                Warn("Invalid translation audio metadata; falling back to unassociated audio.");
                return;
            }
            if (speakers.TryGetValue(metadata.requestId, out var speaker) && speaker != metadata.speakerUuid)
            {
                OnFinished(metadata.requestId);
                Warn("Translation speaker changed within an utterance; ignoring the association.");
                return;
            }
            if (speakers.Count < 64) speakers[metadata.requestId] = metadata.speakerUuid;
            pendingPlay = new MotionAudioIdentity
            {
                RequestId = metadata.requestId,
                SpeakerUuid = metadata.speakerUuid,
                SampleCount = metadata.sampleCount,
                SampleRate = metadata.sampleRate
            };
        }

        private void OnFinished(string requestId)
        {
            if (string.IsNullOrEmpty(requestId)) return;
            replay.Finish(requestId);
            speakers.Remove(requestId);
            if (pendingPlay != null && pendingPlay.RequestId == requestId) pendingPlay = null;
            if (currentPlayback == requestId) currentPlayback = null;
        }

        private void OnAborted(string requestId, string reason)
        {
            OnFinished(requestId);
            Warn($"Playback stopped for {requestId}: {reason}");
        }

        private void SendPing(double now)
        {
            nextPing = now + (++initialPings < 3 ? 0.1 : 5);
            if (!roomClient.JoinedRoom || roomClient.Me == null || networkScene.connectionCount == 0) return;
            scratch.Clear();
            foreach (var entry in pings)
                if (now * 1000 - entry.Value > 10000) scratch.Add(entry.Key);
            foreach (var key in scratch) pings.Remove(key);
            if (pings.Count >= 8) return;
            string nonce = Guid.NewGuid().ToString("N");
            pings.Add(nonce, now * 1000);
            networkScene.SendJson(MetadataId, new MotionMetadata
            {
                version = 1, type = "ping", backendUuid = backendUuid, sessionId = sessionId,
                peerUuid = roomClient.Me.uuid, nonce = nonce, t0 = now * 1000
            });
        }

        private void RefreshBackend()
        {
            string foundUuid = null;
            string foundSession = null;
            int count = 0;
            if (roomClient.JoinedRoom)
            {
                foreach (var peer in roomClient.Peers)
                {
                    string raw = peer[MotionMetadata.BackendProperty];
                    if (string.IsNullOrEmpty(raw) || raw.Length > 1024) continue;
                    MotionMetadata property;
                    try { property = JsonUtility.FromJson<MotionMetadata>(raw); }
                    catch (ArgumentException) { continue; }
                    if (property == null || property.version != 1 || !MotionMetadata.ValidId(property.sessionId)) continue;
                    ++count;
                    foundUuid = peer.uuid;
                    foundSession = property.sessionId;
                }
            }
            if (count != 1) foundUuid = foundSession = null;
            if (backendUuid != foundUuid || sessionId != foundSession)
            {
                ResetState();
                backendUuid = foundUuid;
                sessionId = foundSession;
            }
            status = count == 1 ? "Remote replay enabled; sentence timing is approximate." :
                count > 1 ? "Multiple motion backends: audio only." : "No motion backend: audio only.";
        }

        private void OnPeerChanged(IPeer peer) => RefreshBackend();

        private void OnPeerRemoved(IPeer peer)
        {
            scratch.Clear();
            foreach (var entry in speakers)
                if (entry.Value == peer.uuid) scratch.Add(entry.Key);
            // Callbacks remove from speakers, not from this stable copy.
            foreach (var request in scratch)
            {
                player.Cancel(request);
                OnFinished(request);
            }
            replay.CancelSpeaker(peer.uuid);
            RefreshBackend();
        }

        private void OnJoinedRoom(IRoom room)
        {
            ResetState();
            backendUuid = sessionId = null;
            RefreshBackend();
        }

        private void OnConnectionsModified()
        {
            if (networkScene.connectionCount != 0) return;
            ResetState();
            backendUuid = sessionId = null;
            status = "Disconnected; playback and movement cleared.";
        }

        private void OnAudioConfigurationChanged(bool changed)
        {
            player.ClearPlayback();
            replay.ResetReplay();
            speakers.Clear();
            pendingPlay = null;
            currentPlayback = null;
        }

        private void ResetState()
        {
            if (player) player.ClearPlayback();
            if (replay) replay.ResetReplay();
            speakers.Clear();
            pings.Clear();
            clock.Clear();
            pendingPlay = null;
            currentPlayback = null;
            initialPings = 0;
            nextPing = 0;
        }

        private void Warn(string warning)
        {
            double now = Time.realtimeSinceStartupAsDouble;
            if (now < nextWarning) return;
            nextWarning = now + 1;
            Debug.LogWarning("[Translation Motion] " + warning, this);
        }

        private void OnDisable()
        {
            if (!registered) return;
            registered = false;
            networkScene.RemoveProcessor(AudioId, ReceiveAudio);
            networkScene.RemoveProcessor(MetadataId, ReceiveMetadata);
            networkScene.OnConnectionsModified.RemoveListener(OnConnectionsModified);
            roomClient.OnJoinedRoom.RemoveListener(OnJoinedRoom);
            roomClient.OnPeerAdded.RemoveListener(OnPeerChanged);
            roomClient.OnPeerUpdated.RemoveListener(OnPeerChanged);
            roomClient.OnPeerRemoved.RemoveListener(OnPeerRemoved);
            AudioSettings.OnAudioConfigurationChanged -= OnAudioConfigurationChanged;
            ResetState();
            replay.Initialize(null, null);
            originalVoice.Initialize(null);
            player.PlaybackFinished -= OnFinished;
            player.PlaybackAborted -= OnAborted;
            player.HeadPose = null;
            player.IsLocalSpeaker = null;
            backendUuid = sessionId = null;
        }
    }
}
