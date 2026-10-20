using System;
using System.Security.Cryptography;
using System.Text;
using Ubiq.Messaging;
using Ubiq.Rooms;
using UnityEngine;

namespace Ubiq.Genie.Samples.RealtimeTranslation
{
    /// <summary>Selects a backend-local voice through the existing Ubiq peer property protocol.</summary>
    [DisallowMultipleComponent]
    public sealed class SoVITSVoiceSelector : MonoBehaviour
    {
        [SerializeField, Tooltip("RoomClient for this RealtimeTranslation scene.")]
        private RoomClient roomClient;
        [SerializeField, Tooltip("Restore only a previously confirmed voice for this project and room.")]
        private bool restorePreviousVoice = true;

        private readonly SoVITSVoiceSelectionState state = new SoVITSVoiceSelectionState();
        private bool started;
        private bool subscribed;
        private string preferenceKey = "";
        private string protocolWarning = "";
        private float requestStartedAt;

        public SoVITSVoiceSelectionState State => state;
        public string Status => protocolWarning.Length > 0 ? protocolWarning : state.Status;

        private void Start()
        {
            started = true;
            Subscribe();
        }

        private void OnEnable()
        {
            if (started) Subscribe();
        }

        private void Subscribe()
        {
            if (subscribed) return;
            if (!roomClient)
            {
                var scene = NetworkScene.Find(this);
                if (scene) roomClient = scene.GetComponent<RoomClient>();
            }
            if (!roomClient)
            {
                protocolWarning = "Assign the scene's RoomClient before choosing a voice.";
                return;
            }
            subscribed = true;
            roomClient.OnJoinedRoom.AddListener(OnJoinedRoom);
            roomClient.OnPeerAdded.AddListener(OnPeerChanged);
            roomClient.OnPeerUpdated.AddListener(OnPeerChanged);
            roomClient.OnPeerRemoved.AddListener(OnPeerChanged);
            Refresh();
        }

        private void OnDisable()
        {
            ClearPublishedRequest();
            if (subscribed && roomClient)
            {
                roomClient.OnJoinedRoom.RemoveListener(OnJoinedRoom);
                roomClient.OnPeerAdded.RemoveListener(OnPeerChanged);
                roomClient.OnPeerUpdated.RemoveListener(OnPeerChanged);
                roomClient.OnPeerRemoved.RemoveListener(OnPeerChanged);
            }
            subscribed = false;
            preferenceKey = "";
            state.SetSession("", "", "");
        }

        private void OnJoinedRoom(IRoom room) => Refresh();
        private void OnPeerChanged(IPeer peer) => Refresh();

        private void Update()
        {
            if (state.AwaitingConfirmation && Time.realtimeSinceStartup - requestStartedAt >= 15)
                state.ExpireRequest();
        }

        public void Refresh()
        {
            if (!Application.isPlaying || !roomClient || roomClient.Me == null) return;
            protocolWarning = "";
            var roomUuid = roomClient.JoinedRoom ? roomClient.Room.UUID : "";
            if (state.PeerUuid != roomClient.Me.uuid || state.RoomUuid != roomUuid || preferenceKey.Length == 0)
            {
                preferenceKey = roomUuid.Length > 0 ? GetPreferenceKey(roomUuid) : "";
                var remembered = preferenceKey.Length > 0 ? PlayerPrefs.GetString(preferenceKey, "") : "";
                state.SetSession(roomClient.Me.uuid, roomUuid, remembered);
                ClearPublishedRequest();
            }
            if (roomUuid.Length == 0) return;

            IPeer backend = null;
            SoVITSVoiceCatalog catalog = null;
            var count = 0;
            var invalid = false;
            foreach (var peer in roomClient.Peers)
            {
                var raw = peer[SoVITSVoiceSelectionState.CatalogProperty];
                if (string.IsNullOrEmpty(raw)) continue;
                var candidate = ReadProperty<SoVITSVoiceCatalog>(raw);
                if (!SoVITSVoiceSelectionState.IsValidCatalog(candidate))
                {
                    invalid = true;
                    continue;
                }
                ++count;
                backend = peer;
                catalog = candidate;
            }
            if (count != 1)
            {
                state.SetBackendProblem(count > 1 ?
                    "Multiple translation backends are in this room. Leave only one before choosing a voice." :
                    invalid ? "The backend advertised an invalid voice catalog." :
                    "No translation backend is available in this room. Start the Node app.");
                return;
            }
            state.SetBackend(backend.uuid, catalog.voices);
            ReadAcknowledgement(backend);
            if (restorePreviousVoice && state.ShouldRestore) ApplySelectedVoice();
        }

        public void ChooseVoice(string voiceId)
        {
            state.Choose(voiceId);
        }

        public void ApplySelectedVoice()
        {
            if (!Application.isPlaying || !isActiveAndEnabled || !roomClient || !roomClient.JoinedRoom ||
                roomClient.Me == null || state.PeerUuid != roomClient.Me.uuid || !state.CanApply) return;
            var request = state.BeginRequest(Guid.NewGuid().ToString("D"));
            requestStartedAt = Time.realtimeSinceStartup;
            protocolWarning = "";
            roomClient.Me[SoVITSVoiceSelectionState.SelectionProperty] = JsonUtility.ToJson(request);
        }

        private void ReadAcknowledgement(IPeer backend)
        {
            var raw = backend[SoVITSVoiceSelectionState.BindingsProperty];
            if (string.IsNullOrEmpty(raw)) return;
            var reply = ReadProperty<SoVITSVoiceBindings>(raw);
            if (reply == null || reply.version != 1 || reply.bindings == null || reply.bindings.Length > 256)
            {
                protocolWarning = "The backend advertised an invalid binding response.";
                return;
            }
            foreach (var binding in reply.bindings)
            {
                if (state.Acknowledge(backend.uuid, binding) && preferenceKey.Length > 0)
                {
                    PlayerPrefs.SetString(preferenceKey, state.RememberedVoiceId);
                    PlayerPrefs.Save();
                }
            }
        }

        private void ClearPublishedRequest()
        {
            if (Application.isPlaying && roomClient && roomClient.Me != null)
                roomClient.Me[SoVITSVoiceSelectionState.SelectionProperty] = "";
        }

        private static T ReadProperty<T>(string value) where T : class
        {
            if (value.Length > SoVITSVoiceSelectionState.MaximumPropertyBytes ||
                Encoding.UTF8.GetByteCount(value) > SoVITSVoiceSelectionState.MaximumPropertyBytes) return null;
            try { return JsonUtility.FromJson<T>(value); }
            catch (ArgumentException) { return null; }
        }

        private static string GetPreferenceKey(string roomUuid)
        {
            using (var hash = SHA256.Create())
            {
                var bytes = hash.ComputeHash(Encoding.UTF8.GetBytes(Application.dataPath + "\n" + roomUuid));
                return "ubiq.genie.sovits.voice." + BitConverter.ToString(bytes).Replace("-", "").ToLowerInvariant();
            }
        }
    }
}
