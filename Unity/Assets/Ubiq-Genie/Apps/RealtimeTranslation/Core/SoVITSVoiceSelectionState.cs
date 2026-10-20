using System;
using System.Collections.Generic;

namespace Ubiq.Genie.Samples.RealtimeTranslation
{
    [Serializable]
    public sealed class SoVITSVoiceOption
    {
        public string id;
        public string name;
        public string language;
    }

    [Serializable]
    public sealed class SoVITSVoiceCatalog
    {
        public int version;
        public SoVITSVoiceOption[] voices;
    }

    [Serializable]
    public sealed class SoVITSVoiceRequest
    {
        public int version = 1;
        public string requestId;
        public string voiceId;
        public string backendUuid;
    }

    [Serializable]
    public sealed class SoVITSVoiceBinding
    {
        public string peerUuid;
        public string requestId;
        public string voiceId;
        public string status;
        public string message;
    }

    [Serializable]
    public sealed class SoVITSVoiceBindings
    {
        public int version;
        public SoVITSVoiceBinding[] bindings;
    }

    /// <summary>Selection and acknowledgement state, independent of Unity and networking.</summary>
    public sealed class SoVITSVoiceSelectionState
    {
        public const string CatalogProperty = "ubiq.genie.sovits.catalog";
        public const string SelectionProperty = "ubiq.genie.sovits.selection";
        public const string BindingsProperty = "ubiq.genie.sovits.bindings";
        public const int MaximumPropertyBytes = 128 * 1024;

        private SoVITSVoiceOption[] voices = Array.Empty<SoVITSVoiceOption>();
        private bool restoreAttempted;
        private string backendProblem = "Waiting for a translation backend.";

        public string PeerUuid { get; private set; } = "";
        public string RoomUuid { get; private set; } = "";
        public string BackendUuid { get; private set; } = "";
        public string RememberedVoiceId { get; private set; } = "";
        public string SelectedVoiceId { get; private set; } = "";
        public string BoundVoiceId { get; private set; } = "";
        public string RequestId { get; private set; } = "";
        public string RequestedVoiceId { get; private set; } = "";
        public string Error { get; private set; } = "";
        public bool AwaitingConfirmation { get; private set; }
        public IReadOnlyList<SoVITSVoiceOption> Voices => voices;
        public bool CanApply => BackendUuid.Length > 0 && !AwaitingConfirmation && Contains(SelectedVoiceId);
        public bool ShouldRestore => !restoreAttempted && RememberedVoiceId.Length > 0 &&
            SelectedVoiceId == RememberedVoiceId && CanApply;

        public string Status
        {
            get
            {
                if (RoomUuid.Length == 0) return "Join the translation room to choose a voice.";
                if (BackendUuid.Length == 0) return backendProblem;
                if (SelectedVoiceId.Length > 0 && !Contains(SelectedVoiceId))
                    return "The selected or remembered voice is unavailable on this backend. Choose another voice.";
                if (voices.Length == 0) return "The backend has no valid reference voices available.";
                if (SelectedVoiceId.Length == 0) return "Choose a voice and click Apply. No voice is selected automatically on first use.";
                if (AwaitingConfirmation) return "Waiting for the backend to confirm the voice binding.";
                if (Error.Length > 0) return Error;
                if (BoundVoiceId == SelectedVoiceId) return "Voice bound to this session UUID and remembered for this project and room.";
                return "The selected voice has not been applied.";
            }
        }

        public void SetSession(string peerUuid, string roomUuid, string rememberedVoiceId)
        {
            PeerUuid = peerUuid ?? "";
            RoomUuid = roomUuid ?? "";
            RememberedVoiceId = IsVoiceId(rememberedVoiceId) ? rememberedVoiceId : "";
            SelectedVoiceId = RememberedVoiceId;
            BackendUuid = "";
            voices = Array.Empty<SoVITSVoiceOption>();
            restoreAttempted = false;
            ResetRequest(true);
        }

        public void SetBackend(string backendUuid, SoVITSVoiceOption[] options)
        {
            if (BackendUuid != backendUuid)
            {
                ResetRequest(true);
                SelectedVoiceId = RememberedVoiceId;
                restoreAttempted = false;
            }
            BackendUuid = backendUuid;
            voices = options;
            if (!Contains(BoundVoiceId)) BoundVoiceId = "";
            if (SelectedVoiceId.Length > 0 && !Contains(SelectedVoiceId)) AwaitingConfirmation = false;
        }

        public void SetBackendProblem(string message)
        {
            BackendUuid = "";
            backendProblem = message;
            voices = Array.Empty<SoVITSVoiceOption>();
            SelectedVoiceId = RememberedVoiceId;
            restoreAttempted = false;
            ResetRequest(true);
        }

        public bool Choose(string voiceId)
        {
            if (AwaitingConfirmation || (voiceId.Length > 0 && !Contains(voiceId))) return false;
            if (SelectedVoiceId == voiceId) return true;
            SelectedVoiceId = voiceId;
            ResetRequest();
            restoreAttempted = true;
            return true;
        }

        public SoVITSVoiceRequest BeginRequest(string requestId)
        {
            if (!CanApply) throw new InvalidOperationException("No available voice or unique backend is selected.");
            if (!Guid.TryParse(requestId, out _)) throw new ArgumentException("requestId must be a UUID.", nameof(requestId));
            ResetRequest();
            RequestId = requestId;
            RequestedVoiceId = SelectedVoiceId;
            AwaitingConfirmation = true;
            restoreAttempted = true;
            return new SoVITSVoiceRequest { requestId = requestId, voiceId = RequestedVoiceId, backendUuid = BackendUuid };
        }

        /// <summary>Returns true only when a matching bound acknowledgement should be persisted.</summary>
        public bool Acknowledge(string backendUuid, SoVITSVoiceBinding binding)
        {
            if (binding == null || BackendUuid.Length == 0 || RequestId.Length == 0 ||
                backendUuid != BackendUuid || binding.peerUuid != PeerUuid ||
                binding.requestId != RequestId || binding.voiceId != RequestedVoiceId) return false;
            if (binding.status != "bound" && binding.status != "error") return false;
            if (binding.status == "bound" && !Contains(binding.voiceId)) return false;

            AwaitingConfirmation = false;
            if (binding.status == "error")
            {
                if (BoundVoiceId == binding.voiceId) BoundVoiceId = "";
                Error = string.IsNullOrEmpty(binding.message) ? "The backend rejected the voice binding." :
                    binding.message.Substring(0, Math.Min(binding.message.Length, 512));
                return false;
            }
            var changed = BoundVoiceId != binding.voiceId || RememberedVoiceId != binding.voiceId || Error.Length > 0;
            BoundVoiceId = binding.voiceId;
            RememberedVoiceId = binding.voiceId;
            Error = "";
            return changed;
        }

        public void ExpireRequest()
        {
            if (!AwaitingConfirmation) return;
            AwaitingConfirmation = false;
            Error = "The backend did not confirm within 15 seconds. Check Node, then click Apply to retry.";
        }

        public bool Contains(string voiceId)
        {
            if (string.IsNullOrEmpty(voiceId)) return false;
            foreach (var voice in voices) if (voice.id == voiceId) return true;
            return false;
        }

        public static bool IsVoiceId(string value)
        {
            if (value == null || value.Length != 64) return false;
            foreach (var c in value) if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
            return true;
        }

        public static bool IsValidCatalog(SoVITSVoiceCatalog catalog)
        {
            if (catalog == null || catalog.version != 1 || catalog.voices == null || catalog.voices.Length > 256)
                return false;
            var ids = new HashSet<string>(StringComparer.Ordinal);
            foreach (var voice in catalog.voices)
            {
                if (voice == null || !IsVoiceId(voice.id) || !ids.Add(voice.id) ||
                    string.IsNullOrWhiteSpace(voice.name) || voice.name.Length > 128 ||
                    (voice.language != "zh" && voice.language != "en")) return false;
            }
            return true;
        }

        private void ResetRequest(bool clearBinding = false)
        {
            RequestId = "";
            RequestedVoiceId = "";
            if (clearBinding) BoundVoiceId = "";
            AwaitingConfirmation = false;
            Error = "";
        }
    }
}
