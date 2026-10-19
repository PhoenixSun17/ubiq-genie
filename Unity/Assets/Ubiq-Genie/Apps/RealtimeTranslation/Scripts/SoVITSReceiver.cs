using Ubiq.Messaging;
using Ubiq.Rooms;
using UnityEngine;

/// <summary>Ubiq receiver for the existing Node/Python SoVITS translation pipeline.</summary>
[DisallowMultipleComponent]
public sealed class SoVITSReceiver : MonoBehaviour
{
    private readonly NetworkId networkId = new NetworkId(95);
    private NetworkContext context;
    private RoomClient roomClient;
    private bool started;
    private bool registered;
    private bool receiving;
    private float nextWarningTime;
    private int suppressedWarnings;

    public SoVITSAudioSource audioSource;
    public VirtualAssistantController assistantController;
    public AudioSourceVolume volume;

    private void Start()
    {
        started = true;
        RegisterReceiver();
    }

    private void OnEnable()
    {
        if (started) RegisterReceiver();
    }

    private void RegisterReceiver()
    {
        if (registered) return;
        if (!audioSource || !audioSource.IsReady)
        {
            Debug.LogError("[SoVITS] Assign an enabled SoVITSAudioSource on the copied assistant Speech object.", this);
            enabled = false;
            return;
        }
        var scene = NetworkScene.Find(this);
        if (!scene)
        {
            Debug.LogError("[SoVITS] No Ubiq NetworkScene found.", this);
            enabled = false;
            return;
        }
        foreach (var processor in scene.GetProcessors())
        {
            if (processor.Value.Equals(networkId) && !ReferenceEquals(processor.Key.Target, this))
            {
                Debug.LogError("[SoVITS] Another receiver already owns NetworkId 95; disable the duplicate to avoid repeated playback.", this);
                enabled = false;
                return;
            }
        }
        if (context.Scene == scene)
        {
            // Re-enable without adding another entry to Ubiq's autoProcessors.
            scene.AddProcessor(networkId, ProcessMessage);
        }
        else
        {
            context = NetworkScene.Register(this, networkId);
        }
        registered = true;
        roomClient = scene.GetComponent<RoomClient>();
        if (roomClient) roomClient.OnJoinedRoom.AddListener(OnJoinedRoom);
        Debug.Log("[SoVITS] Receiver ready on NetworkId 95. Source audio comes through the Node app.", this);
    }

    public void ProcessMessage(ReferenceCountedSceneGraphMessage message)
    {
        if (!isActiveAndEnabled || !registered || !audioSource || !audioSource.IsReady) return;
        // Ubiq may provide a slice of a larger pooled byte array.
        var accepted = audioSource.ReceiveMessage(message.bytes, message.start, message.length,
            out var warning, out var resetPlayback);
        if (resetPlayback) receiving = false;
        if (accepted) receiving = true;
        if (!string.IsNullOrEmpty(warning)) Warn(warning);
    }

    private void Update()
    {
        if (assistantController)
        {
            var level = receiving && audioSource && audioSource.IsReady && volume ? Mathf.Max(0, volume.volume) : 0;
            // The backend's targetPeer is empty: broadcast speech uses the
            // existing assistant's usual orientation, rather than a UUID lookup.
            assistantController.UpdateAssistantSpeechStatus(null, level);
        }
    }

    private void Warn(string message)
    {
        if (Time.unscaledTime < nextWarningTime)
        {
            ++suppressedWarnings;
            return;
        }
        var suffix = suppressedWarnings > 0 ? $" ({suppressedWarnings} additional audio warnings suppressed)" : string.Empty;
        Debug.LogWarning($"[SoVITS] {message}{suffix}", this);
        suppressedWarnings = 0;
        nextWarningTime = Time.unscaledTime + 1.0f;
    }

    private void OnJoinedRoom(IRoom room)
    {
        // Ubiq also uses OnJoinedRoom for leaving into an empty room.
        ClearPlayback();
    }

    private void ClearPlayback()
    {
        receiving = false;
        if (audioSource) audioSource.ClearPlayback();
        if (assistantController) assistantController.UpdateAssistantSpeechStatus(null, 0);
    }

    private void OnDisable()
    {
        if (registered && context.Scene) context.Scene.RemoveProcessor(networkId, ProcessMessage);
        registered = false;
        if (roomClient) roomClient.OnJoinedRoom.RemoveListener(OnJoinedRoom);
        roomClient = null;
        ClearPlayback();
    }

    private void OnDestroy()
    {
        if (registered && context.Scene) context.Scene.RemoveProcessor(networkId, ProcessMessage);
        if (roomClient) roomClient.OnJoinedRoom.RemoveListener(OnJoinedRoom);
    }
}
