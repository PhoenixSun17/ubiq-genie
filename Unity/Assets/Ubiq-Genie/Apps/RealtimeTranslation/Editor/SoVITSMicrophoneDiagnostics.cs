using System;
using System.Collections.Generic;
using System.Linq;
using Ubiq.Messaging;
using Ubiq.Rooms;
using Ubiq.Voip;
using UnityEditor;
using UnityEngine;

/// <summary>Reads an existing Ubiq capture session; never starts recording or changes audio settings.</summary>
internal static class SoVITSMicrophoneDiagnostics
{
    private sealed class Capture
    {
        public MonoBehaviour microphone;
        public AudioSource source;
        public float[] buffer;
        public int snapshots, failures, initialTimeSamples;
        public long samples;
        public double squares, peak;
        public string error;
    }

    private sealed class PeerStats
    {
        public VoipPeerConnection peer;
        public string uuid;
        public Action<VoipPeerConnection.AudioStats> handler;
        public long callbacks, samples;
        public double absoluteSum;
    }

    private static readonly object gate = new object();
    private static readonly List<Capture> captures = new List<Capture>();
    private static readonly List<PeerStats> peers = new List<PeerStats>();
    private static bool running;
    private static double started, nextSample, initialDspTime;
    private static int initialPosition;

    [MenuItem("Tools/SoVITS/Log Microphone Diagnostics (5 Seconds)")]
    private static void Start()
    {
        if (running) { Log("A five-second diagnostic is already running."); return; }
        if (!EditorApplication.isPlaying) { Log("Enter Play mode before running this diagnostic."); return; }
        var receivers = UnityEngine.Object.FindObjectsByType<SoVITSReceiver>(FindObjectsInactive.Include, FindObjectsSortMode.None);
        if (receivers.Length != 1) { Log($"Expected one SoVITSReceiver; found {receivers.Length}. No capture inspected."); return; }
        var scene = NetworkScene.Find(receivers[0]);
        if (!scene) { Log("The SoVITSReceiver has no NetworkScene."); return; }
        try
        {
            running = true;
            started = nextSample = EditorApplication.timeSinceStartup;
            initialDspTime = AudioSettings.dspTime;
            initialPosition = Microphone.GetPosition("");
            var room = scene.GetComponent<RoomClient>();
            Log($"Started in scene '{scene.gameObject.scene.name}': local UUID={room?.Me?.uuid ?? "unavailable"}, room={(room && room.JoinedRoom ? room.Room.UUID : "not joined")}.");
            Log("Reading existing capture for five seconds. Raw levels use overlapping whole-clip snapshots, not a continuous recording; no audio is saved.");
            LogAudioSettings();
            LogDevices();
            foreach (var component in scene.GetComponentsInChildren<MonoBehaviour>(true))
            {
                if (!component || component.GetType().FullName != "Ubiq.Voip.Implementations.Unity.PeerConnectionMicrophone") continue;
                var source = component.GetComponent<AudioSource>();
                var capture = new Capture { microphone = component, source = source, initialTimeSamples = source ? source.timeSamples : -1 };
                captures.Add(capture);
                LogCapture(capture, "Initial");
                Log($"Components on '{component.name}' in order: {string.Join(" -> ", component.GetComponents<Component>().Select(item => item ? item.GetType().FullName : "Missing Script"))}.");
            }
            foreach (var peer in scene.GetComponentsInChildren<VoipPeerConnection>(true))
            {
                var stats = new PeerStats { peer = peer, uuid = peer.PeerUuid };
                stats.handler = value => { lock (gate) { stats.callbacks++; stats.samples += value.sampleCount; stats.absoluteSum += value.volumeSum; } };
                peer.recordStatsPushed += stats.handler;
                peers.Add(stats);
                LogPeer(stats, "Initial");
            }
            Log($"Found {captures.Count} existing microphone component(s), {peers.Count} VoIP peer connection(s). No microphone is created when none exists.");
            EditorApplication.update += Update;
            EditorApplication.playModeStateChanged += OnPlayModeChanged;
            AssemblyReloadEvents.beforeAssemblyReload += OnBeforeReload;
        }
        catch (Exception error) { Log($"Cannot start diagnostics: {error.Message}"); Cleanup(); }
    }

    private static void Update()
    {
        if (!running) return;
        if (!EditorApplication.isPlaying) { Finish("Play mode ended"); return; }
        if (EditorApplication.timeSinceStartup - started >= 5) { Finish("Five-second observation complete"); return; }
        if (EditorApplication.timeSinceStartup < nextSample) return;
        nextSample = EditorApplication.timeSinceStartup + 0.2;
        foreach (var capture in captures)
        {
            try
            {
                var clip = capture.source ? capture.source.clip : null;
                if (!clip) throw new InvalidOperationException("No existing microphone AudioClip.");
                if (clip.samples <= 0 || clip.samples > 192000 || clip.channels < 1 || clip.channels > 2)
                    throw new InvalidOperationException("Clip exceeds the diagnostic limit of 192000 frames and two channels.");
                var count = clip.samples * clip.channels;
                if (capture.buffer == null || capture.buffer.Length != count)
                {
                    if (capture.buffer != null) Array.Clear(capture.buffer, 0, capture.buffer.Length);
                    capture.buffer = new float[count];
                }
                if (!clip.GetData(capture.buffer, 0)) throw new InvalidOperationException("AudioClip.GetData returned false; this is not a silence measurement.");
                foreach (var sample in capture.buffer) { capture.peak = Math.Max(capture.peak, Math.Abs(sample)); capture.squares += (double)sample * sample; }
                capture.samples += count;
                capture.snapshots++;
            }
            catch (Exception error) { capture.failures++; capture.error = error.Message; }
        }
    }

    private static void Finish(string reason)
    {
        if (!running) return;
        try
        {
            Log($"{reason}; elapsed={EditorApplication.timeSinceStartup - started:F2}s, DSP progression={AudioSettings.dspTime - initialDspTime:F3}s, default microphone position={initialPosition}->{Microphone.GetPosition("")} (ring positions wrap). This does not identify the default device by name.");
            LogAudioSettings();
            foreach (var capture in captures)
            {
                var rms = capture.samples > 0 ? Math.Sqrt(capture.squares / capture.samples) : 0;
                Log($"Raw clip snapshots: successful={capture.snapshots}, failed={capture.failures}, peakDbFS={(capture.samples > 0 ? Db(capture.peak) : "unavailable")}, rmsDbFS={(capture.samples > 0 ? Db(rms) : "unavailable")}, lastError={capture.error ?? "none"}. Overlapping snapshots include older ring-buffer samples.");
                LogCapture(capture, "Final");
            }
            foreach (var peer in peers) LogPeer(peer, "Final");
        }
        catch (Exception error) { Log($"Diagnostic reporting interrupted: {error.Message}"); }
        finally { Cleanup(); }
    }

    private static void LogCapture(Capture capture, string phase)
    {
        var source = capture.source;
        if (!capture.microphone || !source) { Log($"{phase}: microphone or AudioSource unavailable."); return; }
        var state = capture.microphone.GetType().GetProperty("state")?.GetValue(capture.microphone);
        var clip = source.clip;
        var filters = capture.microphone.GetComponents<Component>().Count(item => item && item.GetType().FullName == "Unity.WebRTC.AudioCustomFilter");
        Log($"{phase} microphone '{capture.microphone.name}': state={state}, active={capture.microphone.isActiveAndEnabled}, sourceEnabled={source.enabled}, playing={source.isPlaying}, virtual={source.isVirtual}, mute={source.mute}, volume={source.volume}, timeSamples={capture.initialTimeSamples}->{source.timeSamples}, customFilters={filters}, clip={(clip ? $"{clip.frequency}Hz/{clip.channels}ch/{clip.samples}frames" : "none")}.");
        if (filters > 1) Log("Multiple WebRTC AudioCustomFilters share this microphone object. Earlier sender filters clear their output; inspect component order before changing anything.");
    }

    private static void LogPeer(PeerStats stats, string phase)
    {
        lock (gate)
        {
            var mean = stats.samples > 0 ? stats.absoluteSum / stats.samples : 0;
            Log($"{phase} remote peer {stats.uuid}: connection={(stats.peer ? stats.peer.peerConnectionState.ToString() : "destroyed")}, ICE={(stats.peer ? stats.peer.iceConnectionState.ToString() : "destroyed")}, recordCallbacks={stats.callbacks}, recordSamples={stats.samples}, UbiqMeanAbsoluteDbFS={(stats.samples > 0 ? Db(mean) : "unavailable")} (not RMS). These callbacks report Ubiq capture statistics; inspect the logged component order relative to WebRTC sender filters.");
        }
    }

    private static void LogAudioSettings() => Log($"EditorMute={EditorUtility.audioMasterMute}, EditorPaused={EditorApplication.isPaused}, listenerPause={AudioListener.pause}, listenerVolume={AudioListener.volume}, outputRate={AudioSettings.outputSampleRate}, DSP={AudioSettings.dspTime:F3}.");

    private static void LogDevices()
    {
        Log($"Default microphone: recording={Microphone.IsRecording("")}, position={Microphone.GetPosition("")}. Default-device identity is unspecified by this API.");
        foreach (var device in Microphone.devices)
            Log($"Device '{device}': recording={Microphone.IsRecording(device)}, position={Microphone.GetPosition(device)}.");
    }

    private static void OnBeforeReload() => Finish("Assembly reload interrupted observation");
    private static void OnPlayModeChanged(PlayModeStateChange state)
    {
        if (state == PlayModeStateChange.ExitingPlayMode || state == PlayModeStateChange.EnteredEditMode) Finish("Play mode exit interrupted observation");
    }

    private static void Cleanup()
    {
        running = false;
        EditorApplication.update -= Update;
        EditorApplication.playModeStateChanged -= OnPlayModeChanged;
        AssemblyReloadEvents.beforeAssemblyReload -= OnBeforeReload;
        foreach (var stats in peers) if (stats.peer) stats.peer.recordStatsPushed -= stats.handler;
        peers.Clear();
        foreach (var capture in captures) if (capture.buffer != null) Array.Clear(capture.buffer, 0, capture.buffer.Length);
        captures.Clear();
    }

    private static string Db(double amplitude) => amplitude > 0 ? (20 * Math.Log10(amplitude)).ToString("F1") : "-inf";
    private static void Log(string message) => Debug.Log($"[SoVITS Diagnostic] {message}");
}
