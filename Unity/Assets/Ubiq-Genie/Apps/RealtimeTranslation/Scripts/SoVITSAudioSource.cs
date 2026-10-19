using System;
using UbiqGenie.RealtimeTranslation;
using UnityEngine;

/// <summary>Plays bounded, mono PCM16LE from the SoVITS Node app at its source sample rate.</summary>
[DisallowMultipleComponent]
[RequireComponent(typeof(AudioSource))]
public sealed class SoVITSAudioSource : MonoBehaviour
{
    [Tooltip("Must match translation.outputSampleRate in Node. Re-enable this component after changing buffer/rate settings.")]
    public int sourceSampleRate = 16000;
    [Min(0)] public int prebufferMilliseconds = 200;
    [Min(1)] public int maxBufferMilliseconds = 2000;

    private readonly object audioGate = new object();
    private SoVITSPcmStream stream;
    private AudioSource output;
    private AudioClip clip;
    private int playbackGeneration;
    private bool callbackEnabled;

    public bool IsReady => isActiveAndEnabled && stream != null && clip != null;
    public int BufferedSamples => stream != null ? stream.BufferedSamples : 0;

    private void OnEnable()
    {
        output = GetComponent<AudioSource>();
        try
        {
            stream = new SoVITSPcmStream(sourceSampleRate, prebufferMilliseconds, maxBufferMilliseconds);
            RecreateClip();
            AudioSettings.OnAudioConfigurationChanged += OnAudioConfigurationChanged;
            Debug.Log($"[SoVITS] PCM source {sourceSampleRate} Hz; device output {AudioSettings.outputSampleRate} Hz; buffer limit {maxBufferMilliseconds} ms.", this);
        }
        catch (Exception error)
        {
            Debug.LogError($"[SoVITS] Cannot initialise audio: {error.Message}", this);
            enabled = false;
        }
    }

    public bool ReceiveMessage(byte[] bytes, int start, int length, out string warning, out bool resetPlayback)
    {
        warning = null;
        resetPlayback = false;
        if (!IsReady) return false;
        var result = stream.Receive(bytes, start, length);
        warning = result.Warning;
        resetPlayback = result.ResetPlayback;
        if (resetPlayback) RecreateClip();
        return result.Accepted;
    }

    public void ClearPlayback()
    {
        // Invalidate old callbacks before clearing; Stop/Destroy stay outside
        // the callback lock because those Unity calls can wait on audio work.
        InvalidateCallbacks();
        stream?.Clear();
        if (isActiveAndEnabled && stream != null) RecreateClip();
    }

    private void OnAudioConfigurationChanged(bool deviceWasChanged)
    {
        if (!isActiveAndEnabled) return;
        ClearPlayback();
        Debug.Log($"[SoVITS] Audio device changed; waiting for a new AudioInfo header. Source {stream.SourceSampleRate} Hz, output {AudioSettings.outputSampleRate} Hz.", this);
    }

    private void RecreateClip()
    {
        ReleaseClip();
        var reader = stream;
        int generation;
        lock (audioGate)
        {
            generation = ++playbackGeneration;
            callbackEnabled = true;
        }
        // A source-rate clip lets Unity resample to the real device rate and
        // retains the copied assistant AudioSource's spatialisation settings.
        clip = AudioClip.Create("SoVITS translation", reader.SourceSampleRate, 1, reader.SourceSampleRate, true,
            destination => ReadPcm(destination, reader, generation));
        output.clip = clip;
        output.playOnAwake = false;
        output.loop = true;
        output.pitch = 1.0f;
        output.dopplerLevel = 0.0f;
        output.Play();
    }

    private void ReadPcm(float[] destination, SoVITSPcmStream reader, int generation)
    {
        // This callback runs on Unity's audio thread. Do not call Unity APIs,
        // log, or allocate here; an obsolete clip must not consume a new stream.
        lock (audioGate)
        {
            if (!callbackEnabled || generation != playbackGeneration)
            {
                Array.Clear(destination, 0, destination.Length);
                return;
            }
            reader.Read(destination);
        }
    }

    private void InvalidateCallbacks()
    {
        lock (audioGate)
        {
            callbackEnabled = false;
            ++playbackGeneration;
        }
    }

    private void ReleaseClip()
    {
        InvalidateCallbacks();
        if (output)
        {
            output.Stop();
            output.clip = null;
        }
        if (clip)
        {
            Destroy(clip);
            clip = null;
        }
    }

    private void OnDisable()
    {
        AudioSettings.OnAudioConfigurationChanged -= OnAudioConfigurationChanged;
        ReleaseClip();
        stream?.Clear();
        stream = null;
    }

    private void OnDestroy()
    {
        ReleaseClip();
    }
}
