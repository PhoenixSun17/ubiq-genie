# RealtimeTranslation / GPT-SoVITS Integration

This scene receives translated speech from the existing `gpt_sovits_translation` app. The path is Unity microphone → Ubiq VoIP → Node app → Python ASR / local translation and synthesis services → Node NetworkId 95 → SoVITSReceiver → assistant audio source. Unity does not call the Python HTTP endpoints directly.

Current status: implementation, scene wiring, and static checks are complete. On 2026-10-13, after approval, script compilation completed in Unity 6000.0.61f1 and all 52 EditMode tests passed without loading models (0 failures, 0 skipped, exit code 0). Playback and live two-user integration have not been tested. Reference voices and end-to-end behavior remain **unverified** and require separate approval before runtime validation.

## Scene and Inspector

Open `RealtimeTranslation.unity` in Unity **6000.0.61f1**.

| Object or setting | Wiring |
| --- | --- |
| `Ubiq-Genie/SoVITSReceiver` | The former MessageReceiver object now hosts SoVITSReceiver, registered on NetworkId **95** |
| `SoVITSReceiver.audioSource` | SoVITSAudioSource on the copied assistant's `Avatar/Speech` object |
| `SoVITSReceiver.assistantController` | VirtualAssistantController on the copied assistant |
| `SoVITSReceiver.volume` | AudioSourceVolume on the copied assistant's Speech object |
| `Avatar/Speech` | Retains AudioSource, AudioSourceVolume, and AudioSourceSpeechIndicator; the player is replaced with SoVITSAudioSource |
| `SoVITSAudioSource.sourceSampleRate` | **16000**; must match Node `translation.outputSampleRate` |
| `SoVITSAudioSource.prebufferMilliseconds` | **200** |
| `SoVITSAudioSource.maxBufferMilliseconds` | **2000** |
| RoomClient server configuration | This directory's independent `SoVITSServerConfig.asset` |

The assistant is the complete Avatar instance copied from `ConversationalAgent.unity`. It retains its prefab overrides, controller, volume indicator, HandMover motion data, and references to existing model assets. Its original position is `(0.632, 1.161, 0.95)`, with a Y rotation of 180°. ConversationalAgentManager was not copied; this scene has exactly one receiver on channel 95. Existing Ubiq user avatars and VoIP components continue to handle microphone capture and the original voice call.

## Server, Room, and Backends

Both sides use the following configuration:

| Setting | Value |
| --- | --- |
| Ubiq TCP | `nexus.cs.ucl.ac.uk:8009` |
| Unity WebGL connection override | `nexus.cs.ucl.ac.uk:8010`, retaining the original WebSocket connection type |
| Room GUID | `6765c52b-3ad6-4fb0-9030-2c9a05dc4731` |
| GPT-SoVITS HTTP | `http://127.0.0.1:9880`, called by Node |
| Ollama HTTP | `http://127.0.0.1:11435`, called by Node |

The scene still references the existing Room Guid asset. Its independent ServerConfig is used only by this scene. The Node configuration is `C:\ubiq-genie\Node\apps\gpt_sovits_translation\config.json`.

After approval for integration testing, run the following in PowerShell from `C:\ubiq-genie\Node`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\apps\gpt_sovits_translation\scripts\start-backends.ps1
npm.cmd start gpt_sovits_translation
```

Using `npm.cmd` avoids PowerShell invoking an `npm.ps1` blocked by execution policy, without changing the system policy. The Node app starts ASR; the first command starts the two local services. Starting a service process does not establish that model inference works.

After a user joins, read `Peer UUID` from the Node log and bind that session UUID in `profiles.json`, following `profiles.example.json` in the same directory. Each person needs a clean 3–10 second reference recording, its accurate transcript, and the recording language. Currently, `profiles.json` is `{}`; users without a voice profile do not enter the translation pipeline. Reconnecting changes the UUID and requires rebinding. For installation and backend troubleshooting, see the existing [Node app README](../../../../Node/apps/gpt_sovits_translation/README.md). Use this document for this scene's Unity wiring and sample-rate settings.

Stop Node with Ctrl+C, then stop this app's backends:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\apps\gpt_sovits_translation\scripts\stop-backends.ps1
```

## Audio Protocol and State Cleanup

Each sentence starts with a UTF-8 JSON header shorter than 100 bytes, followed by mono PCM16LE. The payload is neither WAV nor YOLO JSON:

```json
{"type":"AudioInfo","targetPeer":"","audioLength":"32000"}
```

`audioLength` is the total byte count encoded as a string. It must be even, at least 200 bytes, and no more than 120 seconds at the source sample rate. Each PCM packet must contain an even number of bytes between 200 and 16000; the cumulative count must not exceed the declared length. The receiver uses the exact Ubiq message `bytes/start/length` slice. The backend already redistributes short final packets.

The player uses a locked, fixed-capacity FIFO and a streaming AudioClip created at the specified source sample rate. The default prebuffer is 200 ms; fully received short sentences can play immediately. The callback supplies zeros when data is unavailable. Normally completed sentences retain their unplayed tails when the next sentence arrives. Buffer overflow or damaged data clears playback state, discards subsequent PCM without a header, and waits for a valid header. A new header replacing an incomplete sentence logs the reason and starts the new sentence. Disabling, destruction, room join/leave events, and audio device configuration changes clear state. Callbacks from an old AudioClip cannot read the new stream.

The clip frequency is the **input PCM sample rate**; it does not need to match the sound device's actual output rate. Unity handles device conversion. Keep the Node output and Inspector source-rate settings consistent. After changing sample-rate or buffer settings during Play mode, re-enable SoVITSAudioSource to apply them; playback then resumes from the next valid sentence header. The player uses the [Unity AudioClip.Create streaming callback API](https://docs.unity3d.com/6000.0/Documentation/ScriptReference/AudioClip.Create.html) and supplies silence when data is unavailable. Actual playback duration at different device output rates still requires verification.

Translated speech plays for every client in the room, including the speaker, at the copied assistant's spatial position. `targetPeer` is empty and does not select a recipient or voice. The backend's session UUID configuration continues to select the voice. Run only one app sending translated audio to channel 95 in a room, to avoid interleaved streams from multiple senders.

## Validation and Troubleshooting

If Windows microphone-test playback contains speech but Node reports almost silent PCM, use **Tools > SoVITS > Log Microphone Diagnostics (5 Seconds)** while this scene is in Play mode. Speak during the five-second observation and inspect `[SoVITS Diagnostic]` messages in the Unity Console. This Editor-only command reads the existing Ubiq microphone clip and recording statistics; it does not start another recording device, save audio, change mixer settings, or call a model. The running app continues its normal processing.

The report distinguishes the local RoomClient UUID, microphone recording positions, raw microphone-clip levels, audio-source state, Editor/listener mute or pause settings, WebRTC connection states, and capture-filter component order. A failed clip read is reported separately from a successfully read silent clip. Raw-clip levels are overlapping snapshots rather than a saved continuous recording. A healthy raw clip with no recording-stat callbacks narrows the fault to Unity's audio processing; healthy recording statistics with silent Node input requires inspecting the capture filter, track, and connection. Multiple WebRTC capture filters are a diagnostic clue, not proof of the cause. Match the logged local UUID to the Node voice profile before interpreting a missing-profile counter.

`Tests/SoVITSPcmStreamTests.cs` provides 52 EditMode cases without model loading. They cover message offsets, positive and negative PCM values, short final packets, invalid headers and lengths, prebuffering, underflow, overflow, consecutive and interrupted sentences, FIFO wrapping, clearing, and concurrent reads/writes. The independent Core assembly does not depend on the Unity engine; the test assembly targets the Editor only. Select `UbiqGenie.RealtimeTranslation.Tests` in the Unity Test Runner's EditMode tab to run the group. It does not start Node, models, or a Ubiq room.

The approved run used `-batchmode -nographics -runTests -testPlatform EditMode -testFilter UbiqGenie.RealtimeTranslation.Tests`; **52/52 passed**. Existing SelectionManager, ConversationalAgentManager, and TextureGenerationManager scripts produced deprecated API / unused member warnings. The new scripts produced no compilation errors or warnings. During that implementation and test run, hash checks confirmed that shared scripts, other scenes, the original README, all ProjectSettings files, and Packages files remained unchanged. Raw results are in the local ignored Logs directory: [test XML](../../../Logs/SoVITSVerification/20261013-153144-results.xml) and [Unity log](../../../Logs/SoVITSVerification/20261013-153144-editmode.log).

Static checks cover scene fileID integrity, uniqueness of new GUIDs, completeness of the copied assistant data, a single receiver, Inspector references, the independent ServerConfig, and matching room/sample-rate settings on both sides. These checks do not replace compilation or runtime acceptance testing.

| Symptom | Check |
| --- | --- |
| No translated speech | Both sides use the same room/server; the current UUID has a real reference recording; Node ASR/translation/TTS logs contain no errors |
| Receiver not ready | The AudioSource field references an enabled SoVITSAudioSource, not InjectableAudioSource |
| `Another receiver already owns NetworkId 95` | Another receiver is present; do not also attach ConversationalAgentManager |
| `Invalid AudioInfo` / `PCM exceeds...` | Node sends this app's protocol, not WAV, YOLO, or messages from another app on channel 95 |
| `PCM playback buffer overflow` | Check main-thread stalls, a paused audio device, or a sender not pacing packets by audio duration; the current sentence is discarded and the next valid header recovers |
| Incorrect speed or pitch | Node output matches `sourceSampleRate`; AudioSource pitch remains 1 |
| Assistant audio is too quiet | The original 3D audio source is retained; check listener distance, system volume, and the audio device |
| A startup PID is logged but processing fails | Inspect `Node/.runtime/gpt_sovits_translation/logs`; a PID only confirms process startup |

Runtime acceptance still requiring approval: duration of 16 kHz input at different device output rates; no residual audio after disabling/re-enabling or switching rooms; two users translating Chinese to English and English to Chinese with their respective preset voices; playback to the whole room while retaining original speech; and measurements of stage timings, end-to-end latency, and peak VRAM. Do not mark voices or the full pipeline as verified until reference materials are supplied and live tests complete.
