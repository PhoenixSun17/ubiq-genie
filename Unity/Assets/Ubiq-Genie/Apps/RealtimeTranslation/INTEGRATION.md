# RealtimeTranslation / GPT-SoVITS Integration

This scene receives translated speech from the existing `gpt_sovits_translation` app. The path is Unity microphone → Ubiq VoIP → Node app → Python ASR / local translation and synthesis services → Node NetworkId 95 → SoVITSReceiver → assistant audio source. Unity does not call the Python HTTP endpoints directly.

On 2026-10-20, compilation completed in Unity 6000.0.61f1 and **77/77 EditMode tests passed**: 52 PCM cases and 25 voice-selection state cases (0 failures, 0 skipped, exit code 0). Results: [test XML](../../../../Logs/SoVITSVerification/20261020-voice-selector/results.xml) and [Unity log](../../../../Logs/SoVITSVerification/20261020-voice-selector/unity.log). The app type check and 36 Node tests also passed. The existing voice configuration and 47 checked project/package/shared files were unchanged by verification. No model or room was started.

The user confirmed clear single-user translated playback on 2026-10-19 after correcting a Chinese reference. The new Inspector interaction, actual PlayerPrefs persistence across Editor restarts, and live reconnect workflow remain **unverified**, as does two-user integration. State tests do not replace these runtime checks; obtain approval before live validation.

## Scene and Inspector

Open `Assets/Ubiq-Genie/Apps/RealtimeTranslation/RealtimeTranslation.unity` in the Unity project. Use the version in `Unity/ProjectSettings/ProjectVersion.txt` (currently **6000.0.61f1**).

| Object or setting | Wiring |
| --- | --- |
| `Ubiq-Genie/SoVITSReceiver` | The former MessageReceiver object now hosts SoVITSReceiver, registered on NetworkId **95** |
| `SoVITSVoiceSelector` | On the same receiver object; its custom Inspector displays the local UUID, voice catalogue, selection status, and **Apply Voice** button |
| `SoVITSReceiver.audioSource` | SoVITSAudioSource on the copied assistant's `Avatar/Speech` object |
| `SoVITSReceiver.assistantController` | VirtualAssistantController on the copied assistant |
| `SoVITSReceiver.volume` | AudioSourceVolume on the copied assistant's Speech object |
| `Avatar/Speech` | Retains AudioSource, AudioSourceVolume, and AudioSourceSpeechIndicator; the player is replaced with SoVITSAudioSource |
| `SoVITSAudioSource.sourceSampleRate` | **16000**; must match Node `translation.outputSampleRate` |
| `SoVITSAudioSource.prebufferMilliseconds` | **200** |
| `SoVITSAudioSource.maxBufferMilliseconds` | **2000** |
| RoomClient server configuration | This directory's independent `SoVITSServerConfig.asset` |
| RoomJoiner Guid | This directory's independent `SoVITSRoomGuid.asset` |

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

The scene uses its own `SoVITSRoomGuid.asset` and `SoVITSServerConfig.asset`. The Room Guid value must match `roomGuid` in `C:\ubiq-genie\Node\apps\gpt_sovits_translation\config.json`.

After approval for integration testing, run the following in PowerShell from `C:\ubiq-genie\Node`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\apps\gpt_sovits_translation\scripts\start-backends.ps1
npm.cmd start gpt_sovits_translation
```

Using `npm.cmd` avoids PowerShell invoking an `npm.ps1` blocked by execution policy, without changing the system policy. The Node app starts ASR; the first command starts the two local services. Starting a service process does not establish that model inference works.

Prepare validated references in `profiles.json`, following `profiles.example.json` in the same Node app directory. Each person needs a clean 3–10 second reference recording, its accurate transcript, and the recording language. Use the Inspector workflow below to bind a catalogue voice to this client's current UUID. Existing manual UUID mappings remain supported: read `Peer UUID` from the Node log, and update the file mapping when that UUID changes. An empty catalogue offers no default voice; users without an effective voice binding do not enter the translation pipeline. For installation, the current local reference-material status, and backend troubleshooting, see the [Node app README](../../../../../Node/apps/gpt_sovits_translation/README.md). Use this document for this scene's Unity wiring and sample-rate settings.

Stop Node with Ctrl+C, then stop this app's backends:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\apps\gpt_sovits_translation\scripts\stop-backends.ps1
```

## Inspector Voice Selection

This feature is an Editor Inspector workflow, with no in-game or VR selection UI. After runtime testing is approved, start the Node app in the same room, enter Play mode, and select `Ubiq-Genie/SoVITSReceiver`. Its `SoVITSVoiceSelector` component automatically displays the local RoomClient UUID; there is no field for choosing another user's UUID.

Wait for one compatible backend's catalogue, select a voice labelled by its reference filename and language, and click **Apply Voice**. A submitted request is pending until the targeted Node instance acknowledges the same client, request, and voice with status `bound`. Only then is the voice ID saved in local `PlayerPrefs`, separately for each Unity project and room. A rejected or stale acknowledgement cannot save a new preference or report a successful binding.

Node derives the catalogue from the fully validated `profiles.json` entries (or the configured `translation.profilesPath`). Identical `(refAudioPath, promptText, promptLanguage)` tuples are deduplicated to one SHA-256 voice ID, regardless of their UUID keys. The existing file format and contents are not rewritten when a client chooses a voice. An acknowledged choice creates an in-memory override for that client's UUID, taking priority over its file mapping. Clients without a selection continue using their existing file bindings.

The selector automatically requests the last acknowledged choice after the local UUID changes, Play mode starts again, or Node restarts. It waits for one compatible backend and the same voice ID, and requires a fresh acknowledgement. Changing the validated path, prompt text, or language changes the ID and requires selecting the updated entry. A missing saved voice, offline backend, multiple backends, or rejected request is reported explicitly; the selector does not choose a replacement. There is no default voice when the catalogue is empty.

Normal voice changes affect new utterances while queued utterances retain their original voice snapshots. If a selected voice is removed from the published catalogue, Node invalidates affected runtime overrides and cancels their affected pending jobs. User departure continues to cancel that user's pending work. Catalogue publication and runtime selection do not modify legacy scene bindings.

Control metadata travels through the existing Ubiq peer-property mechanism over Nexus:

| Peer property | Purpose |
| --- | --- |
| `ubiq.genie.sovits.catalog` | Node publishes version 1 catalogue entries containing voice ID, display name, and language |
| `ubiq.genie.sovits.selection` | Unity publishes a version 1 request containing `requestId`, `voiceId`, and the target `backendUuid`; Node derives the requesting user's UUID from the peer event |
| `ubiq.genie.sovits.bindings` | Node publishes version 1 acknowledgements with `peerUuid`, `requestId`, `voiceId`, `status` (`bound` or `error`), and a message |

Only catalogue and binding metadata cross this control path. Reference WAV data, full filesystem paths, and prompt text stay on the Node machine. Channel 95 continues to carry the existing audio protocol below; there is no additional audio message type, Python endpoint, or model change. A `bound` acknowledgement confirms voice selection, not successful synthesis or playback.

## Audio Protocol and State Cleanup

Each sentence starts with a UTF-8 JSON header shorter than 100 bytes, followed by mono PCM16LE. The payload is neither WAV nor YOLO JSON:

```json
{"type":"AudioInfo","targetPeer":"","audioLength":"32000"}
```

`audioLength` is the total byte count encoded as a string. It must be even, at least 200 bytes, and no more than 120 seconds at the source sample rate. Each PCM packet must contain an even number of bytes between 200 and 16000; the cumulative count must not exceed the declared length. The receiver uses the exact Ubiq message `bytes/start/length` slice. The backend already redistributes short final packets.

The player uses a locked, fixed-capacity FIFO and a streaming AudioClip created at the specified source sample rate. The default prebuffer is 200 ms; fully received short sentences can play immediately. The callback supplies zeros when data is unavailable. Normally completed sentences retain their unplayed tails when the next sentence arrives. Buffer overflow or damaged data clears playback state, discards subsequent PCM without a header, and waits for a valid header. A new header replacing an incomplete sentence logs the reason and starts the new sentence. Disabling, destruction, room join/leave events, and audio device configuration changes clear state. Callbacks from an old AudioClip cannot read the new stream.

The clip frequency is the **input PCM sample rate**; it does not need to match the sound device's actual output rate. Unity handles device conversion. Keep the Node output and Inspector source-rate settings consistent. After changing sample-rate or buffer settings during Play mode, re-enable SoVITSAudioSource to apply them; playback then resumes from the next valid sentence header. The player uses the [Unity AudioClip.Create streaming callback API](https://docs.unity3d.com/6000.0/Documentation/ScriptReference/AudioClip.Create.html) and supplies silence when data is unavailable. Actual playback duration at different device output rates still requires verification.

Translated speech plays for every client in the room, including the speaker, at the copied assistant's spatial position. `targetPeer` is empty and does not select a recipient or voice. The backend selects the voice from the current UUID's acknowledged runtime override, or its existing file mapping when no override is active. Run only one app sending translated audio to channel 95 in a room, to avoid interleaved streams from multiple senders.

## Validation and Troubleshooting

If Windows microphone-test playback contains speech but Node reports almost silent PCM, use **Tools > SoVITS > Log Microphone Diagnostics (5 Seconds)** while this scene is in Play mode. Speak during the five-second observation and inspect `[SoVITS Diagnostic]` messages in the Unity Console. This Editor-only command reads the existing Ubiq microphone clip and recording statistics; it does not start another recording device, save audio, change mixer settings, or call a model. The running app continues its normal processing.

The report distinguishes the local RoomClient UUID, microphone recording positions, raw microphone-clip levels, audio-source state, Editor/listener mute or pause settings, WebRTC connection states, and capture-filter component order. A failed clip read is reported separately from a successfully read silent clip. Raw-clip levels are overlapping snapshots rather than a saved continuous recording. A healthy raw clip with no recording-stat callbacks narrows the fault to Unity's audio processing; healthy recording statistics with silent Node input requires inspecting the capture filter, track, and connection. Multiple WebRTC capture filters are a diagnostic clue, not proof of the cause. Match the logged local UUID to the Node voice profile before interpreting a missing-profile counter.

`Tests/SoVITSPcmStreamTests.cs` provides 52 EditMode cases without model loading. They cover message offsets, positive and negative PCM values, short final packets, invalid headers and lengths, prebuffering, underflow, overflow, consecutive and interrupted sentences, FIFO wrapping, clearing, and concurrent reads/writes. The independent Core assembly does not depend on the Unity engine; the test assembly targets the Editor only. Select `UbiqGenie.RealtimeTranslation.Tests` in the Unity Test Runner's EditMode tab to run the group. It does not start Node, models, or a Ubiq room.

The historical PCM test run used `-batchmode -nographics -runTests -testPlatform EditMode -testFilter UbiqGenie.RealtimeTranslation.Tests`; **52/52 passed**. Existing SelectionManager, ConversationalAgentManager, and TextureGenerationManager scripts produced deprecated API / unused member warnings. The PCM receiver scripts produced no compilation errors or warnings in that run. During that implementation and test run, hash checks confirmed that shared scripts, other scenes, the original README, all ProjectSettings files, and Packages files remained unchanged. Raw results are in the local ignored Logs directory: [test XML](../../../../Logs/SoVITSVerification/20261013-153144-results.xml) and [Unity log](../../../../Logs/SoVITSVerification/20261013-153144-editmode.log). This result predates Inspector voice selection and does not validate its new control flow.

Static checks cover scene fileID integrity, uniqueness of new GUIDs, completeness of the copied assistant data, a single receiver, Inspector references, the independent ServerConfig, and matching room/sample-rate settings on both sides. These checks do not replace compilation or runtime acceptance testing.

| Symptom | Check |
| --- | --- |
| `NullReferenceException` in `RoomJoiner.DoJoin` when accessing `Guid.Guid` | Assign this scene's `SoVITSRoomGuid.asset` to RoomJoiner > Guid. After a scene-file repair, exit Play mode and reload the scene from disk before retrying |
| No translated speech | Both sides use the same room/server; the current session has a `bound` acknowledgement or valid UUID file mapping; Node ASR/translation/TTS logs contain no errors |
| Voice catalogue unavailable | Check the selector status and confirm that Node is online in the same room with valid reference profiles; an empty file offers no voices |
| Multiple voice backends | Stop the extra translation app in this room; the selector will not guess which backend to use |
| Saved voice no longer exists | Check for changes to the reference path, prompt text, or language; select the current catalogue entry and wait for a new `bound` acknowledgement |
| Applying a voice never confirms success | Inspect Node's binding error and room connectivity; pending or rejected requests are not saved as successful choices |
| Receiver not ready | The AudioSource field references an enabled SoVITSAudioSource, not InjectableAudioSource |
| `Another receiver already owns NetworkId 95` | Another receiver is present; do not also attach ConversationalAgentManager |
| `Invalid AudioInfo` / `PCM exceeds...` | Node sends this app's protocol, not WAV, YOLO, or messages from another app on channel 95 |
| `PCM playback buffer overflow` | Check main-thread stalls, a paused audio device, or a sender not pacing packets by audio duration; the current sentence is discarded and the next valid header recovers |
| Incorrect speed or pitch | Node output matches `sourceSampleRate`; AudioSource pitch remains 1 |
| Assistant audio is too quiet | The original 3D audio source is retained; check listener distance, system volume, and the audio device |
| A startup PID is logged but processing fails | Inspect `Node/.runtime/gpt_sovits_translation/logs`; a PID only confirms process startup |

Runtime acceptance still requiring approval: duration of 16 kHz input at different device output rates; no residual audio after disabling/re-enabling or switching rooms; two users translating Chinese to English and English to Chinese with their respective preset voices; playback to the whole room while retaining original speech; and measurements of stage timings, end-to-end latency, and peak VRAM. Voice-selection acceptance must also cover saving only after a matching acknowledgement, separate choices for two clients, automatic restoration after UUID/Play/Node changes, missing or edited references, multiple-backend errors, and preservation of queued voice snapshots during ordinary switches. Do not mark voices or the full pipeline as verified until the required reference materials and live tests are complete.
