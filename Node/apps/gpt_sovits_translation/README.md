# GPT-SoVITS Chinese–English Speech Translation

[English](README.md) | [简体中文](README.zh-CN.md)

This app receives speech from users in a Ubiq room, detects complete utterances after pauses, recognizes Chinese or English, translates into the other language, and broadcasts speech using each user's configured reference voice. Original Ubiq voice calls remain active. All AI inference runs locally.

The app follows the [Node architecture](../../README.md), using ApplicationController, VoipReceiver, and ServiceController. The backend implementation does not modify existing apps, shared services, or shared Unity code.

**For the RealtimeTranslation scene, see the [Unity integration guide](../../../Unity/Assets/Ubiq-Genie/Apps/RealtimeTranslation/INTEGRATION.md).** That scene uses `SoVITSReceiver`, `SoVITSAudioSource` for playback at the input sample rate, and `SoVITSVoiceSelector` for local voice selection in the Unity Inspector. Settings for the legacy ConversationalAgent player are identified separately below.

## Current verification status

- Implementation and environment installation records are managed separately. Actual installation results are in `asr-manifest.json`, `receipts/`, and `logs/` under the runtime directory.
- The two independent Python 3.11 environments and the listed models and tools have been installed locally. Both environments pass `pip check`. Model files were checked against pinned upstream revisions; installed versions and hashes are recorded in the receipts.
- On 2026-10-20, the app type check and **36 Node tests passed**: 16 helper tests (including actual FFmpeg conversion), 12 scheduler/app tests, and 8 voice-binding protocol tests. Localhost mock tests ran outside the isolated environment because it blocked loopback connections and temporary-file replacement. The earlier **19 Python worker tests** remain the historical Python result; Python was not changed for voice selection.
- The app type check now passes. The full repository build remains blocked by old `ubiq-server/ubiq` imports in `dynamic_capture_yolo` and `yolo_only_capture`; scoped ESLint cannot load the existing invalid `.eslintrc.json`. These unrelated files were not changed.
- The RealtimeTranslation receiver passed compilation in Unity **6000.0.61f1** and **52/52 EditMode tests without models**. On 2026-10-13, an approved live diagnostic confirmed Unity microphone frames reaching Node, VAD segmentation, and Chinese ASR. Translation failed because the mutable `qwen3:4b` tag resolved to Thinking 2507; the corrected model is described below.
- After correction, the existing translation client returned plain Chinese text with thinking disabled (22.229 seconds including first load). A separate English GPT-SoVITS request returned a valid 4.08-second, 32 kHz mono PCM16 WAV using the user's reference voice (51.396 seconds including first-use work). These component checks do not prove Unity playback or steady-state latency.
- A 5.50-second English reference clip is available locally as `recordings/my-voice-reference.wav`, with an automatically transcribed matching prompt. Manual file bindings still require the current session UUID; Inspector selections restore the remembered voice for a new UUID after backend confirmation.
- On 2026-10-19, the user confirmed clear translated playback after the Chinese reference was corrected. Two-user integration, peak GPU memory, and the new Inspector workflow remain unverified in a live room. Future runtime tests retain the project's approval requirement.
- On 2026-10-20, Unity **6000.0.61f1** compiled successfully and **77/77 EditMode tests passed**: 52 PCM cases and 25 voice-selection state cases. These validate selection, matching acknowledgements, restoration, invalidation, and failure states without models or a room connection; actual Inspector interaction and PlayerPrefs persistence across Editor restarts still require runtime validation. See the [integration guide](../../../Unity/Assets/Ubiq-Genie/Apps/RealtimeTranslation/INTEGRATION.md) for results.

## Architecture and models

```text
Unity/WebRTC PCM16
  → Separate resampler and Silero VAD for each user
  → Complete-utterance FIFO (up to 8 waiting + 1 in progress)
  → Shared Faster-Whisper large-v3-turbo (CUDA FP16)
  → Ollama qwen3:4b-q4_K_M (no chat history; thinking disabled)
  → GPT-SoVITS v2Pro /tts (user reference audio supplied per request)
  → FFmpeg conversion to mono PCM16LE
  → NetworkId 95 → SoVITSReceiver → Unity assistant audio source
```

Recognition uses `transcribe` only. Qwen translates both Chinese to English and English to Chinese. Silero runs on the CPU using the ONNX model bundled with Faster-Whisper. All users share one GPU recognition model; audio buffers and VAD state are isolated by UUID.

Defaults: end an utterance after 700 ms of silence, retain 300 ms before speech, discard speech shorter than 300 ms, and split continuous speech at 15 seconds. Only results recognized as `zh` or `en` are translated. Short utterances and mixed Chinese/English speech may still be assigned the wrong language.

Complete utterances enter the queue in the order they end. Recognition, translation, synthesis, and transmission run sequentially for each utterance; audio capture does not wait for inference. When the queue is full, newly completed utterances are dropped with a logged reason. There is no unbounded buffer or automatic replay. Sustained input above processing capacity will cause utterance drops.

If a microphone's sample rate or channel count changes, only that user's live audio state is rebuilt; completed utterances remain queued. Output is paced by audio duration, and this app's TCP write buffer is limited to approximately one second of PCM. Network congestion causes the remaining audio for the current translation to be dropped and an error logged. If room admission is rejected or the connection closes, the app and ASR stop.

## Installation

Open PowerShell in `C:\ubiq-genie\Node`. The app uses the project's installed Node dependencies and requires no additional npm packages.

By default, the script uses the existing `C:\ProgramData\anaconda3\Scripts\conda.exe` to create two Python environments. Use `-CondaExe` to select another Conda executable. The Chinese frontend dependency `jieba_fast==0.53` has no official Python 3.11 Windows wheel. Its local build using the existing Visual Studio 2022 C++ tools was approved; the receipt is `receipts/jieba-fast-build.json`. Installation into an empty runtime directory requires Visual Studio C++ Build Tools and explicit enablement of this package's build:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\apps\gpt_sovits_translation\scripts\setup.ps1 -BuildJiebaFast
```

The script places environments, downloads, and models in `Node/.runtime/gpt_sovits_translation`: separate Python 3.11 environments for ASR and GPT-SoVITS, CUDA 12.8 TTS dependencies, GPT-SoVITS v2Pro inference weights, Faster-Whisper, a dedicated Ollama installation, and FFmpeg. Installation does not run model inference.

If the wheel already exists, omit `-BuildJiebaFast`. The script does not perform native compilation by default. TTS dependencies are limited to this app's Chinese/English inference workflow, omitting training tools and Japanese/Korean/Cantonese frontends. Downloaded upstream source is retained. This environment is not a complete GPT-SoVITS training or multilingual WebUI installation.

Downloads require network access; routine inference uses local files and loopback interfaces. The installer pins and records the source commit, model revisions, and dependency versions actually used. See the script's initial `param` declaration for optional arguments. Do not overwrite this directory with another project's virtual environment or GPT-SoVITS installation.

| Component | Pinned version / record |
| --- | --- |
| GPT-SoVITS | `20250606v2pro`, commit `d7c2210da8c013e81a94bfc7b811a477c99fd506`; `receipts/tts-source.json` |
| v2Pro weights and Chinese resources | Model revision `8f8857563c040d542de108a2c18b0425abd6c9c5`; `receipts/tts-models.json` |
| PyTorch / torchaudio | `2.7.1+cu128`; complete TTS dependencies in `receipts/tts-packages.txt` |
| Faster-Whisper | `1.2.1`, CTranslate2 `4.8.2`; model revision `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf`; `asr-manifest.json` |
| Ollama / Qwen | Ollama `0.40.3`, `qwen3:4b-q4_K_M`; pinned model digest `2bfd38a7daaf4b1037efe517ccb73d1a3bbd4822cf89f1a82be1569050a114e0` |
| FFmpeg | `9.0.2`; download hash in `receipts/ollama-ffmpeg.json` |

The original mutable `qwen3:4b` tag resolved to **Qwen3 4B Thinking 2507** during local installation. Its template opens a reasoning block even with the app's `think:false` request, causing long generations for short translations. The app now selects the hybrid [qwen3:4b-q4_K_M](https://ollama.com/library/qwen3:4b-q4_K_M) variant, whose template supports disabling thinking. The installer verifies the full manifest digest above and stops if the tag changes. The original `receipts/ollama-ffmpeg.json` remains the historical installation record; this workspace's model correction is recorded separately in `receipts/ollama-model-correction.json`. Fresh installations record the corrected model in `receipts/ollama-ffmpeg.json`. Model selection alone does not establish successful translation or playback.

## Starting and stopping (approval required before running)

Start the two local backends, then the Node app:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\apps\gpt_sovits_translation\scripts\start-backends.ps1
npm.cmd start gpt_sovits_translation
```

Wait for `Both backend HTTP endpoints are ready` before starting Node. The launcher waits up to 10 minutes per backend and reports an error with the log path if a process exits or its HTTP endpoint stays unavailable. These readiness checks do not run translation or synthesis.

On first launch, use the project's existing CLI flow for Ubiq server configuration. The room GUID and server address in `config.json` must match the Unity client. To reconfigure:

```powershell
npm.cmd start gpt_sovits_translation configure
```

Using `npm.cmd` avoids PowerShell selecting an `npm.ps1` blocked by the execution policy, without changing the system execution policy.

The app starts the ASR worker. GPT-SoVITS listens on `127.0.0.1:9880` by default; Ollama listens on `127.0.0.1:11435`. If a backend is unavailable, requests for the affected utterance fail, while subsequent utterances can still be processed. An ASR worker crash or inference timeout stops the app so it cannot start another task while an old GPU task may still be running.

Ctrl+C stops the app and its ASR/conversion child processes. Stop the local backends with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\apps\gpt_sovits_translation\scripts\stop-backends.ps1
```

The stop script only manages backend processes saved by this app whose identities it has verified. It does not stop unrelated Python, Ollama, or Ubiq processes.

## Unity playback settings

For `Unity/Assets/Ubiq-Genie/Apps/RealtimeTranslation/RealtimeTranslation.unity`, follow the [Unity integration guide](../../../Unity/Assets/Ubiq-Genie/Apps/RealtimeTranslation/INTEGRATION.md). The scene is configured for Nexus (`nexus.cs.ucl.ac.uk:8009`) and room `6765c52b-3ad6-4fb0-9030-2c9a05dc4731`. Keep Node's `translation.outputSampleRate` equal to `SoVITSAudioSource.sourceSampleRate`; both default to **16000**. Unity converts to the device output rate, so this player's source rate does not need to match the sound card rate.

Legacy compatibility: use `Unity/Assets/Ubiq-Genie/Apps/ConversationalAgent/ConversationalAgent.unity`, or a scene already configured with `ConversationalAgentManager` and `InjectableAudioSource`. For this legacy player only, Node output must match Unity's actual `AudioSettings.outputSampleRate`. The default **16000** matches the current `Unity/ProjectSettings/AudioManager.asset`; check Unity's `Output sample rate` log during integration testing. The ordinary Transcription scene does not automatically include an audio receiver.

- Do not run another assistant app that sends audio to `NetworkId(95)` in the same room, as the streams will mix.
- Every member hears the original voice and the later translation, including the speaker. Translation audio comes from the assistant's audio source, not from each user's avatar position.
- `targetPeer` is always empty. It is retained for compatibility with the existing audio header and does not select a recipient. The audio header is shorter than 100 bytes; PCM packets are 200–16000 bytes. The final packet is neither zero-padded nor truncated.
- Use headphones to prevent translated playback from re-entering the microphone; neither scene adds echo cancellation. The original RealtimeTranslation scene has no original-voice mute control. The [copied motion scene](../../../Unity/Assets/Ubiq-Genie/Apps/RealtimeTranslation/MOTION_REPLAY.md) adds an Inspector toggle for original remote voice playback.

## Remote avatar motion replay

The copied [`RealtimeTranslationMotion` scene](../../../Unity/Assets/Ubiq-Genie/Apps/RealtimeTranslation/MOTION_REPLAY.md) uses this same backend. The original
`RealtimeTranslation` scene continues to receive the unchanged channel **95** `AudioInfo` and PCM protocol.
The backend additionally advertises peer property `ubiq.genie.sovits.motion` as
`{"version":1,"sessionId":"..."}` and broadcasts JSON sideband messages on **NetworkId 101**.
Do not run multiple translation backends or another channel 95 sender in the same room.

Every sideband message has `version:1`, `backendUuid`, `sessionId`, and `type`. Utterance messages
include `requestId` and `speakerUuid`:

| Type | Additional fields and meaning |
| --- | --- |
| `accepted` | `sourceStartMs`, `sourceEndMs`: approximate received-speech interval, including VAD preroll and silence, in Node's monotonic `performance.now()` milliseconds |
| `play` | `audioLength` (numeric PCM bytes), `sampleRate`, `sampleCount`: synchronously sent immediately before this sentence's legacy channel 95 header |
| `terminal` | `status`: `sent`, `cancelled`, `skipped`, or `failed`; `sent` confirms transmission, not audible completion |
| `ping` | Client supplies `peerUuid`, string `nonce`, `t0`, and the selected backend/session identity |
| `pong` | Echoes `peerUuid`, `nonce`, `t0`; adds Node monotonic receive/send times `s1`, `s2` |

Clients keep 30 seconds of live pose history and pin accepted sentence clips before inference.
Pinned clips may outlive 30 seconds while translation is queued or computed. The copied scene
linearly fits a pinned clip to translated audio playback, renders live movement again afterward,
and never replays motion onto the local tracked rig. Own translated audio remains audible.
This is approximate sentence alignment, not word or phoneme alignment: independently delivered
WebRTC audio, pose packets, and worker scheduling can shift the estimated interval.

The metadata bridge bounds TCP writes using the same one-second audio budget. If required `play`
metadata cannot be queued together with its following header, that sentence's audio transmission
fails rather than sending audio with an ambiguous identity. Accepted metadata is best effort.
Terminal metadata retries every 250 ms with a 32-message cap; exhausting this cap advertises a
fresh replay session to release old client state. Clients must fall back to audio-only when
history or timing is unavailable and release state on backend/session/room changes. Terminal
messages are idempotent and cancellation is published without waiting for an in-flight model computation.

These additions do not change model dependencies, voice selection, VAD, or the Python worker.
They require the copied scene for replay; live two-user playback and headset comfort must be
checked separately from unit tests.

## Selecting a voice in the Unity Inspector

In the RealtimeTranslation scene, `SoVITSVoiceSelector` is attached to the existing `Ubiq-Genie/SoVITSReceiver` object. Its custom Inspector automatically shows the local client's current UUID. It selects a voice for that client only; there is no in-game or VR selection UI.

After runtime testing is approved:

1. Start the Node app in the same room and enter Unity Play mode.
2. Select `Ubiq-Genie/SoVITSReceiver` and find the `SoVITSVoiceSelector` Inspector. Wait for the voice catalogue from one Node backend.
3. Choose a reference voice from the dropdown, labelled with its recording filename and language, then click **Apply Voice**.
4. Wait for Node's matching acknowledgement with status `bound`. A pending request is not a confirmed binding. Only that acknowledgement saves the selected voice ID to local `PlayerPrefs`, scoped to this Unity project and room.

Node builds the catalogue from the validated entries in `profiles.json` (or `translation.profilesPath`). Entries with the same complete `(refAudioPath, promptText, promptLanguage)` tuple share a SHA-256 voice ID and one dropdown item, even if they have different UUID keys. This keeps the existing UUID file and its format intact; selection does not rewrite it. An empty validated file provides no selectable voices and no default voice.

An acknowledged selection becomes an in-memory override for the current UUID and takes priority over that UUID's file mapping. Clients without an Inspector selection keep the existing UUID-based behavior, including legacy scenes. Ordinary voice changes affect new utterances; queued utterances retain their original voice snapshots.

The last acknowledged choice is automatically requested again after a new UUID, another Play session, or a Node restart, when a single compatible backend and the same voice ID are available. This restoration requires a fresh acknowledgement; a saved preference is not proof that the current session is bound. Changing a profile's path, prompt text, or language changes its voice ID and requires selection again. If a selected voice disappears from the published catalogue, its runtime override is invalidated and affected pending work is canceled; the selector does not guess a replacement.

Offline backends, multiple backends, missing saved voices, and rejected requests are shown in the Inspector. Resolve the reported condition before applying or restoring a choice. Selection requests, acknowledgements, and the small catalogue use existing Ubiq peer properties through Nexus. Only voice IDs, display labels/languages, and binding metadata are exchanged; WAV data, full paths, and prompt text stay on the Node machine. This adds no channel 95 audio message format and changes no Python/model interface.

## Configuring reference voices and legacy UUID bindings

When a user joins, Node prints `Peer UUID: ...; name: ...`. Use that **UUID** as the key in `profiles.json`, following the structure in `profiles.example.json`:

```json
{
  "REPLACE_WITH_REAL_UUID_FROM_LOG": {
    "refAudioPath": "REPLACE_WITH_REAL_REFERENCE_WAV_PATH",
    "promptText": "REPLACE_WITH_THE_COMPLETE_ACCURATE_TRANSCRIPT",
    "promptLanguage": "zh"
  }
}
```

This illustrates the structure and is not a valid ready-to-use configuration. Add a separate entry for each user. Reference material requirements:

- A clean, single-speaker, 3–10 second PCM or IEEE-float WAV. Mono is recommended; avoid background music.
- `promptText` must accurately match the recording. `promptLanguage` is the recording's own language, `zh` or `en`; it need not match the output language.
- Paths may be absolute or relative to `profiles.json`. In Windows JSON paths, use `/` or escape `\` as `\\`.
- The local GPT-SoVITS process must be able to read the path. Its API accepts a server-side file path rather than an audio upload.

Saving triggers a hot reload. The current configuration is replaced only if the entire new configuration validates. Invalid JSON from an incomplete edit, or invalid reference audio, is rejected and the previous valid configuration retained. Queued utterances keep a snapshot of their original voice configuration; new utterances use the new configuration. Removing a user's effective voice binding cancels their pending work; removing a catalogue voice also invalidates affected Inspector overrides. An invalid initial configuration prevents app startup.

When a user leaves, their pending utterances and playback are canceled. If the backend has already started computation, the app waits for that request to finish, discards its result, and then processes the next utterance. Closing an HTTP connection alone does not guarantee that backend GPU computation stops immediately. Each backend request still has its configured timeout.

The UUID identifies the current connection and may change after a client restart or rejoin. Manual file bindings require updating the UUID; the Inspector instead requests restoration of its last acknowledged voice for the new UUID. Display names are not used for matching. An empty `profiles.json` does not select a default voice. To keep personal mappings out of version control, set `translation.profilesPath` to the already-ignored `./profiles.local.json`.

## Logs and troubleshooting

To locate a silent pipeline, stop only the Node app with Ctrl+C, then restart it with input diagnostics enabled. Keep Unity in the same Play session to preserve its UUID:

```powershell
$env:SPEECH_TRANSLATION_DIAGNOSTICS = '1'
npm.cmd start gpt_sovits_translation
```

`[Diagnostic] Input` reports each peer's accepted PCM frame count, peak/RMS level in dBFS, and time since its last frame every 10 seconds. With `profile=valid`, `frames=0` means no valid PCM reached the translation service during that interval; inspect Unity's default Windows microphone, microphone permissions, and the Ubiq WebRTC connection. Frames with `peakDbFS=-inf` contain digital silence. Increasing levels show audio arriving but do not prove intelligible speech. `Utterance`, `ASR completed`, `Translation started`, `TTS started`, and `Audio sent` identify the last reached stage. `Audio sent` does not confirm Unity playback. Input diagnostics retain only aggregate counters, not recordings. Disable them with `Remove-Item Env:SPEECH_TRANSLATION_DIAGNOSTICS` before the next Node start. Live speech can trigger model inference; retain the project's runtime approval requirement.

Unity's Ubiq microphone uses the Windows default recording device. Hearing no local echo of your own original voice is expected; translated audio is played separately by the assistant.

Each successfully generated utterance logs `requestId`, `peerUuid`, language direction, original text, translation, input/output duration, and timings for queueing, recognition, translation, synthesis, and conversion. `endpointToPlaybackMs` measures from utterance completion to the start of translated-audio transmission. It includes queueing but excludes the wait for silence detection. It is not a measured latency from the speaker stopping to sound reaching their headphones.

| Symptom | Check |
| --- | --- |
| UUIDs appear but no translation plays | Confirm a `bound` Inspector acknowledgement for this session or a valid file mapping for the current UUID; check the reference audio and transcript |
| Inspector cannot apply a voice | Check its status: Node must be online in the same room, only one compatible backend may publish the catalogue, and the chosen voice must still exist |
| Saved voice is missing after reconnect | Check whether the validated reference path, prompt text, or language changed; select the current catalogue entry and wait for a new `bound` acknowledgement |
| Initial voice validation fails | Check for unfilled example placeholders and confirm that reference audio is 3–10 seconds long |
| `/api/chat` or `/tts` connection fails | Start the backends, verify loopback addresses/ports, and inspect backend logs |
| CUDA DLL / CTranslate2 error | Check the independent environment, CUDA runtime libraries, and installation records; there is no automatic CPU fallback |
| Playback speed changes | RealtimeTranslation: match Node output to `SoVITSAudioSource.sourceSampleRate` and keep pitch at 1. Legacy InjectableAudioSource: match Node output to Unity's actual `AudioSettings.outputSampleRate` |
| Translated audio is repeatedly recognized | Use headphones and check whether the microphone is capturing speaker playback |
| Utterances dropped under load | Inspect queue and stage timings; reduce simultaneous speakers or speaking frequency |

## Verification commands and pending acceptance checks

The following unit tests without models have been approved. They are neither inference nor installation steps:

```powershell
# Run TypeScript unit tests from the Node directory without loading models
$env:SPEECH_TRANSLATION_TEST_FFMPEG = "$PWD\.runtime\gpt_sovits_translation\tools\ffmpeg\bin\ffmpeg.exe"
node --loader ts-node/esm --test .\services\speech_translation\tests\helpers.test.ts .\services\speech_translation\tests\scheduler.test.ts

# Full app type check (currently reports the pre-existing Ubiq type error above)
.\node_modules\.bin\tsc.cmd -p .\apps\gpt_sovits_translation\tsconfig.check.json

# Isolated Python worker unit tests (without loading real models)
.\.runtime\gpt_sovits_translation\asr-env\python.exe -m unittest discover -s .\services\speech_translation\providers\faster_whisper -p "test_*.py"
```

Tests cover framing, split UTF-8 data, audio packet boundaries, 48 kHz stereo to 16 kHz mono conversion, configuration snapshots and invalid hot reloads, HTTP failures/timeouts/cancellation, per-user VAD isolation, utterance order, the eight-utterance waiting limit, cancellation on departure, and discarding late results. Full acceptance still requires:

1. Reference recordings from two users; verify Chinese to English, English to Chinese, and each user's voice assignment.
2. Check concurrent users, cancellation on departure, missing reference files, unsupported languages, empty recognition, backend recovery, and queue overload.
3. In the actual VR scene, inspect CUDA placement, peak GPU memory, stage timings, and end-to-end playback latency. Confirm there is no truncation, speed change, or duplication.
4. Verify Inspector selection and acknowledgement before saving, automatic restoration for new UUIDs and Node restarts, missing/changed voices, multiple-backend errors, and independent choices from two clients. These new selection behaviors have not been validated by the historical tests above.

After runtime testing is approved, GPU time series can be collected in another terminal and matched to Node's per-utterance `requestId` and timing logs. No such measurements are available yet:

```powershell
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total --format=csv -l 1 -f .\.runtime\gpt_sovits_translation\logs\gpu.csv
```

## Upstream references

- [GPT-SoVITS Chinese documentation](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/docs/cn/README.md)
- [GPT-SoVITS API](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/api_v2.py)
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
- [Ollama Windows](https://docs.ollama.com/windows)
- [Ollama Chat API](https://docs.ollama.com/api/chat)
