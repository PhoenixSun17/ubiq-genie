# GPT-SoVITS Chinese–English Speech Translation

[English](README.md) | [简体中文](README.zh-CN.md)

This app receives speech from users in a Ubiq room, detects complete utterances after pauses, recognizes Chinese or English, translates into the other language, and broadcasts speech using each user's configured reference voice. Original Ubiq voice calls remain active. All AI inference runs locally.

The app follows the [Node architecture](../../README.md), using ApplicationController, VoipReceiver, and ServiceController. The backend implementation does not modify existing apps, shared services, or shared Unity code.

**For the RealtimeTranslation scene, see the [Unity integration guide](../../../Unity/Assets/Apps/RealtimeTranslation/INTEGRATION.md).** That scene uses the new `SoVITSReceiver` and `SoVITSAudioSource`, which plays at the input sample rate. Settings for the legacy ConversationalAgent player are identified separately below.

## Current verification status

- Implementation and environment installation records are managed separately. Actual installation results are in `asr-manifest.json`, `receipts/`, and `logs/` under the runtime directory.
- The two independent Python 3.11 environments and the listed models and tools have been installed locally. Both environments pass `pip check`. Model files were checked against pinned upstream revisions; installed versions and hashes are recorded in the receipts.
- Approved tests without models have been run: **19 Python worker tests passed**, **13 Node helper tests passed** (including actual FFmpeg conversion), and **8 scheduler/app lifecycle tests passed**. Node localhost mock tests ran outside the isolated environment because it blocked loopback connections and temporary-file replacement.
- The full app type check currently reports **one pre-existing error**: `components/application.ts:33` references two different Ubiq `NetworkScene` types. A separate baseline check of the original base app reproduced the same diagnostic. New files add no type diagnostics; shared components and existing dependencies were not changed.
- The RealtimeTranslation receiver passed compilation in Unity **6000.0.61f1** and **52/52 EditMode tests without models**. On 2026-10-13, an approved live diagnostic confirmed Unity microphone frames reaching Node, VAD segmentation, and Chinese ASR. Translation failed because the mutable `qwen3:4b` tag resolved to Thinking 2507; the corrected model is described below.
- After correction, the existing translation client returned plain Chinese text with thinking disabled (22.229 seconds including first load). A separate English GPT-SoVITS request returned a valid 4.08-second, 32 kHz mono PCM16 WAV using the user's reference voice (51.396 seconds including first-use work). These component checks do not prove Unity playback or steady-state latency.
- A 5.50-second English reference clip is available locally as `recordings/my-voice-reference.wav`, with an automatically transcribed matching prompt. The current profile binds a session UUID, which must be checked again after Unity restarts. Unconfigured users are ignored.
- The diagnostic Node/ASR session was stopped at the user's request before a complete corrected live playback test. Actual voice quality, Unity playback, two-user integration, peak GPU memory, and end-to-end latency remain unverified. Future runtime tests retain the project's approval requirement.

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

For `Unity/Assets/Apps/RealtimeTranslation/RealtimeTranslation.unity`, follow the [Unity integration guide](../../../Unity/Assets/Apps/RealtimeTranslation/INTEGRATION.md). The scene is configured for Nexus (`nexus.cs.ucl.ac.uk:8009`) and room `6765c52b-3ad6-4fb0-9030-2c9a05dc4731`. Keep Node's `translation.outputSampleRate` equal to `SoVITSAudioSource.sourceSampleRate`; both default to **16000**. Unity converts to the device output rate, so this player's source rate does not need to match the sound card rate.

Legacy compatibility: use `Unity/Assets/Apps/ConversationalAgent/ConversationalAgent.unity`, or a scene already configured with `ConversationalAgentManager` and `InjectableAudioSource`. For this legacy player only, Node output must match Unity's actual `AudioSettings.outputSampleRate`. The default **16000** matches the current `Unity/ProjectSettings/AudioManager.asset`; check Unity's `Output sample rate` log during integration testing. The ordinary Transcription scene does not automatically include an audio receiver.

- Do not run another assistant app that sends audio to `NetworkId(95)` in the same room, as the streams will mix.
- Every member hears the original voice and the later translation, including the speaker. Translation audio comes from the assistant's audio source, not from each user's avatar position.
- `targetPeer` is always empty. It is retained for compatibility with the existing audio header and does not select a recipient. The audio header is shorter than 100 bytes; PCM packets are 200–16000 bytes. The final packet is neither zero-padded nor truncated.
- Use headphones to prevent translated playback from re-entering the microphone. This version adds neither echo cancellation nor controls to mute original voice calls.

## Configuring voices by session UUID

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

Saving triggers a hot reload. The current configuration is replaced only if the entire new configuration validates. Invalid JSON from an incomplete edit, or invalid reference audio, is rejected and the previous valid configuration retained. Queued utterances keep a snapshot of their original voice configuration; new utterances use the new configuration. Removing a user's profile cancels their pending work. An invalid initial configuration prevents app startup.

When a user leaves, their pending utterances and playback are canceled. If the backend has already started computation, the app waits for that request to finish, discards its result, and then processes the next utterance. Closing an HTTP connection alone does not guarantee that backend GPU computation stops immediately. Each backend request still has its configured timeout.

The UUID identifies the current connection and may change after a client restart or rejoin, requiring a new binding. Display names are not used for matching. An empty `profiles.json` does not select a default voice. To keep personal mappings out of version control, set `translation.profilesPath` to the already-ignored `./profiles.local.json`.

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
| UUIDs appear but no translation plays | Confirm that the current UUID has a valid profile and that the audio path and transcript are correct |
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
