# GPT-SoVITS Translation: Deployment and Startup

[English](README.gpt-sovits-translation.md) | [简体中文](README.gpt-sovits-translation.zh-CN.md)

This guide covers the Windows deployment of `gpt_sovits_translation` and the included **RealtimeTranslation** Unity scene. It includes first-time installation, reference recordings, Inspector voice selection, daily startup, shutdown, and troubleshooting. Commands use the checkout at `C:\ubiq-genie`; substitute your checkout location when necessary.

The paths, commands, and settings below were checked against this checkout on **2026-10-23**. Writing this guide did not install software, start services, run models, or repeat live room tests. Existing project READMEs remain unchanged.

**Already installed?** Go to [daily startup](#6-daily-startup-backends---node---unity---voice-selection). For a new machine, follow sections 2–6 in order. For a new reference recording, start with [voice preparation](#4-prepare-reference-recordings-and-the-voice-list).

## 1. What runs locally and what uses the network

```text
Unity microphone
  -> Ubiq WebRTC voice connection -> Node VoipReceiver
  -> Per-user resampling and Silero VAD
  -> Utterance queue -> Faster-Whisper -> Ollama translation -> GPT-SoVITS
  -> FFmpeg -> Ubiq audio messages on NetworkId 95
  -> Unity SoVITSReceiver -> assistant AudioSource
```

| Part | Where it runs / network use |
| --- | --- |
| Microphone, avatars, voice selection, translated playback | On each Unity client |
| VAD, queue, recognition, translation, synthesis, conversion | On the machine running Node; model inference is local |
| Faster-Whisper | Local Python worker started by Node; shared CUDA FP16 model, separate user buffers |
| Ollama | Local HTTP service at `http://127.0.0.1:11435`; model `qwen3:4b-q4_K_M` |
| GPT-SoVITS v2Pro | Local HTTP service at `http://127.0.0.1:9880`; reference audio supplied by local filesystem path |
| Room membership, peer properties, translated audio messages | Through the configured online Ubiq server, currently `nexus.cs.ucl.ac.uk:8009` |
| Original microphone speech | Ubiq/WebRTC between clients and Node; the current config uses Google's STUN service for connection establishment |
| Software/model installation | Internet downloads; model files and environments are stored locally afterward |

**Local inference does not mean the whole application is offline.** With the current Nexus configuration, room communication still needs network access. Reference WAV files, their full paths, and reference transcripts stay on the backend machine; the voice selector exchanges IDs, filenames, languages, and binding status through Ubiq. Microphone audio and generated speech travel over the room's networking paths.

Unity does not call Ollama or GPT-SoVITS directly. Unity can run on another machine, but Node and these two HTTP services must remain together for this deployment: the app accepts loopback model-service addresses. Original calls remain active, and translated speech is heard through the assistant by everyone in the room, including the speaker. Use headphones to avoid feeding translated speech back into the microphone.

## 2. Files and prerequisites

| Path relative to the repository | Purpose |
| --- | --- |
| `Node/apps/gpt_sovits_translation/config.json` | Room/server, runtime paths, model endpoints, VAD, output sample rate |
| `Node/apps/gpt_sovits_translation/profiles.json` | Validated reference voices and legacy UUID mappings |
| `Node/apps/gpt_sovits_translation/profiles.example.json` | Structure template; placeholders are not a usable configuration |
| `Node/apps/gpt_sovits_translation/recordings/` | Reference recordings on the Node machine |
| `Node/apps/gpt_sovits_translation/scripts/` | Install, start, and stop scripts |
| `Node/.runtime/gpt_sovits_translation/` | Independent Python environments, models, tools, receipts, and backend logs |
| `Unity/Assets/Ubiq-Genie/Apps/RealtimeTranslation/RealtimeTranslation.unity` | Standard assistant playback scene |
| `Unity/Assets/Ubiq-Genie/Apps/RealtimeTranslation/INTEGRATION.md` | Detailed scene wiring and protocol documentation |

For a new backend machine, prepare:

- Windows x64, PowerShell, and Node.js **20 or later** with npm.
- An NVIDIA GPU and driver supporting the installed CUDA **12.8** stack. The ASR configuration requires `cuda` / `float16`; it does not fall back to CPU. A minimum VRAM capacity for simultaneous models and Unity has not been established by this project.
- Conda. The default executable is `C:\ProgramData\anaconda3\Scripts\conda.exe`; use `-CondaExe` if yours is elsewhere. The installer creates separate **Python 3.11** environments; no global Python environment activation is needed.
- Visual Studio C++ Build Tools for the initial Windows Python 3.11 build of `jieba_fast==0.53`. Enable that build explicitly with `-BuildJiebaFast`.
- Internet access for installation and sufficient storage for two environments, source, model weights, and download caches. The project does not specify a verified minimum disk budget.
- Unity Hub and the Editor version recorded in [ProjectVersion.txt](Unity/ProjectSettings/ProjectVersion.txt), currently **6000.0.61f1** in this checkout. Use that file as the authority if general framework documentation lists a different version.
- A working microphone and headphones. Test the selected Windows default input device with a recording and playback before opening Unity.

These instructions describe how to run the app. Agent-driven compilation, model inference, and live Ubiq tests remain subject to the project's existing approval requirements.

## 3. First-time installation

Skip this section for an already installed, working runtime. Do not reinstall the models on every start.

### 3.1 Install Node dependencies

Open PowerShell:

```powershell
Set-Location C:\ubiq-genie\Node
node --version
npm.cmd --version
npm.cmd install
```

Use `npm.cmd` throughout this guide. It avoids PowerShell invoking a blocked `npm.ps1`, without changing the system execution policy. Installing Node dependencies does not install the speech models.

### 3.2 Install local environments, models, and tools

From the same `Node` directory, for a new runtime:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\apps\gpt_sovits_translation\scripts\setup.ps1 -BuildJiebaFast
```

If Conda is not at the default location, append `-CondaExe` with your actual executable path. For example, replace the placeholder below before running:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\apps\gpt_sovits_translation\scripts\setup.ps1 -BuildJiebaFast -CondaExe "D:\YOUR_CONDA_DIRECTORY\Scripts\conda.exe"
```

The setup wrapper installs TTS, then ASR, then Ollama/FFmpeg. It downloads GPT-SoVITS v2Pro, the Faster-Whisper `large-v3-turbo` model, and the translation model. It records code/model revisions and dependencies. It does not perform recognition, translation, synthesis, or Ubiq integration tests.

The TTS environment uses PyTorch/torchaudio `2.7.1+cu128`. Although ASR has its own Python environment, its runtime also uses CUDA DLLs from the TTS environment. Install the complete stack for this app. This is a Chinese/English inference environment, not a complete GPT-SoVITS training or multilingual WebUI installation.

Useful options:

| Option | Use |
| --- | --- |
| `-Component all` | Default: install all components |
| `-Component tts`, `asr`, or `ollama` | Retry a particular installation stage after resolving its error; use the `setup.ps1` wrapper |
| `-BuildJiebaFast` | Permit the specific native dependency build; omit when its compatible wheel already exists |
| `-RuntimeRoot "D:\YOUR_RUNTIME_DIRECTORY"` | Change the runtime directory; also update `translation.runtimePath` in app config and pass the same option to start/stop scripts |

Setup stops on an error. Read its output and runtime logs, fix the cause, and rerun the appropriate stage. A printed completion message is an installation result, not proof of successful inference.

### 3.3 Check installation records

The default runtime contains:

```text
Node/.runtime/gpt_sovits_translation/
  asr-env/                  Python 3.11 ASR environment
  tts-env/                  Python 3.11 TTS environment
  GPT-SoVITS/               Pinned upstream source and TTS resources
  models/                   ASR and Ollama model storage
  ollama/                   Dedicated Ollama executable and local state
  tools/ffmpeg/bin/          ffmpeg.exe
  tts_infer.yaml            Generated GPT-SoVITS inference configuration
  asr-manifest.json         ASR installation record
  receipts/                 Source, model, package, and download records
  logs/                     Backend and installation logs
  pids/                     App-owned backend process identities
```

For exact installed versions, inspect `asr-manifest.json` and `receipts/tts-source.json`, `tts-models.json`, `tts-install.json`, `tts-packages.txt`, and `ollama-ffmpeg.json`. This existing workspace also has `ollama-model-correction.json` for a later translation-model correction. Keep `qwen3:4b-q4_K_M` in the app config; the older mutable `qwen3:4b` tag previously selected a thinking variant and caused unsuitable translation behavior.

Runtime directories and recordings are local deployment data. They are not supplied by a source-only clone and must not be committed as model/environment artifacts.

## 4. Prepare reference recordings and the voice list

### 4.1 Recording location and requirements

Put reference files in:

```text
C:\ubiq-genie\Node\apps\gpt_sovits_translation\recordings\
```

Each voice needs a **3–10 second**, clean, single-speaker, uncompressed PCM or IEEE-float **WAV**, plus its exact transcript and recording language. Mono is recommended. Avoid music, long silence, clipped speech, and mismatches between the selected segment and its text. MP3 files renamed to `.wav` are not valid WAV files.

`promptLanguage` describes the reference recording: `zh` for Chinese or `en` for English. It does not force the translated output language. Chinese speech is translated into English, and English speech into Chinese.

To create a separate five-second mono PCM16 copy using the installed FFmpeg, replace the example input filename and choose an appropriate segment. This command keeps the source and refuses to overwrite an existing output:

```powershell
Set-Location C:\ubiq-genie\Node
& .\.runtime\gpt_sovits_translation\tools\ffmpeg\bin\ffmpeg.exe -n -i .\apps\gpt_sovits_translation\recordings\source-recording.wav -ss 0 -t 5 -ac 1 -ar 32000 -c:a pcm_s16le .\apps\gpt_sovits_translation\recordings\my-reference-5s.wav
```

Listen to the resulting file and transcribe **that exact segment**, including the words at its boundaries. Automatic transcription is not part of normal setup and requires a separate model run. A structurally valid recording can still produce poor synthesis if its contents or transcript are unsuitable.

### 4.2 Add a validated entry

The selector reads the Node catalogue derived from `profiles.json`; simply copying a WAV into `recordings` does **not** add a voice. Keep existing valid entries. Add or edit an entry using the template below, replacing every placeholder:

```json
{
  "REPLACE_WITH_A_REAL_PEER_UUID": {
    "refAudioPath": "recordings/my-reference-5s.wav",
    "promptText": "REPLACE_WITH_THE_EXACT_TRANSCRIPT_OF_THIS_RECORDING",
    "promptLanguage": "en"
  }
}
```

This is a structure example, not a file to paste unchanged. The top-level key must have UUID syntax. Reference paths are resolved relative to `profiles.json`, not the PowerShell working directory. Use `/` in JSON paths, or escape each Windows backslash as `\\`. The file must be readable by GPT-SoVITS on the backend machine.

For a **fresh configuration with no UUID yet**, use `{}` as the file contents, start Node and Unity as described below, and read `Local UUID` in the Inspector or `Peer UUID` in the Node console. Use it once to create the first valid reference entry. Save, wait for the catalogue to refresh, then select it in Unity. An empty file permits startup but offers no voices and translates no unbound user's speech. Do not replace an existing populated file with `{}`.

Existing entries remain available in the catalogue even when their original UUIDs are offline. They do not need new keys on every reconnect. The selector binds the selected voice to the current local UUID automatically. Identical normalized path, transcript, and language tuples appear only once in the dropdown.

Node validates the whole file and hot-reloads it. Expect `Loaded N validated voice profile(s)`. Invalid JSON, missing files, unsupported WAV data, or durations outside 3–10 seconds reject the entire update and retain the previous valid configuration. Invalid initial configuration prevents startup. For personal configuration outside version control, use the ignored `profiles.local.json` and set `translation.profilesPath` to `./profiles.local.json` before starting Node.

## 5. Check Node and Unity room settings

Open [the app config](Node/apps/gpt_sovits_translation/config.json) and the Unity scene's settings. For the supplied standard scene, these must agree:

| Setting | Value / Unity location |
| --- | --- |
| Node `roomserver.joinExisting` | `true`; connects to the existing Nexus server |
| Node `roomserver.uri` and TCP port | `nexus.cs.ucl.ac.uk`, `8009` |
| Unity RoomClient server | `SoVITSServerConfig.asset` in the RealtimeTranslation directory |
| Room GUID on both sides | `6765c52b-3ad6-4fb0-9030-2c9a05dc4731` |
| Unity RoomJoiner > Guid | `SoVITSRoomGuid.asset` in the same directory; must not be empty |
| Node `translation.outputSampleRate` | `16000` |
| Unity `SoVITSAudioSource.sourceSampleRate` | `16000`, on the assistant's `Avatar/Speech` object |
| Ollama URL / model | `http://127.0.0.1:11435` / `qwen3:4b-q4_K_M` |
| GPT-SoVITS URL | `http://127.0.0.1:9880` |

The supplied asset also has a WebGL connection override on port `8010`; this guide's primary workflow is the desktop Unity Editor. Do not treat the model service ports as Ubiq room ports or expose them to remote Unity clients. The supplied backend launch scripts use fixed ports `11435` and `9880`; editing only the JSON URLs will not change the listener ports.

Node's runtime and profile paths are relative to the app directory; ASR, model, and FFmpeg paths inside the runtime are relative to `translation.runtimePath`. A custom runtime installation must match this configuration.

For the already configured Nexus workflow, no configuration wizard is required. `npm.cmd start gpt_sovits_translation configure` changes the existing configuration; use it only when intentionally reconfiguring the app, then recheck both sides.

## 6. Daily startup: backends -> Node -> Unity -> voice selection

### Step 1: Start Ollama and GPT-SoVITS

```powershell
Set-Location C:\ubiq-genie\Node
powershell -NoProfile -ExecutionPolicy Bypass -File .\apps\gpt_sovits_translation\scripts\start-backends.ps1
```

Wait for:

```text
Both backend HTTP endpoints are ready. You can now start the Node app; inference has not been tested.
```

The launcher waits up to ten minutes **per service** for HTTP readiness. A PID message alone is not readiness. Existing app-owned processes are reused; an occupied port belonging to an unrelated process is rejected. If startup fails after one service starts, that service may remain running; inspect the error and use this app's stop script when needed.

Optional checks that do not request inference:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:11435/api/tags
(Invoke-WebRequest -UseBasicParsing -Uri http://127.0.0.1:9880/openapi.json).StatusCode
```

The first should list the expected Ollama model; the second should return `200`. These checks do not establish synthesis quality or warm up the first translation.

### Step 2: Start Node and wait for ASR

In the same terminal after the launcher returns:

```powershell
npm.cmd start gpt_sovits_translation
```

Keep this terminal running. Node starts the ASR worker itself; do not manually launch another ASR worker. Look for:

```text
[SpeechTranslation] Loaded N validated voice profile(s)
[SpeechTranslation] ASR ready; translated audio output is 16000 Hz PCM16 mono.
[GPT-SoVITS Translation] Waiting for peers. Select a voice in Unity SoVITSVoiceSelector, or bind a UUID in profiles.json.
```

`N` is the number of validated mappings, not necessarily the number of distinct catalogue voices. Run **one** translation backend per room. Another app sending to channel 95 can interleave audio, and multiple selector backends produce an explicit ambiguity error.

### Step 3: Open the standard Unity scene

1. In Unity Hub, open `C:\ubiq-genie\Unity` with the version in `ProjectVersion.txt`.
2. Wait for package import and script compilation to finish. Resolve compilation errors before Play.
3. Open `Assets/Ubiq-Genie/Apps/RealtimeTranslation/RealtimeTranslation.unity`.
4. Check the server/room references in section 5, the Windows default microphone, and the output device. Ensure Unity Editor audio is not muted.
5. Enter **Play** and wait for the room connection. Node should print `Peer UUID: ...; name: ...`.

The copied assistant's references are already wired in the scene. If checking manually, `SoVITSReceiver` must point to `SoVITSAudioSource`, the assistant controller, and its volume component. Keep one channel 95 receiver; do not add `ConversationalAgentManager` alongside it. Unity converts the player's 16 kHz source audio to the output device rate, so the sound card does not need to run at 16 kHz.

### Step 4: Select and confirm the local voice

1. Select **`Ubiq-Genie/SoVITSReceiver`** in the Hierarchy.
2. In **SoVITS Voice Selector**, verify `Local UUID`, `Room UUID`, and `Backend UUID` are available.
3. Choose a filename/language entry from **Voice**, then click **Apply Voice**.
4. Wait for **Confirmed voice** to show the selected recording. Node also logs `Voice binding bound for ...`. A pending request or `Remembered voice` alone is not a confirmed current binding.

Leave **Restore Previous Voice** enabled to restore the last successfully acknowledged choice after Play restarts, UUID changes, or Node restarts. The preference stores a voice ID in local PlayerPrefs, scoped to the Unity project and room; Node still confirms each new binding. Selection changes only a session override and does not rewrite `profiles.json`.

The Inspector operates on the **local client only**. On another computer or Unity project, select the voice there. No game-view or VR voice picker is provided. First use has no automatic default choice. If the reference path, transcript, or language changes, select the updated voice again because its ID changes. Multiple backends, an offline backend, or a removed voice must be resolved before binding can succeed.

Manual fallback: map the current `Peer UUID` to a reference entry in the file. That approach requires updating the key after UUID changes and does not itself save an Inspector preference.

### Step 5: Test one utterance

Speak a clear Chinese or English sentence, then pause for about two seconds. Wait for its translation before repeating. The first real utterance may be substantially slower because of model loading and first-use work. Do not assume HTTP readiness has removed this delay.

The default VAD ends a sentence after 700 ms of silence, retains 300 ms of preceding audio, requires at least 300 ms of speech, and caps a segment at 15 seconds. Up to eight completed sentences wait behind the active task; overflow drops new sentences with a log message. Only Chinese/English recognition results are translated.

Successful synthesis produces a JSON `type:"translation"` log with the source text, translated text, durations, and stage timings. With diagnostics enabled, `Audio sent` reports transmission. Confirm the final result by listening in Unity: neither a binding acknowledgement nor successful sending proves audible playback.

## 7. Stop and restart

For a normal shutdown:

1. Exit Unity Play mode to stop live capture into this session.
2. Press **Ctrl+C** in the Node terminal to stop the app and its ASR/conversion children.
3. Stop this app's model backends from the `Node` directory:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File .\apps\gpt_sovits_translation\scripts\stop-backends.ps1
   ```

The stop script verifies recorded process identities and manages only this app's backends. It does not stop the Node app or unrelated Python/Ollama/Unity processes. Use the same `-RuntimeRoot` if installation used a custom runtime.

To restart **only Node** during diagnostics, keep Unity in Play, stop Node with Ctrl+C, and run the same npm command again. Existing backends can stay running. The selector requests its remembered voice when the replacement backend appears; wait for fresh confirmation before speaking. New speech while the app is running can trigger model inference.

## 8. Logs, latency, and troubleshooting

Backend files are under `Node/.runtime/gpt_sovits_translation/logs/`:

- `gpt-sovits.out.log` and `gpt-sovits.err.log` for synthesis startup and errors.
- `ollama.out.log` and `ollama.err.log` for translation service issues.
- Stage-specific installation logs when setup fails.

Node output is normally only in its terminal. Copy the relevant lines before closing it. To enable aggregate input and pipeline diagnostics, stop Node and restart it from `Node` with:

```powershell
$env:SPEECH_TRANSLATION_DIAGNOSTICS = '1'
npm.cmd start gpt_sovits_translation
```

To disable diagnostics for the next launch:

```powershell
Remove-Item Env:SPEECH_TRANSLATION_DIAGNOSTICS -ErrorAction SilentlyContinue
```

Check the pipeline in order: **profile -> input frames -> Utterance -> ASR -> Translation -> TTS -> Audio sent -> Unity playback**. With `profile=missing`, `frames=0` does not by itself diagnose a faulty microphone: the service intentionally ignores unbound users.

| Symptom | Action |
| --- | --- |
| PowerShell blocks `npm.ps1` | Run `npm.cmd start gpt_sovits_translation` |
| `Cannot find package '@ucl-vr/ubiq'` | Run `npm.cmd install` in `Node`; confirm installation succeeded before retrying |
| Missing Conda, C++ compiler, or jieba_fast wheel | Check `-CondaExe`, install the C++ tools, and explicitly enable `-BuildJiebaFast` for the initial build |
| Missing installation artifact / CUDA DLL / CTranslate2 failure | Inspect setup logs and manifests; verify both Python environments and their installed dependencies; no automatic CPU fallback exists |
| Port 11435/9880 occupied | Identify the listener; use this app's stop script only for its owned services. Do not terminate arbitrary processes |
| `RoomJoiner.DoJoin` null reference at Guid access | Assign `SoVITSRoomGuid.asset` to RoomJoiner > Guid; exit Play and reload the scene after a serialized repair |
| Unity cannot resolve assistant controller/volume types | Resolve the actual Console compilation error and sample assembly/namespace references before testing playback; do not replace shared scripts blindly |
| `No voice profile` / `profile=missing` | Select and apply a catalogue voice for `Local UUID`, or add a valid current UUID file mapping |
| No dropdown voices | A WAV folder alone is insufficient. Check validated profile entries, Node readiness, and matching room/server settings |
| `Voice profile reload rejected` | Fix JSON commas/quotes, UUID keys, WAV paths, durations, and text/language fields; wait for a successful reload |
| Remembered voice unavailable | Select the current entry after reference material/path changes; wait for a new acknowledgement |
| Multiple backends / binding never confirms | Stop duplicate translation apps in the room; inspect Node's binding errors and room connectivity |
| Valid profile but zero frames | Check the selected Windows microphone, desktop-app microphone permission, Unity capture, and WebRTC connection |
| Frames arrive but peak stays near -80 dBFS or `-inf` | Input is nearly silent. Test an actual Windows recording, check hardware mute/default input, and compare Unity microphone diagnostics |
| ASR/translation succeeds but TTS fails | Check `gpt-sovits.err.log`, local reference file access, exact prompt text/language, and backend readiness |
| `Audio sent` but silence or faint/incomplete speech | Check Unity mute/listener/output, assistant distance and volume, receiver references, and whether the synthesized audio/reference itself is valid |
| Wrong speed or pitch | Match Node output and SoVITSAudioSource source rate; keep AudioSource pitch at 1 |
| Very slow first sentence, then high `queueMs` | Separate cold model work from waiting time. Stop submitting repeated utterances while diagnosing the first one |
| Echo or repeated translations | Use headphones and stop microphone capture of loudspeaker playback |

Unity also provides **Tools > SoVITS > Log Microphone Diagnostics (5 Seconds)** in Play mode. Speak during the observation, then inspect `[SoVITS Diagnostic]` Console messages. This is a logging command, not a popup window. It inspects existing capture; a running Node app can still process the speech normally.

Timing fields separate `queueMs`, `asrMs`, `translationMs`, `ttsMs`, and `convertMs`. `endpointToPlaybackMs` ends at the start of Node audio transmission; it excludes VAD silence detection and is **not** a measurement of sound reaching the headphones. The default Unity prebuffer is 200 ms and its PCM buffer cap is two seconds. Check a second isolated sentence after the first completes before judging steady-state latency.

## 9. Validation and related guides

For a two-client acceptance check, each client should confirm its own voice, then take turns testing Chinese -> English and English -> Chinese. Verify the intended voice and complete playback on both clients, original remote speech, and automatic rebinding after Play and Node restarts. Record the Node request ID, stage timings, and the actual listening result. Component readiness and unit tests do not replace this check.

Historical baseline: on **2026-10-20**, the app type check, **36 Node tests**, Unity compilation, and **77 EditMode tests** passed. These results predate subsequent motion-scene changes and are not a new validation of the current working tree. The user confirmed clear single-user translated playback on 2026-10-19; live Inspector restoration and two-user acceptance remain separate checks. No compilation, inference, or live test was run while preparing this guide.

Related local documentation:

- [App README](Node/apps/gpt_sovits_translation/README.md): implementation, installation records, configuration, and test commands.
- [Unity integration](Unity/Assets/Ubiq-Genie/Apps/RealtimeTranslation/INTEGRATION.md): scene references, playback framing, selector protocol, and diagnostic details.
- [Optional RealtimeTranslationMotion scene](Unity/Assets/Ubiq-Genie/Apps/RealtimeTranslation/MOTION_REPLAY.md): a separate scene using the same backend, avatar-based translated playback, and remote movement replay. Its object names and playback behavior differ; this guide's main procedure uses the standard assistant scene.
- [Framework README](README.md) and [Node architecture](Node/README.md): general project setup and reusable components.
