# RealtimeTranslationMotion

Open `RealtimeTranslationMotion.unity` for the remote movement replay experiment. It is a copy of `RealtimeTranslation.unity`, with its own `SoVITSServerConfigMotion.asset` and `SoVITSRoomGuidMotion.asset`. Both scenes use the existing `gpt_sovits_translation` Node app, reference voices, and Inspector voice selector. Follow [the existing integration guide](INTEGRATION.md) for backend setup and voice selection.

The original scene and its PCM player are preserved. The new scene disables the copied assistant and plays translations from the speaking user's avatar instead.

## Behavior and controls

Select `Ubiq-Genie/TranslationMotionReceiver` in the new scene:

| Component / setting | Default and behavior |
| --- | --- |
| RemoteAvatarReplay | Records only remote avatars: 30 seconds of history at 60 Hz |
| Blend Seconds | 0.2 seconds into replay and back to the current live pose |
| Max Pinned Clips | 10 retained utterance clips across the client |
| OriginalVoiceToggle / Play Original Voice | ON; Inspector-only toggle for this listener's incoming original VoIP |
| MotionAudioPlayer / Source Sample Rate | 16000 Hz; must match Node `translation.outputSampleRate` |
| Prebuffer Milliseconds | 200 ms, or a fully received shorter sentence |
| Max Buffer Milliseconds | 2000 ms, including scheduled audio that has not yet played |

Change history/capture or audio-buffer settings before entering Play mode, or disable and re-enable the relevant component to rebuild its buffers. The original-voice toggle can change during Play mode.

Other people remain live until their translated audio starts. During translation, their recorded movement replays at the original room positions, including walking. Its duration is scaled to the translated sentence. A rendering-only copy shows the replay while the live network avatar continues receiving poses and producing new recordings underneath. At completion, the visual blends back to the current live position.

The local headset, camera, controllers, avatar poses, and locomotion are never driven by replay. The speaker still hears their own translated audio through nonspatial playback. The original-voice toggle changes only receive AudioSources; it does not mute outgoing microphone data or translated playback.

The supported Ubiq Floating Avatar has a head, procedural torso, and two hands. The replay includes the rendered hand bones, preserving grip animation. It does not add legs, full-body tracking, or semantic gesture generation. Live name/voice indicator renderers are hidden during replay rather than showing stale indicators at the live avatar's other position.

## Timing and bounded retention

“30 seconds” means rolling history, not a fixed playback delay or an inference deadline. At queue admission, the client retains the movement for that utterance separately. Retained clips survive inference and queue delays longer than 30 seconds. They are released after local playback finishes or the request is cancelled, skipped, or fails. When all ten slots are occupied, a newly admitted sentence uses audio-only playback; the currently playing clip is never evicted.

The existing VAD segmentation remains unchanged: up to 15 seconds including preroll and trailing silence. Node estimates its interval as `[utterance receipt time - duration, utterance receipt time]` using `performance.now()`. A ping/pong exchange maps this monotonic timeline to Unity's monotonic timeline. Three startup probes are followed by a probe every five seconds; the client chooses the lowest-round-trip fresh sample from its last eight observations.

This is **approximate sentence-level alignment**. Clock mapping cannot remove independent WebRTC and pose-network transit delays or worker scheduling delay. The waveform's preroll and trailing silence are included. The system does not align translated word order to individual gestures.

The player schedules source-rate PCM chunks on Unity's DSP timeline. Motion progress follows the scheduled real-audio samples rather than packet arrival, the next sentence's header, or audio callback prefetch. During an underrun, already scheduled audio finishes, motion holds, and playback resumes after prebuffering. A short final chunk retains its exact length. Audio device conversion remains Unity's responsibility. Hardware/driver latency still requires an audiovisual runtime measurement.

## Network interface

Run only one translation backend per room. NetworkId **95** retains its existing strict `AudioInfo` header and raw mono PCM16LE packets. NetworkId **101** carries version 1 JSON metadata, using the same ordered Ubiq TCP connection.

Node advertises peer property `ubiq.genie.sovits.motion` with `{ "version": 1, "sessionId": "..." }`. The client uses the advertising peer UUID as the backend identity. A changed backend session clears prior timing and replay state.

Every lifecycle message contains `version`, `type`, `backendUuid`, `sessionId`, `requestId`, and `speakerUuid`:

| Type | Additional fields | Meaning |
| --- | --- | --- |
| `accepted` | `sourceStartMs`, `sourceEndMs` | Queue admission, before inference; retain the matching movement now |
| `play` | `audioLength`, `sampleRate`, `sampleCount` | Identity of the immediately following channel 95 header; audioLength is a numeric byte count here |
| `terminal` | `status` | `sent`, `cancelled`, `skipped`, or `failed` |

`sent` means transmission has finished, not that the client's remaining audio has played. The retained clip stays until the local player completes it. Queue rejections before admission do not create retained clips.

Clock `ping` messages contain `version`, `type`, `backendUuid`, `sessionId`, `peerUuid`, `nonce`, and client monotonic `t0` in milliseconds. `pong` echoes those identifiers and adds Node monotonic receive/send times `s1` and `s2`. The requesting client captures `t3` and estimates server-minus-client offset as `((s1-t0)+(s2-t3))/2`. Other clients ignore that response.

No Python/model protocol or new configuration field is required. The original scene ignores channel 101.

## Recovery

- Missing history, unsupported avatars, unavailable clock estimates, and retained-clip capacity limits produce audio-only playback at the live speaker's head.
- Unusable speaker metadata produces nonspatial audio. The client never guesses a speaker from the preceding sentence.
- Multiple advertised motion backends disable motion association; remove the extra backend. As with the original scene, concurrent channel 95 senders can corrupt the untagged audio stream.
- Cancellation stops affected scheduled audio and blends the remote avatar back to live. Departure, voice invalidation, room changes, backend replacement, and component shutdown release affected state.
- Audio device changes clear playback and require the next valid audio header. They do not change the project's audio settings.
- Buffers and clips exist only in client memory. No motion or audio recordings are written to disk.

## Validation

The new EditMode test assembly is `Ubiq.Genie.Samples.RealtimeTranslationMotion.Tests`. It covers the rolling history, retained clips, interpolation, local exclusion, clock estimates, audio metadata/framing, DSP timelines, underruns, cancellation, scene wiring, and cleanup. Run it with the existing `UbiqGenie.RealtimeTranslation.Tests` assembly to retain legacy PCM and voice-selection coverage.

Use the Unity version in `ProjectSettings/ProjectVersion.txt`: **6000.0.61f1** for this workspace. Node's translation unit tests and app type-check remain separate from live ASR/translation/TTS execution.

Two-client VR acceptance remains necessary:

1. Bind a reference voice on each client and verify translation in both directions.
2. Speak while moving hands, gripping, turning, and walking. Verify replay at the original room positions on the listener and uninterrupted local tracking on the speaker.
3. Compare shorter and longer translated utterances; verify blend transitions and speech position.
4. Toggle original voices in one client's Inspector. Confirm only that listener's original voice playback changes and both microphones still reach Node.
5. Verify own translated feedback, queued sentences, inference delays beyond 30 seconds, and missing-history audio-only fallback.
6. Test cancellation/departure, backend restart, room changes, and audio-device reset. Check for stuck hidden avatars or residual speech.
7. Measure audiovisual offset and buffer/proxy memory with the actual avatars and devices. Offline tests do not establish headset comfort, acoustic timing, or live WebRTC mute behavior.

Offline verification on 2026-10-20:

- Unity **6000.0.61f1** compiled the scripts and passed **128/128 EditMode tests**: 51 new motion/audio/scene tests and 77 existing translation/voice-selection tests. The scene loaded successfully with valid references, one active translation receiver, matching independent room settings, and the existing voice selector.
- The headless tests used a temporary validation project with copies of the current assets/settings and the installed package sources. This avoided interrupting the already-open editor. Results are in `Unity/Logs/SoVITSVerification/20261020-motion/results-verified.xml`; the corresponding editor log is `validation-verified.log` in that directory.
- The original translation scene remained byte-for-byte unchanged. Project settings and dependency manifests were preserved.
- The installed floating-avatar prefab has 35 captured transforms and 16 skinned hand-bone references. Its 1,802-frame ring has **2,544,424 bytes (about 2.43 MiB)** of pose-array/timestamp payload per remote avatar. This is an offline layout measurement, excluding managed-object overhead, rendering copies, and retained utterance clips.
- The translation app's TypeScript check passes. The full Node build still fails on existing `ubiq-server/ubiq` imports in the unrelated `dynamic_capture_yolo` and `yolo_only_capture` services. Scoped lint cannot load the existing `.eslintrc.json` because of its trailing comma at line 19. These unrelated files were left unchanged.
- Node's full regression suite passed **45/45** before the final terminal-retry addition. The final sideband test file then passed **8/8**, including both added retry tests: **47 distinct tests passed across those runs**. A final combined rerun was cancelled while awaiting execution approval, so this is not a claim of a single final 47-test run.
- The offline dependency-install attempt could not complete because `@eslint/js` was unavailable in the npm cache (`ENOTCACHED`); checks used the installed dependencies. No model or room service was started for verification.

Live two-client VR, acoustic alignment, total runtime memory, and headset comfort have **not** been validated. Use the acceptance steps above before treating the scene as runtime-verified.
