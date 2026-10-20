# GPT-SoVITS 中英翻译语音

[English](README.md) | [简体中文](README.zh-CN.md)

接收 Ubiq 房间用户的语音，在停顿后自动识别中文或英文，翻译为另一种语言，再用该用户预设的参考音色向全房间播报。原始 Ubiq 通话保持工作。所有 AI 推理在本机运行。

本 app 是新增实现，沿用 [Node 架构](../../README.zh-CN.md) 的 ApplicationController、VoipReceiver 和 ServiceController；后端实现不修改已有 app、共享服务或共享 Unity 代码。

**RealtimeTranslation 场景请参阅 [Unity 接线说明（英文）](../../../Unity/Assets/Ubiq-Genie/Apps/RealtimeTranslation/INTEGRATION.md)。** 该场景使用 `SoVITSReceiver`、按输入采样率播放的 `SoVITSAudioSource`，以及在 Unity Inspector 中为本机用户选择音色的 `SoVITSVoiceSelector`；下文另行标明旧版 ConversationalAgent 播放器的设置。

## 当前验证状态

- 实现与环境安装记录分开管理。实际安装结果见运行目录的 `asr-manifest.json`、`receipts/` 和 `logs/`。
- 本机已完成两个独立 Python 3.11 环境、所列模型和工具的安装；两个环境的 `pip check` 均通过。模型文件按固定上游 revision 校验，安装版本与哈希已写入回执。
- 2026-10-20，app 类型检查及 **36 项 Node 测试全部通过**：helpers 16 项（含实际 FFmpeg 转换）、调度器／app 12 项、音色绑定协议 8 项。localhost 模拟测试在隔离环境外运行，因为隔离环境禁止了回环连接和临时文件替换。此前 **19 项 Python worker 测试通过**为历史结果；本次音色选择未修改 Python。
- 当前 app 类型检查通过。全仓库构建仍受 `dynamic_capture_yolo` 和 `yolo_only_capture` 中旧 `ubiq-server/ubiq` 导入阻挡；针对修改文件的 ESLint 因现有 `.eslintrc.json` 格式错误而无法启动。这些无关文件未修改。
- RealtimeTranslation 接收器已通过 Unity **6000.0.61f1** 编译及 **52/52 个 EditMode 无模型测试**。2026-10-13 获准现场诊断，已确认 Unity 麦克风音频到达 Node、VAD 切句及中文 ASR 成功。翻译失败的原因是可变标签 `qwen3:4b` 指向 Thinking 2507；修正模型见下文。
- 修正后，原翻译客户端在关闭 thinking 时返回了纯中文译文（含首次加载 22.229 秒）。另行调用 GPT-SoVITS，使用用户参考音色生成了有效的 4.08 秒、32 kHz 单声道 PCM16 英文 WAV（含首次处理 51.396 秒）。这些组件检查不代表 Unity 已经播放，也不代表预热后的延迟。
- 本地已准备 5.50 秒英文参考片段 `recordings/my-voice-reference.wav`，并自动转写匹配文字。手工文件绑定仍需核对当前会话 UUID；Inspector 选择会在后端确认后为新 UUID 恢复已记住的音色。
- 2026-10-19，修正中文参考录音后，用户已确认听到清晰译音。双用户联调、峰值显存及新增 Inspector 流程仍待现场验证；后续实测继续遵守项目授权要求。
- 2026-10-20，Unity **6000.0.61f1** 编译通过，**77/77 项 EditMode 测试通过**：PCM 52 项、音色选择状态 25 项。测试覆盖选择、确认匹配、恢复、失效和错误状态，不加载模型或连接房间；实际 Inspector 操作及跨 Editor 重启的 PlayerPrefs 保存仍待运行验证。结果见 [接线说明](../../../Unity/Assets/Ubiq-Genie/Apps/RealtimeTranslation/INTEGRATION.md)。

## 架构与模型

```text
Unity/WebRTC PCM16
  → 每位用户独立的重采样器与 Silero VAD
  → 完整语句 FIFO（最多 8 句等待 + 1 句处理中）
  → 共享 Faster-Whisper large-v3-turbo（CUDA FP16）
  → Ollama qwen3:4b-q4_K_M（无聊天历史，关闭 thinking）
  → GPT-SoVITS v2Pro /tts（每次请求携带用户参考录音）
  → FFmpeg 转 PCM16LE mono
  → NetworkId 95 → SoVITSReceiver → Unity 助手音源
```

识别只执行 `transcribe`；中文→英文、英文→中文均由 Qwen 负责翻译。Silero 使用 Faster-Whisper 自带的 ONNX 模型在 CPU 上运行。所有用户共享一个 GPU 识别模型，音频缓存和 VAD 状态按 UUID 隔离。

默认静音 700 ms 后切句，前置保留 300 ms，语音不足 300 ms 丢弃，连续语音达到 15 秒强制切段。只翻译识别为 `zh` 或 `en` 的结果；短句和中英混说仍可能发生语言误判。

完整语句按结束顺序排队，识别、翻译、合成和发送按句处理；音频采集不会等待推理。队列满时丢弃新完成的句子并记录原因。没有无限缓存或自动重复播报；语速持续超过处理能力时会出现丢句。

麦克风采样率或声道变化时，仅重建该用户的实时音频状态，已完成句子仍保留。输出按音频时长发送，并将本 app 的 TCP 写入缓冲限制在约 1 秒 PCM；网络拥堵时丢弃当前译音剩余部分并记录错误。房间拒绝加入或连接关闭时停止 app 和 ASR。

## 安装

在 `C:\ubiq-genie\Node` 打开 PowerShell。Node 依赖沿用项目已经安装的依赖；无需新增 npm 包。

脚本默认使用已有 `C:\ProgramData\anaconda3\Scripts\conda.exe` 创建两个 Python 环境，可通过 `-CondaExe` 指定其他 Conda。中文前端 `jieba_fast==0.53` 没有官方 Python 3.11 Windows wheel；本机已获准使用现有 Visual Studio 2022 C++ 工具构建，回执为 `receipts/jieba-fast-build.json`。从空运行目录安装时，需要可用的 Visual Studio C++ Build Tools，并明确启用这一包的编译：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\apps\gpt_sovits_translation\scripts\setup.ps1 -BuildJiebaFast
```

脚本将环境、下载与模型放在 `Node/.runtime/gpt_sovits_translation`，包含 ASR 和 GPT-SoVITS 两个 Python 3.11 环境、CUDA 12.8 路线的 TTS 依赖、GPT-SoVITS v2Pro 推理权重、Faster-Whisper、独立 Ollama 和 FFmpeg。安装不执行模型推理。

已存在该 wheel 时可省略 `-BuildJiebaFast`；默认脚本不会自动进行原生编译。TTS 依赖范围限定为本 app 的中英推理，省略训练工具及日／韩／粤语前端；保留下载的上游源码。该环境不作为完整的 GPT-SoVITS 训练或多语言 WebUI 安装使用。

下载需要联网；日常推理使用本地文件和回环接口。安装脚本固定并记录实际使用的源码 commit、模型 revision 和依赖版本。具体可选参数请查看脚本开头的 `param` 定义。不要手动把已有项目的虚拟环境或 GPT-SoVITS 安装覆盖到本目录。

| 组件 | 固定版本／记录位置 |
| --- | --- |
| GPT-SoVITS | `20250606v2pro`，commit `d7c2210da8c013e81a94bfc7b811a477c99fd506`；`receipts/tts-source.json` |
| v2Pro 权重及中文资源 | 模型 revision `8f8857563c040d542de108a2c18b0425abd6c9c5`；`receipts/tts-models.json` |
| PyTorch / torchaudio | `2.7.1+cu128`；完整 TTS 依赖见 `receipts/tts-packages.txt` |
| Faster-Whisper | `1.2.1`、CTranslate2 `4.8.2`；模型 revision `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf`；`asr-manifest.json` |
| Ollama / Qwen | Ollama `0.40.3`、`qwen3:4b-q4_K_M`；固定模型 digest `2bfd38a7daaf4b1037efe517ccb73d1a3bbd4822cf89f1a82be1569050a114e0` |
| FFmpeg | `9.0.2`；下载哈希见 `receipts/ollama-ffmpeg.json` |

原先可变的 `qwen3:4b` 标签在本机安装时指向了 **Qwen3 4B Thinking 2507**。其模板即使收到 app 的 `think:false` 请求仍会开启推理块，导致短句翻译也生成大量内容。app 现改用支持关闭 thinking 的混合模式 [qwen3:4b-q4_K_M](https://ollama.com/library/qwen3:4b-q4_K_M) 版本。安装脚本校验上表中的完整清单 digest，标签变化时会停止安装。原 `receipts/ollama-ffmpeg.json` 保留为历史安装记录；本工作区的模型修正单独记录在 `receipts/ollama-model-correction.json`。全新安装则在 `receipts/ollama-ffmpeg.json` 中记录修正后的模型。选定模型本身不代表翻译或播放已通过验证。

## 启动与停止（运行前需批准）

先启动两个本地后端，再启动 Node app：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\apps\gpt_sovits_translation\scripts\start-backends.ps1
npm.cmd start gpt_sovits_translation
```

等待出现 `Both backend HTTP endpoints are ready` 后再启动 Node。启动脚本为每个后端最多等待 10 分钟；若进程退出或 HTTP 接口持续不可用，会报错并显示日志路径。这些就绪检查不执行翻译或语音合成。

首次启动沿用项目 CLI 的 Ubiq 服务器配置流程。`config.json` 的房间 GUID 和服务器地址必须与 Unity 客户端一致。若需重新配置：

```powershell
npm.cmd start gpt_sovits_translation configure
```

使用 `npm.cmd` 可避免 PowerShell 调用被执行策略阻止的 `npm.ps1`，无需更改系统执行策略。

ASR worker 由 app 启动；GPT-SoVITS 默认监听 `127.0.0.1:9880`，Ollama 监听 `127.0.0.1:11435`。后端未启动时，句子请求会报错，之后的句子仍可继续处理。ASR worker 崩溃或推理超时会停止 app，避免旧 GPU 任务尚未退出又开始新的任务。

Ctrl+C 停止 app 和它的 ASR/转换子进程；本地后端通过以下命令停止：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\apps\gpt_sovits_translation\scripts\stop-backends.ps1
```

停止脚本只处理本 app 保存并验证过进程身份的后端，不停止其他 Python、Ollama 或 Ubiq 进程。

## Unity 播放设置

使用 `Unity/Assets/Ubiq-Genie/Apps/RealtimeTranslation/RealtimeTranslation.unity` 时，按 [Unity 接线说明](../../../Unity/Assets/Ubiq-Genie/Apps/RealtimeTranslation/INTEGRATION.md) 配置。该场景已接入 Nexus（`nexus.cs.ucl.ac.uk:8009`）和房间 `6765c52b-3ad6-4fb0-9030-2c9a05dc4731`。保持 Node 的 `translation.outputSampleRate` 与 `SoVITSAudioSource.sourceSampleRate` 一致，默认均为 **16000**。Unity 负责转换到设备输出采样率，不需要把该播放器的源采样率改成声卡频率。

旧版兼容路径：使用现有 `Unity/Assets/Ubiq-Genie/Apps/ConversationalAgent/ConversationalAgent.unity` 场景，或已经配置好 `ConversationalAgentManager` 与 `InjectableAudioSource` 的场景。仅对此旧播放器，Node 输出必须与 Unity 实际 `AudioSettings.outputSampleRate` 一致；默认 **16000** 匹配当前 `Unity/ProjectSettings/AudioManager.asset`，联调时应查看 Unity 的 `Output sample rate` 日志。普通 Transcription 场景不自动具备音频接收器。

- 不同时启动向同一房间 `NetworkId(95)` 发送音频的其他助手 app，否则流会混合。
- 每位成员都会听到原声及之后的译音，包括说话者本人；译音来自现有助手音源，不是每位用户头像的位置。
- `targetPeer` 固定为空，仅为兼容既有音频头，不表示定向发送。音频头少于 100 字节，PCM 包为 200–16000 字节，末尾不补零、不截断。
- 使用耳机避免扬声器译音再次进入麦克风。此版本不增加回声消除或原声静音控制。

## 在 Unity Inspector 中选择音色

RealtimeTranslation 场景的现有 `Ubiq-Genie/SoVITSReceiver` 对象上挂有 `SoVITSVoiceSelector`。其自定义 Inspector 自动显示本机客户端当前 UUID，仅为本机用户选择音色；没有新增游戏内或 VR 选择界面。

获准运行验证后：

1. 在同一房间启动 Node app，并进入 Unity Play 模式。
2. 选中 `Ubiq-Genie/SoVITSReceiver`，找到 `SoVITSVoiceSelector` Inspector，等待单个 Node 后端发布音色目录。
3. 从下拉列表选择参考音色；条目显示录音文件名和语言。然后点击 **Apply Voice**。
4. 等待 Node 返回与本次请求匹配、状态为 `bound` 的确认。等待中的请求不代表绑定成功；只有收到该确认才会将音色 ID 保存至本地 `PlayerPrefs`，按 Unity 项目和房间隔离。

Node 从 `profiles.json`（或 `translation.profilesPath`）中通过校验的条目生成目录。完整 `(refAudioPath, promptText, promptLanguage)` 三元组相同的条目共享一个 SHA-256 音色 ID 和下拉选项，即使它们使用不同的 UUID 键。原有 UUID 文件及其格式保留不变，选择操作不会重写文件。有效配置为空时，没有可选音色，也不会选择默认音色。

确认后的选择成为当前 UUID 的内存覆盖配置，优先于该 UUID 的文件映射。没有通过 Inspector 选择的客户端（包括旧场景）继续使用原有 UUID 绑定方式。正常切换音色只影响新句子，排队中的句子保留原音色快照。

UUID 变化、再次进入 Play 或 Node 重启后，当单个兼容后端和同一音色 ID 可用时，会自动重新请求上次确认的选择。恢复仍须收到新的确认，已保存的偏好不代表当前会话已绑定。修改参考路径、文字或语言会改变音色 ID，需要重新选择。所选音色从发布目录消失时，相关内存覆盖失效，受影响的待处理任务取消；选择器不会猜测替代音色。

后端离线、存在多个后端、已保存音色缺失或请求被拒绝时，Inspector 会显示错误；解决所报告的问题后再应用或恢复。选择请求、确认及小型音色目录通过现有 Ubiq peer properties 经 Nexus 传递。仅交换音色 ID、显示标签／语言和绑定元数据；WAV、完整路径和参考文字留在 Node 本机。这不增加通道 95 音频消息格式，也不改变 Python／模型接口。

## 配置参考音色与旧版 UUID 绑定

用户加入后，Node 会显示 `Peer UUID: ...; name: ...`。把该 **UUID** 作为 `profiles.json` 的键，参考 `profiles.example.json` 的结构填写：

```json
{
  "这里替换为日志中的真实UUID": {
    "refAudioPath": "这里填写真实参考WAV路径",
    "promptText": "这里填写录音实际说出的完整文字",
    "promptLanguage": "zh"
  }
}
```

以上是结构说明，不能直接作为有效配置。每位用户需单独填写一项，录音要求：

- 干净、单人、3–10 秒的 PCM 或 IEEE-float WAV；建议单声道，避免背景音乐。
- `promptText` 与录音内容准确对应，`promptLanguage` 为录音本身的 `zh` 或 `en`，不必与输出语言相同。
- 路径可使用绝对路径，或者相对于 `profiles.json` 的路径；Windows JSON 路径建议使用 `/`，或者将 `\` 写为 `\\`。
- 路径必须能由本机 GPT-SoVITS 进程读取；该 API 使用服务器文件路径，不上传音频。

保存后热重载：整份配置验证成功才替换当前配置；编辑到一半的无效 JSON 或错误录音会被拒绝并保留上一份有效配置。排队中的句子保留原音色快照，新句子使用新配置。删除用户实际生效的音色绑定会取消其待处理任务；目录中的音色被移除也会使相关 Inspector 覆盖失效。初始配置无效时 app 不启动。

用户离房会取消待处理句子和播放。如果后端已开始计算，会等待当前请求完成后丢弃其结果，再处理下一句；HTTP 连接中断本身不能保证后端 GPU 立即停止。每次后端请求仍有配置的超时上限。

UUID 是当前连接身份，重启客户端或重新加入后可能改变。手动文件绑定需要更新 UUID；Inspector 则为新 UUID 请求恢复上次确认的音色。显示名不参与匹配。`profiles.json` 留空时不选择默认音色。如果希望把个人映射排除在版本控制之外，可把 `translation.profilesPath` 改为已被忽略的 `./profiles.local.json`。

## 日志与故障处理

无声音时，可用 Ctrl+C 仅停止 Node app，再启用输入诊断重新启动。Unity 保持当前 Play 会话，以保留 UUID：

```powershell
$env:SPEECH_TRANSLATION_DIAGNOSTICS = '1'
npm.cmd start gpt_sovits_translation
```

`[Diagnostic] Input` 每 10 秒报告各用户通过校验的 PCM 帧数、峰值／RMS 音量（dBFS）以及距离上一帧的时间。`profile=valid` 时，`frames=0` 表示该时间段内翻译服务未收到有效 PCM，应检查 Windows 默认麦克风、麦克风权限和 Ubiq WebRTC 连接。有帧但 `peakDbFS=-inf` 表示数字静音；音量变化说明收到了信号，但不保证语音清晰可识别。`Utterance`、`ASR completed`、`Translation started`、`TTS started`、`Audio sent` 表示流程所到达的阶段；`Audio sent` 不代表 Unity 已经播放。输入诊断仅保留汇总计数，不保存录音。下次启动 Node 前执行 `Remove-Item Env:SPEECH_TRANSLATION_DIAGNOSTICS` 可关闭诊断。现场说话可能触发模型推理，仍须遵守项目的实测授权要求。

Unity 的 Ubiq 使用 Windows 默认录音设备。本地默认不回放自己的原声，因此听不到原声回声是正常行为；译音由助手单独播放。

每个成功生成的句子记录 `requestId`、`peerUuid`、语言方向、原文、译文、输入／输出时长及队列、识别、翻译、合成、转换耗时。`endpointToPlaybackMs` 从切句完成到开始发送译音计算，包含排队时间但不包含静音判定的等待；不等同于用户停止发声时刻到耳机出声的实测延迟。

| 现象 | 检查 |
| --- | --- |
| 只出现 UUID，无译音 | 当前会话是否收到 Inspector 的 `bound` 确认，或当前 UUID 是否存在有效文件映射；检查参考录音和文字 |
| Inspector 无法应用音色 | 查看状态：Node 需在同一房间在线，只能有一个兼容后端发布目录，所选音色需仍然存在 |
| 重连后已保存音色缺失 | 检查已校验参考路径、文字或语言是否变化；重新选择当前目录条目并等待新的 `bound` 确认 |
| 初始音色校验失败 | 是否错误复制了未填写的 example；录音是否在 3–10 秒内 |
| `/api/chat` 或 `/tts` 连接失败 | 启动后端，核对回环地址和端口，查看后端日志 |
| CUDA DLL / CTranslate2 错误 | 独立环境与 CUDA 运行库是否完整，检查安装记录；不自动切 CPU |
| 语音变速 | RealtimeTranslation：Node 输出与 `SoVITSAudioSource.sourceSampleRate` 一致、pitch 为 1；旧 InjectableAudioSource：Node 输出与 Unity 实际 `AudioSettings.outputSampleRate` 一致 |
| 译音反复被识别 | 使用耳机，检查麦克风是否采集了扬声器回放 |
| 过载丢句 | 日志中查看队列等待与阶段耗时，减少同时发言者或降低发言频率 |

## 验证命令与待验收项目

以下无模型单元测试已获批准；不是模型推理或安装步骤：

```powershell
# 从 Node 目录运行不加载模型的 TypeScript 单元测试
$env:SPEECH_TRANSLATION_TEST_FFMPEG = "$PWD\.runtime\gpt_sovits_translation\tools\ffmpeg\bin\ffmpeg.exe"
node --loader ts-node/esm --test .\services\speech_translation\tests\helpers.test.ts .\services\speech_translation\tests\scheduler.test.ts

# 完整 app 类型检查（当前会报告上面记录的既有 Ubiq 类型错误）
.\node_modules\.bin\tsc.cmd -p .\apps\gpt_sovits_translation\tsconfig.check.json

# Python worker 的隔离单元测试（不加载真实模型）
.\.runtime\gpt_sovits_translation\asr-env\python.exe -m unittest discover -s .\services\speech_translation\providers\faster_whisper -p "test_*.py"
```

测试覆盖分帧、UTF-8 拆包、音频包边界、48 kHz 双声道→16 kHz 单声道转换、配置快照及无效热重载、HTTP 失败／超时／取消、按用户隔离的 VAD、句子顺序、8 句等待上限、离房取消及迟到结果丢弃。完整验收还需：

1. 两名用户分别提供参考录音，检查中文→英文、英文→中文及用户音色匹配。
2. 检查双人并发、离房取消、缺失参考文件、未知语言、空识别、后端停机恢复和队列过载。
3. 在实际 VR 场景中检查 CUDA 放置、峰值显存、每阶段耗时与端到端播放延迟，并核实音频无截断、变速和重复。
4. 检查 Inspector 选择收到确认后才保存、新 UUID 和 Node 重启后的自动恢复、音色缺失／变化、多后端错误及两名客户端独立选择。上述历史测试未验证这些新增选择行为。

获准实测后，可在另一个终端采集 GPU 时间序列，与 Node 每句的 `requestId`、耗时日志对应；当前尚无实测数据：

```powershell
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total --format=csv -l 1 -f .\.runtime\gpt_sovits_translation\logs\gpu.csv
```

## 上游资料

- [GPT-SoVITS 中文说明](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/docs/cn/README.md)
- [GPT-SoVITS API](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/api_v2.py)
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
- [Ollama Windows](https://docs.ollama.com/windows)
- [Ollama Chat API](https://docs.ollama.com/api/chat)
