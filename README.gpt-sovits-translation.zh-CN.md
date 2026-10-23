# GPT-SoVITS 翻译语音：部署与启动

[English](README.gpt-sovits-translation.md) | [简体中文](README.gpt-sovits-translation.zh-CN.md)

本指南介绍如何在 Windows 上部署 `gpt_sovits_translation`，并使用项目内的 **RealtimeTranslation** Unity 场景。内容包括首次安装、参考录音、Inspector 音色选择、日常启动、停止和排错。命令以 `C:\ubiq-genie` 为项目目录；如果你的项目位于其他位置，请替换对应路径。

下文路径、命令和配置已于 **2026-10-23** 对照当前项目文件核查。编写本指南期间没有安装软件、启动服务、运行模型或重复进行真实房间联调。已有项目 README 保持不变。

如果环境已经安装，请直接查看[第 6 节：日常启动](#6-日常启动顺序后端---node---unity---选择音色)。新机器请依次完成第 2–6 节；如果只是新增音色，请先查看[第 4 节：参考录音与音色列表](#4-准备参考录音与音色列表)。

## 1. 本地处理与网络通信的边界

```text
Unity 麦克风
  -> Ubiq WebRTC 语音连接 -> Node VoipReceiver
  -> 按用户独立重采样与 Silero VAD
  -> 句子队列 -> Faster-Whisper -> Ollama 翻译 -> GPT-SoVITS
  -> FFmpeg -> NetworkId 95 上的 Ubiq 音频消息
  -> Unity SoVITSReceiver -> 助手 AudioSource
```

| 部分 | 运行位置／网络使用情况 |
| --- | --- |
| 麦克风、虚拟形象、音色选择、译音播放 | 各 Unity 客户端本地 |
| VAD、队列、识别、翻译、合成、格式转换 | 运行 Node 的机器；模型推理均在本地 |
| Faster-Whisper | 由 Node 启动的本地 Python worker；共享 CUDA FP16 模型，各用户使用独立缓冲 |
| Ollama | 本地 HTTP 服务 `http://127.0.0.1:11435`；模型为 `qwen3:4b-q4_K_M` |
| GPT-SoVITS v2Pro | 本地 HTTP 服务 `http://127.0.0.1:9880`；通过本地文件路径提供参考录音 |
| 房间成员、peer properties、译音消息 | 经过配置的在线 Ubiq 服务器，当前为 `nexus.cs.ucl.ac.uk:8009` |
| 原始麦克风语音 | 通过 Ubiq/WebRTC 在客户端与 Node 之间传输；当前配置使用 Google STUN 服务协助建立连接 |
| 软件和模型安装 | 需要联网下载；安装后模型文件和环境保存在本地 |

**本地推理不代表整个应用离线运行。** 使用当前 Nexus 配置时，房间通信仍然需要网络。参考 WAV 文件、完整文件路径和参考文字保留在后端机器上；音色选择器通过 Ubiq 交换音色 ID、文件名、语言和绑定状态。麦克风音频和合成译音会通过房间的网络链路传输。

Unity 不直接调用 Ollama 或 GPT-SoVITS。Unity 可以运行在另一台机器上，但本部署方式要求 Node 与这两个 HTTP 服务运行在同一台机器：app 仅接受回环地址的模型服务。原声通话继续保留，房间内所有用户（包括说话者）都会从助手听到译音。建议使用耳机，避免译音被麦克风再次采集。

## 2. 文件位置与前置条件

| 相对于仓库根目录的路径 | 用途 |
| --- | --- |
| `Node/apps/gpt_sovits_translation/config.json` | 房间／服务器、运行目录、模型接口、VAD、输出采样率 |
| `Node/apps/gpt_sovits_translation/profiles.json` | 已校验的参考音色和原有 UUID 映射 |
| `Node/apps/gpt_sovits_translation/profiles.example.json` | 结构模板；其中的占位符不是可用配置 |
| `Node/apps/gpt_sovits_translation/recordings/` | Node 机器上的参考录音 |
| `Node/apps/gpt_sovits_translation/scripts/` | 安装、启动和停止脚本 |
| `Node/.runtime/gpt_sovits_translation/` | 独立 Python 环境、模型、工具、安装记录和后端日志 |
| `Unity/Assets/Ubiq-Genie/Apps/RealtimeTranslation/RealtimeTranslation.unity` | 通过助手播放译音的标准场景 |
| `Unity/Assets/Ubiq-Genie/Apps/RealtimeTranslation/INTEGRATION.md` | 场景接线与通信协议的详细说明 |

在新的后端机器上，需要准备：

- Windows x64、PowerShell，以及带 npm 的 **Node.js 20 或以上版本**。
- NVIDIA GPU，以及支持已安装 **CUDA 12.8** 软件栈的驱动。ASR 配置要求 `cuda` / `float16`，不会自动退回 CPU。项目尚未确定多个模型与 Unity 同时运行所需的最低显存容量。
- Conda。默认可执行文件为 `C:\ProgramData\anaconda3\Scripts\conda.exe`；如果安装在其他位置，使用 `-CondaExe` 指定。安装脚本会创建两个独立的 **Python 3.11** 环境，无需激活全局 Python 环境。
- Visual Studio C++ Build Tools，用于首次在 Windows Python 3.11 下构建 `jieba_fast==0.53`。使用 `-BuildJiebaFast` 显式启用此构建。
- 安装时需要联网，并预留足够空间存放两个环境、源码、模型权重和下载缓存。项目尚未给出经过验证的最低磁盘容量。
- Unity Hub，以及 [ProjectVersion.txt](Unity/ProjectSettings/ProjectVersion.txt) 指定的 Editor 版本。当前项目文件记录为 **6000.0.61f1**。如果框架通用文档列出不同版本，以该文件为准。
- 可正常使用的麦克风和耳机。在打开 Unity 之前，先通过实际录音并回放，验证选中的 Windows 默认输入设备。

本指南说明 app 的运行方法。由代理执行编译、模型推理和真实 Ubiq 联调时，仍须遵守项目已有的授权要求。

## 3. 首次安装

如果本地运行环境已经安装且可正常工作，请跳过本节。无需每次启动都重新安装模型。

### 3.1 安装 Node 依赖

打开 PowerShell：

```powershell
Set-Location C:\ubiq-genie\Node
node --version
npm.cmd --version
npm.cmd install
```

本指南统一使用 `npm.cmd`。它可以避免 PowerShell 调用被执行策略阻止的 `npm.ps1`，无需修改系统执行策略。安装 Node 依赖不会安装语音模型。

### 3.2 安装本地环境、模型和工具

仍在同一个 `Node` 目录下，为新环境执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\apps\gpt_sovits_translation\scripts\setup.ps1 -BuildJiebaFast
```

如果 Conda 不在默认位置，在命令末尾添加 `-CondaExe` 和实际可执行文件路径。例如，下例中的占位路径必须先替换后再运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\apps\gpt_sovits_translation\scripts\setup.ps1 -BuildJiebaFast -CondaExe "D:\YOUR_CONDA_DIRECTORY\Scripts\conda.exe"
```

安装入口脚本依次安装 TTS、ASR、Ollama/FFmpeg。它会下载 GPT-SoVITS v2Pro、Faster-Whisper `large-v3-turbo` 和翻译模型，并记录代码／模型版本及依赖。它不会执行识别、翻译、合成或 Ubiq 联调测试。

TTS 环境使用 PyTorch/torchaudio `2.7.1+cu128`。虽然 ASR 有独立 Python 环境，但运行时也会使用 TTS 环境中的 CUDA DLL，因此需要安装本 app 的完整软件栈。这是用于中英文推理的环境，并非完整的 GPT-SoVITS 训练或多语言 WebUI 环境。

常用选项：

| 选项 | 用途 |
| --- | --- |
| `-Component all` | 默认值：安装所有组件 |
| `-Component tts`、`asr` 或 `ollama` | 修复错误后，只重试对应安装阶段；通过 `setup.ps1` 入口调用 |
| `-BuildJiebaFast` | 允许构建指定的原生依赖；如果已有兼容 wheel，可省略 |
| `-RuntimeRoot "D:\YOUR_RUNTIME_DIRECTORY"` | 修改运行目录；同时需要更新 app 配置中的 `translation.runtimePath`，并向启动／停止脚本传入相同选项 |

安装遇到错误会停止。读取终端输出和运行目录中的日志，修复原因后重新执行对应阶段。安装完成提示只说明安装结果，不代表模型推理已经验证成功。

### 3.3 检查安装记录

默认运行目录包含：

```text
Node/.runtime/gpt_sovits_translation/
  asr-env/                  Python 3.11 ASR 环境
  tts-env/                  Python 3.11 TTS 环境
  GPT-SoVITS/               固定版本的上游源码和 TTS 资源
  models/                   ASR 和 Ollama 模型存储
  ollama/                   独立 Ollama 可执行文件和本地状态
  tools/ffmpeg/bin/          ffmpeg.exe
  tts_infer.yaml            自动生成的 GPT-SoVITS 推理配置
  asr-manifest.json         ASR 安装记录
  receipts/                 源码、模型、软件包和下载记录
  logs/                     后端与安装日志
  pids/                     本 app 所属后端的进程身份记录
```

查看实际安装版本时，请检查 `asr-manifest.json`，以及 `receipts/tts-source.json`、`tts-models.json`、`tts-install.json`、`tts-packages.txt` 和 `ollama-ffmpeg.json`。当前工作区还包含后续修正翻译模型的 `ollama-model-correction.json`。请在 app 配置中保留 `qwen3:4b-q4_K_M`；此前可变的 `qwen3:4b` 标签曾选择到 thinking 变体，导致翻译行为不符合要求。

运行目录和录音属于本地部署数据。仅克隆源码不会获得这些文件，也不应将模型和环境产物提交到仓库。

## 4. 准备参考录音与音色列表

### 4.1 录音位置与要求

将参考录音放在：

```text
C:\ubiq-genie\Node\apps\gpt_sovits_translation\recordings\
```

每个音色需要一段 **3–10 秒**、干净、单人说话、未压缩 PCM 或 IEEE-float 格式的 **WAV**，以及准确文字和录音语言。推荐单声道。避免背景音乐、过长静音、削波失真，以及录音片段与文字不对应。将 MP3 扩展名改为 `.wav` 不会得到有效的 WAV。

`promptLanguage` 描述参考录音的语言：中文用 `zh`，英文用 `en`。它不用于强制指定译音语言。输入中文会翻译为英文，输入英文会翻译为中文。

如需使用已安装的 FFmpeg 生成独立的五秒单声道 PCM16 副本，请替换示例输入文件名并选择合适的片段。该命令保留源文件，并拒绝覆盖已有输出：

```powershell
Set-Location C:\ubiq-genie\Node
& .\.runtime\gpt_sovits_translation\tools\ffmpeg\bin\ffmpeg.exe -n -i .\apps\gpt_sovits_translation\recordings\source-recording.wav -ss 0 -t 5 -ac 1 -ar 32000 -c:a pcm_s16le .\apps\gpt_sovits_translation\recordings\my-reference-5s.wav
```

回放生成的文件，并转写**该片段的准确内容**，尤其核对片段开头和结尾的词句。自动转写不属于常规安装流程，需要另行运行模型。即使录音格式校验通过，内容或参考文字不合适仍可能导致合成效果不佳。

### 4.2 添加有效配置条目

选择器读取的是 Node 从 `profiles.json` 生成的音色目录；仅将 WAV 复制到 `recordings` **不会**添加音色。保留已有有效条目，使用以下模板新增或编辑条目，并替换全部占位符：

```json
{
  "REPLACE_WITH_A_REAL_PEER_UUID": {
    "refAudioPath": "recordings/my-reference-5s.wav",
    "promptText": "REPLACE_WITH_THE_EXACT_TRANSCRIPT_OF_THIS_RECORDING",
    "promptLanguage": "en"
  }
}
```

这只是结构示例，不应原样粘贴使用。顶层键必须符合 UUID 格式。参考路径相对于 `profiles.json` 解析，不是相对于 PowerShell 当前目录。JSON 路径建议使用 `/`；如果使用 Windows 反斜杠，需要将每个反斜杠转义为 `\\`。后端机器上的 GPT-SoVITS 必须能够读取该文件。

对于**尚无 UUID 的全新配置**，可以先将文件内容设为 `{}`，按照下文启动 Node 和 Unity，然后从 Inspector 的 `Local UUID` 或 Node 终端的 `Peer UUID` 读取 UUID。用它建立第一个有效参考音色条目，保存并等待音色目录刷新，再在 Unity 中选择。空配置允许 app 启动，但不会提供任何可选音色，也不会翻译未绑定用户的语音。不要将已有内容的配置替换为 `{}`。

即使原 UUID 对应的用户不在线，其配置条目仍然可以出现在音色目录中，不必每次重连都新增键。选择器会自动将所选音色绑定到当前本机 UUID。规范化后的录音路径、参考文字和语言完全相同的条目，在下拉列表中只显示一次。

Node 对整个文件进行校验和热重载。成功时会出现 `Loaded N validated voice profile(s)`。JSON 无效、文件缺失、WAV 数据不受支持或时长不在 3–10 秒范围内时，整次更新会被拒绝，并保留上次有效配置。初始配置无效则无法启动。如果希望个人配置不进入版本控制，可以使用已被忽略的 `profiles.local.json`，并在启动 Node 前将 `translation.profilesPath` 设为 `./profiles.local.json`。

## 5. 核对 Node 与 Unity 房间配置

打开 [app 配置](Node/apps/gpt_sovits_translation/config.json) 和 Unity 场景配置。对于随项目提供的标准场景，两端必须一致：

| 配置项 | 值／Unity 中的位置 |
| --- | --- |
| Node `roomserver.joinExisting` | `true`；连接现有 Nexus 服务器 |
| Node `roomserver.uri` 与 TCP 端口 | `nexus.cs.ucl.ac.uk`、`8009` |
| Unity RoomClient 服务器 | RealtimeTranslation 目录中的 `SoVITSServerConfig.asset` |
| 两端房间 GUID | `6765c52b-3ad6-4fb0-9030-2c9a05dc4731` |
| Unity RoomJoiner > Guid | 同目录中的 `SoVITSRoomGuid.asset`；不能为空 |
| Node `translation.outputSampleRate` | `16000` |
| Unity `SoVITSAudioSource.sourceSampleRate` | `16000`，位于助手的 `Avatar/Speech` 对象上 |
| Ollama URL／模型 | `http://127.0.0.1:11435` / `qwen3:4b-q4_K_M` |
| GPT-SoVITS URL | `http://127.0.0.1:9880` |

提供的资产还包含端口 `8010` 的 WebGL 连接覆盖配置；本指南主要面向桌面 Unity Editor。不要将模型服务端口与 Ubiq 房间端口混淆，也不要将模型服务暴露给远程 Unity 客户端。现有后端启动脚本使用固定端口 `11435` 和 `9880`；仅修改 JSON 中的 URL 不会改变服务监听端口。

Node 的运行目录和音色配置路径相对于 app 目录；运行目录内的 ASR、模型和 FFmpeg 路径相对于 `translation.runtimePath`。自定义安装目录必须与这些配置一致。

使用已经配置好的 Nexus 连接时，无需运行配置向导。`npm.cmd start gpt_sovits_translation configure` 会修改现有配置，仅在明确需要重新配置 app 时使用，并在之后重新核对两端设置。

## 6. 日常启动顺序：后端 -> Node -> Unity -> 选择音色

### 第 1 步：启动 Ollama 和 GPT-SoVITS

```powershell
Set-Location C:\ubiq-genie\Node
powershell -NoProfile -ExecutionPolicy Bypass -File .\apps\gpt_sovits_translation\scripts\start-backends.ps1
```

等待出现：

```text
Both backend HTTP endpoints are ready. You can now start the Node app; inference has not been tested.
```

启动脚本对**每个服务**最多等待十分钟，检查 HTTP 接口是否就绪。仅打印 PID 并不代表服务就绪。已有且属于本 app 的进程会被复用；如果端口被无关进程占用，脚本会拒绝继续。若一个服务启动后另一服务失败，已启动的服务可能继续运行；请检查错误，并在需要时使用本 app 的停止脚本。

以下可选检查不会请求模型推理：

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:11435/api/tags
(Invoke-WebRequest -UseBasicParsing -Uri http://127.0.0.1:9880/openapi.json).StatusCode
```

第一个请求应列出预期 Ollama 模型，第二个应返回 `200`。这些检查不能证明合成质量，也不会为第一次翻译完成模型预热。

### 第 2 步：启动 Node 并等待 ASR 就绪

启动脚本返回后，在同一终端执行：

```powershell
npm.cmd start gpt_sovits_translation
```

保持该终端运行。Node 会自行启动 ASR worker，不要再手动启动另一个。关注以下日志：

```text
[SpeechTranslation] Loaded N validated voice profile(s)
[SpeechTranslation] ASR ready; translated audio output is 16000 Hz PCM16 mono.
[GPT-SoVITS Translation] Waiting for peers. Select a voice in Unity SoVITSVoiceSelector, or bind a UUID in profiles.json.
```

`N` 是通过校验的映射数量，不一定等于去重后的音色数量。每个房间只运行**一个**翻译后端。其他向通道 95 发送数据的 app 可能导致音频交错；多个音色选择后端则会触发明确的目标歧义提示。

### 第 3 步：打开标准 Unity 场景

1. 在 Unity Hub 中，使用 `ProjectVersion.txt` 指定的版本打开 `C:\ubiq-genie\Unity`。
2. 等待包导入和脚本编译完成。进入 Play 前先解决编译错误。
3. 打开 `Assets/Ubiq-Genie/Apps/RealtimeTranslation/RealtimeTranslation.unity`。
4. 核对第 5 节的服务器／房间引用、Windows 默认麦克风和输出设备，并确认 Unity Editor 音频没有被静音。
5. 进入 **Play**，等待房间连接。Node 应打印 `Peer UUID: ...; name: ...`。

场景中已完成复制助手的引用接线。如需手动检查，`SoVITSReceiver` 应引用 `SoVITSAudioSource`、助手控制器及音量组件。保留一个通道 95 接收器，不要再添加 `ConversationalAgentManager`。Unity 会将播放器的 16 kHz 源音频转换为输出设备采样率，因此声卡无需设置为 16 kHz。

### 第 4 步：选择并确认本机用户音色

1. 在 Hierarchy 中选择 **`Ubiq-Genie/SoVITSReceiver`**。
2. 在 **SoVITS Voice Selector** 中确认已显示 `Local UUID`、`Room UUID` 和 `Backend UUID`。
3. 从 **Voice** 选择文件名／语言条目，点击 **Apply Voice**。
4. 等待 **Confirmed voice** 显示所选录音。Node 也会打印 `Voice binding bound for ...`。请求仍在等待，或仅显示 `Remembered voice`，都不能说明本次绑定已经成功。

保持 **Restore Previous Voice** 启用，即可在重新进入 Play、UUID 改变或 Node 重启后恢复上次成功确认的选择。偏好以音色 ID 保存到本机 PlayerPrefs，并按 Unity 项目和房间隔离；每次新绑定仍须得到 Node 确认。选择只建立会话内覆盖，不会改写 `profiles.json`。

Inspector 只操作**本机客户端**。在另一台电脑或另一个 Unity 项目上，需要在对应客户端选择音色。目前没有 Game 视图或 VR 音色面板。首次使用不会自动选取默认音色。如果参考路径、文字或语言发生变化，音色 ID 会改变，需要重新选择更新后的音色。存在多个后端、后端离线或音色被删除时，应先解决对应问题再绑定。

手动备用方式：在配置文件中，将当前 `Peer UUID` 映射到参考录音条目。这种方式需要在 UUID 改变后更新键，并不会自动保存 Inspector 偏好。

### 第 5 步：测试一句话

清楚说一句中文或英文，然后停顿约两秒。等这句译音播放后再重复。第一次真实语句可能因模型加载和首次运行开销而明显较慢，不要将 HTTP 就绪理解为这些延迟已被消除。

默认 VAD 在静音 700 ms 后切句，保留前置 300 ms 音频，要求至少 300 ms 语音，并将单段上限设为 15 秒。一个任务正在处理时，最多另有八句完整语句等待；队列溢出时丢弃新句并记录原因。只有识别结果为中文或英文的语句会进入翻译。

合成成功后，会打印 JSON 格式的 `type:"translation"` 日志，其中包含原文、译文、音频时长和各阶段耗时。启用诊断时，`Audio sent` 表示音频已发送。最终结果仍需在 Unity 中实际试听确认：绑定确认或发送成功都不等于已经听到声音。

## 7. 停止与重启

正常停止顺序：

1. 退出 Unity Play，停止向当前会话采集语音。
2. 在 Node 终端按 **Ctrl+C**，停止 app 及其 ASR／转换子进程。
3. 在 `Node` 目录停止本 app 的模型后端：

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File .\apps\gpt_sovits_translation\scripts\stop-backends.ps1
   ```

停止脚本会核对记录的进程身份，只管理本 app 的后端。它不会停止 Node app，也不会停止无关的 Python、Ollama 或 Unity 进程。如果安装时使用了自定义运行目录，需传入相同的 `-RuntimeRoot`。

诊断时若只需**重启 Node**，可以保持 Unity Play，按 Ctrl+C 停止 Node，再执行相同 npm 启动命令。已有模型后端可保持运行。新的后端出现后，选择器会请求恢复已记住的音色；等新的确认出现后再说话。app 运行期间的新语音可能触发模型推理。

## 8. 日志、延迟与排错

后端日志位于 `Node/.runtime/gpt_sovits_translation/logs/`：

- `gpt-sovits.out.log` 和 `gpt-sovits.err.log`：合成服务启动和错误信息。
- `ollama.out.log` 和 `ollama.err.log`：翻译服务问题。
- 各阶段安装日志：安装失败时查看。

Node 输出通常只保留在终端中，关闭前请复制相关日志。要启用汇总输入和流水线诊断，先停止 Node，再从 `Node` 目录执行：

```powershell
$env:SPEECH_TRANSLATION_DIAGNOSTICS = '1'
npm.cmd start gpt_sovits_translation
```

要在下次启动时禁用诊断：

```powershell
Remove-Item Env:SPEECH_TRANSLATION_DIAGNOSTICS -ErrorAction SilentlyContinue
```

按顺序检查：**profile -> input frames -> Utterance -> ASR -> Translation -> TTS -> Audio sent -> Unity 播放**。当 `profile=missing` 时，单凭 `frames=0` 不能判断麦克风故障，因为服务会主动忽略未绑定用户的音频。

| 现象 | 处理方法 |
| --- | --- |
| PowerShell 阻止 `npm.ps1` | 使用 `npm.cmd start gpt_sovits_translation` |
| `Cannot find package '@ucl-vr/ubiq'` | 在 `Node` 目录运行 `npm.cmd install`，确认依赖安装成功后重试 |
| 缺少 Conda、C++ 编译器或 jieba_fast wheel | 检查 `-CondaExe`，安装 C++ 工具，并在首次构建时显式添加 `-BuildJiebaFast` |
| 缺少安装产物／CUDA DLL／CTranslate2 失败 | 查看安装日志与记录，核对两个 Python 环境及其依赖；没有自动 CPU 回退 |
| 端口 11435/9880 被占用 | 确认监听进程；仅对本 app 所属服务使用其停止脚本，不要任意结束其他进程 |
| `RoomJoiner.DoJoin` 访问 Guid 时出现空引用 | 将 `SoVITSRoomGuid.asset` 赋给 RoomJoiner > Guid；修复序列化配置后退出 Play 并重新加载场景 |
| Unity 无法解析助手 controller／volume 类型 | 先解决 Console 中的实际编译错误及示例程序集／命名空间引用，再测试播放；不要盲目替换共享脚本 |
| `No voice profile` / `profile=missing` | 为 `Local UUID` 选择并应用音色，或添加当前 UUID 的有效文件映射 |
| 音色下拉列表为空 | 仅放置 WAV 不够；检查有效音色条目、Node 是否就绪，以及房间／服务器是否一致 |
| `Voice profile reload rejected` | 修正 JSON 逗号／引号、UUID 键、WAV 路径、时长和文字／语言字段，等待重载成功 |
| 已记住的音色不可用 | 参考材料／路径变化后重新选择当前条目，等待新的确认 |
| 多个后端／绑定始终未确认 | 停止房间内重复的翻译 app，检查 Node 绑定错误和房间连接 |
| 音色有效但帧数为零 | 检查 Windows 所选麦克风、桌面应用麦克风权限、Unity 采集和 WebRTC 连接 |
| 有音频帧但峰值始终接近 -80 dBFS 或 `-inf` | 输入近乎静音；实际进行 Windows 录音测试，检查硬件静音／默认输入，并对照 Unity 麦克风诊断 |
| ASR／翻译成功但 TTS 失败 | 检查 `gpt-sovits.err.log`、本地参考文件访问权限、准确的参考文字／语言和后端就绪状态 |
| `Audio sent` 但静音，或声音很轻／不完整 | 检查 Unity 静音／listener／输出设备、助手距离和音量、接收器引用，以及合成音频或参考录音本身是否有效 |
| 语速或音高异常 | 确保 Node 输出采样率与 SoVITSAudioSource 源采样率一致，AudioSource pitch 保持为 1 |
| 第一句很慢，随后 `queueMs` 很高 | 区分模型冷启动开销与排队时间；诊断第一句时先停止反复提交新语句 |
| 回声或重复翻译 | 使用耳机，避免麦克风采集扬声器播放的译音 |

Unity 在 Play 模式还提供 **Tools > SoVITS > Log Microphone Diagnostics (5 Seconds)**。观察期间说话，然后查看 Console 中的 `[SoVITS Diagnostic]` 日志。该命令只打印日志，不会弹出窗口。它检查现有采集状态；正在运行的 Node app 仍可能正常处理这段语音。

耗时字段分别记录 `queueMs`、`asrMs`、`translationMs`、`ttsMs` 和 `convertMs`。`endpointToPlaybackMs` 计时结束于 Node 开始发送音频，既不包含 VAD 静音判定时间，也**不是**声音到达耳机的实际延迟。Unity 默认预缓冲为 200 ms，PCM 缓冲上限为两秒。判断稳定运行延迟前，请等第一句完成，再单独测试第二句。

## 9. 验证与相关指南

进行双客户端验收时，各客户端先确认自己的音色，再轮流测试中文 -> 英文和英文 -> 中文。检查两个客户端上音色是否符合选择、译音是否完整、对方原声是否保留，以及重新进入 Play 或重启 Node 后是否自动重绑。记录 Node 请求 ID、各阶段耗时和实际听到的结果。组件就绪和单元测试不能替代这项检查。

历史基线：**2026-10-20**，app 类型检查、**36 个 Node 测试**、Unity 编译和 **77 个 EditMode 测试**通过。这些结果早于后续动作场景修改，不能视为对当前工作区的新验证。用户于 2026-10-19 确认听到清晰的单用户译音；Inspector 现场恢复和双用户验收仍是独立检查项。编写本指南期间没有执行编译、推理或现场联调。

相关本地文档：

- [App README](Node/apps/gpt_sovits_translation/README.md)：实现、安装记录、配置和测试命令。
- [Unity 集成说明](Unity/Assets/Ubiq-Genie/Apps/RealtimeTranslation/INTEGRATION.md)：场景引用、播放封包、选择器协议和诊断细节。
- [可选的 RealtimeTranslationMotion 场景](Unity/Assets/Ubiq-Genie/Apps/RealtimeTranslation/MOTION_REPLAY.md)：使用相同后端的独立场景，提供基于虚拟形象的译音播放和远程动作回放。其对象名称和播放行为不同；本指南主要流程使用标准助手场景。
- [框架 README](README.md) 和 [Node 架构](Node/README.md)：通用项目配置和可复用组件。
