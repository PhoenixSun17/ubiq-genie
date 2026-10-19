# 语音转写示例

[English](README.md) | [简体中文](README.zh-CN.md)

本指南介绍如何使用 Ubiq-Genie 框架的语音转写示例，转写房间内的音频流。对于房间中的每个参与端，示例会录制其音频流，并使用语音转文字服务进行转写。转写结果保存为 .csv 文件，录音保存为 .wav 文件。

示例默认使用 **Nemotron Streaming** 服务，通过 NVIDIA 的 NeMo 工具包在本机运行，无需 API 密钥。也可以使用 **Azure Speech Services** 等其他服务；其中一些基于云端，需要 API 密钥。

## 前提条件

运行本示例需要配备麦克风的计算机或 VR 头显。如果使用 Azure 等云端服务，还需要 API 订阅。可通过[此链接](https://azure.microsoft.com/en-us/try/cognitive-services/?api=speech-services)创建免费的 Azure Speech Services 订阅。

> [!IMPORTANT]
> 在继续之前，请确保已正确安装 Ubiq-Genie 框架及本示例所需的依赖。有关详情，请参阅根目录的 [README](../../../README.zh-CN.md) 文件。

## 运行示例

### 服务端（Node.js）

建议使用 VS Code 运行和修改服务端应用，并将 `Node` 文件夹作为工作区根目录。运行服务端应用：

1. 打开终端，进入 `Node/apps/transcription` 目录，确保已激活 conda 或 venv 环境。
2. 执行以下命令，按引导完成配置，包括服务器信息。配置流程仅在首次启动应用时运行。请确保 Unity 客户端（`Unity/Assets/ServerConfig.asset`）采用相同的服务器配置。

    ```bash
    npm start transcription
    ```

如果需要重新配置应用，可运行 `npm start transcription configure`。也可以直接编辑 `config.json` 文件，手动配置应用。

**切换服务：** 如需使用其他语音转文字服务，请修改 `app.ts` 中导入的服务。使用 Azure 等云端服务时，需要设置环境变量，例如在 `.env` 文件中设置 `SPEECH_KEY` 和 `SPEECH_REGION`。

### 客户端（Unity）

1. 启动 Unity，进入 `Unity/Assets/Apps/Transcription` 目录，打开 `Transcription.unity` 场景。
2. 确保 `Network Scene` 对象下的 `Room Client` 使用正确的服务器 IP 地址和端口。如果服务器与 Unity Editor 在同一台计算机上运行，IP 地址应为 `localhost`。该配置应与服务端一致。
3. 在 Unity Editor 中点击 `Play` 按钮，启动应用。
4. 对着麦克风说话。每位用户的语音都会录制并保存为 .wav 文件，转写结果保存为 .csv 文件；两类文件默认均位于 `Node/samples/apps/transcription/recordings` 目录。

## 支持

如有任何问题，请使用 GitHub 的 Discussions 页面，或在 [Ubiq Discord 服务器](https://discord.gg/cZYzdcxAAB)的 *ubiq-genie* 频道中留言。
