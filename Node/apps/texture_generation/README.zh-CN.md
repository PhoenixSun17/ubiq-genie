# 语音纹理生成示例

[English](README.md) | [简体中文](README.zh-CN.md)

本指南介绍如何使用 Ubiq-Genie 框架创建根据语音命令生成纹理的应用。用户也可以通过射线选择指定目标对象。

## 前提条件

运行本示例需要能够支持 Stable Diffusion 2.0 的计算机，通常要求 GPU 至少具有 4GB 显存，系统至少具有 16GB 内存。本示例已在配备 NVIDIA GeForce RTX 3080 Ti GPU 和 64GB 内存的 Windows 11 计算机上测试。通过 PyTorch 的 MPS 支持，也可在 macOS 上运行，但需要 Apple Silicon。

此外，还需要 Azure Speech Services 订阅。可通过[此链接](https://azure.microsoft.com/en-us/try/cognitive-services/?api=speech-services)创建免费订阅。

> [!IMPORTANT]
> 在继续之前，请确保已正确安装 Ubiq-Genie 框架及本示例所需的依赖。有关详情，请参阅根目录的 [README](../../../README.zh-CN.md) 文件。

## 运行示例

### 服务端（Node.js）

建议使用 VS Code 运行和修改服务端应用，并将 `Node` 文件夹作为工作区根目录。运行服务端应用：

1. 打开终端，进入 `Node/apps/texture_generation` 目录，确保已激活 conda 或 venv 环境。
2. 执行以下命令，按引导完成配置，包括服务器信息及所需的环境变量。配置流程仅在首次启动应用时运行。请确保 Unity 客户端（`Unity/Assets/ServerConfig.asset`）采用相同的服务器配置。

    ```bash
    npm start texture_generation
    ```

如果需要重新配置应用，可运行 `npm start texture_generation configure`。也可以直接修改 `config.json` 和 `.env` 文件，手动配置应用。`config.json` 包含服务器配置，`.env` 包含 Azure Speech Services 订阅密钥和区域对应的环境变量（`SPEECH_KEY` 和 `SPEECH_REGION`）。

### 客户端（Unity）

1. 启动 Unity，进入 `Unity/Assets/Apps/TextureGeneration` 目录，打开 `TextureGeneration.unity` 场景。
2. 确保 `Network Scene` 对象下的 `Room Client` 使用正确的服务器 IP 地址和端口。如果服务器与 Unity Editor 在同一台计算机上运行，IP 地址应为 `localhost`。该配置应与服务端一致。
3. 在 Unity Editor 中点击 `Play` 按钮启动应用，或将其部署到 VR 头显。
4. 按住扳机按钮以激活语音命令系统。
5. 指向场景中的目标对象，同时说出类似“make this look like lava”（让它看起来像熔岩）的命令。

## 演示视频

有关本示例的演示，请参阅 [Ubiq-Genie 演示视频](https://youtu.be/cGz0z9BIgQk)。

## 支持

如有任何问题，请使用 GitHub 的 Discussions 页面，或在 [Ubiq Discord 服务器](https://discord.gg/cZYzdcxAAB)的 *ubiq-genie* 频道中留言。
