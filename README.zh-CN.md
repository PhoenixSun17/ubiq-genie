# 欢迎使用 Ubiq-Genie

[English](README.md) | [简体中文](README.zh-CN.md)

![Ubiq-Genie 中两个示例演示的插图](header.png)

Ubiq-Genie 是一个基于 [Ubiq](https://ubiq.online) 框架的开发框架，可让你使用 Unity 构建由服务器辅助的协作式混合现实应用。它尤其适合需要服务器端处理的多用户应用，例如生成模型、对话代理和实时转写。更多信息请参阅 [Ubiq-Genie 论文](https://ubiq.online/publication/ubiq-genie/)。

> [!NOTE]
> 开始使用 Ubiq-Genie 之前，建议先熟悉 Ubiq 框架。更多信息请参阅 Ubiq 的[文档](https://ucl-vr.github.io/ubiq/)和[网站](https://ubiq.online)。Ubiq-Genie 当前使用 Ubiq [v1.0.0-pre.16](https://github.com/UCL-VR/ubiq/releases/tag/unity-v1.0.0-pre.16)。

## 初始设置

以下步骤将帮助你获取并运行项目，以便运行示例并开始构建自己的应用。Ubiq-Genie 支持 Windows、macOS 和 Linux。Ubiq-Genie 采用服务器与客户端架构，因此你可能需要将服务器与 Unity 客户端运行在不同的机器上。

### 服务器（Node.js）

0. 安装 [Node.js](https://nodejs.org/en/download/)（v20 或更高版本）和 [Python](https://www.python.org/downloads/)（v3.10 或更高版本）。

1. 将此仓库克隆到你的机器上（本地或远程机器均可）。

2. 在 `Node` 文件夹中打开终端，运行 `npm install` 安装依赖。

3. 为计划使用的服务安装 Python 依赖。每个服务提供器都有自己的 `requirements.txt` 文件，位于其提供器文件夹内（例如 `Node/services/speech_to_text/providers/azure/requirements.txt`）。使用 `pip install -r <path_to_requirements.txt>` 安装依赖。如果使用虚拟环境，请先激活环境再运行该命令。服务启动时会对缺失的依赖发出警告。请确保安装了正确版本的 PyTorch 和 CUDA（更多信息请参阅 [PyTorch 网站](https://pytorch.org/get-started/locally/)）。

### 客户端（Unity）

1. 安装 [Unity](https://unity3d.com/get-unity/download) **6.0 LTS**（当前使用的版本为 *Unity 6000.0.67f1*）。

2. 将此仓库克隆到本地机器上。

3. 将 `Unity` 文件夹添加到 Unity Hub，并在 Unity 中打开项目。

4. 打开 Package Manager，点击 Ubiq 包（com.ucl.ubiq），进入 “Samples” 选项卡，然后导入 “Demo (XRI)” 示例。这会将 Unity XR Interaction Toolkit 包以及 Ubiq-Genie 示例应用使用的一些脚本添加到项目中。

> [!NOTE]
> 请阅读 `Node/apps` 中对应文件夹内的 README 文件，了解进一步的设置步骤。可用示例列表见下方的[示例](#示例)部分。

## 文档

有关如何使用 Ubiq-Genie 的更多信息，请参阅 `Node` 文件夹中的 README 文件。

项目文档使用英文。每份 README 均维护内容一致的两个文件：英文版 `README.md` 和简体中文版 `README.zh-CN.md`。

## 示例

`Node/apps` 文件夹包含多个示例，展示如何使用 Ubiq-Genie；每个示例使用 `Node/services` 文件夹中定义的一项或多项服务。有关这些示例的使用方法，请参阅对应文件夹中的 README 文件。目前提供以下协作式示例应用：

- [**纹理生成**](Node/apps/texture_generation/README.zh-CN.md)：根据语音输入生成纹理，也可通过射线选择目标对象。
- [**多用户对话代理**](Node/apps/conversational_agent/README.zh-CN.md)：支持多名用户交互的对话代理。
- [**转写**](Node/apps/transcription/README.zh-CN.md)：将房间中每位用户的转写内容和音频分别保存到文件中。

示例演示视频见 [Ubiq-Genie 演示视频](https://youtu.be/cGz0z9BIgQk)。

## 支持

如有问题，请使用 GitHub 的 Discussions 选项卡，或在 [Ubiq Discord 服务器](https://discord.gg/cZYzdcxAAB)的 *ubiq-genie* 频道中留言。如需报告缺陷，请使用 GitHub 的 Issues 选项卡。
