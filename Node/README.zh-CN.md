# Ubiq-Genie

[English](README.md) | [简体中文](README.zh-CN.md)

Ubiq-Genie 是一个基于 [Ubiq](https://ubiq.online) 框架的开发框架，可让你使用 Unity 构建由服务器辅助的协作式混合现实应用。Ubiq-Genie 采用模块化架构，便于集成新服务，以及在不影响整个系统的情况下更新或替换单项服务。架构包含三个主要部分：Unity 场景、应用和服务。

## 系统架构

- **Unity 场景**是 VR 用户的交互界面，包含应用专用的 Unity 组件。这些组件通过 TCP 连接，使用 Ubiq 的 `Networking` 组件与服务器端的 `ApplicationController` 通信。这些客户端组件使用 C# 编写，确保发出和接收的数据得到正确处理与路由。每个应用的 Unity 场景位于 `Unity/Assets/Apps` 文件夹中。

- **应用**应具有对应的 Unity 场景和 `ApplicationController`。`ApplicationController` 负责初始化和管理应用所需的服务，并处理服务与 Unity 场景之间的通信。`ApplicationController` 使用 TypeScript（ESM）编写，在服务器上运行。每个示例应用的 `ApplicationController` 位于 `Node/apps` 下对应文件夹中的 `app.ts` 文件内。

- **服务**是可在不同应用中复用的模块。每项服务负责一项特定任务，由 `ServiceController` 管理。服务使用**提供器（provider）**，即定义后端运行方式及其生命周期管理方式的轻量配置对象。例如，`ImageGenerationService` 使用 Stable Diffusion 提供器，启动 Python 子进程来生成图像。无需修改服务或应用代码即可替换提供器，从而方便地切换后端（例如在 Azure Speech 与 Whisper.cpp 之间切换语音识别后端）。`ServiceController` 使用 TypeScript（ESM）编写，在服务器上运行。每个示例服务的 `ServiceController` 位于 `Node/services` 下对应文件夹中的 `service.ts` 文件内。

## 定义新服务

按以下步骤定义新服务：

1. 复制 `Node/services/base` 文件夹，并将其重命名为你的服务名称（例如 `my_service`）。将 `service.ts` 文件中的类名 `BaseService` 替换为你的服务类名（例如 `MyService`）。

2. 为服务创建提供器。提供器是一个 `ServiceProvider` 对象，用于指定运行的命令、命令参数和进程模式。在 `providers/` 下创建文件夹（例如 `providers/my_provider/`），并添加用于导出提供器配置的 `provider.ts` 文件。将后端脚本（例如 Python 脚本）和 `requirements.txt` 放在同一文件夹中。

    ```typescript
    import type { ServiceProvider } from '../../../../components/service';
    import path from 'path';
    import { fileURLToPath } from 'url';

    const __dirname = path.dirname(fileURLToPath(import.meta.url));

    export const MyProvider: ServiceProvider = {
        name: 'my-provider',
        command: 'python',
        args: ['-u', path.join(__dirname, 'my_script.py')],
        processMode: 'singleton',
        requirements: path.join(__dirname, 'requirements.txt'),
    };
    ```

3. 在 `service.ts` 中导入该提供器，并将其传给 `ServiceController`：

    ```typescript
    import { ServiceController } from '../../components/service';
    import { NetworkScene } from 'ubiq-server/ubiq';
    import { MyProvider } from './providers/my_provider/provider';

    class MyService extends ServiceController {
        constructor(scene: NetworkScene) {
            super(scene, 'MyService', MyProvider);
        }
    }
    ```

### 进程模式

每个提供器必须指定 `processMode`，以确定子进程的管理方式：

| 模式 | 行为 |
| --- | --- |
| `per-peer` | 每个已连接的对等端各有一个子进程。对等端加入时启动，离开时终止。适用于每个对等端都需要独立后端的情况（例如语音识别）。 |
| `singleton` | 服务创建后立即启动一个共享子进程。适用于共享状态的后端（例如文本生成）。 |
| `lazy-singleton` | 第一个对等端加入时启动一个共享子进程，所有对等端离开时终止。适用于资源消耗较大且只需按需运行的后端（例如图像生成）。 |

### 提供器目录结构

每个提供器的所需文件都放在自己的文件夹中：

```text
services/my_service/
├── service.ts
└── providers/
    └── my_provider/
        ├── provider.ts        # ServiceProvider configuration
        ├── my_script.py       # Backend script
        └── requirements.txt   # Python dependencies
```

当提供器指定了 `requirements` 路径时，`ServiceController` 会在启动时自动检查所需的 Python 包是否已安装，并在缺少包时记录警告。

> [!NOTE]
> `BaseService` 提供了使用提供器的最小服务示例。更完整的示例见 `Node/services` 文件夹中的现有服务。并非所有服务都需要提供器，例如 `AudioRecorder` 直接使用 TypeScript 录制音频，不会启动子进程。

现在可以在应用中使用你的新服务。有关定义新应用的更多信息，请参阅下方的“定义新应用”部分。

## 定义新应用

按以下步骤定义新应用：

1. 复制 `Node/apps/base` 文件夹，并将其重命名为你的应用名称（例如 `my_application`）。同时将 `app.ts` 文件中的类名 `BaseApplication` 替换为你的应用类名（例如 `MyApplication`）。

> [!NOTE]
> `BaseApplication` 提供了使用 `BaseService` 服务创建应用的最小示例。更完整的示例见 `Node/apps` 文件夹中的现有应用。`registerComponents` 方法定义应用组件，并将其保存在名为 `components` 的字典中。`definePipeline` 方法定义应用的处理流程。`start` 方法通过注册组件、定义处理流程以及加入配置文件中指定的 Ubiq 服务器上的房间来启动应用。

2. 查看新建文件夹中的 `config.json` 文件。该文件包含应用配置，包括应用名称、应用将加入的房间 GUID、加入或启动 Ubiq 服务器所需的信息，以及 WebRTC 连接使用的 ICE 服务器。如果需要加入现有服务器，应将 `joinExisting` 设为 `true`；如果需要启动新服务器，则设为 `false`。有关 Ubiq 服务器和消息的更多信息，请参阅 [Ubiq 文档](https://ucl-vr.github.io/ubiq/serverintroduction/)。

3. 在 Unity 中，复制 `Unity/Assets/Apps/Base` 文件夹，并将其重命名为你的应用名称（例如 `MyApplication`）。该文件夹将包含应用的 Unity 场景。

4. 在 Unity 场景层级中找到 `Ubiq-Genie/`。建议将与 Ubiq-Genie 应用服务器端进程通信的应用专用组件放在这里。在 Base 应用中，这里只有一个 `MessageReceiver` 组件，用于接收服务器消息。这些消息由 `BaseApplication` 应用中 `BaseService` 服务的 Python 进程发送。
