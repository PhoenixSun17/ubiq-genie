# 基础应用

[English](README.md) | [简体中文](README.zh-CN.md)

此文件夹包含基础应用的代码，可作为使用 Ubiq-Genie 框架创建新应用的模板。有关创建自己的应用的方法，请参阅 [README](../../README.zh-CN.md)。

> [!IMPORTANT]
> 在继续之前，请确保已正确安装 Ubiq-Genie 框架及本示例所需的依赖。有关详情，请参阅根目录的 [README](../../../README.zh-CN.md) 文件。

## 运行示例

### 服务端（Node.js）

建议使用 VS Code 运行和修改服务端应用，并将 `Node` 文件夹作为工作区根目录。运行服务端应用：

1. 打开终端，进入 `Node/apps/base` 目录。
2. 执行以下命令，按引导完成配置，包括服务器信息。配置流程仅在首次启动应用时运行。请确保 Unity 客户端（`Unity/Assets/ServerConfig.asset`）采用相同的服务器配置。

    ```bash
    npm start base
    ```

如果需要重新配置应用，可运行 `npm start base configure`。也可以直接修改包含服务器配置的 `config.json`，手动配置应用。

### 客户端（Unity）

1. 启动 Unity，进入 `Unity/Assets/Apps/Base` 目录，打开 `Base.unity` 场景。
2. 确保 `Network Scene` 对象下的 `Room Client` 使用正确的服务器 IP 地址和端口。如果服务器与 Unity Editor 在同一台计算机上运行，IP 地址应为 `localhost`。该配置应与服务端一致。
3. 在 Unity Editor 中点击 `Play` 按钮，启动应用。
