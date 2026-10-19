# DynamicCaptureYolo

[English](README.md) | [简体中文](README.zh-CN.md)

此独立应用为原始 DynamicCapture 前景采集流程添加 YOLO 对象类别。它使用自己的应用、服务、Python 环境和 CSV 文件。现有 DynamicCapture、Yolo、共享组件和 Unity 代码保持不变。

## 采集行为

默认输入为摄像头 **1**，请求分辨率为 **1280 x 720**。保持摄像头静止。按顺时针顺序点击四个地面点：左上、右上、右下、左下。第四次点击会采集参考背景；所选区域映射到原有的 **6 x 6** Unity 坐标。采集背景时，请确保区域内没有要跟踪的对象。

按 **b** 重新采集背景，按 **q** 退出。重新采集背景不会立即重置会话 ID 计数器或已保存的轨迹。

Python 后端在标定、背景／运动检测和 YOLO 之前应用选定的**中等强度镜头校正（`k1 = -0.12`）**。请在**校正后的预览**中选择四个地面点。这是一项未经标定的实验性校正：它会裁剪并拉伸边缘内容，在 1920 x 1080 实验中保留约 80% 的源图像面积；识别效果是否改善，以及度量准确性，均未得到证实。校正映射使用实际帧尺寸并进行缓存；如果帧尺寸变化，需要重启并重新标定。请求的摄像头分辨率保持不变，不同摄像头模式可能具有不同投影。将 `services/dynamic_capture_yolo/Capture.py` 中的 `LENS_CORRECTION_K1` 设为 `0` 可禁用校正。该集成修改尚未测试。

前景检测保留原有的模糊灰度背景差分、阈值 25、两次膨胀迭代、轮廓面积大于 800 像素以及质心位于区域内的要求。Dynamic/Static 状态仍根据与上一帧的差异计算，沿用运动分数阈值 5。位置和尺寸来自前景轮廓及标定，而非 YOLO 检测框。

YOLOv8n 在未绘制标记的图像帧上运行，只有在边界框能够以至少 0.3 的 IoU 一对一匹配时，才为前景轮廓添加类别。默认检测置信度为 0.5。未匹配的前景对象仍会输出，使用 `label: "unknown"`、`class_id: null` 和 `confidence: 0.0`。YOLO 检测到的背景对象不会独立生成输出对象。因此，分裂的轮廓、重叠对象以及模型类别词表之外的类别可能仍为 unknown。

ID 是由现有质心方法分配的会话跟踪 ID，与标签无关。轨迹可在 50 像素范围内匹配，连续十帧漏检期间在内部保留，第十一帧时移除。仅输出当前观测到的对象。对象交叉、轮廓合并、快速移动或长时间遮挡后，ID 可能变化或交换；它们不用于识别人，也不永久标识某个物理对象。

## 安装

这些文件应位于 `C:\ubiq-genie\Node` 下。应用复用现有本地模型 `Node\apps\Yolo\yolov8n.pt`，会检查文件是否存在，不会请求下载模型。

使用以下 PowerShell 命令创建独立服务环境并安装依赖。安装与测试分开进行；这些命令不会启动采集应用。依赖文件锁定了当前计算机的软件包版本，包括 Ultralytics 8.4.19 和 PyTorch 2.10.0。初始 Windows 环境使用 CPU 版 PyTorch，因此 `device: "auto"` 会选择 CPU。不会升级任何现有 Python 环境。

```powershell
Set-Location C:\ubiq-genie\Node\services\dynamic_capture_yolo
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Windows 上的默认解释器为 `Node\services\dynamic_capture_yolo\.venv\Scripts\python.exe`，其他系统使用 `.venv/bin/python`。可在本应用 `config.json` 的顶层通过可选字段 `pythonCommand` 指定其他解释器的绝对路径。环境缺失时会提供可操作的错误信息，不会回退到全局解释器。Ultralytics 设置保存在本服务的 `.venv/yolo-config` 目录中，子进程禁用 Python 用户级 site-packages。

在用户另行批准测试前，不要运行测试、编译／导入检查、模型推理、摄像头采集或此应用。准备此副本时未执行这些检查；运行时兼容性和摄像头行为仍未验证。

## 配置与获准后的启动方式

复制的房间配置使用 GUID `6765c52b-3ad6-4fb0-9030-2c9a05dc4731`，加入现有的 `localhost:8009` 服务器。服务器与客户端必须已使用同一房间。不修改任何 Unity 接收器或场景。评估此应用时，请用它替代原有采集应用，因为两个应用都向网络 ID 99 发布消息。

新应用的 `config.json` 包含：

| `capture` 下的键 | 默认值 | 含义 |
| --- | --- | --- |
| `source` | `1` | 摄像头索引；也可以提供视频文件路径。 |
| `modelPath` | `../Yolo/yolov8n.pt` | 现有本地 YOLO 权重。 |
| `logPath` | `dynamiccapture_yolo_tracking.csv` | 独立输出日志。 |
| `confidence` | `0.5` | 最低 YOLO 检测置信度。 |
| `matchIou` | `0.3` | 前景与 YOLO 匹配所需的最小重叠比例。 |
| `device` | `auto` | 自动选择设备；`cpu` 强制使用 CPU 推理。 |

模型和日志路径相对于本应用目录解析，与终端初始目录无关。相对视频路径也以应用目录为基准，因为该目录是子进程的工作目录。每次运行都将写入配置的独立 CSV，并保留原有列：`frame count`、`object count` 和 `udpstring`。

**获得明确的测试批准后**，在 Node 目录中启动：

```powershell
Set-Location C:\ubiq-genie\Node
npm start DynamicCaptureYolo
```

## 输出约定

每条已完成的采集记录保留原始 Python stdout 表示形式：

```text
[frame_count, object_count, '{"count": ..., "objects": [...]}']
```

Node 应用组装完整的 UTF-8 行，将每条记录以 `{ "data": "<original line>" }` 的形式转发到 **NetworkId(99)**。内部行是 Python 列表的文本表示，不是 JSON 数组；新应用不会解析或重写它。诊断信息写入 stderr。CSV 数据包保留相同的 JSON 结构。

每个对象保留 `id`（整数）、`pos`（数值型 `x`、`y`）、`scale`（数值型 `w`、`h`）和 `status`（`"Dynamic"` 或 `"Static"`），并新增 `label`（字符串）、`class_id`（整数或 null）和 `confidence`（数值）。顶层数据包仍包含 `count`（整数）和 `objects`（数组）。当前没有前景对象的帧包含 `count: 0` 和空数组。

标识字段通过网络消息传送，并记录在新 CSV 中。此副本未实现将这些字段显示到现有 Unity UI 的功能。
