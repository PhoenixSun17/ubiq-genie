# YoloOnlyCapture

[English](README.md) | [简体中文](README.zh-CN.md)

这是用于与 `DynamicCaptureYolo` 比较的独立 YOLO 检测应用。现有 `Yolo`、`DynamicCapture`、`DynamicCaptureYolo`、共享组件、VRDC 和 Unity 文件均未修改。OpenCV 仍负责摄像头输入、实验性镜头校正、标定和绘制；检测出的对象及其边界框完全由 YOLO 决定。此应用不进行前景／背景差分、轮廓检测或前景与 YOLO 的 IoU 匹配。

## 对比行为

| 属性 | DynamicCaptureYolo | YoloOnlyCapture |
| --- | --- | --- |
| 对象数量与几何形状 | 前景轮廓 | YOLO 检测框 |
| 类别 | 与轮廓匹配的 YOLO 类别 | YOLO 检测框的类别 |
| 漏检／无法识别的对象 | 可能保留为前景 `unknown` 对象 | YOLO 未检测到时不会输出 |
| 启动时已存在的对象 | 被采集的背景抑制 | 如果在标定区域内被检测到，则会输出 |
| 会话 ID | 根据质心匹配，与标签无关 | 相同的匹配策略 |
| Static/Dynamic 状态 | 与上一帧的像素差异 | 相邻处理帧之间的检测框中心位移 |
| 摄像头 | 索引 1，Windows 使用 DirectShow，请求 1280 x 720 | 相同 |
| 镜头校正 | 实验性中等强度 `k1=-0.12` | 相同 |
| 模型 | 本地 YOLOv8n，置信度 0.5，推理尺寸 640 | 相同 |
| 坐标 | 四点地面映射到 6 x 6 | 使用相同映射，但输入为检测框中心与角点 |
| 传输 | 网络 ID 99，完整 stdout 记录 | 相同 |

比较运行结果时，应使用相同的四个实际地面点和摄像头模式。每次只运行一个采集应用：两个应用都使用摄像头 1，并向网络 ID 99 发送消息。经批准后，可将录制的视频设为 `capture.source`，进行可重复的比较。不要仅根据对象数量变化推断准确率有所提高；两个程序的纳入规则不同，YOLO 检测框也不等于对象的实际占地轮廓。

原有 `apps/Yolo` 后端使用摄像头 0、硬编码的全画面标定、逐帧 ID 和较小的 `{id,x,y,label}` 消息。本应用保留混合检测应用的信息与标定方式，而不修改那个较早的程序。

## 标定与控制

保持摄像头静止。在校正后的预览中，按顺时针顺序点击四个地面点：左上、右上、右下、左下。区域有效后开始跟踪和输出数据包。当对象的 YOLO 检测框中心位于所选多边形内部或边界上时，该对象会被纳入；检测框本身不会裁剪到该多边形。

镜头校正未经标定。它会裁剪并拉伸边缘内容；临时进行的 1920 x 1080 实验在此设置下保留了约 80% 的源图像面积。映射根据实际采集尺寸构建并缓存。如果帧尺寸发生变化，采集会停止，需要重新启动并标定。请求的分辨率保持为 1280 x 720；不同摄像头模式可能具有不同投影。识别效果是否改善，以及度量准确性，均未得到证实。

- **q**：退出并释放摄像头。
- **b**：清除运动基准，同时保留会话 ID 和标定。此操作不会采集或减去背景。下一次观测将建立新的运动基准。

ID 在 50 图像像素范围内进行一对一匹配，连续十一帧未检测到后过期。只输出当前检测到的对象。标签变化不会重置 ID。对象交叉、检测框大幅移动、检测结果合并、长时间遮挡和检测框抖动都可能导致 ID 变化或交换；这些是会话 ID，并非永久身份。

如果匹配对象的检测框中心相对于紧邻的上一处理帧移动**超过** `motionThresholdPx` 像素，`status` 为 `Dynamic`；否则为 `Static`。默认阈值为 5 像素。新 ID 或在漏检后重新出现的对象以 `Static` 开始，以建立基准；这一初始状态不能证明对象在物理上静止。这种简单估计依赖处理速率和检测框抖动，可能漏掉缓慢运动，或对象在不变检测框内部的运动。按照此次比较的选择，它有意采用了不同于混合应用像素差分状态的方法。

## 配置与环境

应用位于 `C:\ubiq-genie\Node\apps\YoloOnlyCapture`；独立后端为 `Node\services\yolo_only_capture\Capture.py`。

提供的配置明确使用 `C:\ubiq-genie\Node\services\dynamic_capture_yolo\.venv\Scripts\python.exe` 和现有 `apps/Yolo/yolov8n.pt`，使比较双方的软件包及权重版本一致。不安装或升级任何软件包，不导入现有后端，也不自动下载模型。应用使用自己的 `services/yolo_only_capture/.runtime/yolo-config` 设置目录。封装层禁用 Python 用户级 site-packages 和字节码输出。

| 配置键 | 默认值 | 含义 |
| --- | --- | --- |
| `pythonCommand`（顶层） | 上述绝对路径 | 现有独立 Python 解释器；不回退到全局解释器 |
| `capture.source` | `1` | 摄像头索引或本地视频路径 |
| `capture.modelPath` | `../Yolo/yolov8n.pt` | 现有本地权重 |
| `capture.logPath` | `yolo_only_tracking.csv` | 此应用独立的 CSV，下次运行时覆盖 |
| `capture.confidence` | `0.5` | 最低检测置信度 |
| `capture.device` | `auto` | 设备选择；现有 CPU 环境仍使用 CPU |
| `capture.motionThresholdPx` | `5` | 每个处理帧的非负质心运动阈值 |

模型、CSV 和相对视频路径均以新应用目录为基准解析。如需之后改用独立环境，请创建 `services/yolo_only_capture/.venv`，在其中安装本服务锁定版本的 `requirements.txt`，然后移除 `pythonCommand` 以选择该环境，或将其替换为其他解释器的绝对路径。镜头设置为后端常量 `LENS_CORRECTION_K1`；设为 `0` 可禁用校正。

房间配置与混合应用一致：加入现有 `localhost:8009` 服务器，房间 GUID 为 `6765c52b-3ad6-4fb0-9030-2c9a05dc4731`。请单独启动该服务器，并让接收端使用相同房间。

## 测试获准后的启动方式

新增文件仅经过源代码审阅。尚未执行导入、编译、模型推理、离线运行检查、摄像头采集或 Ubiq 传输。执行这些检查前，请先获得用户批准。

获得批准后，在 Node 目录中启动：

```powershell
Set-Location C:\ubiq-genie\Node
npm start YoloOnlyCapture
```

## 输出

外层网络消息在网络 ID **99** 上发送，格式为 `{ "data": "<complete stdout line>" }`。每行 Python stdout 保留原来的列表表示形式：

```text
[frame_count, object_count, '{"count": ..., "objects": [...]}']
```

每个对象包含 `id`、`pos: {x,y}`、`scale: {w,h}`、`status`、`label`、`class_id` 和 `confidence`。位置将检测框中心映射到 6 x 6 坐标区域，并保留两位小数。尺寸将左上角和右下角映射后，取坐标差的绝对值，并保留三位小数，与混合应用的计算一致。置信度是模型的检测分数，并非经过校准的物理正确性概率。

无检测结果的帧输出 `count: 0` 和 `objects: []`；漏检轨迹会在内部保留，但不会作为过时对象输出。YOLO 未检测到对象时，不会凭空创建 `unknown` 回退对象。

CSV 列保持为 `frame count,object count,udpstring`。CSV 和 stdout 来自同一数据包。诊断信息写入 stderr；Node 应用会将拆分或合并的 UTF-8 stdout 数据块重新组装为完整行后再传送。UDP 仍未使用。本应用不包含 Unity 接收器修改。

预测 API 参考：[Ultralytics 预测接口](https://docs.ultralytics.com/modes/predict/)。
