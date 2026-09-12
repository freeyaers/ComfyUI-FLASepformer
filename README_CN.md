# ComfyUI-FLASepformer

> 📖 [English README](README.md)

基于通义实验室 FLASepformer 模型的语音分离 ComfyUI 节点，可将双人混合语音拆分为两条独立音轨。

---

## 简介

**FLASepformer** 是通义实验室（阿里云 DAMO Academy）提出的高效时域语音分离模型，采用 Gated Focused Linear Attention（Gated FLA）架构，将长序列注意力复杂度从二次降低至线性，在保持高性能的同时大幅降低显存占用和推理时间。

本插件将 FLASepformer 模型封装为 ComfyUI 节点，支持：
- 输入任意采样率、任意声道数的音频，自动重采样至 8kHz 单声道
- 输出两路独立的说话人语音（8kHz 单声道）
- 支持 GPU（CUDA / ROCm）和 CPU 推理
- 可调节分离阶段数（4 / 6 / 8）和输出归一化

### 模型规格

| 项目 | 参数 |
|------|------|
| 模型名称 | speech_flatsepreformer_separation_temporal_8k_base_libri2mix100 |
| 架构 | FLA-SepReformer-B（纯时域） |
| 训练数据 | Libri2Mix-100 |
| 输入 | 8kHz 单声道混合语音 |
| 输出 | 2 路 8kHz 单声道独立语音 |
| ONNX Opset | 17 |
| 许可证 | CC BY-NC 4.0（非商业学术用途） |

### 节点参数

| 参数 | 选项 | 默认 | 说明 |
|------|------|------|------|
| `num_stages` | 4 / 6 / 8 | 4 | 分离阶段数，越大分离效果越好但推理越慢 |
| `normalize_output` | enabled / disabled | enabled | 输出归一化，防止削波（峰值缩放至 0.5） |
| `use_gpu` | auto / cpu / gpu | auto | 推理设备选择（auto 自动检测） |

---

## 工作流示例

![工作流示例](workflows/workflow.png)

工作流说明：
1. 输入音频（任意格式/采样率）连接至 `FY FLASepformer` 节点
2. 节点输出 `audio_1` 和 `audio_2` 两路分离结果
3. 分别连接至 `SaveAudio` 节点保存为独立音频文件

---

## 文件列表

```
ComfyUI-FLASepformer/
├── __init__.py          # 插件入口，模型路径与 ONNX 会话管理
├── nodes.py             # FY_FLASepformer 节点实现
├── requirements.txt     # 依赖声明（无额外依赖）
├── workflows/
│   ├── workflow.png     # 工作流截图
│   └── workflows.json   # ComfyUI 工作流文件
└── temp/                # 运行时临时目录（自动创建）
```

---

## 模型下载及存放位置

### 存放路径

插件会自动在以下路径查找模型，若不存在则自动下载：

```
ComfyUI/models/diffusers/speech_flatsepreformer_separation_temporal_8k_base_libri2mix100/
└── onnx_model.onnx
```

### 自动下载

首次运行节点时，插件会**自动检测**模型文件是否存在：
1. 若 `onnx_model.onnx` 已存在，直接加载使用
2. 若不存在，优先通过 **ModelScope CLI** 自动下载
3. ModelScope 失败时，自动回退到 **HuggingFace CLI** 下载
4. 两者均失败则报错并提示手动下载链接

> 无需手动下载，首次执行节点时插件会自动完成模型获取。

---

## 依赖说明

本插件无需额外安装 Python 依赖包，以下依赖已包含在 ComfyUI 内置 Python 环境中：

| 包名 | 版本 |
|------|------|
| onnxruntime | ≥ 1.23.2 |
| soundfile | ≥ 0.12.1 |
| torch | ≥ 2.13.0+rocm10.0.0 |
| torchaudio | ≥ 2.11.0.2+rocm10.0.0 |
| numpy | ≥ 2.5.1 |

---

## 注意事项

1. **输入音频**：可接受任意采样率和声道数，插件会自动处理。但输入为纯静音时推理可能异常。
2. **输出采样率**：固定为 8000 Hz 单声道，如需更高音质请使用其他模型。
3. **说话人顺序**：两路输出的说话人顺序不固定，同一说话人在不同音频中不一定对应相同序号。
4. **模型局限性**：本模型在干净条件下的双说话人场景表现最佳，强噪声、混响、音乐或歌声环境下性能可能下降。
5. **许可证**：模型采用 CC BY-NC 4.0 协议，仅限非商业学术研究用途。

---

## 感谢开源

- **通义实验室**（阿里云 DAMO Academy）：FLASepformer 模型及 ONNX 导出
- 论文：[FLASepformer: Efficient Speech Separation with Gated Focused Linear Attention Transformer](https://www.isca-archive.org/interspeech_2025/wang25j_interspeech.html)，Interspeech 2025

```bibtex
@inproceedings{wang25j_interspeech,
  title     = {{FLASepformer: Efficient Speech Separation with Gated Focused Linear Attention Transformer}},
  author    = {Haoxu Wang and Yiheng Jiang and Gang Qiao and Pengteng Shi and Biao Tian},
  year      = {2025},
  booktitle = {{Interspeech 2025}},
  pages     = {1468--1472},
  doi       = {10.21437/Interspeech.2025-1315},
}
```
