# Qwen3VL 钢材表面缺陷识别微调

用 QLoRA 对 Qwen3-VL-2B-Instruct 做参数高效微调，让通用多模态大模型学会定位钢材表面缺陷，并在公开基准 NEU-DET 上评估。训练与评估均在 Kaggle（T4 x2）上完成。

## 数据

[NEU-DET](https://www.kaggle.com/datasets/gmy723/neu-det-corrected)（东北大学钢材表面缺陷数据集，Hugging Face 镜像 `KeenForgeAI/NEU-DET-corrected`）：1,797 张灰度图（200×200），6 类缺陷（crazing / inclusion / patches / pitted_surface / rolled-in_scale / scratches），共 4,177 个标注框。

- 训练：1,437 张图按（图，类别）拆为 1,534 条样本，YOLO 标注转绝对坐标定位格式，30 条中文指令模板
- 测试：180 张图，195 个评测项，420 个真实框
- 损失只对答案 token 计算；灰度图转 RGB 后输入

## 方法

- 基座：Qwen3-VL-2B-Instruct，QLoRA（NF4 4-bit，r=8，alpha=16，dropout 0.05，target 全部 linear 层含视觉塔）
- 可训练参数 12,320,768 / 2,139,852,800（0.58%），lr 1e-4，等效 batch 16，2 epochs（192 步）
- 训练 loss 2.92 → 0.57，约 51 分钟；适配器见 [neudet-adapter-v3](https://www.kaggle.com/datasets/gmy723/neudet-adapter-v3)

## 评估结果

测试集（180 张、420 框），同一指令、贪心解码：

| 指标 | 基线 | 微调后 |
| --- | --- | --- |
| 平均 IoU | 0.211 | **0.325** |
| Acc@0.5 | 16.43% | **22.14%** |
| Acc@0.25 | 30.95% | **55.00%** |
| 格式合规率 | 100% | 100% |

逐类 Acc@0.5（基线 → 微调）：crazing 0% → 18.75%，inclusion 11.93% → 10.09%，patches 26.58% → 21.52%，pitted_surface 47.73% → 68.18%，rolled-in_scale 2.78% → 9.72%，scratches 23.08% → 30.77%。

如实说明：inclusion 与 patches 两类 Acc@0.5 微降；微调预测框 766 个多于基线的 468 个，提升部分来自多出框，Prec@0.5 仅 14.74% → 16.71%；pitted_surface 的框常接近整图，该类 IoU 易虚高。完整数字见 [results/eval-v3-neudet.md](results/eval-v3-neudet.md)。

## 部署

权重合并回主干模型 → GGUF 量化（Q4_K_M）→ Docker + llama.cpp HTTP 服务，树莓派 4B 端侧推理。说明：v3 当前只保存了适配器，GGUF 合并与端侧部署按此前同一管线进行。

## DPO（进行中）

偏好数据已备好（`dpo/`，训练 1,534 对 / 验证 192 对；chosen 为该类全部真实框，rejected 为全部框各平移 ±5% 框尺寸），训练笔记本见 `notebooks/qwen3vl-dpo-v3-neudet.ipynb`，训练进行中，结果待补充。

## 文件说明

| 目录 | 内容 |
| --- | --- |
| `notebooks/` | Kaggle 训练 / 评估 / DPO 笔记本（直接在 Kaggle 上跑） |
| `scripts/` | 生成上述笔记本的脚本（本地生成，改完脚本重新生成笔记本） |
| `dpo/` | DPO 偏好对数据（JSON） |
| `results/` | v3 评估完整结果 |

## 在 Kaggle 上复现

1. 打开 `notebooks/qwen3vl-retrain-v3-neudet.ipynb`，加速器选 GPU T4 x2，打开 Internet
2. 挂两个输入数据集：[neu-det-corrected](https://www.kaggle.com/datasets/gmy723/neu-det-corrected)（数据）与训练输出的适配器
3. Save & Run All；评估笔记本同理，挂适配器数据集后运行

## 相关链接

- 训练笔记本：https://www.kaggle.com/code/gmy723/qwen3vl-retrain-v3-neudet
- 评估笔记本：https://www.kaggle.com/code/gmy723/qwen3vl-eval-kaggle-v3-neudet
- DPO 笔记本：https://www.kaggle.com/code/gmy723/qwen3vl-dpo-v3-neudet
