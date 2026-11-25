---
title: Qwen3VL Steel Defect Detection
emoji: 🔍
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
---

Qwen3-VL-2B-Instruct 经 QLoRA 在公开基准 NEU-DET 上微调,定位钢材表面缺陷。
测试集:平均 IoU 0.325(基线 0.211)、Acc@0.5 22.14%(基线 16.43%)、Acc@0.25 55.00%(基线 30.95%)。

代码与训练细节:[blankGMY/qwen3vl-steel-defect](https://github.com/blankGMY/qwen3vl-steel-defect)
