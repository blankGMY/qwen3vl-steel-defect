"""Qwen3VL 钢材表面缺陷识别 - 在线演示.

上传一张钢材表面图片,选择缺陷类别,模型输出缺陷边界框.
左侧为微调后模型,中间为真实标注(绿色框,仅 NEU-DET 测试集图片有标注),
右侧为未微调的基线模型,便于直接对比.

推理逻辑与评估笔记本完全一致:同一指令、贪心解码、max_new_tokens=256.
模型:Qwen3-VL-2B-Instruct + QLoRA 适配器(在 NEU-DET 上微调,2 epochs).
测试集成绩:平均 IoU 0.325(基线 0.211)、Acc@0.5 22.14%(基线 16.43%)、
Acc@0.25 55.00%(基线 30.95%)、格式合规率 100%.
"""

import hashlib
import json
import os
import re

import gradio as gr
import torch
from PIL import Image, ImageDraw
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

try:
    import spaces

    gpu = spaces.GPU(duration=120)
except Exception:  # 本地运行无 spaces 包时直接跑
    def gpu(fn):
        return fn


BASE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
ADAPTER_REPO = os.environ.get("ADAPTER_REPO", "GuoMingYu/neudet-adapter-v3")

CLS_CN = {
    "crazing": "裂纹",
    "inclusion": "夹杂物",
    "patches": "斑块",
    "pitted_surface": "麻点",
    "rolled-in_scale": "氧化皮",
    "scratches": "划痕",
}
CLS_CHOICES = [(f"{cn} ({en})", en) for en, cn in CLS_CN.items()]


# ---- 真实标注 (GT) 对照 ----
# gt_lookup.json 由 NEU-DET 测试集逐项评估结果生成:主键为图片像素内容哈希
# (PIL 转 RGB 后 tobytes() 的 md5),文件名作备用键;每项含原图尺寸与各类别 GT 框
# (像素坐标)。命中时画绿框对照,未命中(非测试集图片)不报错、只显示原图。
try:
    with open(os.path.join(os.path.dirname(__file__), "gt_lookup.json"), encoding="utf-8") as _f:
        _GT_LOOKUP = json.load(_f)
except Exception:
    _GT_LOOKUP = {"by_hash": {}, "by_name": {}}


def lookup_gt(img, cls_key):
    """返回按上传图尺寸缩放后的 GT 像素坐标框列表;未命中返回 None."""
    w, h = img.size
    rgb = img.convert("RGB")
    key = hashlib.md5(rgb.tobytes()).hexdigest()
    entry = _GT_LOOKUP.get("by_hash", {}).get(key)
    if entry is None:
        fname = getattr(img, "filename", None)
        if fname:
            entry = _GT_LOOKUP.get("by_name", {}).get(os.path.basename(fname))
    if entry is None:
        return None
    sw, sh = entry["size"]
    boxes = entry["boxes"].get(cls_key)
    if not boxes:
        return None
    sx, sy = w / sw, h / sh
    return [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in boxes]


def instruction_for(cls_name):
    return f"请识别图片中的钢材表面缺陷（{CLS_CN[cls_name]}），并用边界框标出所有位置。"


def parse_boxes(text, w, h):
    boxes = []
    for m in re.finditer(
        r"<\|object_ref\|>\s*([0-9]+)[,\s]+([0-9]+)[,\s]+([0-9]+)[,\s]+([0-9]+)\s*<\|/object_ref\|>",
        text,
    ):
        boxes.append([float(m.group(i)) for i in range(1, 5)])
    if boxes:
        return boxes
    for m in re.finditer(
        r'"x_min"\D+([0-9.]+)\D+"y_min"\D+([0-9.]+)\D+"x_max"\D+([0-9.]+)\D+"y_max"\D+([0-9.]+)',
        text,
    ):
        x1, y1, x2, y2 = (float(m.group(i)) for i in range(1, 5))
        boxes.append([x1 / 1000 * w, y1 / 1000 * h, x2 / 1000 * w, y2 / 1000 * h])
    if boxes:
        return boxes
    for m in re.finditer(r"\[\s*([0-9.]+)[,\s]+([0-9.]+)[,\s]+([0-9.]+)[,\s]+([0-9.]+)\s*\]", text):
        vals = [float(m.group(i)) for i in range(1, 5)]
        if max(vals) <= 1000 and (w > 1000 or h > 1000 or max(vals) > max(w, h)):
            vals = [vals[0] / 1000 * w, vals[1] / 1000 * h, vals[2] / 1000 * w, vals[3] / 1000 * h]
        boxes.append(vals)
    return boxes


def draw_boxes(img, boxes, color, label):
    out = img.copy()
    d = ImageDraw.Draw(out)
    for b in boxes:
        d.rectangle([b[0], b[1], b[2], b[3]], outline=color, width=max(2, img.width // 100))
    d.text((4, 4), label, fill=color)
    return out


print("loading processor ...")
processor = AutoProcessor.from_pretrained(BASE_MODEL)
print("processor ready (model loads on first GPU request)")

_model = None


def _get_model():
    # ZeroGPU gives no CUDA at app startup: load the model lazily inside the
    # @spaces.GPU-decorated predict call, where a GPU is allocated.
    global _model
    if _model is None:
        print("loading base model ...")
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
        )
        print("loading adapter ...", ADAPTER_REPO)
        _model = PeftModel.from_pretrained(base, ADAPTER_REPO)
        _model.eval()
        print("model ready")
    return _model


@torch.no_grad()
def _generate(model, img, instruction):
    msgs = [
        {
            "role": "user",
            "content": [{"type": "image", "image": img}, {"type": "text", "text": instruction}],
        }
    ]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=256, do_sample=False)
    gen = out[0][enc["input_ids"].shape[1]:]
    return processor.decode(gen, skip_special_tokens=False)


@gpu
def predict(img, cls_key):
    if img is None:
        return None, None, None, "请先上传图片。", ""
    img = img.convert("RGB")
    w, h = img.size
    instruction = instruction_for(cls_key)
    model = _get_model()

    raw_ft = _generate(model, img, instruction)
    boxes_ft = parse_boxes(raw_ft, w, h)
    annotated_ft = draw_boxes(img, boxes_ft, (255, 0, 0), f"微调后: {len(boxes_ft)} 个框")

    with model.disable_adapter():
        raw_base = _generate(model, img, instruction)
    boxes_base = parse_boxes(raw_base, w, h)
    annotated_base = draw_boxes(img, boxes_base, (255, 165, 0), f"基线: {len(boxes_base)} 个框")

    gt_boxes = lookup_gt(img, cls_key)
    if gt_boxes is not None:
        annotated_gt = draw_boxes(img, gt_boxes, (0, 200, 0), f"真实标注: {len(gt_boxes)} 个框")
        gt_note = f"真实标注 {len(gt_boxes)} 个框。"
    else:
        annotated_gt = img
        gt_note = "该图片无真实标注（仅 NEU-DET 测试集图片有标注）。"

    info = (
        f"微调后检出 {len(boxes_ft)} 个框;基线检出 {len(boxes_base)} 个框。"
        f"类别:{CLS_CN[cls_key]}({cls_key})。{gt_note}"
    )
    return annotated_ft, annotated_gt, annotated_base, info, raw_ft


with gr.Blocks(title="Qwen3VL 钢材表面缺陷识别") as demo:
    gr.Markdown(
        """# Qwen3VL 钢材表面缺陷识别

        Qwen3-VL-2B-Instruct 经 QLoRA 在公开基准 NEU-DET 上微调,定位钢材表面缺陷。
        测试集(180 张、420 框):平均 IoU **0.325**(基线 0.211)、Acc@0.5 **22.14%**(基线 16.43%)、
        Acc@0.25 **55.00%**(基线 30.95%)。上传图片并选择缺陷类别,左侧为微调后模型,
        中间为真实标注(绿色框,仅 NEU-DET 测试集图片有标注),右侧为基线模型。
        """
    )
    with gr.Row():
        img_in = gr.Image(type="pil", label="上传钢材表面图片")
        cls_in = gr.Dropdown(choices=CLS_CHOICES, value="crazing", label="缺陷类别")
    btn = gr.Button("检测", variant="primary")
    with gr.Row():
        img_ft = gr.Image(label="微调后 (Qwen3VL + NEU-DET 适配器)")
        img_gt = gr.Image(label="真实标注 (GT)")
        img_base = gr.Image(label="基线 (Qwen3-VL-2B-Instruct 原版)")
    info_out = gr.Textbox(label="结果")
    raw_out = gr.Textbox(label="模型原始输出 (微调后)", lines=4)
    gr.Examples(
        examples=[
            ["examples/patches_219.jpg", "patches"],
            ["examples/scratches_201.jpg", "scratches"],
            ["examples/inclusion_82.jpg", "inclusion"],
        ],
        inputs=[img_in, cls_in],
        label="示例图片 (NEU-DET 测试集,点一下试试)",
    )
    btn.click(predict, inputs=[img_in, cls_in], outputs=[img_ft, img_gt, img_base, info_out, raw_out])

if __name__ == "__main__":
    demo.launch()
