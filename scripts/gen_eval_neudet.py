# Build the Kaggle evaluation notebook v3-NEU-DET: base Qwen3-VL-2B-Instruct
# vs fine-tuned v3 (adapter neudet-adapter-v3) on the NEU-DET test split
# (180 images, 6 classes). Same fixed instruction style, greedy decoding,
# IoU metrics as the FabricSpotDefect evals, PLUS per-class breakdown
# (pitted_surface / crazing boxes are loose by annotation nature and would
# otherwise inflate the overall number silently).
# Data arrives from the attached Kaggle dataset neu-det-corrected (HF mirror
# only as fallback - HF 429-blocked Kaggle egress on 2026-11-24, V3-V6 died
# there); the adapter arrives as the attached Kaggle dataset
# neudet-adapter-v3 (searched under /kaggle/input, fallback /kaggle/working).
# Training v3 DONE 2026-11-24 ~15:16 CST: 192/192 steps, loss 2.922 -> 0.5672.
import json

NB_PATH = '/home/hatch/workspace/job-search/qwen3vl-eval-kaggle-v3-neudet.ipynb'

def md(src):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': src}

def code(src):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': src}

cells = []

cells.append(md(
    '# Qwen3-VL 钢材缺陷定位评估（Kaggle 版）：基线 vs 补训 v3（NEU-DET）\n'
    '留出测试集 NEU-DET test（180 张图、195 个（图，类别）评测项，15 张图含 2 类）、每图一条固定指令（写明该图缺陷类别）、贪心解码。\n'
    '指标：格式合规率、mean IoU、Acc@IoU>=0.5、Acc@IoU>=0.25、Prec@IoU>=0.5，另附逐类 mean IoU / Acc@0.5\n'
    '（crazing 与 pitted_surface 的标注框本身偏松，总体数需对照逐类数看）。结果存 /kaggle/working/eval_results_neudet/。'))

cells.append(code(
    '# ==== 1. 数据（Kaggle 输入优先，HF 后备）与 v3 适配器（Kaggle 数据集输入） ====\n'
    'import os, subprocess, sys, glob\n'
    "WORK = '/kaggle/working'\n"
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'transformers==4.57.1', 'accelerate', 'peft', 'huggingface_hub'], check=False)\n"
    "import transformers\n"
    "assert transformers.__version__ == '4.57.1', transformers.__version__\n"
    "print('transformers', transformers.__version__)\n"
    '\n'
    '# NEU-DET: Kaggle input first (nested path), HF download only as fallback\n'
    '# (HF 429-blocked Kaggle egress IPs on 2026-11-24, V3-V6 all died here).\n'
    "def find_data_root():\n"
    "    for cand in ['/kaggle/input/datasets/gmy723/neu-det-corrected',\n"
    "                '/kaggle/input/neu-det-corrected']:\n"
    "        if os.path.isdir(os.path.join(cand, 'images')):\n"
    "            return cand\n"
    "    return None\n"
    '\n'
    "DATA_ROOT = find_data_root()\n"
    "if DATA_ROOT is None:\n"
    "    from huggingface_hub import snapshot_download\n"
    "    DATA_ROOT = snapshot_download(repo_id='KeenForgeAI/NEU-DET-corrected', repo_type='dataset',\n"
    "                                  allow_patterns=['images/test/**', 'labels/test/**', 'classes.txt'])\n"
    "print('NEU-DET test ->', DATA_ROOT)\n"
    '\n'
    "def find_dir(root, marker):\n"
    '    for d, _, files in os.walk(root):\n'
    '        if marker in files:\n'
    '            return d\n'
    '    return None\n'
    '\n'
    "ADAPTER = find_dir('/kaggle/input', 'adapter_config.json') or find_dir(WORK, 'adapter_config.json')\n"
    "print('ADAPTER dir:', ADAPTER)\n"
    "assert ADAPTER, 'adapter not found - attach the neudet-adapter-v3 dataset as input'\n"))

cells.append(code(
    '# ==== 2. 加载测试集（180 张，逐类计数） ====\n'
    'import os, re, json, glob, collections\n'
    'import torch\n'
    'from PIL import Image\n'
    'from transformers import AutoProcessor, Qwen3VLForConditionalGeneration\n'
    '\n'
    "CLS_CN = {'crazing': '裂纹', 'inclusion': '夹杂物', 'patches': '斑块',\n"
    "          'pitted_surface': '麻点', 'rolled-in_scale': '氧化皮', 'scratches': '划痕'}\n"
    "CLASSES = open(os.path.join(DATA_ROOT, 'classes.txt')).read().split()\n"
    "TEST_LBL = os.path.join(DATA_ROOT, 'labels', 'test')\n"
    "TEST_IMG = os.path.join(DATA_ROOT, 'images', 'test')\n"
    "OUT_DIR = os.path.join(WORK, 'eval_results_neudet')\n"
    'os.makedirs(OUT_DIR, exist_ok=True)\n'
    '\n'
    'def instruction_for(cls_name):\n'
    "    return f'请识别图片中的钢材表面缺陷（{CLS_CN[cls_name]}），并用边界框标出所有位置。'\n"
    '\n'
    'def load_items():\n'
    '    items = []\n'
    "    for lp in sorted(glob.glob(os.path.join(TEST_LBL, '*.txt'))):\n"
    '        name = os.path.splitext(os.path.basename(lp))[0]\n'
    "        ip = os.path.join(TEST_IMG, name + '.jpg')\n"
    '        if not os.path.exists(ip):\n'
    '            continue\n'
    '        try:\n'
    '            w, h = Image.open(ip).size\n'
    '        except Exception:\n'
    '            continue\n'
    '        by_cls = {}\n'
    "        for line in open(lp, encoding='utf-8'):\n"
    '            p = line.strip().split()\n'
    '            if len(p) < 5:\n'
    '                continue\n'
    '            c = int(p[0])\n'
    '            cx, cy, bw, bh = map(float, p[1:5])\n'
    '            by_cls.setdefault(c, []).append([(cx - bw / 2) * w, (cy - bh / 2) * h,\n'
    '                                              (cx + bw / 2) * w, (cy + bh / 2) * h])\n'
    '        # One eval item per (image, class), same as training samples:\n'
    '        # 15 test images contain 2 classes -> 180 images = 195 items.\n'
    '        for c, boxes in sorted(by_cls.items()):\n'
    "            items.append({'path': ip, 'w': w, 'h': h, 'gt': boxes,\n"
    "                          'cls': CLASSES[c],\n"
    "                          'instruction': instruction_for(CLASSES[c])})\n"
    '    return items\n'
    '\n'
    'items = load_items()\n'
    "print('test images:', len({it['path'] for it in items}), '| eval items (image, class):', len(items))\n"
    "print('per-class test counts:', dict(collections.Counter(it['cls'] for it in items)))\n"
    "print('total GT boxes:', sum(len(it['gt']) for it in items))\n"
    "assert len(items) == 195, 'expected 195 eval items, got %d' % len(items)"))

cells.append(code(
    '# ==== 3. 解析 / IoU / 评估函数（与织物版评估完全一致，指令改为逐图类别） ====\n'
    'def parse_boxes(text, w, h):\n'
    '    boxes = []\n'
    "    for m in re.finditer(r'<\\|object_ref\\|>\\s*([0-9]+)[,\\s]+([0-9]+)[,\\s]+([0-9]+)[,\\s]+([0-9]+)\\s*<\\|/object_ref\\|>', text):\n"
    '        boxes.append([float(m.group(i)) for i in range(1, 5)])\n'
    '    if boxes:\n'
    '        return boxes\n'
    "    for m in re.finditer(r'\"x_min\"\\D+([0-9.]+)\\D+\"y_min\"\\D+([0-9.]+)\\D+\"x_max\"\\D+([0-9.]+)\\D+\"y_max\"\\D+([0-9.]+)', text):\n"
    '        x1, y1, x2, y2 = (float(m.group(i)) for i in range(1, 5))\n'
    '        boxes.append([x1 / 1000 * w, y1 / 1000 * h, x2 / 1000 * w, y2 / 1000 * h])\n'
    '    if boxes:\n'
    '        return boxes\n'
    "    for m in re.finditer(r'\\[\\s*([0-9.]+)[,\\s]+([0-9.]+)[,\\s]+([0-9.]+)[,\\s]+([0-9.]+)\\s*\\]', text):\n"
    '        vals = [float(m.group(i)) for i in range(1, 5)]\n'
    '        if max(vals) <= 1000 and (w > 1000 or h > 1000 or max(vals) > max(w, h)):\n'
    '            vals = [vals[0] / 1000 * w, vals[1] / 1000 * h, vals[2] / 1000 * w, vals[3] / 1000 * h]\n'
    '        boxes.append(vals)\n'
    '    return boxes\n'
    '\n'
    'def iou(a, b):\n'
    '    x1, y1 = max(a[0], b[0]), max(a[1], b[1])\n'
    '    x2, y2 = min(a[2], b[2]), min(a[3], b[3])\n'
    '    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)\n'
    '    area = lambda r: max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])\n'
    '    union = area(a) + area(b) - inter\n'
    '    return inter / union if union > 0 else 0.0\n'
    '\n'
    '@torch.no_grad()\n'
    'def run_eval(model, processor, tag):\n'
    '    per_image = []\n'
    "    save_path = os.path.join(OUT_DIR, 'results_' + tag + '.json')\n"
    '    for idx, it in enumerate(items):\n'
    "        img = Image.open(it['path']).convert('RGB')\n"
    "        msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': img}, {'type': 'text', 'text': it['instruction']}]}]\n"
    '        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)\n'
    "        enc = processor(text=[text], images=[img], return_tensors='pt').to('cuda')\n"
    '        out = model.generate(**enc, max_new_tokens=256, do_sample=False)\n'
    "        gen = out[0][enc['input_ids'].shape[1]:]\n"
    "        resp = processor.decode(gen, skip_special_tokens=False)\n"
    "        preds = parse_boxes(resp, it['w'], it['h'])\n"
    "        per_image.append({'path': it['path'], 'cls': it['cls'], 'gt': it['gt'], 'pred': preds, 'raw': resp[:300]})\n"
    '        if (idx + 1) % 10 == 0 or idx + 1 == len(items):\n'
    "            print(tag, idx + 1, '/', len(items))\n"
    "            with open(save_path, 'w', encoding='utf-8') as f:\n"
    '                json.dump(per_image, f, ensure_ascii=False)\n'
    '    return per_image\n'
    '\n'
    'def summarize(per_image):\n'
    '    n_items = len(per_image)\n'
    "    n_img = len({r['path'] for r in per_image})\n"
    "    fmt = sum(1 for r in per_image if r['pred']) / max(n_items, 1)\n"
    '    gt_best, pred_hit, n_gt, n_pred = [], 0, 0, 0\n'
    '    for r in per_image:\n'
    "        for g in r['gt']:\n"
    "            gt_best.append(max([iou(g, p) for p in r['pred']], default=0.0))\n"
    '            n_gt += 1\n'
    "        for p in r['pred']:\n"
    '            n_pred += 1\n'
    "            if max([iou(p, g) for g in r['gt']], default=0.0) >= 0.5:\n"
    '                pred_hit += 1\n'
    '    mean_iou = sum(gt_best) / max(len(gt_best), 1)\n'
    '    acc50 = sum(1 for v in gt_best if v >= 0.5) / max(len(gt_best), 1)\n'
    '    acc25 = sum(1 for v in gt_best if v >= 0.25) / max(len(gt_best), 1)\n'
    '    prec50 = pred_hit / max(n_pred, 1)\n'
    "    # duplicate-box check (v2 collapse signature): dedup changes nothing if healthy\n"
    '    n_pred_uniq = 0\n'
    '    for r in per_image:\n'
    "        n_pred_uniq += len({tuple(p) for p in r['pred']})\n"
    "    per_cls = {}\n"
    '    for cls in CLASSES:\n'
    "        sub = [r for r in per_image if r['cls'] == cls]\n"
    '        if not sub:\n'
    '            continue\n'
    '        best = []\n'
    '        for r in sub:\n'
    "            for g in r['gt']:\n"
    '                best.append(max([iou(g, p) for p in r["pred"]], default=0.0))\n'
    "        per_cls[cls] = {'images': len(sub), 'gt_boxes': len(best),\n"
    "                        'mean_iou': round(sum(best) / max(len(best), 1), 4),\n"
    "                        'acc_iou50': round(sum(1 for v in best if v >= 0.5) / max(len(best), 1), 4)}\n"
    "    return {'images': n_img, 'eval_items': n_items, 'gt_boxes': n_gt, 'pred_boxes': n_pred,\n"
    "            'pred_boxes_unique': n_pred_uniq,\n"
    "            'format_rate': round(fmt, 4), 'mean_iou': round(mean_iou, 4),\n"
    "            'acc_iou50': round(acc50, 4), 'acc_iou25': round(acc25, 4),\n"
    "            'prec_iou50': round(prec50, 4), 'per_class': per_cls}\n"
    "print('helpers ready')"))

cells.append(code(
    '# ==== 4. 基线：原版 Qwen3-VL-2B-Instruct ====\n'
    "base = Qwen3VLForConditionalGeneration.from_pretrained('Qwen/Qwen3-VL-2B-Instruct', torch_dtype=torch.float16, device_map='cuda')\n"
    "proc = AutoProcessor.from_pretrained('Qwen/Qwen3-VL-2B-Instruct')\n"
    "res_base = run_eval(base, proc, 'neudet_base')\n"
    'sum_base = summarize(res_base)\n'
    "print('BASE summary:', json.dumps(sum_base, ensure_ascii=False))\n"
    'del base\n'
    'torch.cuda.empty_cache()'))

cells.append(code(
    '# ==== 5. 微调 v3：基座 + 适配器 neudet-adapter-v3 ====\n'
    'from peft import PeftModel\n'
    "base2 = Qwen3VLForConditionalGeneration.from_pretrained('Qwen/Qwen3-VL-2B-Instruct', torch_dtype=torch.float16, device_map='cuda')\n"
    'ft = PeftModel.from_pretrained(base2, ADAPTER)\n'
    'ft.eval()\n'
    "proc_ft = AutoProcessor.from_pretrained('Qwen/Qwen3-VL-2B-Instruct')\n"
    "res_ft = run_eval(ft, proc_ft, 'neudet_finetuned_v3')\n"
    'sum_ft = summarize(res_ft)\n'
    "print('FINETUNED_V3 summary:', json.dumps(sum_ft, ensure_ascii=False))\n"
    'del ft, base2\n'
    'torch.cuda.empty_cache()'))

cells.append(code(
    '# ==== 6. 汇总保存 ====\n'
    "final = {'dataset': 'NEU-DET test (180 images)', 'base': sum_base, 'finetuned_v3': sum_ft}\n"
    "with open(os.path.join(OUT_DIR, 'eval_summary_neudet.json'), 'w', encoding='utf-8') as f:\n"
    '    json.dump(final, f, ensure_ascii=False, indent=2)\n'
    'print(json.dumps(final, ensure_ascii=False, indent=2))\n'
    "print('saved to', OUT_DIR)"))

nb = {'nbformat': 4, 'nbformat_minor': 0,
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
                   'accelerator': 'GPU'},
      'cells': cells}

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('written', NB_PATH)
