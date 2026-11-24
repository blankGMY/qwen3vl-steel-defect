# Build the Kaggle training notebook v3: Qwen3-VL-2B QLoRA on NEU-DET
# (steel surface defect detection, 6 classes, 1797 images).
# Why a new dataset: FabricSpotDefect v2 collapsed (mode collapse, mean IoU
# 0.0173 vs base 0.1084). Diagnosis showed the fabric targets are too small
# to learn (median box 0.46% of image, 33% < 30x30). NEU-DET boxes are ~26x
# larger (median 11.82% of image, 0% < 30x30), verified from the actual
# labels on 2026-11-24; overlay spot-check (6 classes) shows boxes align
# with visible defects; centers are spread (only 11.9% in central 20%
# square), so there is no positional shortcut to exploit.
# Changes vs v2 (keep everything else identical for clean attribution):
#   1. Data: NEU-DET train split (1437 images) from HF mirror
#      KeenForgeAI/NEU-DET-corrected (YOLO labels + 1437/180/180 split).
#   2. Epochs 3 -> 2 (collapse guard: v2 overfit the repeat-one-box pattern).
#   3. Data-verification cell BEFORE training: per-class counts, center
#      spread, dedupe of identical boxes, 6 overlay images saved.
#   4. EOS check added to the masking self-check: kept labels must end with
#      <|im_end|>, otherwise the model never learns to stop (v2 failure mode).
#   5. No merge step: adapter only (eval notebook loads base + adapter).
#      Vision tower is already trained: target_modules="all-linear" covers
#      vision-tower linears too (same as v2).
# Output: adapter /kaggle/working/neudet-adapter-v3 + train_log_v3.json.
import json

NB_PATH = '/home/hatch/workspace/job-search/qwen3vl-retrain-v3-neudet.ipynb'

CLS_CN = {
    'crazing': '裂纹',
    'inclusion': '夹杂物',
    'patches': '斑块',
    'pitted_surface': '麻点',
    'rolled-in_scale': '氧化皮',
    'scratches': '划痕',
}

TEMPLATES = [
    # 直接定位
    '请识别图片中的钢材表面缺陷（{cn}），并用边界框标出所有位置。',
    '找出图片中所有的钢材表面缺陷（{cn}），用边界框标出。',
    '定位图片中的钢材表面缺陷（{cn}），给出每个缺陷的边界框。',
    '检测图片中的钢材表面缺陷（{cn}），并框出它们的位置。',
    '请标出这张钢材图片中所有缺陷（{cn}）的位置。',
    '在这张图片中找到钢材表面缺陷（{cn}），用边界框标出。',
    # 框出/圈出
    '请用边界框框出图片中的所有钢材表面缺陷（{cn}）。',
    '把图片中钢材表面缺陷（{cn}）的位置用框圈出来。',
    '用矩形框标出图片里所有的钢材表面缺陷（{cn}）。',
    '请圈出图中所有的钢材表面缺陷（{cn}）。',
    '找出图中的钢材表面缺陷（{cn}）并用方框标出。',
    # 检测类
    '检测这张钢材图片，定位其中的缺陷（{cn}）。',
    '对图片进行缺陷检测，标出所有钢材表面缺陷（{cn}）。',
    '请检测图片中的钢材表面缺陷（{cn}）并给出边界框。',
    '执行钢材表面缺陷检测，标出图中所有{cn}的位置。',
    '检查这张图片，找出其中的钢材表面缺陷（{cn}）并定位。',
    # 识别类
    '识别图片中的钢材表面缺陷（{cn}），给出边界框坐标。',
    '请识别图中所有的钢材表面缺陷（{cn}）并标出位置。',
    '这张图片里有哪些钢材表面缺陷（{cn}）？请用边界框标出。',
    '识别并定位图片中的钢材表面缺陷（{cn}）。',
    # 位置询问
    '钢材表面缺陷（{cn}）在图片的什么位置？请用边界框标出。',
    '图中的钢材表面缺陷（{cn}）在哪里？请框出来。',
    '请指出图片中钢材表面缺陷（{cn}）的位置。',
    # 数量/全部
    '请标出图片中每一个钢材表面缺陷（{cn}）。',
    '不要漏掉任何一个，标出图片中所有的钢材表面缺陷（{cn}）。',
    '图中共有几个钢材表面缺陷（{cn}）？请全部框出来。',
    '列出图片中全部钢材表面缺陷（{cn}）的边界框。',
    # 简洁指令
    '标出缺陷（{cn}）。',
    '定位缺陷（{cn}）。',
    '框出所有{cn}。',
]
assert len(TEMPLATES) == 30, len(TEMPLATES)

def md(src):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': src}

def code(src):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': src}

cells = []

cells.append(md(
    '# Qwen3-VL 补训 v3：NEU-DET 钢材表面缺陷定位（只训答案 + 2 epochs）\n'
    '\n'
    '背景：FabricSpotDefect 补训 v2 模式坍缩（mean IoU 0.0173，低于基线 0.1084；151/152 张输出 9 个框、134 张所有框一模一样）。\n'
    '诊断结论：织物斑点目标太小（框中位数只占图 0.46%），模型学不到定位信号；同时 3 epochs 过拟合了重复模式。\n'
    '\n'
    '本版改动（其余与 v2 完全相同，便于归因）：\n'
    '1. **换数据集 NEU-DET**：1797 张钢材表面缺陷，6 类；框中位数占图 11.82%（斑点的约 26 倍），0% 的框小于 30×30；\n'
    '   框中心分布散开（仅 11.9% 落在中心 20% 区域内），无位置捷径；6 类叠加抽查框与可见缺陷对得上。\n'
    '2. **2 epochs**（v2 是 3）：防坍缩。训练集 1,437 张图、1,534 个（图，类别）样本（95 张图含 2 类、2 张含 3 类，混类图按类别拆成多条样本），等效 batch 16，约 192 步，T4 上约 1 小时。\n'
    '3. **训练前先验数据**：逐类计数、中心分布、去重、存 6 张叠加图，任一项异常即停，不烧 GPU。\n'
    '4. **EOS 自查**：kept labels 必须以 `<|im_end|>` 结尾，否则模型学不会停（v2 的失败模式）。\n'
    '5. **只存适配器**（`neudet-adapter-v3`），不合并——评估本直接加载基座 + 适配器。\n'
    '6. **2026-11-24 修复（首跑失败后）**：① transformers 锁定 4.57.1——首跑时 `pip install -U` 拉到最新 5.x，\n'
    '   其 Qwen3-VL 视觉路径与本管线不兼容，训练第一步即报 `hidden_states + pos_embeds` 尺寸不匹配；\n'
    '   ② **修掉 v2 就有的 pixel_values[0] 取首补丁 bug**：处理器返回的 pixel_values 已是切块后的 (256, 1536)、无 batch 维，\n'
    '   `[0]` 只剩第一个补丁，模型看到的是同一补丁在 256 个位置上重复——视觉信号基本是空的，\n'
    '   这很可能就是 v2 只学会格式、定位坍缩到众数框的根因；自查 cell 已加补丁数与 grid 一致性断言。\n'
    '7. **2026-11-24 绕过 HF 429（V3–V6 均死于此）**：Kaggle 出口 IP 匿名下载 NEU-DET 被 Hugging Face 限流，\n'
    '   三次重跑换了三个出口 IP 全撞 429。改走 Kaggle 输入：数据集 zip（neu-det-corrected.zip，本地同名文件）\n'
    '   随笔记本挂为输入，首 cell 先解压它，HF 下载只作后备。\n'
    '\n'
    '其余与 v2 相同：基座 Qwen/Qwen3-VL-2B-Instruct、QLoRA NF4 r=8 alpha=16、lr 1e-4、等效 batch 16、损失只算答案。\n'
    '验收线（NEU-DET 测试集 180 张，同口径评估）：格式合规率接近 100%，且 Acc@IoU>=0.5 高于基线。'))

cells.append(code(
    '!pip install -q -U "transformers==4.57.1" accelerate peft bitsandbytes huggingface_hub\n'
    'import os\n'
    '# ---- Dataset source: Kaggle input first, HF as fallback ----\n'
    '# HF 429: anonymous downloads from Kaggle egress IPs are rate-limited\n'
    '# (V3-V6 all died here). The dataset is attached as Kaggle input.\n'
    '# Probe (2026-11-24) found the real mount path is nested:\n'
    '# /kaggle/input/datasets/gmy723/neu-det-corrected (extracted files,\n'
    '# NOT a zip) - the flat /kaggle/input/neu-det-corrected guess was wrong.\n'
    'DATA_ROOT = None\n'
    "for _cand in ['/kaggle/input/datasets/gmy723/neu-det-corrected',\n"
    "              '/kaggle/input/neu-det-corrected']:\n"
    "    _z = os.path.join(_cand, 'neu-det-corrected.zip')\n"
    '    if os.path.isfile(_z):\n'
    '        import zipfile\n'
    "        DATA_ROOT = '/kaggle/working/neudet'\n"
    "        print('unzipping dataset from', _z)\n"
    '        zipfile.ZipFile(_z).extractall(DATA_ROOT)\n'
    '        break\n'
    "    if os.path.isdir(os.path.join(_cand, 'images')):\n"
    '        DATA_ROOT = _cand\n'
    '        break\n'
    'if DATA_ROOT is None:\n'
    '    from huggingface_hub import snapshot_download\n'
    "    DATA_ROOT = snapshot_download(repo_id='KeenForgeAI/NEU-DET-corrected', repo_type='dataset')\n"
    "print('NEU-DET ->', DATA_ROOT)\n"
    'import transformers\n'
    "assert transformers.__version__ == '4.57.1', f\"pinned transformers 4.57.1 required, got {transformers.__version__}\"\n"
    "print('transformers', transformers.__version__, '(pinned: latest 5.x breaks the Qwen3-VL vision path)')\n"
    "import torch\n"
    "print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"))

cells.append(code(
    'import os, glob, random, json\n'
    'from PIL import Image\n'
    '\n'
    "CLS_CN = " + repr(CLS_CN) + '\n'
    'TEMPLATES = ' + repr(TEMPLATES) + '\n'
    'assert len(TEMPLATES) == 30\n'
    '\n'
    "TRAIN_LBL = os.path.join(DATA_ROOT, 'labels', 'train')\n"
    "TRAIN_IMG = os.path.join(DATA_ROOT, 'images', 'train')\n"
    "CLASSES = open(os.path.join(DATA_ROOT, 'classes.txt')).read().split()\n"
    "WORK = '/kaggle/working'\n"
    "ADAPTER_DIR = os.path.join(WORK, 'neudet-adapter-v3')\n"
    '\n'
    'def build_samples(seed=42):\n'
    '    rng = random.Random(seed)\n'
    '    samples, skipped = [], {"no_image": 0, "no_box": 0, "dup_box": 0, "bad_image": 0}\n'
    "    for lp in sorted(glob.glob(os.path.join(TRAIN_LBL, '*.txt'))):\n"
    '        name = os.path.splitext(os.path.basename(lp))[0]\n'
    "        ip = os.path.join(TRAIN_IMG, name + '.jpg')\n"
    '        if not os.path.exists(ip):\n'
    '            skipped["no_image"] += 1\n'
    '            continue\n'
    '        try:\n'
    '            w, h = Image.open(ip).size\n'
    '        except Exception:\n'
    '            skipped["bad_image"] += 1\n'
    '            continue\n'
    '        by_cls = {}\n'
    "        for line in open(lp, encoding='utf-8'):\n"
    '            p = line.strip().split()\n'
    '            if len(p) < 5:\n'
    '                continue\n'
    '            c = int(p[0])\n'
    '            cx, cy, bw, bh = map(float, p[1:5])\n'
    '            x1, y1 = int(round((cx - bw / 2) * w)), int(round((cy - bh / 2) * h))\n'
    '            x2, y2 = int(round((cx + bw / 2) * w)), int(round((cy + bh / 2) * h))\n'
    '            by_cls.setdefault(c, []).append((x1, y1, x2, y2))\n'
    '        if not by_cls:\n'
    '            skipped["no_box"] += 1\n'
    '            continue\n'
    '        # One sample per (image, class): 95 train images contain 2 classes,\n'
    '        # 2 contain 3 classes. Each sample asks for ONE named class and\n'
    '        # lists only that class boxes, so instruction and answer stay aligned.\n'
    '        for c, boxes in sorted(by_cls.items()):\n'
    '            cls_name = CLASSES[c]\n'
    '            uniq = sorted(set(boxes))\n'
    '            skipped["dup_box"] += len(boxes) - len(uniq)\n'
    "            ans = '<|object_ref_start|>' + ''.join(\n"
    "                f'<|object_ref|>{x1},{y1},{x2},{y2}<|/object_ref|>' for x1, y1, x2, y2 in uniq\n"
    "            ) + '<|object_ref_end|>'\n"
    "            samples.append({'path': ip, 'cls': cls_name, 'n_boxes': len(uniq),\n"
    "                            'instruction': rng.choice(TEMPLATES).format(cn=CLS_CN[cls_name]),\n"
    "                            'answer': ans})\n"
    '    return samples, skipped\n'
    '\n'
    "CACHE = os.path.join(WORK, 'samples_cache_v3.json')\n"
    'if os.path.exists(CACHE):\n'
    "    with open(CACHE, encoding='utf-8') as f:\n"
    '        samples = json.load(f)\n'
    "    skipped = {'cached': True}\n"
    "    print('loaded cached samples:', len(samples))\n"
    'else:\n'
    '    samples, skipped = build_samples()\n'
    "    with open(CACHE, 'w', encoding='utf-8') as f:\n"
    '        json.dump(samples, f, ensure_ascii=False)\n'
    "    print('samples cached to', CACHE)\n"
    "print('train samples:', len(samples), '| skipped:', skipped)\n"
    "print('example instruction:', samples[0]['instruction'])\n"
    "print('example answer:', samples[0]['answer'][:200])"))

cells.append(code(
    '# ---- Data verification BEFORE training (fail fast, do not burn GPU) ----\n'
    'import collections\n'
    'from PIL import ImageDraw\n'
    'per_cls = collections.Counter(s["cls"] for s in samples)\n'
    "print('per-class train counts:', dict(per_cls))\n"
    'assert set(per_cls) == set(CLASSES), "missing class in train split"\n'
    'assert all(v >= 200 for v in per_cls.values()), "a class has too few samples"\n'
    'nbox = [s["n_boxes"] for s in samples]\n'
    "print('boxes/image: mean %.2f max %d | images with 1 box: %d' % (\n"
    '    sum(nbox) / len(nbox), max(nbox), sum(1 for n in nbox if n == 1)))\n'
    '# center-spread check on raw labels (no positional shortcut allowed)\n'
    'cxs, cys = [], []\n'
    "for lp in glob.glob(os.path.join(TRAIN_LBL, '*.txt')):\n"
    "    for line in open(lp, encoding='utf-8'):\n"
    '        p = line.strip().split()\n'
    '        if len(p) >= 5:\n'
    '            cxs.append(float(p[1])); cys.append(float(p[2]))\n'
    'central = sum(1 for x, y in zip(cxs, cys) if 0.4 <= x <= 0.6 and 0.4 <= y <= 0.6) / len(cxs)\n'
    "print('box centers in central 20%% square: %.1f%% (must be < 25%%)' % (central * 100))\n"
    "assert central < 0.25, 'boxes are center-clustered - positional shortcut risk'\n"
    '# save one overlay per class for eyeballing\n'
    "ov_dir = os.path.join(WORK, 'overlays_v3')\n"
    'os.makedirs(ov_dir, exist_ok=True)\n'
    'seen = set()\n'
    'for s in samples:\n'
    "    if s['cls'] in seen:\n"
    '        continue\n'
    "    seen.add(s['cls'])\n"
    '    img = Image.open(s["path"]).convert("RGB").resize((400, 400), Image.NEAREST)\n'
    '    d = ImageDraw.Draw(img)\n'
    "    import re\n"
    "    for m in re.finditer(r'<\\|object_ref\\|>(\\d+),(\\d+),(\\d+),(\\d+)<\\|/object_ref\\|>', s['answer']):\n"
    '        x1, y1, x2, y2 = (int(m.group(i)) * 2 for i in range(1, 5))\n'
    '        d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)\n'
    "    img.save(os.path.join(ov_dir, s['cls'] + '.png'))\n"
    "    if len(seen) == 6:\n"
    '        break\n'
    "print('overlays saved to', ov_dir, '- eyeball them before trusting the run')\n"
    "print('DATA CHECKS PASSED')"))

cells.append(code(
    'from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig\n'
    'from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training\n'
    '\n'
    "BASE = 'Qwen/Qwen3-VL-2B-Instruct'\n"
    'processor = AutoProcessor.from_pretrained(BASE)\n'
    'tok = processor.tokenizer\n'
    '\n'
    'bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",\n'
    '                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)\n'
    'model = Qwen3VLForConditionalGeneration.from_pretrained(BASE, quantization_config=bnb, device_map="cuda")\n'
    'model.config.use_cache = False\n'
    'model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)\n'
    '# all-linear covers language + vision-tower + projector linears (same as v2)\n'
    'peft_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",\n'
    '                      task_type="CAUSAL_LM", target_modules="all-linear")\n'
    'model = get_peft_model(model, peft_cfg)\n'
    "model.print_trainable_parameters()\n"
    "print('model ready')"))

cells.append(code(
    'import torch\n'
    'from torch.utils.data import Dataset\n'
    '\n'
    'ANS_START_ID = tok.convert_tokens_to_ids("<|object_ref_start|>")\n'
    'MAX_LEN = 2048\n'
    '\n'
    'class DefectDataset(Dataset):\n'
    '    def __init__(self, samples):\n'
    '        self.samples = samples\n'
    '    def __len__(self):\n'
    '        return len(self.samples)\n'
    '    def __getitem__(self, i):\n'
    '        s = self.samples[i]\n'
    '        img = Image.open(s["path"]).convert("RGB")\n'
    "        msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': img}, {'type': 'text', 'text': s['instruction']}]},\n"
    "                {'role': 'assistant', 'content': [{'type': 'text', 'text': s['answer']}]}]\n"
    '        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)\n'
    "        enc = processor(text=[text], images=[img], return_tensors='pt')\n"
    "        ids = enc['input_ids'][0]\n"
    "        attn = enc['attention_mask'][0]\n"
    '        labels = ids.clone()\n'
    '        # Loss only on the answer: mask everything before the first <|object_ref_start|> token.\n'
    '        pos = (ids == ANS_START_ID).nonzero(as_tuple=True)[0]\n'
    '        assert len(pos) > 0, "answer start token missing in sample %d" % i\n'
    '        labels[:int(pos[0])] = -100\n'
    "        mm = enc['mm_token_type_ids'][0] if 'mm_token_type_ids' in enc else None\n"
    '        if len(ids) > MAX_LEN:\n'
    '            ids, labels, attn = ids[:MAX_LEN], labels[:MAX_LEN], attn[:MAX_LEN]\n'
    '            if mm is not None:\n'
    '                mm = mm[:MAX_LEN]\n'
    "        out = {'input_ids': ids, 'labels': labels,\n"
    "               'pixel_values': enc['pixel_values'],\n"
    "               'image_grid_thw': enc['image_grid_thw'][0],\n"
    "               'attention_mask': attn}\n"
    '        # NOTE: pixel_values has NO batch dim - it is already patchified\n'
    "        # (num_patches, 1536). Do NOT take [0]: that keeps only the first\n"
    "        # patch and the model sees one patch repeated at all positions.\n"
    "        # That exact bug was in v2 and is the likely root cause of its\n"
    "        # mode collapse (format learned from text, no visual signal).\n"
    "        # image_grid_thw DOES have a leading num_images dim, so [0] is right there.\n"
    '        if mm is not None:\n'
    "            out['mm_token_type_ids'] = mm\n"
    '        return out\n'
    '\n'
    'def collate(batch):\n'
    '    maxlen = max(len(b["input_ids"]) for b in batch)\n'
    '    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id\n'
    '    input_ids, labels, attn = [], [], []\n'
    '    for b in batch:\n'
    '        n = maxlen - len(b["input_ids"])\n'
    '        input_ids.append(torch.cat([b["input_ids"], torch.full((n,), pad, dtype=torch.long)]))\n'
    '        labels.append(torch.cat([b["labels"], torch.full((n,), -100, dtype=torch.long)]))\n'
    '        attn.append(torch.cat([b["attention_mask"], torch.zeros(n, dtype=torch.long)]))\n'
    "    out = {'input_ids': torch.stack(input_ids), 'labels': torch.stack(labels),\n"
    "           'attention_mask': torch.stack(attn),\n"
    "           'pixel_values': torch.stack([b['pixel_values'] for b in batch]),\n"
    "           'image_grid_thw': torch.stack([b['image_grid_thw'] for b in batch])}\n"
    "    if 'mm_token_type_ids' in batch[0]:\n"
    '        mms = []\n'
    '        for b in batch:\n'
    '            n = maxlen - len(b["input_ids"])\n'
    '            mms.append(torch.cat([b["mm_token_type_ids"], torch.zeros(n, dtype=torch.long)]))\n'
    "        out['mm_token_type_ids'] = torch.stack(mms)\n"
    '    return out\n'
    '\n'
    'ds = DefectDataset(samples)\n'
    "print('dataset size:', len(ds))"))

cells.append(code(
    '# ---- Masking + EOS self-check: run BEFORE training, not after ----\n'
    'ex = ds[0]\n'
    'kept = ex["labels"][ex["labels"] != -100]\n'
    'masked_n = int((ex["labels"] == -100).sum())\n'
    'decoded_kept = tok.decode(kept, skip_special_tokens=False)\n'
    "print('total tokens:', len(ex['input_ids']), '| masked (prompt+image):', masked_n,\n"
    "      '| kept (answer):', len(kept))\n"
    "print('kept text starts with:', repr(decoded_kept[:60]))\n"
    "print('kept text ends with:', repr(decoded_kept[-40:]))\n"
    "assert decoded_kept.startswith('<|object_ref_start|>'), 'MASKING WRONG: answer does not start with object_ref_start'\n"
    "assert len(kept) < len(ex['input_ids']) * 0.5, 'MASKING WRONG: kept part too long'\n"
    "assert decoded_kept.rstrip().endswith('<|im_end|>'), 'EOS MISSING: kept labels do not end with <|im_end|> - model will never learn to stop (v2 failure mode)'\n"
    "print('masking + EOS OK: loss on answer tokens only, ending with EOS')\n"
    '# ---- Pixel/grid consistency guard: catches the v2 first-patch bug ----\n'
    "n_patches = int(ex['pixel_values'].shape[0])\n"
    "grid_prod = int(ex['image_grid_thw'].prod())\n"
    "print('pixel_values patches:', n_patches, '| grid t*h*w:', grid_prod,\n"
    "      '| pixel_values shape:', tuple(ex['pixel_values'].shape))\n"
    "assert n_patches == grid_prod, 'PIXEL BUG: patch count does not match image_grid_thw'\n"
    "assert n_patches > 100, 'PIXEL BUG: suspiciously few patches - probably only the first patch survived'\n"
    "print('pixels OK: full image patches reach the model')"))

cells.append(code(
    'from transformers import Trainer, TrainingArguments\n'
    '\n'
    'import inspect\n'
    '_wanted = dict(\n'
    '    output_dir=ADAPTER_DIR,\n'
    '    per_device_train_batch_size=1,\n'
    '    gradient_accumulation_steps=16,\n'
    '    num_train_epochs=2,\n'
    '    learning_rate=1e-4,\n'
    '    lr_scheduler_type="cosine",\n'
    '    warmup_steps=10,\n'
    '    optim="paged_adamw_8bit",\n'
    '    max_grad_norm=1.0,\n'
    '    logging_steps=10,\n'
    '    save_strategy="epoch",\n'
    '    report_to=[],\n'
    '    remove_unused_columns=False,\n'
    '    dataloader_num_workers=0,\n'
    '    gradient_checkpointing=True,\n'
    '    gradient_checkpointing_kwargs={"use_reentrant": False},\n'
    ')\n'
    '_ok = set(inspect.signature(TrainingArguments.__init__).parameters)\n'
    '_dropped = sorted(k for k in _wanted if k not in _ok)\n'
    'args = TrainingArguments(**{k: v for k, v in _wanted.items() if k in _ok})\n'
    "print('TrainingArguments dropped unsupported kwargs:', _dropped)\n"
    'trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collate)\n'
    'trainer.train()\n'
    "print('training done')\n"
    'trainer.save_model(ADAPTER_DIR)\n'
    "hist = [{'step': h.get('step'), 'loss': h.get('loss')} for h in trainer.state.log_history if 'loss' in h]\n"
    "print('first loss:', hist[0], '| last loss:', hist[-1])\n"
    "with open(os.path.join(WORK, 'train_log_v3.json'), 'w', encoding='utf-8') as f:\n"
    '    json.dump(hist, f, ensure_ascii=False, indent=2)\n'
    "import glob as _glob\n"
    "print('adapter files:', sorted(_glob.glob(os.path.join(ADAPTER_DIR, '*'))))\n"
    "print('TRAINING DONE -> adapter at', ADAPTER_DIR)"))

cells.append(md(
    '## 下一步\n'
    '\n'
    '1. 在 Kaggle 上把本笔记本的输出（`neudet-adapter-v3`）作为输入数据挂到评估本 `qwen3vl-eval-kaggle-v3-neudet.ipynb`，Run all。\n'
    '2. 对照同本内测出的基线看三项：格式合规率是否接近 100%、Acc@IoU>=0.5 是否超过基线、是否还有重复框（结果 JSON 里检查每图预测框去重前后数量）。\n'
    '3. 只有复测确实有提升，才把数字写进简历；写法须注明是 NEU-DET 公开基准上的微调验证。'))

nb = {'nbformat': 4, 'nbformat_minor': 0,
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
                   'accelerator': 'GPU'},
      'cells': cells}

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('written', NB_PATH)
