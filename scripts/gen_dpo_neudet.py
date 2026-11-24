# Build the DPO preference JSON for NEU-DET (v3 follow-up stage, after SFT).
#
# Why a rewrite: the original DPO attempt in the v1 fabric notebook
# (Qwen3VL-2B-Qlora微调.ipynb, cells 19/21/23) never produced data - it died
# in data processing on a regex bug, and the dpo_training folder in Drive is
# empty. Two things are wrong with the original design:
#   1. REGEX BUG (fatal): perturb_bbox used
#      re.search(r'<|object_ref|>(\d+),(\d+),(\d+),(\d+)<|/object_ref|>', ...)
#      where the unescaped | is alternation, so re.search matches the first
#      '<' with all groups None -> int(None) raises TypeError. This rewrite
#      never parses boxes with regex at all: boxes are perturbed numerically
#      from parsed int tuples, and formatting is done by construction.
#   2. SINGLE-BOX DEFAULT: the fabric DPO path was built around one box per
#      image (the fabric SFT JSON's modal sample has exactly 1 box, 653/1817).
#      NEU-DET samples are (image, class) with ALL boxes of that class
#      (mean 2.16, max 9). This rewrite keeps the full box set on BOTH sides:
#      chosen = all ground-truth boxes, rejected = ALL boxes perturbed.
#      No sample is ever reduced to a single box.
# Additional fix vs the original: a +/-5% shift on NEU-DET's small boxes can
# round to 0 px, leaving rejected == chosen (a useless DPO pair). Here every
# box is forced to move by >= 1 px, and rejected != chosen is asserted.
#
# Design (same as the original, bugs fixed):
#   prompt   = v3 eval instruction (TEMPLATES[0]), deterministic, so DPO
#              trains on exactly the prompt the eval uses.
#   chosen   = v3 SFT answer format, all GT boxes of the class.
#   rejected = same format, every box shifted by U(-5%, +5%) of its own size.
#   image    = path relative to the dataset root (images/train/xxx.jpg).
# RNG seeded (42) so the file is reproducible.
# Outputs: neudet_dpo_train.json, neudet_dpo_valid.json (JSON lists).
import glob
import json
import os
import random
from collections import Counter

from PIL import Image

DATA_ROOT = '/home/hatch/workspace/job-search/diag/datasets/neu-det'
OUT_DIR = '/home/hatch/workspace/job-search'
SEED = 42
PERTURB_RATIO = 0.05

CLS_CN = {
    'crazing': '裂纹',
    'inclusion': '夹杂物',
    'patches': '斑块',
    'pitted_surface': '麻点',
    'rolled-in_scale': '氧化皮',
    'scratches': '划痕',
}
# TEMPLATES[0] is the eval instruction; DPO prompts use it for all pairs.
PROMPT_TMPL = '请识别图片中的钢材表面缺陷（{cn}），并用边界框标出所有位置。'


def fmt_answer(boxes):
    return '<|object_ref_start|>' + ''.join(
        f'<|object_ref|>{x1},{y1},{x2},{y2}<|/object_ref|>'
        for x1, y1, x2, y2 in boxes
    ) + '<|object_ref_end|>'


def perturb_box(box, w, h, rng):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    dx = rng.uniform(-PERTURB_RATIO * bw, PERTURB_RATIO * bw)
    dy = rng.uniform(-PERTURB_RATIO * bh, PERTURB_RATIO * bh)
    ix, iy = int(round(dx)), int(round(dy))
    # Guarantee the box actually moves: a sub-pixel shift would round to 0
    # and leave rejected == chosen for this box.
    if ix == 0 and iy == 0:
        ix = rng.choice([-1, 1])
    nx1 = max(0, min(x1 + ix, w - 1))
    ny1 = max(0, min(y1 + iy, h - 1))
    nx2 = max(0, min(x2 + ix, w - 1))
    ny2 = max(0, min(y2 + iy, h - 1))
    if nx1 >= nx2:
        nx2 = min(nx1 + 1, w - 1)
        if nx1 >= nx2:
            nx1 = nx2 - 1
    if ny1 >= ny2:
        ny2 = min(ny1 + 1, h - 1)
        if ny1 >= ny2:
            ny1 = ny2 - 1
    assert 0 <= nx1 < nx2 < w and 0 <= ny1 < ny2 < h, (box, (nx1, ny1, nx2, ny2), w, h)
    return (nx1, ny1, nx2, ny2)


def build_split(split, rng):
    # NB: on disk the split is called 'val', the output file 'valid'.
    split_dir = {'train': 'train', 'valid': 'val'}[split]
    lbl_dir = os.path.join(DATA_ROOT, 'labels', split_dir)
    img_dir = os.path.join(DATA_ROOT, 'images', split_dir)
    classes = open(os.path.join(DATA_ROOT, 'classes.txt')).read().split()
    records, skipped = [], {'no_image': 0, 'no_box': 0, 'dup_box': 0, 'bad_image': 0}
    for lp in sorted(glob.glob(os.path.join(lbl_dir, '*.txt'))):
        name = os.path.splitext(os.path.basename(lp))[0]
        ip = os.path.join(img_dir, name + '.jpg')
        if not os.path.exists(ip):
            skipped['no_image'] += 1
            continue
        try:
            w, h = Image.open(ip).size
        except Exception:
            skipped['bad_image'] += 1
            continue
        by_cls = {}
        for line in open(lp, encoding='utf-8'):
            p = line.strip().split()
            if len(p) < 5:
                continue
            c = int(p[0])
            cx, cy, bw, bh = map(float, p[1:5])
            x1, y1 = int(round((cx - bw / 2) * w)), int(round((cy - bh / 2) * h))
            x2, y2 = int(round((cx + bw / 2) * w)), int(round((cy + bh / 2) * h))
            by_cls.setdefault(c, []).append((x1, y1, x2, y2))
        if not by_cls:
            skipped['no_box'] += 1
            continue
        for c, boxes in sorted(by_cls.items()):
            cls_name = classes[c]
            uniq = sorted(set(boxes))
            skipped['dup_box'] += len(boxes) - len(uniq)
            chosen_boxes = uniq
            rejected_boxes = [perturb_box(b, w, h, rng) for b in uniq]
            assert rejected_boxes != chosen_boxes, 'rejected == chosen for %s' % ip
            records.append({
                'prompt': PROMPT_TMPL.format(cn=CLS_CN[cls_name]),
                'chosen': fmt_answer(chosen_boxes),
                'rejected': fmt_answer(rejected_boxes),
                'image': os.path.join('images', split_dir, name + '.jpg'),
                'cls': cls_name,
                'n_boxes': len(uniq),
            })
    return records, skipped


def main():
    rng = random.Random(SEED)
    for split in ('train', 'valid'):
        recs, skipped = build_split(split, rng)
        out = os.path.join(OUT_DIR, f'neudet_dpo_{split}.json')
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(recs, f, ensure_ascii=False)
        # ---- verification ----
        import re
        box_re = re.compile(r'<\|object_ref\|>(\d+),(\d+),(\d+),(\d+)<\|/object_ref\|>')
        assert recs, 'no records for %s' % split
        assert all(r['rejected'] != r['chosen'] for r in recs), 'some rejected == chosen'
        for r in recs:
            cb, rb = box_re.findall(r['chosen']), box_re.findall(r['rejected'])
            assert len(cb) == len(rb) == r['n_boxes'], 'box count mismatch'
            for x1, y1, x2, y2 in rb:
                assert int(x1) < int(x2) and int(y1) < int(y2), 'invalid rejected box'
        nbox = [r['n_boxes'] for r in recs]
        print(f'{split}: {len(recs)} pairs | skipped: {skipped}')
        print(f'  per-class: {dict(Counter(r["cls"] for r in recs))}')
        print(f'  boxes/pair: mean {sum(nbox)/len(nbox):.2f} max {max(nbox)} '
              f'| single-box pairs: {sum(1 for n in nbox if n == 1)} '
              f'| multi-box pairs: {sum(1 for n in nbox if n > 1)}')
        print(f'  rejected != chosen: {sum(1 for r in recs if r["rejected"] != r["chosen"])}/{len(recs)}')
        print(f'  wrote {out}')
    print('sample train[0]:')
    print(json.dumps(json.load(open(os.path.join(OUT_DIR, 'neudet_dpo_train.json')))[:1],
                     ensure_ascii=False, indent=2)[:800])


if __name__ == '__main__':
    main()
