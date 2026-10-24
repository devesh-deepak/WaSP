#!/usr/bin/env python3
"""
data_overlap.py — near-duplicate search between test reference images and training data.

Searches every test reference image for a copy, crop or rescaled version of itself in the
training sets of the evaluated models (Supplementary Section S8).

Queries (test references):
  TID2013 (25), KADID-10K (81), PIPAL training split (200),
  Set5 (5), Set14 (14), BSD100 (100), Urban100 (100) HR images
Training sets searched:
  DIV2K train (800)  — our fine-tuning data; RCAN, EDSR, IMDN, SRResNet
  DIV2K valid (100)  — used for the training-loss weight search
  Flickr2K (2650)    — SwinIR (Swin2SR), RealESRNet, RealESRGAN (with DIV2K)
  OST train (10324)  — RealESRNet, RealESRGAN

Method
  1. Retrieval: ImageNet ResNet-50 descriptors (global average pool, L2-normalised) of every
     training image (whole image) and of square windows of 288/432/648/972 px (stride = half
     the side). Each query is described by its whole image and its central square; the
     similarity to a training image is the maximum over windows. The top-K training images per
     query and training set are kept.
  2. Verification: SIFT + ratio test + RANSAC similarity transform between the query and each
     candidate (training images above 2048 px are downscaled for SIFT). A pair is a duplicate if
     it has >= MIN_INLIERS inliers and, after alignment, the luma PSNR is >= MIN_PSNR dB or the
     normalised cross-correlation is >= MIN_NCC (copies with tone or colour edits). Repetitive
     structures (facades, grids) can give many geometrically consistent inliers between
     different photographs; the pixel test rejects these.
  3. Exact-crop test: smooth crops give too few keypoints for step 2, so each candidate is also
     searched by normalised cross-correlation at native scale (PSNR >= MIN_PSNR and NCC >= MIN_NCC
     at the best location).
  PIPAL references were cropped from DIV2K and Flickr2K (Jinjin et al., 2020), so they act as
  a positive control for the detector.

Output: results/data_overlap.json (paths relative to the dataset roots); descriptor cache in
results/overlap_desc/ (several GB, not part of the repository). Needs opencv-python.

  python evaluation/data_overlap.py --flickr2k <Flickr2K_HR dir> --ost <OutdoorSceneTrain_v2 dir>
"""

import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision.ops import roi_align

from utils.paths import IQA_DATA, RESULTS_DIR, SR_DATA

_AP = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_AP.add_argument('--flickr2k', default=os.path.join(SR_DATA, 'Flickr2K', 'Flickr2K_HR'))
_AP.add_argument('--ost', default=os.path.join(SR_DATA, 'OST', 'OutdoorSceneTrain_v2'))
_AP.add_argument('--workers', type=int, default=min(32, os.cpu_count() or 1))
ARGS    = _AP.parse_args() if __name__ == '__main__' else _AP.parse_args([])
DATA    = SR_DATA
MOS     = IQA_DATA
OUT     = os.path.join(RESULTS_DIR, 'data_overlap.json')
CACHE   = os.path.join(RESULTS_DIR, 'overlap_desc')
DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
SIDES   = [288, 432, 648, 972]
TOPK    = 10
MIN_INLIERS = 15
MIN_PSNR    = 25.0
MIN_NCC     = 0.9

QUERIES = {
    'TID2013':   sorted(p for p in glob.glob(f'{MOS}/TID2013/reference_images/*') if p.lower().endswith('.bmp')),
    'KADID-10K': sorted(p for p in glob.glob(f'{MOS}/KADID-10K/**/images/I*.png', recursive=True)
                        if '_' not in os.path.basename(p)),
    'PIPAL':     sorted(glob.glob(f'{MOS}/PIPAL/**/Train_Ref/*.bmp', recursive=True)),
    'Set5':      sorted(glob.glob(f'{DATA}/Set5/image_SRF_4/*HR.png')),
    'Set14':     sorted(glob.glob(f'{DATA}/Set14/image_SRF_4/*HR.png')),
    'BSD100':    sorted(glob.glob(f'{DATA}/BSD100/image_SRF_4/*HR.png')),
    'Urban100':  sorted(glob.glob(f'{DATA}/Urban100/image_SRF_4/*HR.png')),
}
TRAIN = {
    'DIV2K-train': sorted(glob.glob(f'{DATA}/DIV2K/DIV2K_train_HR/*.png')),
    'DIV2K-valid': sorted(glob.glob(f'{DATA}/DIV2K/DIV2K_valid_HR/*.png')),
    'Flickr2K':    sorted(glob.glob(f'{ARGS.flickr2k}/*.png')),
    'OST':         sorted(glob.glob(f'{ARGS.ost}/*/*.png')),
}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_rgb(p):
    return np.asarray(Image.open(p).convert('RGB'))


def windows(h, w):
    """Boxes (x1, y1, x2, y2): the whole image, then square windows at several sizes."""
    boxes = [(0, 0, w, h)]
    for s in SIDES:
        if s > min(h, w):
            continue
        st = s // 2
        ys = list(range(0, h - s + 1, st)); xs = list(range(0, w - s + 1, st))
        if ys[-1] != h - s: ys.append(h - s)
        if xs[-1] != w - s: xs.append(w - s)
        boxes += [(x, y, x + s, y + s) for y in ys for x in xs]
    return boxes


class Desc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        self.body = torch.nn.Sequential(*list(m.children())[:-2])

    @torch.no_grad()
    def forward(self, crops):                       # [N, 3, 224, 224] in [0, 1]
        x = (crops - MEAN.to(crops.device)) / STD.to(crops.device)
        f = self.body(x).mean((2, 3))
        return F.normalize(f.float(), dim=1)


@torch.no_grad()
def describe(net, img, boxes, bs=512):
    t = torch.from_numpy(img).permute(2, 0, 1).float().div(255).unsqueeze(0).to(DEVICE)
    b = torch.tensor(boxes, dtype=torch.float32, device=DEVICE)
    out = []
    for i in range(0, len(b), bs):
        bb = torch.cat([torch.zeros(len(b[i:i + bs]), 1, device=DEVICE), b[i:i + bs]], 1)
        crops = roi_align(t, bb, output_size=(224, 224), spatial_scale=1.0, sampling_ratio=2, aligned=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out.append(net(crops))
    return torch.cat(out).cpu()


def _load(p):
    return p, load_rgb(p)


def train_descriptors(net, name, paths):
    f = os.path.join(CACHE, f'{name}.pt')
    if os.path.exists(f):
        return torch.load(f)
    descs, owner = [], []
    t0 = time.time()
    with Pool(32) as pool:
        for i, (p, img) in enumerate(pool.imap(_load, paths, chunksize=4)):
            d = describe(net, img, windows(*img.shape[:2]))
            descs.append(d.half()); owner.append(torch.full((len(d),), i, dtype=torch.int32))
            if (i + 1) % 500 == 0:
                print(f'  [{name}] {i + 1}/{len(paths)} ({time.time() - t0:.0f}s)', flush=True)
    res = {'desc': torch.cat(descs), 'owner': torch.cat(owner)}
    torch.save(res, f)
    return res


def query_descriptors(net, paths):
    out = []
    for p in paths:
        img = load_rgb(p); h, w = img.shape[:2]; s = min(h, w)
        boxes = [(0, 0, w, h), ((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)]
        out.append(describe(net, img, boxes))
    return out


def candidates(qd, tr, n_train):
    """Top-K training images per query by max cosine similarity over windows."""
    D = tr['desc'].to(DEVICE).float(); own = tr['owner'].to(DEVICE).long()
    res = []
    for q in qd:
        sim = (q.to(DEVICE) @ D.T).max(0).values                     # best query view per window
        best = torch.full((n_train,), -1.0, device=DEVICE).scatter_reduce(0, own, sim, 'amax')
        v, i = best.topk(min(TOPK, n_train))
        res.append(list(zip(i.tolist(), v.tolist())))
    return res


_SIFT = None


def sift(gray):
    global _SIFT
    if _SIFT is None:
        _SIFT = cv2.SIFT_create(nfeatures=4000)
    return _SIFT.detectAndCompute(gray, None)


def luma(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def template_check(rec, qg, tg_full, tg_half):
    """Exact-crop test that does not need keypoints (smooth crops): normalised cross-correlation
    of the query against the training image at native scale, located on 2x-downsampled images
    and confirmed at full resolution."""
    if qg.shape[0] > tg_full.shape[0] or qg.shape[1] > tg_full.shape[1]:
        return
    qh = cv2.resize(qg, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    if qh.shape[0] > tg_half.shape[0] or qh.shape[1] > tg_half.shape[1]:
        return
    r = cv2.matchTemplate(tg_half, qh, cv2.TM_CCOEFF_NORMED)
    _, _, _, (x, y) = cv2.minMaxLoc(r)
    best = None
    for dy in (-1, 0, 1):                              # refine the location at full resolution
        for dx in (-1, 0, 1):
            X, Y = 2 * x + dx, 2 * y + dy
            if X < 0 or Y < 0 or Y + qg.shape[0] > tg_full.shape[0] or X + qg.shape[1] > tg_full.shape[1]:
                continue
            a = qg.astype(np.float64); b = tg_full[Y:Y + qg.shape[0], X:X + qg.shape[1]].astype(np.float64)
            psnr = 10 * np.log10(255 ** 2 / max(np.mean((a - b) ** 2), 1e-10))
            if best is None or psnr > best[0]:
                a0, b0 = a - a.mean(), b - b.mean()
                best = (psnr, float((a0 * b0).sum() / (np.sqrt((a0 ** 2).sum() * (b0 ** 2).sum()) + 1e-10)))
    if best is not None:
        rec['tm_psnr'], rec['tm_ncc'] = float(best[0]), best[1]


def verify(task):
    """task: (train_path, [(query_path, query_name), ...]) -> list of match records."""
    tp, qs = task
    timg = load_rgb(tp); tg = tg_full = luma(timg)
    tg_half = cv2.resize(tg_full, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    scale = min(1.0, 2048 / max(tg.shape))
    if scale < 1:
        tg = cv2.resize(tg, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    kt, dt = sift(tg)
    out = []
    bf = cv2.BFMatcher(cv2.NORM_L2)
    for qp, qn in qs:
        qg = luma(load_rgb(qp))
        kq, dq = sift(qg)
        rec = {'query': qn, 'query_path': qp, 'train_path': tp, 'inliers': 0, 'psnr': None}
        template_check(rec, qg, tg_full, tg_half)
        if dq is None or dt is None or len(kq) < 10 or len(kt) < 10:
            out.append(rec); continue
        good = [m for m, n in (x for x in bf.knnMatch(dq, dt, k=2) if len(x) == 2) if m.distance < 0.75 * n.distance]
        if len(good) >= 10:
            src = np.float32([kq[m.queryIdx].pt for m in good]); dst = np.float32([kt[m.trainIdx].pt for m in good])
            M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0,
                                                 maxIters=5000, confidence=0.999)
            if M is not None and inl is not None:
                rec['inliers'] = int(inl.sum()); rec['n_good'] = len(good)
                rec['scale'] = float(np.sqrt(abs(np.linalg.det(M[:, :2])))) / scale
                if rec['inliers'] >= 10:
                    src_img, Mw = tg, M.copy()
                    s_m = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
                    if s_m > 1.1:                     # training image larger: downscale with anti-aliasing first
                        src_img = cv2.resize(tg, None, fx=1 / s_m, fy=1 / s_m, interpolation=cv2.INTER_AREA)
                        Mw = M / s_m
                    warped = cv2.warpAffine(src_img, Mw, (qg.shape[1], qg.shape[0]),
                                            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
                    valid = cv2.warpAffine(np.ones_like(src_img), Mw, (qg.shape[1], qg.shape[0]),
                                           flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
                    valid = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                    if valid.mean() > 0.5:
                        a = qg[valid].astype(np.float64); b = warped[valid].astype(np.float64)
                        mse = np.mean((a - b) ** 2)
                        rec['psnr'] = float(10 * np.log10(255 ** 2 / max(mse, 1e-10)))
                        a0, b0 = a - a.mean(), b - b.mean()
                        rec['ncc'] = float((a0 * b0).sum() / (np.sqrt((a0 ** 2).sum() * (b0 ** 2).sum()) + 1e-10))
                        rec['overlap'] = float(valid.mean())
        out.append(rec)
    return out


def relative_paths(x, roots=None):
    """Strip the dataset roots from every path in the result (portable output)."""
    roots = roots or sorted({os.path.join(MOS, ''), os.path.join(DATA, ''),
                             os.path.join(os.path.dirname(ARGS.flickr2k.rstrip('/')), ''),
                             os.path.join(os.path.dirname(ARGS.ost.rstrip('/')), '')}, key=len, reverse=True)
    if isinstance(x, dict):
        return {k: relative_paths(v, roots) for k, v in x.items()}
    if isinstance(x, list):
        return [relative_paths(v, roots) for v in x]
    if isinstance(x, str):
        for r in roots:
            if x.startswith(r):
                return x[len(r):]
    return x


def main():
    os.makedirs(CACHE, exist_ok=True)
    for k, v in {**QUERIES, **TRAIN}.items():
        print(f'  {k}: {len(v)} images', flush=True)
        assert v, k
    net = Desc().to(DEVICE).eval().to(memory_format=torch.channels_last)
    qdesc = {k: query_descriptors(net, v) for k, v in QUERIES.items()}
    tasks = {}                                       # train path -> [(query path, query set)]
    retrieval = {}
    for tn, tp in TRAIN.items():
        tr = train_descriptors(net, tn, tp)
        for qn, qp in QUERIES.items():
            cand = candidates(qdesc[qn], tr, len(tp))
            retrieval[f'{qn}|{tn}'] = [c[0][1] for c in cand]
            for q, cs in zip(qp, cand):
                for i, _ in cs:
                    tasks.setdefault(tp[i], []).append((q, qn))
    print(f'  verifying {sum(len(v) for v in tasks.values())} pairs on {len(tasks)} training images', flush=True)
    t0 = time.time()
    recs = []
    with Pool(ARGS.workers) as pool:
        for r in pool.imap_unordered(verify, list(tasks.items()), chunksize=1):
            recs += r
    print(f'  verification done in {time.time() - t0:.0f}s', flush=True)

    owner = {p: tn for tn, tp in TRAIN.items() for p in tp}
    which = owner.__getitem__
    sift_dup = lambda r: (r['inliers'] >= MIN_INLIERS and r['psnr'] is not None
                          and (r['psnr'] >= MIN_PSNR or r['ncc'] >= MIN_NCC))
    tm_dup = lambda r: r.get('tm_psnr') is not None and r['tm_psnr'] >= MIN_PSNR and r['tm_ncc'] >= MIN_NCC
    dup = [r for r in recs if sift_dup(r) or tm_dup(r)]
    for r in dup:
        r['found_by'] = '+'.join(k for k, f in (('sift', sift_dup), ('template', tm_dup)) if f(r))
    summary = {}
    for qn, qp in QUERIES.items():
        summary[qn] = {'n': len(qp)}
        for tn in TRAIN:
            hits = {r['query_path'] for r in dup if r['query'] == qn and which(r['train_path']) == tn}
            summary[qn][tn] = len(hits)
        summary[qn]['any'] = len({r['query_path'] for r in dup if r['query'] == qn})
        best = {}
        for r in recs:
            if r['query'] == qn:
                best[r['query_path']] = max(best.get(r['query_path'], 0), r['inliers'])
        nondup = [v for p, v in best.items() if p not in {r['query_path'] for r in dup}]
        summary[qn]['max_inliers_non_duplicates'] = int(max(nondup)) if nondup else None
    res = {'method': {'descriptor': 'ResNet-50 (ImageNet) GAP, windows ' + str(SIDES) + ' + whole image',
                      'top_k': TOPK, 'min_inliers': MIN_INLIERS, 'min_psnr_db': MIN_PSNR, 'min_ncc': MIN_NCC},
           'n_train': {k: len(v) for k, v in TRAIN.items()},
           'summary': summary,
           'duplicates': sorted([{k: r.get(k) for k in ('query', 'query_path', 'train_path', 'found_by', 'inliers',
                                                        'psnr', 'ncc', 'tm_psnr', 'tm_ncc', 'scale')}
                                 for r in dup], key=lambda r: (r['query'], r['query_path'])),
           'closest_template_non_duplicates': {qn: max(([{k: r.get(k) for k in ('query_path', 'train_path', 'tm_psnr', 'tm_ncc')}
                                                         for r in recs if r['query'] == qn and r not in dup
                                                         and r.get('tm_ncc') is not None] or [None]),
                                                       key=lambda x: -1 if x is None else x['tm_ncc'])
                                               for qn in QUERIES},
           'candidate_pairs': sorted([{k: r.get(k) for k in ('query', 'query_path', 'train_path', 'inliers', 'psnr', 'ncc')}
                                      for r in recs if r['inliers'] >= 15 and r not in dup],
                                     key=lambda r: -r['inliers']),
           'unmatched': {qn: sorted(set(qp) - {r['query_path'] for r in dup if r['query'] == qn})
                         for qn, qp in QUERIES.items() if qn == 'PIPAL'}}
    json.dump(relative_paths(res), open(OUT, 'w'), indent=1)
    print(json.dumps(summary, indent=1))
    print(f'saved -> {OUT}')


if __name__ == '__main__':
    main()
