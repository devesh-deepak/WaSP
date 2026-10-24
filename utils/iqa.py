"""Full-reference IQA datasets with human scores: TID2013, KADID-10K and PIPAL.

Expected layout under the IQA root (utils.paths.IQA_DATA), as the official archives extract:
    TID2013/    distorted_images/  reference_images/  mos_with_names.txt
    KADID-10K/  images/  dmos.csv                     (or KADID-10K/kadid10k/...)
    PIPAL/      Train_Ref/  Distortion_1..4/  Train_Label/    (or PIPAL/PIPAL/...)

Every loader returns (entries, ref_path_fn, dist_path_fn) with entries a list of
(distorted name, reference name, score); higher score = better quality on all three.
PIPAL is evaluated on a fixed sample of 5,000 of its 23,200 training pairs (seed 42).
"""

import csv
import glob
import os

import numpy as np
import torch.nn.functional as F

from .data import to_tensor as img_tensor  # noqa: F401  (re-exported)

IQA_DATASETS = ('TID2013', 'KADID-10K', 'PIPAL')
PIPAL_SAMPLE = 5000
PIPAL_SEED = 42


def _root(root):
    from utils.paths import IQA_DATA
    return root or IQA_DATA


def _nested(path, sub):
    """Use path/sub if the archive was extracted one level deeper."""
    return os.path.join(path, sub) if os.path.isdir(os.path.join(path, sub)) else path


def load_tid2013(root=None):
    base = os.path.join(_root(root), 'TID2013')
    dist_dir = os.path.join(base, 'distorted_images')
    ref_dir = os.path.join(base, 'reference_images')
    dist_map = {f.lower(): os.path.join(dist_dir, f) for f in os.listdir(dist_dir)}
    ref_map = {f.lower(): os.path.join(ref_dir, f) for f in os.listdir(ref_dir)}
    entries = []
    with open(os.path.join(base, 'mos_with_names.txt')) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                dk = parts[1].lower()
                entries.append((dk, dk.split('_')[0] + '.bmp', float(parts[0])))
    return entries, ref_map.get, dist_map.get


def load_kadid10k(root=None):
    base = _nested(os.path.join(_root(root), 'KADID-10K'), 'kadid10k')
    img_dir = os.path.join(base, 'images')
    entries = []
    with open(os.path.join(base, 'dmos.csv'), newline='') as f:
        for row in csv.DictReader(f):
            entries.append((row['dist_img'].strip(), row['ref_img'].strip(), float(row['dmos'])))
    return entries, (lambda rk: os.path.join(img_dir, rk)), (lambda dk: os.path.join(img_dir, dk))


def load_pipal(root=None):
    """All 23,200 PIPAL training pairs; see pipal_sample() for the 5,000 used in the paper."""
    base = _nested(os.path.join(_root(root), 'PIPAL'), 'PIPAL')
    ref_dir = os.path.join(base, 'Train_Ref')
    dist_map = {}
    for i in range(1, 5):
        dd = os.path.join(base, f'Distortion_{i}')
        if os.path.isdir(dd):
            for f in os.listdir(dd):
                dist_map[f] = os.path.join(dd, f)
    entries = []
    for label_file in sorted(glob.glob(os.path.join(base, 'Train_Label', '*.txt'))):
        ref_name = os.path.basename(label_file).replace('.txt', '.bmp')
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    entries.append((parts[0].strip(), ref_name, float(parts[1].strip())))
    return entries, (lambda rk: os.path.join(ref_dir, rk)), dist_map.get


def pipal_sample(entries, n=PIPAL_SAMPLE, seed=PIPAL_SEED):
    """The fixed 5,000-pair PIPAL sample of the paper (NumPy global seed 42, sorted indices)."""
    np.random.seed(seed)
    idx = np.random.choice(len(entries), min(n, len(entries)), replace=False)
    return [entries[i] for i in sorted(idx)]


def load_iqa(name, root=None):
    """entries, ref_fn, dist_fn of one dataset, with PIPAL restricted to the paper's sample."""
    if name == 'TID2013':
        return load_tid2013(root)
    if name == 'KADID-10K':
        return load_kadid10k(root)
    if name == 'PIPAL':
        entries, ref_fn, dist_fn = load_pipal(root)
        return pipal_sample(entries), ref_fn, dist_fn
    raise ValueError(f'unknown IQA dataset {name!r} (choose from {IQA_DATASETS})')


def iter_pairs(name, root=None):
    """Yield (distorted name, distorted path, reference path, score) of the existing pairs."""
    entries, ref_fn, dist_fn = load_iqa(name, root)
    for dk, rk, score in entries:
        dp, rp = dist_fn(dk), ref_fn(rk)
        if dp and rp and os.path.exists(dp) and os.path.exists(rp):
            yield dk, dp, rp, score


def align(d, r):
    """Resize the distorted image to the reference size (bilinear) if they differ."""
    if d.shape[-2:] != r.shape[-2:]:
        d = F.interpolate(d, size=r.shape[-2:], mode='bilinear', align_corners=False)
    return d
