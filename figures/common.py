"""Shared style and helpers of the figure scripts (figures are written to results/figures/)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from models import super_resolve
from utils.paths import RESULTS_DIR

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_DIR = os.path.join(RESULTS_DIR, 'figures')

STYLE = {
    'font.family': 'DejaVu Sans', 'font.size': 10, 'text.color': 'black',
    'axes.labelcolor': 'black', 'axes.titlecolor': 'black', 'axes.edgecolor': '#666666',
    'xtick.color': 'black', 'ytick.color': 'black', 'axes.facecolor': 'none',
    'figure.facecolor': 'none', 'savefig.transparent': True, 'legend.facecolor': 'white',
    'legend.edgecolor': '#aaaaaa', 'legend.labelcolor': 'black',
}

MODEL_PALETTE = {
    'Bicubic': '#9e9e9e', 'IMDN': '#FF8F00',
    'SRResNet-L1': '#795548', 'SRResNet+WaSP': '#FF7043',
    'RCAN': '#1565C0', 'RCAN+WaSP': '#42A5F5',
    'EDSR': '#2E7D32', 'EDSR+WaSP': '#81C784',
    'RealESRNet': '#00838F', 'RealESRNet+WaSP': '#80DEEA',
    'RealESRGAN': '#B71C1C', 'RealESRGAN+WaSP': '#EF9A9A',
    'SwinIR': '#6A1B9A', 'SwinIR+WaSP': '#CE93D8',
    'SinSR': '#283593', 'SinSR+WaSP': '#9FA8DA',
}


def out_path(name):
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, name)


def savefig(name, **kw):
    path = out_path(name)
    plt.savefig(path, **({'dpi': 220, 'bbox_inches': 'tight', 'transparent': True} | kw))
    plt.close()
    print(f'  saved -> {path}')


@torch.no_grad()
def infer(model, lr, hr, seed=42):
    """SR output aligned to the HR size. The RNG is re-seeded on every call because SinSR
    samples prior noise in each forward pass."""
    torch.manual_seed(seed)
    return super_resolve(model, lr, hr.shape[-2:])


def t2np(t):
    return t.squeeze(0).permute(1, 2, 0).cpu().numpy()


def overlay(ax, txt, loc='tl', fontsize=8.5, alpha=0.50, pad=1.5):
    """Small white label in a translucent black box inside a panel."""
    x, y, ha = (0.03, 0.97, 'left') if loc == 'tl' else (0.97, 0.97, 'right')
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=fontsize, color='white', fontweight='bold',
            va='top', ha=ha, bbox=dict(facecolor='black', alpha=alpha, pad=pad, edgecolor='none'))
