#!/usr/bin/env python3
"""Generate polished fig5 behavior probe confidence."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

matplotlib.rcParams.update({
    'font.size': 18,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.labelsize': 19,
    'xtick.labelsize': 15,
    'ytick.labelsize': 17,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'mathtext.fontset': 'stix',
})

# Colors — gradient from muted blue to green
BLUE   = '#4878A8'
GREEN  = '#2E8B57'
GRAY   = '#999999'

# 5-step gradient
cmap = LinearSegmentedColormap.from_list('bg', [BLUE, '#3A8A7A', GREEN], N=5)
bar_colors = [cmap(i / 4) for i in range(5)]

with open("claim_ab/paper/paper/figures/figure_data.json") as f:
    D = json.load(f)

pc = D["probe_confidence"]
bins_labels = pc["bins"]
confidence  = pc["confidence"]
std         = pc["std"]
n_per_bin   = pc["n"]
sem = [s / np.sqrt(n) for s, n in zip(std, n_per_bin)]
n_bins = len(bins_labels)

fig, ax = plt.subplots(1, 1, figsize=(6, 3.8))

x = np.arange(n_bins)

bars = ax.bar(x, confidence, color=bar_colors,
              edgecolor='white', linewidth=0.8,
              alpha=0.92, width=0.62)

# Light error caps
ax.errorbar(x, confidence, yerr=sem, fmt='none',
            ecolor='#777777', capsize=4, linewidth=1.0, capthick=1.0)

# Value labels
for i, (bar, val) in enumerate(zip(bars, confidence)):
    ax.text(bar.get_x() + bar.get_width() / 2, val + sem[i] + 0.008,
            f'{val:.3f}', ha='center', va='bottom', fontsize=15,
            fontweight='bold', color=bar_colors[i])

# E control line
ax.axhline(0.805, color=GRAY, linestyle='--', linewidth=1.0, alpha=0.5)
ax.text(0.42, 0.815, 'E control (0.805)', fontsize=13, color=GRAY,
        va='bottom', ha='center', transform=ax.get_yaxis_transform(),
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=1.5))

# Grid and labels
ax.yaxis.grid(True, linestyle=':', linewidth=0.5, color='#CCCCCC', alpha=0.7)
ax.set_axisbelow(True)
ax.set_xticks(x)
ax.set_xticklabels(bins_labels)
ax.set_xlabel('Stability Bin')
ax.set_ylabel(r'Probe Confidence $P(\mathrm{R{+}T})$')
ax.set_ylim(0.65, 1.02)

plt.tight_layout()
fig.savefig('overleaf/figures/fig5_probe_confidence.pdf')
print('Saved fig5_probe_confidence.pdf')
plt.close()
