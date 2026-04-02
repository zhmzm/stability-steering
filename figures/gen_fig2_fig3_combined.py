#!/usr/bin/env python3
"""Generate fig2 (stability distribution) and fig3 (threshold sweep) as separate PDFs for minipage."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Style ──────────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.size': 18,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.labelsize': 19,
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'mathtext.fontset': 'stix',
})

# ── Colors ─────────────────────────────────────────────────
BLUE       = '#4878A8'
GREEN      = '#2E8B57'
RED        = '#C44E52'
LIGHT_RED  = '#FDEAEA'
LIGHT_BLUE = '#D6E4F0'
DARK_BLUE  = '#2B5C8A'

# ── Data ───────────────────────────────────────────────────
with open("claim_ab/paper/paper/figures/figure_data.json") as f:
    D = json.load(f)

scores = np.array(D["stability_distribution"]["scores"])
n_stable = D["stability_distribution"]["n_stable_08"]
n_total  = D["stability_distribution"]["n_total"]
n_unstable = n_total - n_stable
ts = D["threshold_sweep"]

# ============================================================
# Fig 2a: Stability Distribution
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5))

bins = np.arange(-0.05, 1.15, 0.1)
counts, edges, patches = ax.hist(scores, bins=bins, color=BLUE,
                                  edgecolor='white', linewidth=0.8,
                                  alpha=0.90, rwidth=0.88)

for i, patch in enumerate(patches):
    if bins[i] >= 0.75:
        patch.set_facecolor(GREEN)
        patch.set_alpha(0.90)

ax.axvspan(-0.1, 0.75, color=LIGHT_RED, zorder=0)

ax.axvline(0.75, color='#333333', linestyle='--', linewidth=1.2, alpha=0.6, zorder=3)
ax.text(0.77, max(counts) * 0.55, r'$\tau = 0.8$', fontsize=17,
        color='#333333', va='center', zorder=5,
        bbox=dict(facecolor='white', edgecolor='#999999', linewidth=0.5,
                  alpha=0.95, boxstyle='round,pad=0.3'))

idx_max = int(np.argmax(counts))
ax.text(edges[idx_max] + 0.12, counts[idx_max] + 8,
        f'{int(counts[idx_max])} (54%)', ha='center', va='bottom',
        fontsize=16, color='#3a3a3a', fontweight='bold')

unstable_text = f'Unstable: {n_unstable/n_total*100:.1f}%'
ax.text(0.35, 0.78, unstable_text, transform=ax.transAxes,
        fontsize=13, color=RED, fontweight='bold', ha='center', va='top',
        bbox=dict(boxstyle='round,pad=0.35', facecolor=LIGHT_RED,
                  edgecolor=RED, linewidth=1.0, alpha=0.95))

stable_text = f'Stable: {n_stable/n_total*100:.1f}%'
ax.text(0.97, 0.90, stable_text, transform=ax.transAxes,
        fontsize=13, color=GREEN, fontweight='bold', ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                  edgecolor=GREEN, linewidth=1.0, alpha=0.95))

ax.yaxis.grid(True, linestyle=':', linewidth=0.5, color='#CCCCCC', alpha=0.7)
ax.set_axisbelow(True)
ax.set_xlabel(r'Stability Score $s(b)$')
ax.set_ylabel('Number of Boundaries')
ax.set_xlim(-0.1, 1.15)
ax.set_ylim(0, max(counts) * 1.15)

plt.tight_layout()
fig.savefig('overleaf/figures/fig2a_stability_dist.pdf')
print('Saved fig2a_stability_dist.pdf')
plt.close()

# ============================================================
# Fig 2b: Threshold Sweep
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5))

thresholds  = ts["thresholds"]
accuracy    = ts["accuracy"]

ax.plot(thresholds, accuracy, 'o-', color=GREEN, linewidth=2.2, markersize=9,
        markeredgecolor='white', markeredgewidth=1.5, zorder=4)

# Highlight optimal point (τ=0.8)
opt_idx = thresholds.index(0.8)
ax.plot(thresholds[opt_idx], accuracy[opt_idx], 'o', color=GREEN,
        markersize=13, markeredgecolor='#1a5c2e', markeredgewidth=2.2, zorder=5)
ax.annotate(f'{accuracy[opt_idx]:.3f}',
            xy=(thresholds[opt_idx], accuracy[opt_idx]),
            xytext=(14, 10), textcoords='offset points',
            fontsize=17, fontweight='bold', color=GREEN,
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.3))

# SEAL baseline
ax.axhline(0.426, color=DARK_BLUE, linestyle='--', linewidth=1.2, alpha=0.6)
ax.text(0.42, 0.4268, 'SEAL baseline', fontsize=15, color=DARK_BLUE,
        va='bottom', ha='right',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1))

# Random control band
ax.axhspan(0.425 - 0.023, 0.425 + 0.023, color=LIGHT_BLUE, alpha=0.35,
           zorder=0, label=r'Random $\pm 1\sigma$')

ax.yaxis.grid(True, linestyle=':', linewidth=0.5, color='#CCCCCC', alpha=0.7)
ax.set_axisbelow(True)
ax.set_xlabel(r'Stability Threshold $\tau$')
ax.set_ylabel('MATH-500 Accuracy')
ax.set_xlim(-0.05, 0.98)
ax.set_ylim(0.40, 0.52)
ax.legend(frameon=True, fancybox=True, framealpha=0.9, fontsize=15,
          loc='lower right', edgecolor='#CCCCCC')

plt.tight_layout()
fig.savefig('overleaf/figures/fig2b_threshold_sweep.pdf')
print('Saved fig2b_threshold_sweep.pdf')
plt.close()
