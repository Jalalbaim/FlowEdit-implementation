#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_plot_fig7.py

Reads outputs_fig7/metrics_agg.csv and produces a 2-panel plot similar to Fig.7:
  - Left: Stable Diffusion 3
  - Right: FLUX

Run (from repo root):
  python scripts/fig7/03_plot_fig7.py \
    --metrics outputs_fig7/metrics_agg.csv \
    --out outputs_fig7/fig7_repro.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import matplotlib.pyplot as plt


STYLE = {
    # SD3 panel
    ("sd3", "ode_inv"):  dict(marker="^", linestyle="-",  color="#2ca02c", label="ODE Inv."),
    ("sd3", "sdedit"):   dict(marker="o", linestyle="-",  color="#ffbf00", label="SDEdit"),
    ("sd3", "irfds"):    dict(marker="o", linestyle="None", color="#d62728", label="iRFDS"),
    ("sd3", "flowedit"): dict(marker="s", linestyle="-",  color="#1f77b4", label="Ours"),
    # FLUX panel
    ("flux", "ode_inv"):  dict(marker="^", linestyle="--", color="#2ca02c", label="ODE Inv."),
    ("flux", "sdedit"):   dict(marker="o", linestyle="--", color="#ffbf00", label="SDEdit"),
    ("flux", "rf_inv"):   dict(marker="D", linestyle="--", color="#9467bd", label="RF Inv."),
    ("flux", "rf_edit"):  dict(marker="p", linestyle="--", color="#ff7f0e", label="RF Edit"),
    ("flux", "flowedit"): dict(marker="s", linestyle="--", color="#1f77b4", label="Ours"),
}


def plot_panel(ax, df, title: str, xlim=None, ylim=None):
    ax.set_title(title)
    ax.set_xlabel("CLIP →")
    ax.set_ylabel("← LPIPS")

    # plot each method, connect by order_idx
    for method, sub in df.groupby("method"):
        key = (df["model"].iloc[0], method)
        st = STYLE.get(key, dict(marker="o", linestyle="-", color=None, label=method))

        sub = sub.sort_values("order_idx")
        ax.plot(
            sub["clip_t_mean"],
            sub["lpips_mean"],
            marker=st["marker"],
            linestyle=st["linestyle"],
            color=st["color"],
            label=st["label"],
        )

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(True, alpha=0.25)


def dedup_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    new_h, new_l = [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        new_h.append(h)
        new_l.append(l)
    ax.legend(new_h, new_l, loc="best", frameon=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--out_pdf", type=str, default=None)
    args = ap.parse_args()

    metrics_csv = Path(args.metrics).resolve()
    out_png = Path(args.out).resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf = Path(args.out_pdf).resolve() if args.out_pdf else out_png.with_suffix(".pdf")

    df = pd.read_csv(metrics_csv)

    # Normalize model naming
    df["model"] = df["model"].astype(str).str.lower()
    df["method"] = df["method"].astype(str).str.lower()

    df_sd3 = df[df["model"] == "sd3"].copy()
    df_flux = df[df["model"] == "flux"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=160)

    plot_panel(axes[0], df_sd3, "Stable Diffusion 3", xlim=(0.329, 0.355), ylim=(0.12, 0.47))
    dedup_legend(axes[0])

    plot_panel(axes[1], df_flux, "FLUX", xlim=(0.309, 0.345), ylim=(0.10, 0.38))
    dedup_legend(axes[1])

    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
