from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import matplotlib.pyplot as plt


STYLE = {
    # SD3 panel
    # ("sd3", "ode_inv"):  dict(marker="^", linestyle="-",  color="#2ca02c", label="ODE Inv."),
    ("sd3", "sdedit"):   dict(marker="o", linestyle="-",  color="#ffbf00", label="SDEdit"),
    # ("sd3", "irfds"):    dict(marker="o", linestyle="None", color="#d62728", label="iRFDS"),
    ("sd3", "flowedit"): dict(marker="s", linestyle="-",  color="#1f77b4", label="Ours"),
}


def plot_panel(ax, df, title: str, xlim=None, ylim=None, auto_limits=False):
    ax.set_title(title)
    ax.set_xlabel("CLIP →")
    ax.set_ylabel("← LPIPS")

    # Check if dataframe is empty
    if df.empty:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=ax.transAxes)
        ax.grid(True, alpha=0.25)
        return

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

    if auto_limits and len(df) > 0:
        # Auto-calculate limits with padding
        clip_vals = df["clip_t_mean"].values
        lpips_vals = df["lpips_mean"].values
        clip_range = clip_vals.max() - clip_vals.min()
        lpips_range = lpips_vals.max() - lpips_vals.min()
        clip_pad = clip_range * 0.1 if clip_range > 0 else 0.01
        lpips_pad = lpips_range * 0.1 if lpips_range > 0 else 0.01
        xlim = (clip_vals.min() - clip_pad, clip_vals.max() + clip_pad)
        ylim = (lpips_vals.min() - lpips_pad, lpips_vals.max() + lpips_pad)
    
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

    fig, axes = plt.subplots(1, 1, figsize=(5, 4), dpi=160)

    # Use auto_limits=True to adjust axes to your actual data
    plot_panel(axes, df_sd3, "Stable Diffusion 3", auto_limits=True)
    dedup_legend(axes)

    fig.tight_layout()
    fig.savefig(out_png)
    # fig.savefig(out_pdf)
    print(f"Saved: {out_png}")
    # print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
