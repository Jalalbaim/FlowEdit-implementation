# Flow edit implementation to reproduce Fig.3

import numpy as np
import matplotlib.pyplot as plt

MU_SRC = np.array([[-15*np.sqrt(2), -15*np.sqrt(2)],
                   [ 15*np.sqrt(2),  15*np.sqrt(2)]], dtype=np.float64)

MU_TAR = np.array([[-15.0, 0.0],
                   [  0.0, 15.0]], dtype=np.float64)


def flowedit(X_src, mode, T=60, seed=123):
    """
      Z_src_t = (1-t) X_src + t N
      Z_tar_t = Z + (Z_src_t - X_src)
      Z <- Z + dt * (V_tar - V_src)
    """
    rng = np.random.default_rng(seed)
    Z = X_src.copy()

    delta = MU_TAR[mode] - MU_SRC[mode]
    dt = 1.0 / T

    ts = np.linspace(1.0, 0.0, T, endpoint=True)
    
    # Algorithm

    for t in ts:
        N = rng.standard_normal(X_src.shape)

        Z_src_t = (1.0 - t) * X_src + t * N
        Z_tar_t = Z + (Z_src_t - X_src)

        V_src = X_src - N
        V_tar = (X_src + delta) - N

        Z = Z + dt * (V_tar - V_src)

    return Z


def editing_by_inversion(X_src, mode, sigma=1.8):

    # invert to noise
    n_hat = (X_src - MU_SRC[mode]) / sigma

    k_tar = (n_hat[:, 0] > 0).astype(int)

    # generate target sample
    Z_inv = MU_TAR[k_tar] + sigma * n_hat

    return Z_inv, n_hat


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    N_PER_MODE = 2000
    sigma = 1.8

    # two Gaussian modes
    eps0 = rng.standard_normal((N_PER_MODE, 2))
    eps1 = rng.standard_normal((N_PER_MODE, 2))
    X0 = MU_SRC[0] + sigma * eps0
    X1 = MU_SRC[1] + sigma * eps1
    X_src = np.vstack([X0, X1])

    mode = np.array([0]*N_PER_MODE + [1]*N_PER_MODE)

    Z_fe = flowedit(X_src, mode, T=60, seed=123)
    Z_inv, X_noise = editing_by_inversion(X_src, mode, sigma=sigma)

    # Plot
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 7.0), dpi=200)
    (ax1, ax2), (ax3, ax4) = axs

    def scat(ax, X, title):
        ax.scatter(X[mode==0, 0], X[mode==0, 1], s=6, alpha=0.65)
        ax.scatter(X[mode==1, 0], X[mode==1, 1], s=6, alpha=0.65)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.3, alpha=0.3)
        ax.set_xticks([]); ax.set_yticks([])

    scat(ax1, X_src,   "Source distribution")
    scat(ax2, Z_fe,    "Target distribution\nFlowEdit")
    scat(ax3, X_noise, "Gaussian / noise space")
    scat(ax4, Z_inv,   "Editing-by-inversion")

    all_xy = np.vstack([X_src, Z_fe, Z_inv, MU_SRC, MU_TAR])
    xmin, ymin = all_xy.min(axis=0) - 5
    xmax, ymax = all_xy.max(axis=0) + 5
    for ax in [ax1, ax2, ax4]:
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

    fig.text(0.50, 0.76, "FlowEdit →", ha="center", va="center", fontsize=12)
    fig.text(0.50, 0.28, "inv →",     ha="center", va="center", fontsize=12)

    plt.tight_layout(pad=1.2)
    plt.show()


