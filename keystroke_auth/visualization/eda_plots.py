
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from keystroke_auth.constants import RAW_FEATURES
from keystroke_auth.data.splits import sample_impostors
from keystroke_auth.visualization._style import save_and_show, BLUE, ORANGE

logger = logging.getLogger(__name__)

def run_eda(df: pd.DataFrame, n_users: int = 5, show: bool = True) -> None:
    """
    Plot keystroke timing histograms for the first n_users, one subplot
    per feature, overlaid by user.

    Interpretation
    --------------
    If two users' distributions overlap heavily on every feature, even a
    perfect classifier cannot separate them — that's a limit of the
    biometric signal itself, not a modelling failure.
    """
    subjects = df["subject"].unique()[:n_users]
    candidate_feats = ["H.period", "H.t", "H.i", "DD.period.t", "DD.t.i", "total_time"]
    plot_features = [f for f in candidate_feats if f in df.columns][:6]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()
    colors = plt.cm.Set2(np.linspace(0, 1, n_users))

    for idx, feat in enumerate(plot_features):
        for subj, col in zip(subjects, colors):
            vals = df[df["subject"] == subj][feat].dropna().values
            axes[idx].hist(vals, bins=25, alpha=0.5, color=col, label=subj, density=True)
        axes[idx].set_title(feat, fontsize=10)
        axes[idx].set_xlabel("Time (s)", fontsize=9)
        axes[idx].set_ylabel("Density", fontsize=9)
        axes[idx].grid(True, alpha=0.2)

    axes[0].legend(fontsize=8, title="User")
    fig.suptitle(
        f"Keystroke timing distributions — first {n_users} users\n"
        "Non-overlapping peaks = distinctive typist = easier to authenticate",
        fontsize=11, fontweight="bold",
    )
    save_and_show(fig, "eda_distributions.png", show=show)


def plot_user_heatmap(df: pd.DataFrame, show: bool = True) -> None:
    """
    Plot pairwise mean-absolute-distance between every pair of users'
    average timing profiles.

    Dark cells indicate users with similar typing rhythm — these pairs
    are the hardest for any model (SVM or otherwise) to separate.

    """
    means = df.groupby("subject")[RAW_FEATURES].mean().values
    dist  = np.mean(np.abs(means[:, None, :] - means[None, :, :]), axis=2)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(dist, cmap="YlOrRd", aspect="auto")
    ax.set_title(
        "Pairwise user distance in keystroke space\n"
        "Dark = similar style (harder to authenticate)",
        fontsize=11,
    )
    ax.set_xlabel("User index")
    ax.set_ylabel("User index")
    fig.colorbar(im, ax=ax, label="Mean |difference| (s)")
    save_and_show(fig, "eda_user_heatmap.png", show=show)


def plot_pca(
    df: pd.DataFrame,
    user: str,
    features: list[str],
    n_impostors: int = 500,
    show: bool = True,
) -> None:
    """
    Project a genuine user's samples and a pool of impostors into 2D PCA
    space to visually assess separability before any model is trained.

    A tight, well-separated genuine cluster predicts a low EER.
    A diffuse cluster overlapping impostors predicts a high EER.

    """
    scaler = StandardScaler()
    X_gen  = df[df["subject"] == user][features].values
    X_imp  = sample_impostors(df, user, features, n=n_impostors)
    X_all  = scaler.fit_transform(np.vstack([X_gen, X_imp]))
    labels = np.array([1] * len(X_gen) + [0] * len(X_imp))

    pca  = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_all)
    ev   = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(X_2d[labels == 0, 0], X_2d[labels == 0, 1],
               c=ORANGE, alpha=0.3, s=15, label="Impostors")
    ax.scatter(X_2d[labels == 1, 0], X_2d[labels == 1, 1],
               c=BLUE, alpha=0.75, s=40, zorder=5,
               edgecolors="white", linewidths=0.4, label=f"Genuine: {user}")
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}% var)")
    ax.set_title(
        f"PCA projection — {user} vs impostors\n"
        "Tight cluster = distinctive typist = lower EER"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    save_and_show(fig, f"pca_{user}.png", show=show)
